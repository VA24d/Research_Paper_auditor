"""Checks what a paper says about its evaluation data against its repository.

Covers the keys `data_split` ("80/10/10"), `test_fraction`, `validation_fraction` and
`dataset`. Evidence is taken from the first of these that says anything about the key:

1. code and configs (`auditor.agents.code.extract`);
2. split-list files such as `splits/test.txt`;
3. a Hugging Face dataset card at the pinned revision: the repository's own when it is a
   dataset, or the card of a dataset the code loads with `load_dataset`;
4. the Hub's size service, which describes the latest revision only and so can raise a
   suspicion but never a mismatch.

Without evidence the task is unverifiable with a reason (decision D6); a disagreement is
a `data_split_mismatch` finding that quotes the paper and points at the file and line.
"""

import asyncio
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

from auditor.agents.code.extract import (
    parse_card,
    scan_json,
    scan_python,
    scan_split_lists,
    scan_yaml,
)
from auditor.agents.code.extract.normalise import percents_match, split_parts
from auditor.infra.context import RunContext
from auditor.infra.http import HttpError
from auditor.infra.snapshots import (
    Snapshot,
    SnapshotUnavailableError,
    SourceError,
    SourceKind,
    fetch_snapshot,
    parse_source,
)
from auditor.schemas import (
    AuditTask,
    CandidateFinding,
    ErrorClass,
    Evidence,
    EvidenceKind,
    ExtractedSetting,
    RepositoryRef,
    RepositorySource,
    SentenceAnchor,
    SourceRef,
    TaskOutcome,
    TaskResult,
)
from auditor.schemas import SourceKind as EvidenceSourceKind

logger = logging.getLogger(__name__)

DATA_KEYS = frozenset({"data_split", "test_fraction", "validation_fraction", "dataset"})
# Data files that hold the rows a split is made of; they are never downloaded by default.
DATA_SUFFIXES = frozenset(
    {".csv", ".tsv", ".jsonl", ".parquet", ".arrow", ".feather", ".h5", ".hdf5", ".npz", ".npy"}
)
SIZE_SERVICE_PATH = "datasets-server.huggingface.co/size"
# Datasets followed from `load_dataset` calls per repository.
MAX_FOLLOWED_DATASETS = 3
_TRAIN_BASE = re.compile(r"\bof\s+(?:the\s+)?train(?:ing)?\b", re.IGNORECASE)
_VALIDATION = ("validation", "valid", "dev")


class Tier(StrEnum):
    CODE = "code"
    SPLIT_LIST = "split_list"
    HF_CARD = "hf_card"
    HF_SIZE_LATEST = "hf_size_latest"


@dataclass(frozen=True)
class Candidate:
    tier: Tier
    setting: ExtractedSetting
    snapshot: Snapshot


_ORDER = (Tier.CODE, Tier.SPLIT_LIST, Tier.HF_CARD, Tier.HF_SIZE_LATEST)
_TIER_NAMES = {
    Tier.CODE: "the repository's code",
    Tier.SPLIT_LIST: "the repository's split-list files",
    Tier.HF_CARD: "the Hugging Face dataset card",
    Tier.HF_SIZE_LATEST: "the Hugging Face Hub's size service",
}
_cache: dict[str, list[Candidate]] = {}
_locks: dict[str, asyncio.Lock] = {}


async def check_data_setting(task: AuditTask, ctx: RunContext) -> TaskResult:
    """The result of a code task whose key is in DATA_KEYS."""
    setting, repository = task.setting, task.repository
    if setting is None or repository is None or setting.key not in DATA_KEYS:
        raise ValueError(f"{task.task_id} is not a code task about the evaluation data")
    try:
        snapshot = await fetch_snapshot(repository, ctx.http, ctx.settings)
    except (SnapshotUnavailableError, SourceError) as exc:
        return TaskResult.for_task(
            task, TaskOutcome.UNVERIFIABLE, reason=f"the repository could not be fetched: {exc}"
        )
    sentence = _sentence(task, ctx)
    candidates = await _static_candidates(snapshot)
    if not _related(setting, sentence, candidates, (Tier.CODE, Tier.SPLIT_LIST, Tier.HF_CARD)):
        candidates = [*candidates, *await _followed_candidates(snapshot, candidates, ctx)]
    return _decide(task, setting, sentence, snapshot, candidates)


def _sentence(task: AuditTask, ctx: RunContext) -> str:
    anchor = task.anchor
    assert isinstance(anchor, SentenceAnchor)
    return ctx.paper.sentence(anchor.sentence_id).text


async def _static_candidates(snapshot: Snapshot) -> list[Candidate]:
    """Evidence inside the snapshot itself, read once per snapshot."""
    key = str(snapshot.root)
    async with _locks.setdefault(key, asyncio.Lock()):
        if key not in _cache:
            _cache[key] = await asyncio.to_thread(_scan, snapshot)
        return _cache[key]


def _scan(snapshot: Snapshot) -> list[Candidate]:
    manifest = snapshot.manifest
    ref = RepositoryRef(url=manifest.url, commit=manifest.revision)
    candidates = []
    lists = {}
    for file in manifest.files:
        suffix = PurePosixPath(file.path).suffix.lower()
        text = snapshot.read_text(file.path)
        found: list[ExtractedSetting] = []
        if suffix == ".py":
            found = scan_python(file.path, text, ref)
        elif suffix in {".yaml", ".yml"}:
            found = scan_yaml(file.path, text, ref)
        elif suffix == ".json":
            found = scan_json(file.path, text, ref)
        elif suffix == ".txt":
            lists[file.path] = text
        candidates += [Candidate(Tier.CODE, s, snapshot) for s in found]
    candidates += [Candidate(Tier.SPLIT_LIST, s, snapshot) for s in scan_split_lists(lists, ref)]
    candidates += _dataset_candidates(snapshot, None)
    return candidates


def _is_hf_dataset(snapshot: Snapshot) -> bool:
    manifest = snapshot.manifest
    return manifest.source is SourceKind.HUGGINGFACE and "/datasets/" in manifest.url


def _dataset_candidates(snapshot: Snapshot, config: str | None) -> list[Candidate]:
    """The card and the Hub's sizes of a Hugging Face dataset snapshot."""
    if not _is_hf_dataset(snapshot):
        return []
    manifest = snapshot.manifest
    ref = RepositoryRef(url=manifest.url, commit=manifest.revision)
    found = []
    if "README.md" in {f.path for f in manifest.files}:
        card = parse_card(snapshot.read_text("README.md"))
        if card is not None:
            found += [
                Candidate(Tier.HF_CARD, s, snapshot)
                for s in card.settings("README.md", ref, config)
            ]
    found += [Candidate(Tier.HF_SIZE_LATEST, s, snapshot) for s in _size_settings(snapshot, config)]
    return found


def _size_settings(snapshot: Snapshot, config: str | None) -> list[ExtractedSetting]:
    size = snapshot.manifest.metadata.get("dataset_size")
    inner = size.get("size") if isinstance(size, dict) else None
    splits = inner.get("splits") if isinstance(inner, dict) else None
    if not isinstance(splits, list):
        return []
    by_config: dict[str, dict[str, int]] = {}
    for row in splits:
        rows = row.get("num_rows") if isinstance(row, dict) else None
        if isinstance(row, dict) and isinstance(rows, int):
            config_rows = by_config.setdefault(str(row.get("config", "default")), {})
            config_rows[str(row.get("split"))] = rows
    if config is None and len(by_config) == 1:
        config = next(iter(by_config))
    counts = by_config.get(config or "", {})
    if "train" not in counts or "test" not in counts:
        return []
    ordered = ["train", *[s for s in _VALIDATION if s in counts][:1], "test"]
    total = sum(counts[s] for s in ordered)
    value = "/".join(f"{counts[s] / total * 100:.4g}" for s in ordered)
    when = snapshot.manifest.fetched_at.date().isoformat()
    return [
        ExtractedSetting(
            key="data_split",
            raw_name="Hub size service",
            raw_value=" / ".join(f"{s} {counts[s]}" for s in ordered)
            + f" (latest revision as of {when})",
            value=value,
            context=config,
            source=RepositorySource(
                repository=RepositoryRef(url=snapshot.manifest.url), path=SIZE_SERVICE_PATH, line=1
            ),
        )
    ]


async def _followed_candidates(
    snapshot: Snapshot, candidates: Sequence[Candidate], ctx: RunContext
) -> list[Candidate]:
    """Cards of the Hugging Face datasets the code loads by name."""
    names: list[tuple[str, str | None]] = []
    for candidate in candidates:
        setting = candidate.setting
        if (
            candidate.tier is Tier.CODE
            and setting.key == "dataset"
            and setting.raw_name == "load_dataset"
            and isinstance(setting.value, str)
            and (setting.value, setting.context) not in names
        ):
            names.append((setting.value, setting.context))
    found: list[Candidate] = []
    for name, config in names[:MAX_FOLLOWED_DATASETS]:
        try:
            spec = parse_source(f"https://huggingface.co/datasets/{name}")
            followed = await fetch_snapshot(spec, ctx.http, ctx.settings)
        except (SourceError, SnapshotUnavailableError, HttpError) as exc:
            logger.info("could not follow dataset %s: %s", name, exc)
            continue
        found += _dataset_candidates(followed, config)
    return found


# --- comparing -------------------------------------------------------------------------


def _components(setting: ExtractedSetting) -> dict[str, float]:
    """A split setting as shares of train, validation and test, in percent."""
    if setting.key == "test_fraction" and isinstance(setting.value, int | float):
        return {"test": float(setting.value)}
    if setting.key == "validation_fraction" and isinstance(setting.value, int | float):
        return {"validation": float(setting.value)}
    if setting.key == "data_split" and isinstance(setting.value, str):
        parts = split_parts(setting.value)
        names = {2: ("train", "test"), 3: ("train", "validation", "test")}.get(len(parts))
        return dict(zip(names, parts, strict=True)) if names else {}
    return {}


def _base(setting: ExtractedSetting, sentence: str | None) -> str:
    """What a share is a share of: the training split, or the whole dataset."""
    if sentence is not None:
        return (
            "train"
            if setting.key == "validation_fraction" and _TRAIN_BASE.search(sentence)
            else "all"
        )
    return "train" if (setting.context or "").startswith("of the train") else "all"


def _name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _agrees(paper: ExtractedSetting, sentence: str, repo: ExtractedSetting) -> bool | None:
    """Whether a repository setting agrees with the paper's, or None if it is about
    something else."""
    if paper.key == "dataset" or repo.key == "dataset":
        if paper.key != repo.key:
            return None
        wanted = _name(paper.value)
        offered = {
            _name(repo.value),
            _name(repo.context or ""),
            _name(str(repo.value).split("/")[-1]),
        }
        offered.discard("")
        return any(
            name == wanted
            or (min(len(name), len(wanted)) >= 3 and (name in wanted or wanted in name))
            for name in offered
        )
    if _base(paper, sentence) != _base(repo, None):
        return None
    ours, theirs = _components(paper), _components(repo)
    shared = ours.keys() & theirs.keys()
    if not shared:
        return None
    return all(percents_match(ours[k], theirs[k]) for k in shared)


def _related(
    paper: ExtractedSetting, sentence: str, candidates: Sequence[Candidate], tiers: Sequence[Tier]
) -> bool:
    return any(
        c.tier in tiers and _agrees(paper, sentence, c.setting) is not None for c in candidates
    )


def _decide(
    task: AuditTask,
    paper: ExtractedSetting,
    sentence: str,
    snapshot: Snapshot,
    candidates: Sequence[Candidate],
) -> TaskResult:
    for tier in _ORDER:
        related = [
            (c, agrees)
            for c in candidates
            if c.tier is tier and (agrees := _agrees(paper, sentence, c.setting)) is not None
        ]
        if not related:
            continue
        if any(agrees for _, agrees in related):
            return TaskResult.for_task(task, TaskOutcome.CHECKED)
        disagreeing = [c for c, _ in related]
        values = {str(c.setting.value) for c in disagreeing}
        latest_only = tier is Tier.HF_SIZE_LATEST
        finding = CandidateFinding.for_task(
            task,
            0,
            error_class=ErrorClass.DATA_SPLIT_MISMATCH,
            summary=(
                f"The paper states {_describe(paper)}; {_TIER_NAMES[tier]} gives "
                + ", ".join(_describe(c.setting) for c in disagreeing[:3])
            ),
            explanation=_explain(tier, disagreeing),
            evidence=[_paper_quote(task, sentence), *_repository_evidence(disagreeing)],
            uncertain=latest_only or len(values) > 1 or paper.key == "dataset",
        )
        if latest_only:
            when = disagreeing[0].snapshot.manifest.fetched_at.date().isoformat()
            return TaskResult.for_task(
                task,
                TaskOutcome.UNVERIFIABLE,
                reason=(
                    "only the Hugging Face Hub's sizes for the latest revision (as of "
                    f"{when}) were available; they may not describe the pinned revision"
                ),
                findings=[finding],
            )
        return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=[finding])
    return TaskResult.for_task(
        task, TaskOutcome.UNVERIFIABLE, reason=_why_unverifiable(paper, snapshot)
    )


def _describe(setting: ExtractedSetting) -> str:
    value = setting.value
    where = ""
    if isinstance(setting.source, RepositorySource) and setting.source.path != SIZE_SERVICE_PATH:
        where = f" ({setting.source.path}:{setting.source.line})"
    match setting.key:
        case "data_split":
            text = f"a {value} split"
        case "test_fraction":
            text = (
                f"{value:g}% test data" if isinstance(value, int | float) else f"test data {value}"
            )
        case "validation_fraction":
            text = (
                f"{value:g}% validation data"
                if isinstance(value, int | float)
                else f"validation data {value}"
            )
            if setting.context:
                text += f" {setting.context}"
        case _:
            text = f"dataset {value}" + (f" ({setting.context})" if setting.context else "")
    return text + where


def _explain(tier: Tier, candidates: Sequence[Candidate]) -> str:
    first = candidates[0].snapshot.manifest
    text = f"Evidence from {_TIER_NAMES[tier]} in {first.url} at revision {first.revision}."
    if len({str(c.setting.value) for c in candidates}) > 1:
        text += " The repository holds several different values and none matches the paper."
    return text


def _paper_quote(task: AuditTask, sentence: str) -> Evidence:
    return Evidence(
        kind=EvidenceKind.QUOTE,
        source=SourceRef(kind=EvidenceSourceKind.SUBMISSION, id=task.paper_id),
        text=sentence,
    )


def _repository_evidence(candidates: Sequence[Candidate]) -> list[Evidence]:
    evidence = []
    seen = set()
    for candidate in candidates:
        source = candidate.setting.source
        assert isinstance(source, RepositorySource)
        if (source.path, source.line) in seen:
            continue
        seen.add((source.path, source.line))
        snapshot = candidate.snapshot
        ref = SourceRef(
            kind=EvidenceSourceKind.REPOSITORY, id=snapshot.source_id, url=snapshot.manifest.url
        )
        if candidate.tier is Tier.CODE:
            lines = snapshot.read_text(source.path).splitlines()
            text = lines[source.line - 1] if source.line <= len(lines) else ""
            evidence.append(
                Evidence(
                    kind=EvidenceKind.CODE,
                    source=ref,
                    text=text if text.strip() else candidate.setting.raw_value,
                    path=source.path,
                    line_start=source.line,
                )
            )
        else:
            evidence.append(
                Evidence(
                    kind=EvidenceKind.LOOKUP,
                    source=ref,
                    text=f"{candidate.setting.raw_name}: {candidate.setting.raw_value}",
                    path=source.path,
                    line_start=source.line,
                )
            )
    return evidence


def _why_unverifiable(paper: ExtractedSetting, snapshot: Snapshot) -> str:
    manifest = snapshot.manifest
    if paper.key == "dataset":
        reason = "no dataset is named in the repository's code, configs or dataset card"
    else:
        reason = (
            "no split settings found in the repository's code, configs, split lists or dataset card"
        )
        has_data = any(
            PurePosixPath(s.path).suffix.lower() in DATA_SUFFIXES
            or s.reason == "stored in large file storage"
            for s in manifest.skipped
        ) or bool(manifest.metadata.get("files"))
        if has_data:
            reason += "; row counts need the data files, which are not downloaded"
    if manifest.incomplete:
        reason += f" (the snapshot is incomplete: {'; '.join(manifest.incomplete)})"
    return reason
