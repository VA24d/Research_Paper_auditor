"""Building and scoring set C, the paper-code consistency set (P3.8, P3.9, decision D18).

- `candidates`: every setting the planner finds in a paper, with what the code agent finds
  in the repository and which kinds of evidence exist, to pick instances from.
- `inject_paper` / `inject_code`: make an error on the paper side (a copy of the parsed
  paper) or the code side (a patched copy of the repository), with its gold label and an
  injection record saying which evidence the label rests on.
- `clean_label`: a clean control item.
- `validate`: the D18 checks before labels are frozen.
- `score_code_module`: precision, recall and F1 of the code and data classes.
"""

import random
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from auditor.agents.code import check
from auditor.agents.code.data_checks import (
    COUNT_KEYS,
    DATA_FILE_KEYS,
    DATA_KEYS,
    ERROR_CLASSES,
    Tier,
    data_values_agree,
    repository_evidence,
    static_candidates,
)
from auditor.agents.code.settings_check import values_agree
from auditor.infra.context import RunContext
from auditor.infra.snapshots import SnapshotUnavailableError, SourceError, fetch_snapshot
from auditor.planner.tables import paper_settings
from auditor.planner.tasks import code_tasks
from auditor.schemas import (
    DocumentAnchor,
    ErrorClass,
    Evidence,
    EvidenceKind,
    ExtractedSetting,
    GoldLabel,
    PaperSource,
    PaperVariant,
    ParsedPaper,
    Prediction,
    RepositoryRef,
    RepositorySource,
    SentenceAnchor,
    SettingTarget,
    SourceKind,
    SourceRef,
    TaskOutcome,
    VerificationState,
)
from auditor.schemas.evaluation import matches

# The share of labels a second person must review before freezing (decision D18).
REVIEW_SHARE = 0.2
CODE_CLASSES = frozenset(
    {
        ErrorClass.SETTING_MISMATCH,
        ErrorClass.DATA_SPLIT_MISMATCH,
        ErrorClass.METRIC_MISMATCH,
        ErrorClass.DATASET_SIZE_MISMATCH,
        ErrorClass.LABEL_DISTRIBUTION_MISMATCH,
        ErrorClass.DATA_LEAKAGE,
        ErrorClass.METHOD_MISMATCH,
    }
)


class EvidenceType(StrEnum):
    """What a label's ground truth rests on, recorded so recall can be split by it."""

    CODE = "code"
    SPLIT_LIST = "split_list"
    HF_CARD = "hf_card"
    DATA_ANALYSIS = "data_analysis"


_TIER_EVIDENCE = {
    Tier.CODE: EvidenceType.CODE,
    Tier.SPLIT_LIST: EvidenceType.SPLIT_LIST,
    Tier.HF_CARD: EvidenceType.HF_CARD,
    Tier.DATA: EvidenceType.DATA_ANALYSIS,
}


def error_class_for(key: str) -> ErrorClass:
    if key == "eval_metric":
        return ErrorClass.METRIC_MISMATCH
    if key == "method":
        return ErrorClass.METHOD_MISMATCH
    return ERROR_CLASSES.get(key, ErrorClass.SETTING_MISMATCH)


class InjectionRecord(BaseModel):
    """One line of the injection log, kept next to the gold labels."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label_id: str
    paper_id: str
    key: str
    side: Literal["paper", "code"]
    evidence_type: EvidenceType
    original: str
    injected: str
    path: str | None = None
    line: int | None = None
    # Where a paper-side injection was made, so it can be re-applied to a fresh parse.
    sentence_id: str | None = None
    start: int | None = None
    end: int | None = None


class CandidateRow(BaseModel):
    """What the code agent makes of one paper setting (saved per pair while collecting)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    sentence_id: str
    paper_value: str
    outcome: str
    agrees: bool | None
    reason: str | None
    repository_values: tuple[str, ...]
    evidence_types: tuple[EvidenceType, ...]
    anchor: SentenceAnchor
    setting: ExtractedSetting
    # Repository evidence for the values that agree with the paper (empty if none do).
    agreeing_evidence: tuple[Evidence, ...] = ()
    # Every repository value found for the key, to check a mutation is really an error.
    repository_settings: tuple[ExtractedSetting, ...] = ()


class Pair(BaseModel):
    """One paper and its code, from SciCoQA (CC BY 4.0, Baumgärtner et al., ACL 2026)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str
    arxiv_id: str
    pdf_url: str
    repository: str
    commit: str
    scicoqa_ids: tuple[str, ...]
    real_issue_ids: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    pdf_sha256: str | None = None


_ARXIV_PDF = re.compile(r"arxiv\.org/pdf/(?P<id>\d{4}\.\d{4,5})(?P<version>v\d+)")
_GITHUB_TREE = re.compile(r"^https://github\.com/(?P<repo>[^/]+/[^/]+)/tree/(?P<sha>[0-9a-f]{40})$")


def pool_from_scicoqa(rows: Sequence[Mapping[str, object]]) -> list[Pair]:
    """Paper and code pairs from SciCoQA rows: versioned arXiv PDFs and GitHub repositories
    pinned to a commit, with the discrepancy ids reported for each. Pairs with a real
    (not synthetic) discrepancy come first; the order is otherwise stable."""
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in rows:
        pdf = _ARXIV_PDF.search(str(row.get("paper_url_versioned") or ""))
        code = _GITHUB_TREE.match(str(row.get("code_url_versioned") or ""))
        if pdf is None or code is None:
            continue
        key = (pdf.group("id") + pdf.group("version"), code.group(0))
        grouped.setdefault(key, []).append(row)
    pairs = []
    for (pair_id, tree), members in grouped.items():
        code = _GITHUB_TREE.match(tree)
        assert code is not None
        real = [str(m["discrepancy_id"]) for m in members if m.get("origin_type") != "synthetic"]
        pairs.append(
            Pair(
                pair_id=pair_id,
                arxiv_id=pair_id.split("v")[0],
                pdf_url=f"https://arxiv.org/pdf/{pair_id}",
                repository=f"https://github.com/{code.group('repo')}",
                commit=code.group("sha"),
                scicoqa_ids=tuple(sorted(str(m["discrepancy_id"]) for m in members)),
                real_issue_ids=tuple(sorted(real)),
                categories=tuple(
                    sorted(
                        {
                            str(m["discrepancy_category"])
                            for m in members
                            if m.get("discrepancy_category")
                        }
                    )
                ),
            )
        )
    # One pair per paper: when SciCoQA gives a paper at two commits or in two repositories,
    # the first in order (real issues first, then by repository and commit) is kept.
    kept: dict[str, Pair] = {}
    for pair in sorted(
        pairs, key=lambda p: (not p.real_issue_ids, p.pair_id, p.repository, p.commit)
    ):
        kept.setdefault(pair.pair_id, pair)
    return list(kept.values())


def apply_injections(paper: ParsedPaper, records: Sequence[InjectionRecord]) -> ParsedPaper:
    """Re-apply recorded paper-side injections to a fresh parse of the original, in the
    order they were made, checking each span still holds the original value."""
    current = paper
    for record in records:
        if record.side != "paper" or record.sentence_id is None:
            continue
        assert record.start is not None
        assert record.end is not None
        anchor = SentenceAnchor(sentence_id=record.sentence_id, start=record.start, end=record.end)
        found = current.sentence(record.sentence_id).text[record.start : record.end]
        if found != record.original:
            raise ValueError(
                f"{record.label_id}: expected {record.original!r} at {record.sentence_id}"
                f"[{record.start}:{record.end}], found {found!r}; was the paper parsed with "
                "another GROBID version?"
            )
        current, _, _ = inject_paper(
            current,
            anchor,
            record.injected,
            key=record.key,
            rationale="rebuild",
            evidence_type=record.evidence_type,
            evidence=[_REBUILD_EVIDENCE],
        )
    return current


async def candidates(
    paper: ParsedPaper, repository: RepositoryRef, ctx: RunContext
) -> list[CandidateRow]:
    """What the code agent makes of each setting the paper states, against the
    repository: the starting point for choosing set C instances."""
    try:
        snapshot = await fetch_snapshot(repository, ctx.http, ctx.settings)
        found = await static_candidates(snapshot)
    except (SnapshotUnavailableError, SourceError):
        found = []
    rows = []
    for task in code_tasks(paper, repository, paper_settings(paper)):
        setting = task.setting
        assert setting is not None
        assert isinstance(task.anchor, SentenceAnchor)
        result = await check(task, ctx)
        same_key = [c for c in found if c.setting.key == setting.key]
        values = []
        for candidate in same_key:
            source = candidate.setting.source
            assert isinstance(source, RepositorySource)
            values.append(f"{candidate.setting.value} ({source.path}:{source.line})")
        types = sorted({_TIER_EVIDENCE[c.tier] for c in same_key if c.tier in _TIER_EVIDENCE})
        if ctx.settings.data_analysis_enabled and setting.key in DATA_FILE_KEYS | COUNT_KEYS:
            types = sorted({*types, EvidenceType.DATA_ANALYSIS})
        checked = result.outcome is TaskOutcome.CHECKED
        sentence = paper.sentence(task.anchor.sentence_id).text
        agreeing = [c for c in same_key if _agree(setting, sentence, c.setting)]
        rows.append(
            CandidateRow(
                key=setting.key,
                sentence_id=task.anchor.sentence_id,
                paper_value=setting.raw_value,
                outcome=result.outcome.value,
                agrees=(not result.findings) if checked else None,
                reason=result.reason,
                repository_values=tuple(values),
                evidence_types=tuple(types),
                anchor=task.anchor,
                setting=setting,
                agreeing_evidence=tuple(repository_evidence(agreeing)),
                repository_settings=tuple(c.setting for c in same_key),
            )
        )
    return rows


def _agree(paper: ExtractedSetting, sentence: str, repo: ExtractedSetting) -> bool:
    if paper.key in DATA_KEYS:
        return bool(data_values_agree(paper, sentence, repo))
    return paper.key == repo.key and values_agree(paper, repo)


_REBUILD_EVIDENCE = Evidence(
    kind=EvidenceKind.LOOKUP,
    source=SourceRef(kind=SourceKind.REPOSITORY, id="rebuild"),
    text="re-applied from the injection log",
)


def _label_id(paper_id: str, key: str, anchor: SentenceAnchor, extra: str) -> str:
    digest = sha256(f"{paper_id}|{key}|{anchor.model_dump_json()}|{extra}".encode()).hexdigest()
    return f"c-{digest[:12]}"


def inject_paper(
    paper: ParsedPaper,
    anchor: SentenceAnchor,
    new_value: str,
    *,
    key: str,
    rationale: str,
    evidence_type: EvidenceType,
    evidence: Sequence[Evidence],
) -> tuple[ParsedPaper, GoldLabel, InjectionRecord]:
    """Replace the value at `anchor` in a copy of the paper. Citation markers later in the
    sentence move with the text, and the copy is marked as injected."""
    if anchor.start is None or anchor.end is None:
        raise ValueError("a paper-side injection needs an anchor with a span")
    sentence = paper.sentence(anchor.sentence_id)
    original = sentence.text[anchor.start : anchor.end]
    if original == new_value:
        raise ValueError(f"the new value is the same as the old one ({original})")
    text = sentence.text[: anchor.start] + new_value + sentence.text[anchor.end :]
    delta = len(new_value) - len(original)
    data = paper.model_dump()
    data["variant"] = PaperVariant.INJECTED.value
    for row in data["sentences"]:
        if row["id"] == anchor.sentence_id:
            row["text"] = text
    for mention in data["citations"]:
        if mention["sentence_id"] == anchor.sentence_id and mention["start"] >= anchor.end:
            mention["start"] += delta
            mention["end"] += delta
    injected = ParsedPaper.model_validate(data)
    new_anchor = SentenceAnchor(
        sentence_id=anchor.sentence_id, start=anchor.start, end=anchor.start + len(new_value)
    )
    label_id = _label_id(paper.paper_id, key, anchor, new_value)
    label = GoldLabel(
        label_id=label_id,
        paper_id=paper.paper_id,
        variant=PaperVariant.INJECTED,
        error_class=error_class_for(key),
        anchor=new_anchor,
        target=SettingTarget(key=key),
        rationale=rationale,
        evidence=tuple(evidence),
    )
    record = InjectionRecord(
        label_id=label_id,
        paper_id=paper.paper_id,
        key=key,
        side="paper",
        evidence_type=evidence_type,
        original=original,
        injected=new_value,
        sentence_id=anchor.sentence_id,
        start=anchor.start,
        end=anchor.end,
    )
    return injected, label, record


def inject_code(
    repository: Path, destination: Path, path: str, line: int, old: str, new: str
) -> Path:
    """A copy of `repository` (without `.git`) in which `old` is replaced by `new` on one
    line. The text must occur exactly once on that line."""
    target = repository / path
    lines = target.read_text(encoding="utf-8").splitlines(keepends=True)
    if not 1 <= line <= len(lines) or lines[line - 1].count(old) != 1:
        raise ValueError(f"{path} line {line} does not contain {old!r} exactly once")
    shutil.copytree(repository, destination, ignore=shutil.ignore_patterns(".git"))
    lines[line - 1] = lines[line - 1].replace(old, new)
    (destination / path).write_text("".join(lines), encoding="utf-8", newline="")
    return destination


def clean_label(
    paper: ParsedPaper,
    anchor: SentenceAnchor,
    key: str,
    evidence: Sequence[Evidence],
    rationale: str,
) -> GoldLabel:
    """A clean control: the paper's value is the one the experiment uses."""
    return GoldLabel(
        label_id=_label_id(paper.paper_id, key, anchor, f"none-{paper.variant.value}"),
        paper_id=paper.paper_id,
        variant=paper.variant,
        error_class="none",
        anchor=anchor,
        target=SettingTarget(key=key),
        rationale=rationale,
        evidence=tuple(evidence),
    )


def validate(
    labels: Sequence[GoldLabel],
    records: Sequence[InjectionRecord],
    *,
    data_mode: bool,
    method_ids: frozenset[str] = frozenset(),
) -> list[str]:
    """Problems that stop set C from being frozen (decision D18). `data_mode` says whether
    the experiments run with --analyse-data."""
    problems = []
    by_id = {r.label_id: r for r in records}
    ids = Counter(label.label_id for label in labels)
    problems += [f"{i}: duplicate label id" for i, n in ids.items() if n > 1]
    for label in labels:
        if not isinstance(label.target, SettingTarget):
            problems.append(f"{label.label_id}: set C labels target a setting key")
            continue
        key = label.target.key
        if label.is_error:
            expected = error_class_for(key)
            if label.error_class != expected:
                problems.append(
                    f"{label.label_id}: {key} errors are {expected.value}, not {label.error_class}"
                )
            if key == "method":
                source_id = label.label_id.removeprefix("m-").rsplit("-", 1)[0]
                if source_id not in method_ids:
                    problems.append(f"{label.label_id}: no method issue record")
                continue
            record = by_id.get(label.label_id)
            if record is None:
                problems.append(f"{label.label_id}: no injection record")
                continue
            needs_data = record.evidence_type is EvidenceType.DATA_ANALYSIS or key in DATA_FILE_KEYS
            if needs_data and not data_mode:
                problems.append(
                    f"{label.label_id}: rests on the data files; run the experiments with "
                    "--analyse-data or drop the label"
                )
    for record in records:
        if record.label_id not in ids:
            problems.append(f"{record.label_id}: injection record without a label")
    reviewed = sum(1 for label in labels if label.reviewed_by)
    if labels and reviewed < REVIEW_SHARE * len(labels):
        problems.append(
            f"only {reviewed} of {len(labels)} labels have a second review; D18 asks for 20%"
        )
    return problems


@dataclass
class ClassScore:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        found = self.true_positives + self.false_positives
        return self.true_positives / found if found else 0.0

    @property
    def recall(self) -> float:
        wanted = self.true_positives + self.false_negatives
        return self.true_positives / wanted if wanted else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


@dataclass
class CodeScores:
    per_class: dict[ErrorClass, ClassScore] = field(default_factory=dict)
    false_positive_rate: float = 0.0
    unlabelled: int = 0
    # The same flag raised on both copies of a paper counts once here.
    unlabelled_unique: int = 0

    @property
    def macro_f1(self) -> float:
        """Over the classes the labels contain (decision D4)."""
        scores = [s for s in self.per_class.values() if s.true_positives + s.false_negatives]
        return sum(s.f1 for s in scores) / len(scores) if scores else 0.0


def score_code_module(
    labels: Sequence[GoldLabel],
    predictions: Sequence[Prediction],
    *,
    before_verifier: bool = False,
) -> CodeScores:
    """Scores of the code and data classes (decision D3 matching, one to one; only
    positive predictions count). A positive prediction matching a clean control is a false
    positive; one matching no label is unlabelled, not a false positive."""
    scores = CodeScores()
    in_scope = [
        label for label in labels if not label.is_error or label.error_class in CODE_CLASSES
    ]
    for label in in_scope:
        if label.is_error:
            scores.per_class.setdefault(ErrorClass(label.error_class), ClassScore())
    unmatched = list(in_scope)
    clean_hit = 0
    unlabelled: set[tuple[str, ErrorClass, str, str]] = set()

    def positive(prediction: Prediction) -> bool:
        # Before the verifier, every finding the specialists raised counts, except dropped.
        if before_verifier:
            return prediction.state is not VerificationState.DROPPED
        return prediction.is_positive

    for prediction in (p for p in predictions if positive(p) and p.error_class in CODE_CLASSES):
        matched = next((lbl for lbl in unmatched if matches(prediction, lbl)), None)
        if matched is None:
            scores.unlabelled += 1
            unlabelled.add(
                (
                    prediction.paper_id,
                    prediction.error_class,
                    prediction.anchor.model_dump_json(include={"kind", "sentence_id"}),
                    prediction.target.model_dump_json(),
                )
            )
            continue
        score = scores.per_class.setdefault(prediction.error_class, ClassScore())
        unmatched.remove(matched)
        label = matched
        if not label.is_error:
            clean_hit += 1
            score.false_positives += 1
        elif label.error_class == prediction.error_class:
            score.true_positives += 1
        else:
            score.false_positives += 1
            scores.per_class[ErrorClass(label.error_class)].false_negatives += 1
    for label in unmatched:
        if label.is_error:
            scores.per_class[ErrorClass(label.error_class)].false_negatives += 1
    clean = sum(1 for label in in_scope if not label.is_error)
    scores.false_positive_rate = clean_hit / clean if clean else 0.0
    scores.unlabelled_unique = len(unlabelled)
    return scores


# --- proposing instances ---------------------------------------------------------------

_SCALE_TEN = frozenset({"learning_rate", "weight_decay"})
_DOUBLE = frozenset(
    {
        "batch_size",
        "epochs",
        "training_steps",
        "warmup_steps",
        "hidden_size",
        "intermediate_size",
        "num_layers",
        "num_attention_heads",
        "max_sequence_length",
        *COUNT_KEYS,
    }
)
_OPTIMIZER_SWAP = {
    "adamw": "SGD",
    "adam": "SGD",
    "sgd": "Adam",
    "adafactor": "AdamW",
    "lamb": "AdamW",
}
# Keys whose errors are spread across classes first, so every class gets instances.
_KEY_PRIORITY = (
    "data_split",
    "test_fraction",
    "validation_fraction",
    "eval_metric",
    "train_examples",
    "test_examples",
    "num_classes",
)


def mutate(key: str, raw: str) -> str | None:
    """A different value for `key`, written the way the paper wrote `raw`, or None if
    there is no safe way to change it."""
    text = raw.strip()
    if key in _SCALE_TEN:
        exponent = re.fullmatch(r"(\d+(?:\.\d+)?)\s*[eE]\s*([-+]?\d+)", text)
        if exponent:
            return f"{exponent.group(1)}e{int(exponent.group(2)) + 1}"
        if re.fullmatch(r"\d*\.\d+", text):
            decimals = len(text.split(".")[1])
            return f"{float(text) * 10:.{max(decimals - 1, 1)}f}".rstrip("0").rstrip(".")
        return None
    if key in _DOUBLE or key == "num_classes":
        match = re.fullmatch(r"(\d{1,3}(?:,\d{3})+|\d+)(\s?[kKM])?", text)
        if match is None:
            return None
        number = int(match.group(1).replace(",", ""))
        changed = number + 1 if key == "num_classes" else number * 2
        written = f"{changed:,}" if "," in match.group(1) else str(changed)
        return written + (match.group(2) or "")
    if key == "seed":
        return str(int(text) + 1) if text.isdigit() else None
    if key == "dropout":
        percent = text.endswith("%")
        digits = text.rstrip("%").strip()
        if not re.fullmatch(r"\d*\.?\d+", digits):
            return None
        value = float(digits) + (20 if percent else 0.2)
        if value >= (100 if percent else 1):
            return None
        shown = f"{value:g}"
        return f"{shown}%" if percent else shown
    if key == "data_split":
        parts = text.split("/")
        if len(parts) not in (2, 3) or not all(p.strip().isdigit() for p in parts):
            return None
        numbers = [int(p) for p in parts]
        if sum(numbers) != 100 or numbers[0] <= 10:
            return None
        rest = len(numbers) - 1
        if 10 % rest:
            return None
        new_parts = [numbers[0] - 10, *(n + 10 // rest for n in numbers[1:])]
        return "/".join(map(str, new_parts))
    if key in {"test_fraction", "validation_fraction"}:
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", text)
        if match is None or float(match.group(1)) + 10 >= 100:
            return None
        return f"{float(match.group(1)) + 10:g}%"
    if key == "eval_metric":
        return "macro-F1" if text.lower() == "accuracy" else "accuracy"
    if key == "optimizer":
        return _OPTIMIZER_SWAP.get(text.lower())
    return None


@dataclass
class Proposal:
    """Gold labels and injection records for one paper, and its injected copy."""

    injected: ParsedPaper
    labels: list[GoldLabel] = field(default_factory=list)
    records: list[InjectionRecord] = field(default_factory=list)


def _usable(row: CandidateRow) -> bool:
    return (
        row.outcome == TaskOutcome.CHECKED.value
        and row.agrees is True
        and bool(row.agreeing_evidence)
        and row.anchor.start is not None
    )


def _evidence_type(row: CandidateRow) -> EvidenceType:
    kinds = {e.kind for e in row.agreeing_evidence}
    if EvidenceKind.CODE in kinds:
        return EvidenceType.CODE
    text = " ".join(e.text for e in row.agreeing_evidence)
    if "dataset_info" in text:
        return EvidenceType.HF_CARD
    return EvidenceType.SPLIT_LIST


def propose(
    paper: ParsedPaper,
    rows: Sequence[CandidateRow],
    *,
    errors: int = 2,
    controls: int = 2,
    seed: int = 0,
) -> Proposal:
    """Choose settings on which paper and repository agree, inject errors into some of them
    (one injected copy of the paper holds all of them) and keep others as clean controls,
    labelled in both the original and the injected copy. Deterministic for a seed.

    An injection is kept only if the planner reads the new value back at the same place and
    no repository value for the key equals it, so the label is really an error (D18)."""
    rng = random.Random(seed)
    usable: dict[str, CandidateRow] = {}
    for row in rows:
        if _usable(row) and row.key not in usable:
            usable[row.key] = row
    keys = sorted(usable)
    rng.shuffle(keys)
    keys.sort(key=lambda k: _KEY_PRIORITY.index(k) if k in _KEY_PRIORITY else len(_KEY_PRIORITY))

    current = paper
    proposal = Proposal(injected=paper)
    injected_keys: list[str] = []
    # (sentence id, end of the replaced span in the original, change in length)
    edits: list[tuple[str, int, int]] = []

    def shifted(anchor: SentenceAnchor) -> SentenceAnchor:
        """`anchor` in the injected copy, after the edits made so far."""
        assert anchor.start is not None
        assert anchor.end is not None
        delta = sum(d for sid, end, d in edits if sid == anchor.sentence_id and end <= anchor.start)
        return SentenceAnchor(
            sentence_id=anchor.sentence_id, start=anchor.start + delta, end=anchor.end + delta
        )

    for key in keys:
        if len(injected_keys) >= errors:
            break
        row = usable[key]
        new_value = mutate(key, row.paper_value)
        if new_value is None:
            continue
        where = shifted(row.anchor)
        sentence_now = current.sentence(where.sentence_id).text
        if sentence_now[where.start : where.end] != row.paper_value:
            continue
        try:
            candidate, label, record = inject_paper(
                current,
                where,
                new_value,
                key=key,
                rationale=_rationale(row, new_value),
                evidence_type=_evidence_type(row),
                evidence=row.agreeing_evidence,
            )
        except ValueError:
            continue
        reread = [
            s
            for s in paper_settings(candidate)
            if s.key == key
            and isinstance(s.source, PaperSource)
            and s.source.anchor == label.anchor
        ]
        sentence = candidate.sentence(row.anchor.sentence_id).text
        if len(reread) != 1 or any(
            _agree(reread[0], sentence, value) for value in row.repository_settings
        ):
            continue
        current = candidate
        injected_keys.append(key)
        assert row.anchor.end is not None
        edits.append(
            (row.anchor.sentence_id, row.anchor.end, len(new_value) - len(row.paper_value))
        )
        proposal.labels.append(label)
        proposal.records.append(record)
    proposal.injected = current if injected_keys else paper

    chosen = [k for k in keys if k not in injected_keys][:controls]
    for key in chosen:
        row = usable[key]
        rationale = (
            f"The paper's {key} {row.paper_value} matches the repository "
            f"({'; '.join(e.text for e in row.agreeing_evidence[:2])})."
        )
        proposal.labels.append(
            clean_label(paper, row.anchor, key, row.agreeing_evidence, rationale)
        )
        if injected_keys:
            proposal.labels.append(
                clean_label(
                    proposal.injected, shifted(row.anchor), key, row.agreeing_evidence, rationale
                )
            )
    return proposal


def _rationale(row: CandidateRow, new_value: str) -> str:
    where = "; ".join(
        f"{e.path}:{e.line_start}" if e.path else e.text for e in row.agreeing_evidence[:3]
    )
    return (
        f"The repository uses {row.key} {row.paper_value} ({where}), as the original paper "
        f"states; the injected copy says {new_value}, and no repository value for "
        f"{row.key} equals it."
    )


# --- method discrepancies from SciCoQA -------------------------------------------------


class CodeChange(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    original: str
    changed: str


class MethodIssue(BaseModel):
    """A SciCoQA discrepancy for one pair: real (reported in an issue or reproducibility
    paper, present in the original code) or synthetic (made by changing the code)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    discrepancy_id: str
    pair_id: str
    real: bool
    origin_url: str | None
    category: str | None
    description: str
    paper_quotes: tuple[str, ...]
    changes: tuple[CodeChange, ...] = ()


def method_issue(pair_id: str, row: Mapping[str, object]) -> MethodIssue:
    real = row.get("origin_type") != "synthetic"
    description = str(
        row.get("discrepancy_description_gpt") or row.get("discrepancy_description_gemini") or ""
    )
    quotes = row.get("relevant_paper_sections_gpt") or row.get("relevant_paper_sections_gemini")
    changes: list[CodeChange] = []
    snippets = row.get("changed_code_snippets")
    if not real and isinstance(snippets, Mapping):
        changes = [
            CodeChange(path=str(path), original=str(old), changed=str(new))
            for path, old, new in zip(
                snippets.get("original_file") or [],
                snippets.get("original_code") or [],
                snippets.get("changed_code") or [],
                strict=False,
            )
        ]
    return MethodIssue(
        discrepancy_id=str(row["discrepancy_id"]),
        pair_id=pair_id,
        real=real,
        origin_url=str(row["origin_url"]) if row.get("origin_url") else None,
        category=str(row["discrepancy_category"]) if row.get("discrepancy_category") else None,
        description=description,
        paper_quotes=tuple(str(q) for q in quotes) if isinstance(quotes, list | tuple) else (),
        changes=tuple(changes),
    )


def apply_code_changes(source: Path, destination: Path, issues: Sequence[MethodIssue]) -> set[str]:
    """Copy a snapshot's files and apply the synthetic issues' code changes. An issue is
    applied only if each of its original snippets occurs exactly once in its file; the ids
    of the applied issues are returned."""
    shutil.copytree(source, destination, dirs_exist_ok=True)
    applied = set()
    for issue in issues:
        if issue.real or not issue.changes:
            continue
        texts: dict[Path, str] = {}
        ok = True
        for change in issue.changes:
            target = destination / change.path
            if not target.is_file():
                ok = False
                break
            text = texts.get(target, target.read_text(encoding="utf-8"))
            if text.count(change.original) != 1:
                ok = False
                break
            texts[target] = text.replace(change.original, change.changed)
        if ok:
            for target, text in texts.items():
                target.write_text(text, encoding="utf-8", newline="")
            applied.add(issue.discrepancy_id)
    return applied


def method_labels(
    paper_id: str, issues: Sequence[MethodIssue], *, patched: set[str]
) -> list[GoldLabel]:
    """Gold labels for method discrepancies (document anchor, target `method`). A real
    issue is in both the original and the injected copy; a synthetic one only in the
    injected copy, whose repository carries the change."""
    labels = []
    for issue in issues:
        variants = (
            [PaperVariant.ORIGINAL, PaperVariant.INJECTED]
            if issue.real
            else [PaperVariant.INJECTED]
            if issue.discrepancy_id in patched
            else []
        )
        source = SourceRef(
            kind=SourceKind.SUBMISSION,
            id=paper_id,
            title="SciCoQA (CC BY 4.0)",
            url=issue.origin_url,
        )
        evidence = [
            Evidence(kind=EvidenceKind.QUOTE, source=source, text=quote)
            for quote in issue.paper_quotes[:3]
            if quote.strip()
        ] or [Evidence(kind=EvidenceKind.LOOKUP, source=source, text=issue.description)]
        for variant in variants:
            labels.append(
                GoldLabel(
                    label_id=f"m-{issue.discrepancy_id}-{variant.value}",
                    paper_id=paper_id,
                    variant=variant,
                    error_class=ErrorClass.METHOD_MISMATCH,
                    anchor=DocumentAnchor(),
                    target=SettingTarget(key="method"),
                    rationale=f"SciCoQA {issue.discrepancy_id} "
                    f"({'reported' if issue.real else 'synthetic'}): {issue.description}",
                    evidence=tuple(evidence),
                )
            )
    return labels


__all__ = [
    "CandidateRow",
    "ClassScore",
    "CodeChange",
    "CodeScores",
    "EvidenceType",
    "InjectionRecord",
    "MethodIssue",
    "Pair",
    "Proposal",
    "apply_code_changes",
    "apply_injections",
    "candidates",
    "clean_label",
    "error_class_for",
    "inject_code",
    "inject_paper",
    "method_issue",
    "method_labels",
    "mutate",
    "pool_from_scicoqa",
    "propose",
    "score_code_module",
    "validate",
]
