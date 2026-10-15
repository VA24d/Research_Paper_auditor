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
from collections.abc import Sequence
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
    ErrorClass,
    ExtractedSetting,
    PaperSource,
    Evidence,
    EvidenceKind,
    GoldLabel,
    PaperVariant,
    ParsedPaper,
    Prediction,
    RepositoryRef,
    RepositorySource,
    SentenceAnchor,
    SettingTarget,
    TaskOutcome,
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


@dataclass(frozen=True)
class CandidateRow:
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
    labels: Sequence[GoldLabel], records: Sequence[InjectionRecord], *, data_mode: bool
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

    @property
    def macro_f1(self) -> float:
        scores = list(self.per_class.values())
        return sum(s.f1 for s in scores) / len(scores) if scores else 0.0


def score_code_module(labels: Sequence[GoldLabel], predictions: Sequence[Prediction]) -> CodeScores:
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
    for prediction in (p for p in predictions if p.is_positive and p.error_class in CODE_CLASSES):
        matched = next((lbl for lbl in unmatched if matches(prediction, lbl)), None)
        score = scores.per_class.setdefault(prediction.error_class, ClassScore())
        if matched is None:
            scores.unlabelled += 1
            continue
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
        number = text.rstrip("%").strip()
        if not re.fullmatch(r"\d*\.?\d+", number):
            return None
        value = float(number) + (20 if percent else 0.2)
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
        changed = [numbers[0] - 10, *(n + 10 // rest for n in numbers[1:])]
        return "/".join(map(str, changed))
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
        assert anchor.start is not None and anchor.end is not None
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


__all__ = [
    "CandidateRow",
    "ClassScore",
    "CodeScores",
    "EvidenceType",
    "InjectionRecord",
    "candidates",
    "clean_label",
    "error_class_for",
    "inject_code",
    "inject_paper",
    "mutate",
    "propose",
    "Proposal",
    "score_code_module",
    "validate",
]
