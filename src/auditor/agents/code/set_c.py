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
    ERROR_CLASSES,
    Tier,
    static_candidates,
)
from auditor.infra.context import RunContext
from auditor.infra.snapshots import SnapshotUnavailableError, SourceError, fetch_snapshot
from auditor.planner.tables import paper_settings
from auditor.planner.tasks import code_tasks
from auditor.schemas import (
    ErrorClass,
    Evidence,
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
            )
        )
    return rows


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
        label_id=_label_id(paper.paper_id, key, anchor, "none"),
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
    "score_code_module",
    "validate",
]
