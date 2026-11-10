"""System outputs, gold labels and run metadata used for evaluation (decisions D3, D10, D18)."""

import re
from enum import StrEnum
from typing import Literal, Self

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from auditor.schemas.anchors import Anchor, DocumentAnchor, anchors_match
from auditor.schemas.base import SCHEMA_VERSION, NonEmptyStr, Record, SchemaVersion
from auditor.schemas.findings import ErrorClass, Evidence, EvidenceKind, Severity, VerificationState
from auditor.schemas.paper import PaperVariant
from auditor.schemas.tasks import SettingTarget, Target

NO_ERROR: Literal["none"] = "none"


class SystemName(StrEnum):
    B0 = "b0"
    B1 = "b1"
    FULL = "full"
    FULL_NO_VERIFIER = "full_no_verifier"
    FULL_ABSTRACT_ONLY = "full_abstract_only"


# Systems with no verifier step; every flag they raise counts as positive (decision D3).
UNVERIFIED_SYSTEMS = frozenset({SystemName.B0, SystemName.B1, SystemName.FULL_NO_VERIFIER})


class Prediction(Record):
    """One error a system claims to have found.

    `state` is None exactly for systems without a verifier (`UNVERIFIED_SYSTEMS`); all their
    predictions count as positive. For the others only VERIFIED_FLAG counts as positive.
    """

    schema_version: SchemaVersion = SCHEMA_VERSION
    run_id: NonEmptyStr
    system: SystemName
    paper_id: NonEmptyStr
    variant: PaperVariant
    error_class: ErrorClass
    anchor: Anchor
    target: Target
    state: VerificationState | None = None
    summary: NonEmptyStr
    evidence: tuple[Evidence, ...] = ()
    finding_id: str | None = None
    # From the full system's synthesis step (decision D20); scoring ignores them.
    severity: Severity | None = None
    duplicate_of: str | None = None

    @model_validator(mode="after")
    def _state_matches_system(self) -> Self:
        if self.system in UNVERIFIED_SYSTEMS and self.state is not None:
            raise ValueError(f"{self.system.value} has no verifier, so state must be empty")
        if self.system not in UNVERIFIED_SYSTEMS and self.state is None:
            raise ValueError(f"{self.system.value} predictions need the verifier's state")
        return self

    @property
    def is_positive(self) -> bool:
        if self.system in UNVERIFIED_SYSTEMS:
            return True
        return self.state is VerificationState.VERIFIED_FLAG


class GoldLabel(Record):
    """Ground truth for one injected error or one clean control item (`error_class` "none")."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    label_id: NonEmptyStr
    paper_id: NonEmptyStr
    variant: PaperVariant
    error_class: ErrorClass | Literal["none"]
    anchor: Anchor
    target: Target
    rationale: NonEmptyStr
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    reviewed_by: tuple[str, ...] = ()

    @property
    def is_error(self) -> bool:
        return self.error_class != NO_ERROR


METHOD_OVERLAP_THRESHOLD: float = 0.38

_METHOD_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "up",
        "about",
        "into",
        "over",
        "after",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "as",
        "not",
        "paper",
        "code",
        "states",
        "codebase",
        "instead",
        "specified",
        "specifies",
        "describes",
        "described",
        "used",
        "uses",
    }
)
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]{3,}")


def method_match_terms(text: str) -> frozenset[str]:
    """Salient terms for method discrepancy comparison."""
    return frozenset(w.lower() for w in _TOKEN_PATTERN.findall(text)) - _METHOD_STOPWORDS


def _text_overlap(s1: str, s2: str) -> float:
    """Token-level Szymkiewicz-Simpson overlap coefficient |A ∩ B| / min(|A|, |B|)."""
    t1 = method_match_terms(s1)
    t2 = method_match_terms(s2)
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / min(len(t1), len(t2))


def method_overlap(prediction: Prediction, label: GoldLabel) -> float:
    """Szymkiewicz-Simpson overlap coefficient between prediction and label."""
    p_texts = [prediction.summary] + [
        e.text for e in prediction.evidence if e.kind == EvidenceKind.QUOTE
    ]
    l_texts = [label.rationale] + [e.text for e in label.evidence if e.kind != EvidenceKind.CODE]
    return _text_overlap(" ".join(p_texts), " ".join(l_texts))


def is_method_comparison(prediction: Prediction, label: GoldLabel) -> bool:
    """Whether this match is a document-level method mismatch check."""
    return (
        isinstance(prediction.anchor, DocumentAnchor)
        and isinstance(label.anchor, DocumentAnchor)
        and (
            (isinstance(prediction.target, SettingTarget) and prediction.target.key == "method")
            or (isinstance(label.target, SettingTarget) and label.target.key == "method")
            or prediction.error_class == ErrorClass.METHOD_MISMATCH
            or label.error_class == ErrorClass.METHOD_MISMATCH
        )
    )


def matches(prediction: Prediction, label: GoldLabel) -> bool:
    """Whether a prediction is about the same item as a gold label (class not compared)."""
    base_match = (
        prediction.paper_id == label.paper_id
        and prediction.variant == label.variant
        and prediction.target == label.target
        and anchors_match(prediction.anchor, label.anchor)
    )
    if not base_match:
        return False
    if is_method_comparison(prediction, label):
        return method_overlap(prediction, label) >= METHOD_OVERLAP_THRESHOLD
    return True


class DatasetEntry(Record):
    """One paper in an experiment dataset file (JSON Lines, read by `auditor batch`).

    `paper` is a ParsedPaper JSON file (for example an injected copy) or a PDF; relative
    paths are relative to the dataset file. Without `repository`, the repository is taken
    from the paper when it announces one. `repository_commit` pins the snapshot the gold
    labels were checked against; a /tree/<ref> or /commit/<sha> URL does the same.
    """

    paper: NonEmptyStr
    repository: str | None = None
    repository_commit: str | None = None
    venue: str | None = None


class RunManifest(Record):
    schema_version: SchemaVersion = SCHEMA_VERSION
    run_id: NonEmptyStr
    system: SystemName
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    git_commit: str | None = None
    git_dirty: bool | None = None
    model: str | None = None
    llm_cache: bool = False
    seed: int | None = None
    dataset_version: str | None = None
    python_version: NonEmptyStr
    platform: NonEmptyStr
    packages: dict[str, str] = Field(default_factory=dict)
    config: dict[str, JsonValue] = Field(default_factory=dict)
