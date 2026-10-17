"""System outputs, gold labels and run metadata used for evaluation (decisions D3, D10, D18)."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from auditor.schemas.anchors import Anchor, anchors_match
from auditor.schemas.base import SCHEMA_VERSION, NonEmptyStr, Record, SchemaVersion
from auditor.schemas.findings import ErrorClass, Evidence, Severity, VerificationState
from auditor.schemas.paper import PaperVariant
from auditor.schemas.tasks import Target

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


def matches(prediction: Prediction, label: GoldLabel) -> bool:
    """Whether a prediction is about the same item as a gold label (class not compared)."""
    return (
        prediction.paper_id == label.paper_id
        and prediction.variant == label.variant
        and prediction.target == label.target
        and anchors_match(prediction.anchor, label.anchor)
    )


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
