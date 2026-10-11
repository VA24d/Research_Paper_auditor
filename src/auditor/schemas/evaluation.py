"""System outputs, gold labels and run metadata used for evaluation (decisions D3, D10, D18)."""

from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, Field, JsonValue

from auditor.schemas.anchors import Anchor, anchors_match
from auditor.schemas.base import SCHEMA_VERSION, NonEmptyStr, Record, SchemaVersion
from auditor.schemas.findings import ErrorClass, Evidence, VerificationState
from auditor.schemas.paper import PaperVariant
from auditor.schemas.tasks import Target

NO_ERROR: Literal["none"] = "none"


class SystemName(StrEnum):
    B0 = "b0"
    B1 = "b1"
    FULL = "full"
    FULL_NO_VERIFIER = "full_no_verifier"
    FULL_ABSTRACT_ONLY = "full_abstract_only"


class Prediction(Record):
    """One error a system claims to have found.

    `state` is None for systems without a verifier (B0, B1); all their predictions count
    as positive. For the full system only VERIFIED_FLAG counts as positive.
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

    @property
    def is_positive(self) -> bool:
        return self.state is None or self.state is VerificationState.VERIFIED_FLAG


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
