"""What specialists report, and what the verifier decides about it."""

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from auditor.schemas.anchors import Anchor
from auditor.schemas.base import SCHEMA_VERSION, NonEmptyStr, Record, SchemaVersion
from auditor.schemas.paper import PaperVariant
from auditor.schemas.tasks import Target, TaskKind


class ErrorClass(StrEnum):
    """The error classes scored by Macro-F1 (decision D4)."""

    REFERENCE_NOT_FOUND = "reference_not_found"
    REFERENCE_METADATA_MISMATCH = "reference_metadata_mismatch"
    CLAIM_NOT_SUPPORTED = "claim_not_supported"
    CLAIM_NUMBER_MISMATCH = "claim_number_mismatch"
    SETTING_MISMATCH = "setting_mismatch"
    DATA_SPLIT_MISMATCH = "data_split_mismatch"
    METRIC_MISMATCH = "metric_mismatch"
    TEMPLATE_VIOLATION = "template_violation"


class EvidenceKind(StrEnum):
    QUOTE = "quote"
    CODE = "code"
    RULE = "rule"
    LOOKUP = "lookup"


class SourceKind(StrEnum):
    SUBMISSION = "submission"
    CITED_WORK = "cited_work"
    REPOSITORY = "repository"
    VENUE_RULE = "venue_rule"
    SCHOLARLY_INDEX = "scholarly_index"


class SourceRef(Record):
    """Identifies where evidence came from, e.g. an arXiv id, a DOI or `url@commit`."""

    kind: SourceKind
    id: NonEmptyStr
    title: str | None = None
    url: str | None = None


class Evidence(Record):
    """One piece of evidence.

    `text` is a verbatim quote for QUOTE, the exact lines for CODE, the rule and the measured
    value for RULE, and the query and its result for LOOKUP.
    """

    kind: EvidenceKind
    source: SourceRef
    text: NonEmptyStr
    page: int | None = Field(default=None, ge=1)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    path: str | None = None
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end must be given together")
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end <= self.char_start
        ):
            raise ValueError("char_end must be greater than char_start")
        if self.kind is EvidenceKind.CODE and (self.path is None or self.line_start is None):
            raise ValueError("code evidence needs a path and line_start")
        if self.line_end is not None and (
            self.line_start is None or self.line_end < self.line_start
        ):
            raise ValueError("line_end needs a line_start that is not after it")
        return self


class CandidateFinding(Record):
    """A suspected error, reported by a specialist and not yet verified."""

    finding_id: NonEmptyStr
    task_id: NonEmptyStr
    paper_id: NonEmptyStr
    variant: PaperVariant
    error_class: ErrorClass
    anchor: Anchor
    target: Target
    summary: NonEmptyStr
    explanation: str = ""
    evidence: tuple[Evidence, ...] = ()
    uncertain: bool = False


class TaskOutcome(StrEnum):
    CHECKED = "checked"
    UNVERIFIABLE = "unverifiable"
    FAILED = "failed"


class TaskResult(Record):
    """How one task ended (decision D6)."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    task_id: NonEmptyStr
    paper_id: NonEmptyStr
    variant: PaperVariant
    kind: TaskKind
    outcome: TaskOutcome
    reason: str | None = None
    error: str | None = None
    findings: tuple[CandidateFinding, ...] = ()
    attempts: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.outcome is TaskOutcome.UNVERIFIABLE and not self.reason:
            raise ValueError("an unverifiable result needs a reason")
        if self.outcome is TaskOutcome.FAILED and not self.error:
            raise ValueError("a failed result needs the error")
        if self.outcome is not TaskOutcome.FAILED and self.error is not None:
            raise ValueError("only failed results carry an error")
        for finding in self.findings:
            if (finding.task_id, finding.paper_id, finding.variant) != (
                self.task_id,
                self.paper_id,
                self.variant,
            ):
                raise ValueError(f"finding {finding.finding_id} belongs to a different task")
        ids = [f.finding_id for f in self.findings]
        if len(ids) != len(set(ids)):
            raise ValueError("finding ids must be unique within a task result")
        return self


class VerificationState(StrEnum):
    VERIFIED_FLAG = "verified_flag"
    NEEDS_HUMAN_REVIEW = "needs_human_review"
    DROPPED = "dropped"


class VerifiedFinding(Record):
    """The verifier's decision on one candidate finding."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    finding: CandidateFinding
    state: VerificationState
    reason: NonEmptyStr
    accepted_evidence: tuple[Evidence, ...] = ()

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.state is VerificationState.VERIFIED_FLAG and not self.accepted_evidence:
            raise ValueError("a verified flag must keep at least one piece of accepted evidence")
        return self
