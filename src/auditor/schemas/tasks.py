"""Audit tasks produced by the planner and what each one is about."""

import hashlib
import json
from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from auditor.schemas.anchors import Anchor, BibliographyAnchor, DocumentAnchor, SentenceAnchor
from auditor.schemas.base import SCHEMA_VERSION, NonEmptyStr, Record, SchemaVersion
from auditor.schemas.experiment import ExtractedSetting, RepositoryRef
from auditor.schemas.paper import PaperVariant
from auditor.schemas.venue import VenueRule


class BibTarget(Record):
    kind: Literal["bibliography"] = "bibliography"
    bib_id: NonEmptyStr


class MarkerTarget(Record):
    """A citation marker the parser could not link to any bibliography entry."""

    kind: Literal["marker"] = "marker"
    marker: NonEmptyStr


class SettingTarget(Record):
    kind: Literal["setting"] = "setting"
    key: NonEmptyStr = Field(pattern=r"^[a-z][a-z0-9_]*$")


class RuleTarget(Record):
    kind: Literal["rule"] = "rule"
    rule_id: NonEmptyStr


Target = Annotated[
    BibTarget | MarkerTarget | SettingTarget | RuleTarget,
    Field(discriminator="kind"),
]


class TheoryClaim(Record):
    """One formal claim the planner found for the theory checker (decision D21): what it
    says, where it is stated and argued, what it builds on, and how risky it looks."""

    claim_id: NonEmptyStr
    # The model's own word for it (theorem, lemma, estimate, identity, ...), not a fixed list.
    kind: str = ""
    statement: str = ""
    # The sentences that state and argue the claim, then those it depends on.
    sentence_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    depends_on: tuple[NonEmptyStr, ...] = ()
    risk: Literal["low", "medium", "high"] = "medium"
    # Whether checking it takes real reasoning (a derivation), not just reading.
    needs_reasoning: bool = True
    # Why a referee reading the whole paper suspects it (the triage step); a claim with a
    # suspicion goes straight to a deep check.
    suspicion: str = ""


class TaskKind(StrEnum):
    REFERENCE = "reference"
    CITATION = "citation"
    CODE = "code"
    TEMPLATE = "template"
    # The mathematics of one segment of the paper: equations, derivations, proofs (D21).
    THEORY = "theory"
    # Problems in the paper's own text that need no outside source (D22).
    INTEGRITY = "integrity"


def compute_task_id(
    paper_id: str,
    variant: PaperVariant,
    kind: TaskKind,
    anchor: Anchor,
    target: Target,
    claim: str | None = None,
) -> str:
    """Deterministic id, identical across runs, processes and operating systems.

    A citation task's claim is part of its identity: one sentence can make two different
    claims about the same work ("90% on A [3] and 80% on B [3]"), and each needs a check.
    """
    parts = [
        paper_id,
        variant.value,
        kind.value,
        anchor.model_dump(mode="json"),
        target.model_dump(mode="json"),
    ]
    if claim is not None:
        parts.append(claim)
    payload = json.dumps(
        parts,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    digest = hashlib.sha256(payload.encode("ascii")).hexdigest()
    return f"{kind.value}-{digest[:16]}"


class AuditTask(Record):
    """One check to run. Fields after `claim` apply only to some task kinds."""

    schema_version: SchemaVersion = SCHEMA_VERSION
    task_id: NonEmptyStr
    paper_id: NonEmptyStr
    variant: PaperVariant
    kind: TaskKind
    anchor: Anchor
    target: Target
    claim: str | None = None
    setting: ExtractedSetting | None = None
    repository: RepositoryRef | None = None
    rule: VenueRule | None = None
    # Theory tasks: the claims the task checks, in reading order.
    claims: tuple[TheoryClaim, ...] | None = None

    @classmethod
    def create(
        cls,
        *,
        paper_id: str,
        variant: PaperVariant,
        kind: TaskKind,
        anchor: Anchor,
        target: Target,
        claim: str | None = None,
        setting: ExtractedSetting | None = None,
        repository: RepositoryRef | None = None,
        rule: VenueRule | None = None,
        claims: Sequence[TheoryClaim] | None = None,
    ) -> Self:
        return cls(
            task_id=compute_task_id(paper_id, variant, kind, anchor, target, claim),
            paper_id=paper_id,
            variant=variant,
            kind=kind,
            anchor=anchor,
            target=target,
            claim=claim,
            setting=setting,
            repository=repository,
            rule=rule,
            claims=tuple(claims) if claims is not None else None,
        )

    @model_validator(mode="after")
    def _check(self) -> Self:
        expected = compute_task_id(
            self.paper_id, self.variant, self.kind, self.anchor, self.target, self.claim
        )
        if self.task_id != expected:
            raise ValueError(f"task_id should be {expected}; build tasks with AuditTask.create")

        match self.kind:
            case TaskKind.REFERENCE:
                _expect(self, BibliographyAnchor, (BibTarget,), set())
            case TaskKind.CITATION:
                _expect(self, SentenceAnchor, (BibTarget, MarkerTarget), {"claim"})
            case TaskKind.CODE:
                _expect(self, SentenceAnchor, (SettingTarget,), {"setting", "repository"})
                if (
                    isinstance(self.target, SettingTarget)
                    and self.setting is not None
                    and self.setting.key != self.target.key
                ):
                    raise ValueError("code task setting key must match its target")
            case TaskKind.TEMPLATE:
                _expect(self, DocumentAnchor, (RuleTarget,), {"rule"})
                if (
                    isinstance(self.target, RuleTarget)
                    and self.rule is not None
                    and self.rule.id != self.target.rule_id
                ):
                    raise ValueError("template task rule must match its target")
            case TaskKind.THEORY:
                _expect(self, SentenceAnchor, (SettingTarget,), {"claims"})
            case TaskKind.INTEGRITY:
                _expect(self, DocumentAnchor, (RuleTarget,), set())
        return self


_DETAIL_FIELDS = ("claim", "setting", "repository", "rule", "claims")


def _expect(
    task: AuditTask,
    anchor_type: type[Record],
    target_types: tuple[type[Record], ...],
    required: set[str],
) -> None:
    kind = task.kind.value
    if not isinstance(task.anchor, anchor_type):
        raise ValueError(f"{kind} tasks need a {anchor_type.__name__}")
    if not isinstance(task.target, target_types):
        names = " or ".join(t.__name__ for t in target_types)
        raise ValueError(f"{kind} tasks need a {names}")
    for name in _DETAIL_FIELDS:
        present = getattr(task, name) is not None
        if name in required and not present:
            raise ValueError(f"{kind} tasks need '{name}'")
        if name not in required and present:
            raise ValueError(f"{kind} tasks do not take '{name}'")
