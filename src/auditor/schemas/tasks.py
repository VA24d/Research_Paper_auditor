"""Audit tasks produced by the planner and what each one is about."""

import hashlib
import json
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
    key: NonEmptyStr


class RuleTarget(Record):
    kind: Literal["rule"] = "rule"
    rule_id: NonEmptyStr


Target = Annotated[
    BibTarget | MarkerTarget | SettingTarget | RuleTarget,
    Field(discriminator="kind"),
]


class TaskKind(StrEnum):
    REFERENCE = "reference"
    CITATION = "citation"
    CODE = "code"
    TEMPLATE = "template"


def compute_task_id(
    paper_id: str,
    variant: PaperVariant,
    kind: TaskKind,
    anchor: Anchor,
    target: Target,
) -> str:
    """Deterministic id, identical across runs, processes and operating systems."""
    payload = json.dumps(
        [
            paper_id,
            variant.value,
            kind.value,
            anchor.model_dump(mode="json"),
            target.model_dump(mode="json"),
        ],
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
    ) -> Self:
        return cls(
            task_id=compute_task_id(paper_id, variant, kind, anchor, target),
            paper_id=paper_id,
            variant=variant,
            kind=kind,
            anchor=anchor,
            target=target,
            claim=claim,
            setting=setting,
            repository=repository,
            rule=rule,
        )

    @model_validator(mode="after")
    def _check(self) -> Self:
        expected = compute_task_id(self.paper_id, self.variant, self.kind, self.anchor, self.target)
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
        return self


_DETAIL_FIELDS = ("claim", "setting", "repository", "rule")


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
