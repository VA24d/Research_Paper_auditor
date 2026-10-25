"""Reference checks (decision P2.5): does the bibliography entry exist and match?

A reference task is checked without a model. The entry is resolved against the scholarly
indexes, and its metadata is compared with the entry's fields field by field. A reference
that resolves to nothing is a `reference_not_found` finding; one that resolves to a
different work is a `reference_metadata_mismatch` finding.
"""

from auditor.infra.context import RunContext
from auditor.retrieval.clients import Work
from auditor.retrieval.resolve import Resolution, titles_match, venues_match
from auditor.schemas import (
    AuditTask,
    BibEntry,
    BibTarget,
    CandidateFinding,
    ErrorClass,
    Evidence,
    EvidenceKind,
    SourceKind,
    SourceRef,
    TaskKind,
    TaskOutcome,
    TaskResult,
)


def _lookup_evidence(entry: BibEntry, resolution: Resolution) -> Evidence:
    """The LOOKUP evidence for a reference check: what was looked up and what came back."""
    work = resolution.work
    identifier = work.id if work else entry.id
    return Evidence(
        kind=EvidenceKind.LOOKUP,
        source=SourceRef(kind=SourceKind.SCHOLARLY_INDEX, id=identifier),
        text=(
            f"Resolved '{entry.raw or entry.title}' "
            f"(queries: {', '.join(q for q, _ in resolution.queries) or 'none'}); "
            f"{'found' if resolution.found else 'not found: ' + (resolution.reason or 'unknown')}"
        ),
    )


def _metadata_findings(task: AuditTask, entry: BibEntry, work: Work) -> list[CandidateFinding]:
    """Findings for the metadata fields that do not match."""
    findings: list[CandidateFinding] = []
    problems: list[str] = []

    if entry.title and work.title and not titles_match(entry.title, work.title):
        problems.append(f"title: the entry says '{entry.title}', the index says '{work.title}'")
    if entry.year and work.year and entry.year != work.year:
        problems.append(f"year: the entry says {entry.year}, the index says {work.year}")
    if entry.venue and work.venue and not venues_match(entry.venue, work.venue):
        problems.append(f"venue: the entry says '{entry.venue}', the index says '{work.venue}'")

    if not problems:
        return findings

    findings.append(
        CandidateFinding.for_task(
            task,
            0,
            error_class=ErrorClass.REFERENCE_METADATA_MISMATCH,
            summary="The reference's metadata does not match the scholarly indexes: "
            + "; ".join(problems),
            explanation=entry.raw or entry.title or entry.id,
            evidence=(_lookup_evidence(entry, Resolution(work=work)),),
        )
    )
    return findings


async def check_reference(task: AuditTask, ctx: RunContext, resolution: Resolution) -> TaskResult:
    """Whether the bibliography entry exists and its metadata matches (decision P2.5)."""
    if task.kind is not TaskKind.REFERENCE:
        raise ValueError(f"{task.task_id} is not a reference task")
    if not isinstance(task.target, BibTarget):
        raise ValueError(f"{task.task_id} is not a reference task")

    try:
        entry = ctx.paper.bib_entry(task.target.bib_id)
    except KeyError:
        entry = None
    if entry is None:
        return TaskResult.for_task(
            task,
            TaskOutcome.UNVERIFIABLE,
            reason=f"the bibliography entry {task.target.bib_id} is not in the paper",
        )
    findings: list[CandidateFinding] = []
    if not resolution.found:
        findings.append(
            CandidateFinding.for_task(
                task,
                0,
                error_class=ErrorClass.REFERENCE_NOT_FOUND,
                summary=f"The reference could not be resolved: {resolution.reason}",
                explanation=entry.raw or entry.title or entry.id,
                evidence=(_lookup_evidence(entry, resolution),),
            )
        )
    else:
        assert resolution.work is not None
        findings.extend(_metadata_findings(task, entry, resolution.work))

    if not findings:
        return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=[])
    return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=findings)
