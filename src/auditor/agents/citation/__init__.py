"""The citation agent (Part 2): references and claim support.

A reference task asks whether a bibliography entry exists and its metadata matches; a
citation task asks whether the cited work supports the claim the paper makes about it.
Both go through the same resolution (auditor.retrieval.resolve) and, for citation tasks,
the same retrieval (auditor.retrieval.fulltext) and passage ranking
(auditor.retrieval.passages) that baseline B1's tools use.
"""

from auditor.agents.citation.reference import check_reference
from auditor.agents.citation.support import check_citation
from auditor.infra.context import RunContext
from auditor.retrieval.fulltext import retrieve_source
from auditor.retrieval.resolve import resolve_entry, resolve_marker
from auditor.schemas import (
    AuditTask,
    BibTarget,
    MarkerTarget,
    TaskKind,
    TaskOutcome,
    TaskResult,
)


async def check(task: AuditTask, ctx: RunContext) -> TaskResult:
    """The entry point for the citation agent (decision P2.5)."""
    if task.kind is TaskKind.REFERENCE:
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
        resolution = await resolve_entry(ctx.http, entry)
        return await check_reference(task, ctx, resolution)
    if task.kind is TaskKind.CITATION:
        if isinstance(task.target, BibTarget):
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
            resolution = await resolve_entry(ctx.http, entry)
        elif isinstance(task.target, MarkerTarget):
            resolution = await resolve_marker(ctx.http, task.target.marker, ctx.paper.bibliography)
        else:
            raise ValueError(f"unsupported target {task.target}")
        if not resolution.found or resolution.work is None:
            # The work could not be resolved: the claim cannot be checked.
            return TaskResult.for_task(
                task,
                TaskOutcome.UNVERIFIABLE,
                reason=resolution.reason or "the cited work could not be resolved",
            )
        source = await retrieve_source(
            ctx.http,
            resolution.work,
            ctx.settings,
            abstract_only=ctx.settings.citation_abstract_only,
        )
        return await check_citation(task, ctx, resolution, source)
    raise ValueError(f"{task.task_id} is not a citation or reference task")
