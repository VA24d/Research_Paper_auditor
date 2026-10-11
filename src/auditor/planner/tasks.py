"""Turns a parsed paper into AuditTasks (decisions D3 and D6)."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from auditor.infra.context import RunContext
from auditor.orchestration.graph import PlanFn
from auditor.planner.claims import ClaimCandidate, citation_candidates, refine_candidates
from auditor.planner.repository import find_repository
from auditor.planner.table_reader import read_tables
from auditor.planner.tables import paper_settings
from auditor.schemas import (
    AuditTask,
    BibliographyAnchor,
    BibTarget,
    DocumentAnchor,
    ExtractedSetting,
    MarkerTarget,
    PaperSource,
    ParsedPaper,
    RepositoryRef,
    RuleTarget,
    SentenceAnchor,
    SettingTarget,
    TaskKind,
    VenueProfile,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlannerOptions:
    repository: RepositoryRef | None = None
    venue: VenueProfile | None = None
    use_llm: bool = True
    detect_repository: bool = True


def reference_tasks(paper: ParsedPaper) -> list[AuditTask]:
    return [
        AuditTask.create(
            paper_id=paper.paper_id,
            variant=paper.variant,
            kind=TaskKind.REFERENCE,
            anchor=BibliographyAnchor(bib_id=entry.id),
            target=BibTarget(bib_id=entry.id),
        )
        for entry in paper.bibliography
    ]


def citation_tasks(paper: ParsedPaper, candidates: list[ClaimCandidate]) -> list[AuditTask]:
    tasks = []
    for candidate in candidates:
        if not candidate.check_worthy:
            continue
        target = (
            BibTarget(bib_id=candidate.bib_id)
            if candidate.bib_id is not None
            else MarkerTarget(marker=candidate.marker)
        )
        tasks.append(
            AuditTask.create(
                paper_id=paper.paper_id,
                variant=paper.variant,
                kind=TaskKind.CITATION,
                anchor=SentenceAnchor(sentence_id=candidate.sentence_id),
                target=target,
                claim=candidate.claim,
            )
        )
    return tasks


def code_tasks(
    paper: ParsedPaper,
    repository: RepositoryRef | None,
    settings: Sequence[ExtractedSetting] | None = None,
) -> list[AuditTask]:
    if repository is None:
        return []
    tasks = []
    for setting in paper_settings(paper) if settings is None else settings:
        if not isinstance(setting.source, PaperSource):
            continue
        # The value's span keeps two values of one setting in one sentence apart
        # ("1e-4 for BERT and 2e-5 for RoBERTa"); a span-less gold anchor still matches.
        tasks.append(
            AuditTask.create(
                paper_id=paper.paper_id,
                variant=paper.variant,
                kind=TaskKind.CODE,
                anchor=setting.source.anchor,
                target=SettingTarget(key=setting.key),
                setting=setting,
                repository=repository,
            )
        )
    return tasks


def template_tasks(paper: ParsedPaper, venue: VenueProfile | None) -> list[AuditTask]:
    if venue is None:
        return []
    return [
        AuditTask.create(
            paper_id=paper.paper_id,
            variant=paper.variant,
            kind=TaskKind.TEMPLATE,
            anchor=DocumentAnchor(),
            target=RuleTarget(rule_id=rule.id),
            rule=rule,
        )
        for rule in venue.rules
    ]


def _unique(tasks: list[AuditTask]) -> list[AuditTask]:
    """Drop repeats, e.g. the same work cited twice in one sentence."""
    seen: dict[str, AuditTask] = {}
    for task in tasks:
        seen.setdefault(task.task_id, task)
    return list(seen.values())


async def plan_paper(
    paper: ParsedPaper, ctx: RunContext, options: PlannerOptions | None = None
) -> list[AuditTask]:
    options = options or PlannerOptions()
    candidates = citation_candidates(paper)
    if options.use_llm and ctx.llm is not None and candidates:
        candidates = await refine_candidates(paper, candidates, ctx.llm)

    repository = options.repository
    if repository is None and options.detect_repository:
        repository = find_repository(paper)
        if repository is not None:
            logger.info("using repository %s found in the paper", repository.url)

    settings = paper_settings(paper)
    # Reading messy tables costs LLM calls, so only when there is code to check against.
    if repository is not None and options.use_llm and ctx.llm is not None:
        settings += await read_tables(paper, ctx.llm, settings)

    tasks = [
        *reference_tasks(paper),
        *citation_tasks(paper, candidates),
        *code_tasks(paper, repository, settings),
        *template_tasks(paper, options.venue),
    ]
    return _unique(tasks)


def make_planner(options: PlannerOptions | None = None) -> PlanFn:
    async def plan(paper: ParsedPaper, ctx: RunContext) -> list[AuditTask]:
        return await plan_paper(paper, ctx, options)

    return plan
