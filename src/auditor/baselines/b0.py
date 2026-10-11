"""Baseline B0: the same model reads the whole paper once, with no tools."""

from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from auditor.baselines.common import (
    ANSWER_GUIDE,
    ERROR_CLASS_GUIDE,
    ReportedErrors,
    render_paper,
    to_predictions,
)
from auditor.infra.llm import StructuredLLM
from auditor.infra.scope import Scope, enter_scope
from auditor.infra.usage import UsageLog
from auditor.schemas import ParsedPaper, Prediction, RepositoryRef, SystemName, VenueProfile

SYSTEM_PROMPT = f"""You audit a research paper submission for factual errors before peer review.
You cannot open cited papers or the code repository; use only the paper text and what you
already know.

{ERROR_CLASS_GUIDE}

{ANSWER_GUIDE}"""


@dataclass(frozen=True)
class BaselineResult:
    predictions: list[Prediction]
    dropped: list[str]


async def run_b0(
    paper: ParsedPaper,
    llm: StructuredLLM,
    *,
    run_id: str,
    usage: UsageLog | None = None,
    venue: VenueProfile | None = None,
    repository: RepositoryRef | None = None,
) -> BaselineResult:
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(render_paper(paper, venue=venue, repository=repository)),
    ]
    with enter_scope(Scope.for_paper(run_id, paper, usage)):
        result = await llm.generate(ReportedErrors, messages)
    predictions, dropped = to_predictions(result.value, paper, run_id=run_id, system=SystemName.B0)
    return BaselineResult(predictions=predictions, dropped=dropped)
