"""Baseline B1: one tool-calling agent with the specialists' tools, one pass, no verifier."""

from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from auditor.baselines.b0 import BaselineResult
from auditor.baselines.common import (
    ANSWER_GUIDE,
    ERROR_CLASS_GUIDE,
    ReportedErrors,
    render_paper,
    to_predictions,
)
from auditor.infra.llm import UsageCallback
from auditor.infra.scope import Scope, enter_scope
from auditor.infra.usage import UsageLog
from auditor.schemas import ParsedPaper, RepositoryRef, SystemName, VenueProfile

SYSTEM_PROMPT = f"""You audit a research paper submission for factual errors before peer review.
You have tools to look up cited works and to inspect the paper's code repository. Tool
calls are limited, so spend them on the items most likely to be wrong. When you are done,
give your final answer in the required structured format.

{ERROR_CLASS_GUIDE}

{ANSWER_GUIDE}"""


async def run_b1(
    paper: ParsedPaper,
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    *,
    run_id: str,
    model_name: str,
    usage: UsageLog | None = None,
    venue: VenueProfile | None = None,
    repository: RepositoryRef | None = None,
    max_model_calls: int = 20,
    max_tool_calls: int = 40,
) -> BaselineResult:
    # Each middleware is generic over its own state type; the list is typed loosely so
    # mypy accepts the mix.
    limits: list[AgentMiddleware[Any, Any]] = [
        ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=max_tool_calls, exit_behavior="continue"),
    ]
    agent = create_agent(
        model,
        tools=list(tools),
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(ReportedErrors),
        middleware=limits,
    )
    scope = Scope.for_paper(run_id, paper, usage)
    config = RunnableConfig(
        callbacks=[UsageCallback(scope, model_name)],
        recursion_limit=4 * max_model_calls + 10,
    )
    message = HumanMessage(render_paper(paper, venue=venue, repository=repository))
    with enter_scope(scope):
        state = await agent.ainvoke({"messages": [message]}, config)

    answer = state.get("structured_response")
    if not isinstance(answer, ReportedErrors):
        return BaselineResult(predictions=[], dropped=["the agent stopped without a final answer"])
    predictions, dropped = to_predictions(answer, paper, run_id=run_id, system=SystemName.B1)
    return BaselineResult(predictions=predictions, dropped=dropped)
