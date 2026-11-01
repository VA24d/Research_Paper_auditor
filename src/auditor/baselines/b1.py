"""Baseline B1: one tool-calling agent with the specialists' tools, one pass, no verifier."""

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ToolCallLimitMiddleware,
)
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from auditor.baselines.b0 import BaselineResult
from auditor.baselines.common import (
    ANSWER_GUIDE,
    ERROR_CLASS_GUIDE,
    IncompleteAnswerError,
    ReportedErrors,
    render_paper,
    to_predictions,
)
from auditor.infra.llm import StructuredLLM, TokenUsage, prompt_of
from auditor.infra.scope import Scope, enter_scope
from auditor.infra.usage import UsageLog
from auditor.schemas import ParsedPaper, RepositoryRef, SystemName, VenueProfile

SYSTEM_PROMPT = f"""You audit a research paper submission for factual errors before peer review.
You have tools to look up cited works and to inspect the paper's code repository. Tool
calls are limited, so spend them on the items most likely to be wrong. When you are done,
give your final answer in the required structured format.

{ERROR_CLASS_GUIDE}

{ANSWER_GUIDE}"""


PURPOSE = "b1"


def _usage(response: ModelResponse[Any]) -> TokenUsage:
    total = TokenUsage()
    for message in response.result:
        if isinstance(message, AIMessage):
            total += TokenUsage.from_message(message)
    return total


class _Accounted(AgentMiddleware[Any, Any, Any]):
    """Sends each model call through the run's LLM accounting (the dollar and call limits,
    the spend ledger and the usage log), and takes the tools away on the last allowed call
    so the agent has to give its answer."""

    def __init__(self, llm: StructuredLLM, max_model_calls: int) -> None:
        super().__init__()
        self.llm = llm
        self.max_model_calls = max_model_calls
        self.calls = 0

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        self.calls += 1
        if self.calls >= self.max_model_calls:
            request = request.override(tools=[])
        return await self.llm.direct_call(
            lambda: handler(request), _usage, purpose=PURPOSE, prompt=prompt_of(request)
        )


async def run_b1(
    paper: ParsedPaper,
    llm: StructuredLLM,
    tools: Sequence[BaseTool],
    *,
    run_id: str,
    usage: UsageLog | None = None,
    venue: VenueProfile | None = None,
    repository: RepositoryRef | None = None,
    max_model_calls: int = 20,
    max_tool_calls: int = 40,
) -> BaselineResult:
    # Each middleware is generic over its own state type; the list is typed loosely so
    # mypy accepts the mix.
    limits: list[AgentMiddleware[Any, Any]] = [
        _Accounted(llm, max_model_calls),
        ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=max_tool_calls, exit_behavior="continue"),
    ]
    agent = create_agent(
        llm.model,
        tools=list(tools),
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(ReportedErrors),
        middleware=limits,
    )
    scope = Scope.for_paper(run_id, paper, usage)
    # A model turn takes up to five graph steps: the two limits' hooks, the model and the
    # tools.
    config = RunnableConfig(recursion_limit=6 * max_model_calls + 10)
    message = HumanMessage(render_paper(paper, venue=venue, repository=repository))
    with enter_scope(scope):
        state = await agent.ainvoke({"messages": [message]}, config)

    answer = state.get("structured_response")
    if not isinstance(answer, ReportedErrors):
        raise IncompleteAnswerError(
            f"B1 stopped without a final answer (limit: {max_model_calls} model calls)"
        )
    predictions, dropped = to_predictions(answer, paper, run_id=run_id, system=SystemName.B1)
    return BaselineResult(predictions=predictions, dropped=dropped)
