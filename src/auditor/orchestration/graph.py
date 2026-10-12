"""The audit workflow: plan, run one branch per task in parallel, collect, verify."""

import operator
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, Protocol, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from auditor.infra.context import RunContext
from auditor.infra.scope import Scope, enter_scope
from auditor.infra.usage import UsageLog
from auditor.orchestration.mocks import check_nothing, send_all_to_review
from auditor.orchestration.runner import CheckFn, TaskPolicy, run_task
from auditor.schemas import (
    AuditTask,
    ParsedPaper,
    SystemName,
    TaskKind,
    TaskResult,
    VerifiedFinding,
)

PlanFn = Callable[[ParsedPaper, RunContext], Awaitable[list[AuditTask]]]
VerifyFn = Callable[[Sequence[TaskResult], RunContext], Awaitable[list[VerifiedFinding]]]

NODE_FOR_KIND = {
    TaskKind.REFERENCE: "citation",
    TaskKind.CITATION: "citation",
    TaskKind.CODE: "code",
    TaskKind.TEMPLATE: "template",
}
SPECIALIST_NODES = ("citation", "code", "template")


class ThreadInUseError(RuntimeError):
    """A fresh run was started on a checkpoint thread that already holds state."""


class NothingToResumeError(RuntimeError):
    """A resume was requested for a checkpoint thread with no saved state."""


class AuditState(TypedDict):
    paper: ParsedPaper
    tasks: list[AuditTask]
    results: Annotated[list[TaskResult], operator.add]
    verified: list[VerifiedFinding]


class TaskInput(TypedDict):
    task: AuditTask


class _TaskNode(Protocol):
    def __call__(self, state: TaskInput) -> Awaitable[dict[str, Any]]: ...


@dataclass(frozen=True)
class Specialists:
    """One check function per specialist node; mocks until the real modules land."""

    citation: CheckFn = check_nothing
    code: CheckFn = check_nothing
    template: CheckFn = check_nothing


@dataclass(frozen=True)
class AuditPipeline:
    ctx: RunContext
    planner: PlanFn
    specialists: Specialists = field(default_factory=Specialists)
    verifier: VerifyFn = send_all_to_review
    policy: TaskPolicy = field(default_factory=TaskPolicy)
    max_concurrency: int = 8


@dataclass(frozen=True)
class AuditOutcome:
    tasks: list[AuditTask]
    results: list[TaskResult]
    verified: list[VerifiedFinding]


def thread_id(run_id: str, system: SystemName, paper: ParsedPaper) -> str:
    """One checkpoint thread per execution (decision D10)."""
    return f"{run_id}/{system.value}/{paper.paper_id}/{paper.variant.value}"


def build_graph(
    pipeline: AuditPipeline, *, run_id: str, usage: UsageLog | None
) -> StateGraph[AuditState, None, AuditState, AuditState]:
    ctx = pipeline.ctx

    async def plan(state: AuditState) -> dict[str, Any]:
        paper = state["paper"]
        with enter_scope(Scope.for_paper(run_id, paper, usage)):
            tasks = await pipeline.planner(paper, ctx)
        return {"tasks": tasks}

    def fan_out(state: AuditState) -> list[Send] | str:
        if not state["tasks"]:
            return "verify"
        return [Send(NODE_FOR_KIND[task.kind], {"task": task}) for task in state["tasks"]]

    def specialist_node(check: CheckFn) -> _TaskNode:
        async def node(state: TaskInput) -> dict[str, Any]:
            result = await run_task(
                check, state["task"], ctx, policy=pipeline.policy, run_id=run_id, usage=usage
            )
            return {"results": [result]}

        return node

    async def verify(state: AuditState) -> dict[str, Any]:
        paper = state["paper"]
        with enter_scope(Scope.for_paper(run_id, paper, usage)):
            verified = await pipeline.verifier(state["results"], ctx)
        return {"verified": verified}

    graph = StateGraph(AuditState)
    graph.add_node("plan", plan)
    for name, check in (
        ("citation", pipeline.specialists.citation),
        ("code", pipeline.specialists.code),
        ("template", pipeline.specialists.template),
    ):
        graph.add_node(name, specialist_node(check), input_schema=TaskInput)
    graph.add_node("verify", verify)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", fan_out, [*SPECIALIST_NODES, "verify"])
    # Separate edges, so the verifier still runs when some specialists get no tasks.
    for name in SPECIALIST_NODES:
        graph.add_edge(name, "verify")
    graph.add_edge("verify", END)
    return graph


def compile_graph(
    pipeline: AuditPipeline,
    *,
    run_id: str,
    usage: UsageLog | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[AuditState, None, AuditState, AuditState]:
    return build_graph(pipeline, run_id=run_id, usage=usage).compile(checkpointer=checkpointer)


async def run_audit(
    pipeline: AuditPipeline,
    paper: ParsedPaper,
    *,
    run_id: str,
    system: SystemName,
    usage: UsageLog | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    resume: bool = False,
) -> AuditOutcome:
    """Audit one paper. With `resume`, continue an interrupted run on the same thread."""
    graph = compile_graph(pipeline, run_id=run_id, usage=usage, checkpointer=checkpointer)
    config = RunnableConfig(
        configurable={"thread_id": thread_id(run_id, system, paper)},
        max_concurrency=pipeline.max_concurrency,
    )
    initial: AuditState | None = None
    has_state = checkpointer is not None and bool((await graph.aget_state(config)).values)
    if resume and not has_state:
        raise NothingToResumeError(
            f"thread {thread_id(run_id, system, paper)} has no saved state to resume"
        )
    if not resume:
        if has_state:
            raise ThreadInUseError(
                f"thread {thread_id(run_id, system, paper)} already has state; "
                "use a new run id, or resume=True to continue it"
            )
        initial = {"paper": paper, "tasks": [], "results": [], "verified": []}
    state = await graph.ainvoke(initial, config)
    tasks: list[AuditTask] = state["tasks"]
    order = {task.task_id: index for index, task in enumerate(tasks)}
    results = sorted(state["results"], key=lambda r: order.get(r.task_id, len(order)))
    return AuditOutcome(tasks=tasks, results=results, verified=state["verified"])
