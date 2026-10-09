"""Pins LangGraph behaviour that the orchestration design depends on.

If a LangGraph upgrade makes this test fail, the upstream behaviour changed and the
failure handling in `auditor.orchestration.runner` can be reconsidered.
"""

import operator
from typing import Annotated, Any, TypedDict

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import NodeError
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, Send


class State(TypedDict):
    tasks: list[str]
    findings: Annotated[list[str], operator.add]


class Branch(TypedDict):
    task: str


async def work(state: Branch) -> dict[str, Any]:
    if state["task"] == "bad":
        raise ValueError("boom")
    return {"findings": [f"ok:{state['task']}"]}


def handler(state: Branch, error: NodeError) -> Command[Any]:
    return Command(update={"findings": [f"failed:{state['task']}"]}, goto="collect")


def build() -> Any:
    graph = StateGraph(State)
    graph.add_node("plan", lambda state: {})
    graph.add_node(
        "work",
        work,
        input_schema=Branch,
        retry_policy=RetryPolicy(max_attempts=1),
        error_handler=handler,
    )
    graph.add_node("collect", lambda state: {})
    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan", lambda state: [Send("work", {"task": t}) for t in state["tasks"]], ["work"]
    )
    graph.add_edge("work", "collect")
    graph.add_edge("collect", END)
    return graph.compile(checkpointer=InMemorySaver())


async def test_error_handler_does_not_isolate_parallel_branch_failures() -> None:
    graph = build()
    config = {"configurable": {"thread_id": "t"}}
    with pytest.raises(ValueError, match="boom"):
        await graph.ainvoke({"tasks": ["good", "bad"], "findings": []}, config)
    # The handler did record the failure before the run was aborted.
    assert sorted(graph.get_state(config).values["findings"]) == ["failed:bad", "ok:good"]


async def test_reinvoking_a_thread_keeps_earlier_reducer_values() -> None:
    graph = build()
    config = {"configurable": {"thread_id": "t"}}
    await graph.ainvoke({"tasks": ["a"], "findings": []}, config)
    second = await graph.ainvoke({"tasks": ["a"], "findings": []}, config)
    assert second["findings"] == ["ok:a", "ok:a"]
