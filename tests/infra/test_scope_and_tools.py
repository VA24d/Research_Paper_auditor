import asyncio
import json
from pathlib import Path

import pytest

from auditor.infra.scope import Budget, BudgetExceededError, Scope, current_scope, enter_scope
from auditor.infra.tools import args_hash, tracked_tool
from auditor.infra.usage import UsageLog
from auditor.schemas import CallStatus, SentenceAnchor


@tracked_tool()
async def lookup(title: str) -> str:
    await asyncio.sleep(0)
    return title.upper()


@tracked_tool("read_file")
async def read_local(path: str) -> str:
    raise FileNotFoundError(path)


def test_scope_is_restored_after_exit() -> None:
    assert current_scope() is None
    with enter_scope(Scope(run_id="r1")) as active:
        assert current_scope() is active
    assert current_scope() is None


def test_budget_raises_when_used_up() -> None:
    budget = Budget(max_tool_calls=2, max_llm_calls=0)
    budget.charge_tool_call()
    budget.charge_tool_call()
    with pytest.raises(BudgetExceededError):
        budget.charge_tool_call()
    with pytest.raises(BudgetExceededError):
        budget.charge_llm_call()


async def test_tool_calls_are_recorded_and_charged() -> None:
    usage = UsageLog()
    budget = Budget(max_tool_calls=5)
    with enter_scope(Scope(run_id="r1", paper_id="p1", task_id="t1", budget=budget, usage=usage)):
        assert await lookup("attention") == "ATTENTION"
        with pytest.raises(FileNotFoundError):
            await read_local("train.py")
    assert budget.tool_calls == 2
    ok, failed = usage.tool_calls
    assert (ok.tool, ok.status, ok.task_id) == ("lookup", CallStatus.OK, "t1")
    assert (failed.tool, failed.status) == ("read_file", CallStatus.ERROR)
    assert failed.error is not None
    assert "FileNotFoundError" in failed.error


async def test_tool_call_over_budget_is_not_run() -> None:
    ran = False

    @tracked_tool()
    async def expensive() -> None:
        nonlocal ran
        ran = True

    with (
        enter_scope(Scope(run_id="r1", budget=Budget(max_tool_calls=0))),
        pytest.raises(BudgetExceededError),
    ):
        await expensive()
    assert not ran


async def test_tools_work_without_a_scope() -> None:
    assert await lookup("x") == "X"


async def test_parallel_tasks_keep_their_own_scope() -> None:
    usage = UsageLog()

    async def run(task_id: str) -> None:
        with enter_scope(Scope(run_id="r1", task_id=task_id, usage=usage)):
            for _ in range(3):
                await lookup(task_id)

    await asyncio.gather(run("a"), run("b"))
    by_task = {r.task_id for r in usage.tool_calls if r.args_hash == args_hash(("a",), {})}
    assert by_task == {"a"}
    assert len(usage.tool_calls) == 6


def test_args_hash_is_stable_and_ignores_object_identity() -> None:
    class Client:
        pass

    anchor = SentenceAnchor(sentence_id="s1")
    first = args_hash((Client(), anchor, [1, 2]), {"q": "x"})
    second = args_hash((Client(), anchor, (1, 2)), {"q": "x"})
    assert first == second
    assert first != args_hash((Client(), anchor, [1, 3]), {"q": "x"})


def test_usage_log_writes_json_lines(tmp_path: Path) -> None:
    usage = UsageLog(tmp_path / "usage")

    async def go() -> None:
        with enter_scope(Scope(run_id="r1", usage=usage)):
            await lookup("a")
            await lookup("b")

    asyncio.run(go())
    lines = (tmp_path / "usage" / "tool_calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["tool"] for line in lines] == ["lookup", "lookup"]
    assert b"\r\n" not in (tmp_path / "usage" / "tool_calls.jsonl").read_bytes()
