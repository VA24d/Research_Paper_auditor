import asyncio
import random
from pathlib import Path

import pytest
from langchain_core.exceptions import ModelAuthenticationError, ModelRateLimitError

from auditor.infra.context import RunContext
from auditor.infra.http import HttpRequestError
from auditor.infra.scope import current_scope
from auditor.infra.tools import tracked_tool
from auditor.infra.usage import UsageLog
from auditor.orchestration.runner import CheckFn, TaskPolicy, is_transient, run_task
from auditor.schemas import AuditTask, TaskOutcome, TaskResult
from auditor.schemas.examples import example_tasks
from auditor.testing import checked_result, offline_context

TASK = example_tasks()[1]
FAST = TaskPolicy(max_attempts=3, timeout_s=5.0, backoff_base_s=0.0)


async def no_sleep(_: float) -> None:
    return None


async def run(check: CheckFn, tmp_path: Path, policy: TaskPolicy = FAST) -> TaskResult:
    return await run_task(
        check,
        TASK,
        offline_context(tmp_path),
        policy=policy,
        run_id="r1",
        usage=UsageLog(),
        sleep=no_sleep,
        rng=random.Random(0),
    )


async def test_successful_check_runs_inside_the_task_scope(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    async def check(task: AuditTask, ctx: RunContext) -> TaskResult:
        active = current_scope()
        assert active is not None
        seen.update(
            task_id=active.task_id,
            paper_id=active.paper_id,
            variant=active.variant,
            run_id=active.run_id,
        )
        return checked_result(task)

    result = await run(check, tmp_path)
    assert result.outcome is TaskOutcome.CHECKED
    assert result.attempts == 1
    assert seen == {
        "task_id": TASK.task_id,
        "paper_id": TASK.paper_id,
        "variant": TASK.variant,
        "run_id": "r1",
    }


async def test_transient_error_is_retried(tmp_path: Path) -> None:
    calls = 0

    async def check(task: AuditTask, ctx: RunContext) -> TaskResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise HttpRequestError("connection reset")
        return checked_result(task)

    result = await run(check, tmp_path)
    assert result.outcome is TaskOutcome.CHECKED
    assert result.attempts == 2


async def test_permanent_error_fails_without_retry(tmp_path: Path) -> None:
    calls = 0

    async def check(task: AuditTask, ctx: RunContext) -> TaskResult:
        nonlocal calls
        calls += 1
        raise ValueError("bad input")

    result = await run(check, tmp_path)
    assert result.outcome is TaskOutcome.FAILED
    assert result.error == "ValueError: bad input"
    assert result.attempts == 1
    assert calls == 1


async def test_transient_errors_give_up_after_max_attempts(tmp_path: Path) -> None:
    async def check(task: AuditTask, ctx: RunContext) -> TaskResult:
        raise ConnectionError("down")

    result = await run(check, tmp_path)
    assert result.outcome is TaskOutcome.FAILED
    assert result.attempts == 3


async def test_slow_check_times_out_and_fails(tmp_path: Path) -> None:
    async def check(task: AuditTask, ctx: RunContext) -> TaskResult:
        await asyncio.sleep(10)
        return checked_result(task)

    policy = TaskPolicy(max_attempts=2, timeout_s=0.05, backoff_base_s=0.0)
    result = await run(check, tmp_path, policy)
    assert result.outcome is TaskOutcome.FAILED
    assert result.attempts == 2
    assert result.error is not None
    assert result.error.startswith("TimeoutError")


async def test_budget_overrun_fails_the_task(tmp_path: Path) -> None:
    @tracked_tool()
    async def lookup() -> None:
        return None

    async def check(task: AuditTask, ctx: RunContext) -> TaskResult:
        for _ in range(5):
            await lookup()
        return checked_result(task)

    policy = TaskPolicy(max_attempts=3, timeout_s=5.0, backoff_base_s=0.0, max_tool_calls=3)
    result = await run(check, tmp_path, policy)
    assert result.outcome is TaskOutcome.FAILED
    assert result.error is not None
    assert "BudgetExceededError" in result.error
    assert result.attempts == 1


async def test_result_for_another_task_is_rejected(tmp_path: Path) -> None:
    async def check(task: AuditTask, ctx: RunContext) -> TaskResult:
        return checked_result(example_tasks()[0])

    result = await run(check, tmp_path)
    assert result.outcome is TaskOutcome.FAILED
    assert result.error is not None
    assert "different task" in result.error


async def test_cancellation_is_not_swallowed(tmp_path: Path) -> None:
    started = asyncio.Event()

    async def check(task: AuditTask, ctx: RunContext) -> TaskResult:
        started.set()
        await asyncio.sleep(10)
        return checked_result(task)

    pending = asyncio.create_task(run(check, tmp_path))
    await started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending


def test_transient_classification() -> None:
    assert is_transient(TimeoutError())
    assert is_transient(HttpRequestError("x"))
    assert is_transient(ModelRateLimitError("slow down"))
    assert not is_transient(ModelAuthenticationError("bad key"))
    assert not is_transient(ValueError())
