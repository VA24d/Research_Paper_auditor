"""Runs one specialist check with retries, a timeout and failure isolation.

LangGraph 1.2.14 aborts the whole run when an exception leaves one of several parallel
`Send` branches, even if the node has an `error_handler` (see
`tests/orchestration/test_langgraph_behaviour.py`). So nothing may escape a specialist
node: transient errors are retried here and anything left becomes a `failed` result.
"""

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from langchain_core.exceptions import ModelError

from auditor.infra.context import RunContext
from auditor.infra.http import HttpRequestError
from auditor.infra.scope import Budget, Scope, enter_scope
from auditor.infra.usage import UsageLog
from auditor.schemas import AuditTask, TaskOutcome, TaskResult

logger = logging.getLogger(__name__)

CheckFn = Callable[[AuditTask, RunContext], Awaitable[TaskResult]]


@dataclass(frozen=True)
class TaskPolicy:
    max_attempts: int = 3
    timeout_s: float = 300.0
    backoff_base_s: float = 2.0
    max_tool_calls: int | None = 30
    max_llm_calls: int | None = 10


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, ModelError):
        return exc.is_retryable
    return isinstance(exc, TimeoutError | ConnectionError | HttpRequestError)


async def run_task(
    check: CheckFn,
    task: AuditTask,
    ctx: RunContext,
    *,
    policy: TaskPolicy,
    run_id: str,
    usage: UsageLog | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rng: random.Random | None = None,
) -> TaskResult:
    """Never raises (except on cancellation); failures come back as `failed` results."""
    rng = rng or random.Random()
    budget = Budget(max_tool_calls=policy.max_tool_calls, max_llm_calls=policy.max_llm_calls)
    task_scope = Scope(
        run_id=run_id, paper_id=task.paper_id, task_id=task.task_id, budget=budget, usage=usage
    )
    attempt = 0
    with enter_scope(task_scope):
        while True:
            attempt += 1
            try:
                async with asyncio.timeout(policy.timeout_s):
                    result = await check(task, ctx)
                _check_belongs_to(result, task)
                return result.model_copy(update={"attempts": attempt})
            except Exception as exc:
                if is_transient(exc) and attempt < policy.max_attempts:
                    delay = rng.uniform(0, policy.backoff_base_s * 2 ** (attempt - 1))
                    logger.warning("attempt %d failed (%r); retrying in %.1fs", attempt, exc, delay)
                    await sleep(delay)
                    continue
                logger.error("task failed after %d attempt(s): %r", attempt, exc)
                return TaskResult(
                    task_id=task.task_id,
                    paper_id=task.paper_id,
                    variant=task.variant,
                    kind=task.kind,
                    outcome=TaskOutcome.FAILED,
                    error=f"{type(exc).__name__}: {exc}"[:2000],
                    attempts=attempt,
                )


def _check_belongs_to(result: TaskResult, task: AuditTask) -> None:
    if (result.task_id, result.paper_id, result.variant, result.kind) != (
        task.task_id,
        task.paper_id,
        task.variant,
        task.kind,
    ):
        raise ValueError(f"specialist returned a result for a different task ({result.task_id})")
