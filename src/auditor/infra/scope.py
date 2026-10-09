"""Which run and task the current coroutine is working for.

The orchestrator enters a scope around every task, so the HTTP client, tool counting and
LLM usage tracking can attribute calls to the right task without passing ids around.
Each asyncio task gets its own copy of the scope, so parallel tasks do not interfere.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from auditor.infra.usage import UsageLog, UsageRecord


class BudgetExceededError(Exception):
    """A task used more tool or LLM calls than its budget allows."""


@dataclass
class Budget:
    max_tool_calls: int | None = None
    max_llm_calls: int | None = None
    tool_calls: int = field(default=0, init=False)
    llm_calls: int = field(default=0, init=False)

    def charge_tool_call(self) -> None:
        if self.max_tool_calls is not None and self.tool_calls >= self.max_tool_calls:
            raise BudgetExceededError(f"tool call budget of {self.max_tool_calls} used up")
        self.tool_calls += 1

    def charge_llm_call(self) -> None:
        if self.max_llm_calls is not None and self.llm_calls >= self.max_llm_calls:
            raise BudgetExceededError(f"LLM call budget of {self.max_llm_calls} used up")
        self.llm_calls += 1


@dataclass(frozen=True)
class Scope:
    run_id: str
    paper_id: str | None = None
    task_id: str | None = None
    budget: Budget = field(default_factory=Budget)
    usage: UsageLog | None = None

    def record(self, record: UsageRecord) -> None:
        if self.usage is not None:
            self.usage.add(record)


_current: ContextVar[Scope | None] = ContextVar("auditor_scope", default=None)


def current_scope() -> Scope | None:
    return _current.get()


@contextmanager
def enter_scope(value: Scope) -> Iterator[Scope]:
    token = _current.set(value)
    try:
        yield value
    finally:
        _current.reset(token)
