"""Counting tool calls (decision D8).

Every shared tool function (decision D7) is decorated with `tracked_tool`, so each call is
charged to the current task's budget and recorded, whether the tool goes over the network
or only reads local files.
"""

import functools
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from auditor.infra.scope import current_scope
from auditor.schemas import CallStatus, ToolCallRecord

_JSON_SCALARS = (str, int, float, bool, type(None))


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, _JSON_SCALARS):
        return value
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    # Objects such as clients or contexts do not identify the call; keep only their type.
    return f"<{type(value).__name__}>"


def args_hash(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    payload = json.dumps(
        {"args": _jsonable(args), "kwargs": _jsonable(kwargs)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def tracked_tool[**P, R](
    name: str | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    def decorate(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        tool_name = name or fn.__name__

        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            active = current_scope()
            if active is not None:
                active.budget.charge_tool_call()
            started_at = datetime.now(UTC)
            started = time.perf_counter()
            error: str | None = None
            try:
                return await fn(*args, **kwargs)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                if active is not None:
                    active.record(
                        ToolCallRecord(
                            run_id=active.run_id,
                            paper_id=active.paper_id,
                            task_id=active.task_id,
                            tool=tool_name,
                            args_hash=args_hash(args, kwargs),
                            started_at=started_at,
                            duration_s=time.perf_counter() - started,
                            status=CallStatus.ERROR if error else CallStatus.OK,
                            error=error,
                        )
                    )

        return wrapper

    return decorate
