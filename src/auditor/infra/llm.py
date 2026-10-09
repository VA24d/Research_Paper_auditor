"""The one place chat models are created and called for structured output (decision D9).

The provider is chosen in config (`AUDITOR_LLM_PROVIDER`, `AUDITOR_LLM_MODEL`), so no
other module depends on which provider the team picks. Structured calls are validated
against a Pydantic model, retried with the validation error when the reply does not fit,
recorded against the current task, and optionally cached on disk.
"""

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from pydantic import BaseModel, ValidationError

from auditor.config import Settings
from auditor.infra.files import atomic_write
from auditor.infra.scope import current_scope
from auditor.schemas import CallStatus, LLMCallRecord


class LLMNotConfiguredError(RuntimeError):
    pass


class StructuredOutputError(RuntimeError):
    """The model did not produce output matching the schema within the allowed attempts."""


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0

    @classmethod
    def from_message(cls, message: BaseMessage) -> "TokenUsage":
        usage = getattr(message, "usage_metadata", None) or {}
        input_details = usage.get("input_token_details") or {}
        output_details = usage.get("output_token_details") or {}
        return cls(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            reasoning_tokens=output_details.get("reasoning", 0) or 0,
            cached_input_tokens=input_details.get("cache_read", 0) or 0,
        )

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
            self.cached_input_tokens + other.cached_input_tokens,
        )


@dataclass(frozen=True)
class StructuredResult[T: BaseModel]:
    value: T
    attempts: int
    usage: TokenUsage
    from_cache: bool = False


def model_id(settings: Settings) -> str:
    return f"{settings.llm_provider}:{settings.llm_model}"


def create_chat_model(settings: Settings) -> BaseChatModel:
    if not settings.llm_provider or not settings.llm_model:
        raise LLMNotConfiguredError(
            "Set AUDITOR_LLM_PROVIDER and AUDITOR_LLM_MODEL (see .env.example)."
        )
    kwargs: dict[str, Any] = {
        "timeout": settings.llm_timeout_s,
        "max_retries": settings.llm_max_retries,
        **settings.llm_options,
    }
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    model = init_chat_model(settings.llm_model, model_provider=settings.llm_provider, **kwargs)
    if not isinstance(model, BaseChatModel):
        raise TypeError(f"expected a chat model, got {type(model).__name__}")
    return model


class LLMCache:
    """Validated structured results on disk, keyed by model, schema and messages."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    @staticmethod
    def key(
        model: str,
        method: str | None,
        schema: type[BaseModel],
        messages: Sequence[BaseMessage],
    ) -> str:
        payload = json.dumps(
            {
                "model": model,
                "method": method,
                "schema": schema.model_json_schema(),
                "messages": [{"type": m.type, "content": m.content} for m in messages],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get[T: BaseModel](self, key: str, schema: type[T]) -> T | None:
        try:
            data = json.loads(self._path(key).read_text(encoding="utf-8"))
            return schema.model_validate(data["value"])
        except (OSError, ValueError, KeyError, ValidationError):
            return None

    def put(self, key: str, value: BaseModel) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "value": value.model_dump(mode="json"),
            "stored_at": datetime.now(UTC).isoformat(),
        }
        atomic_write(path, json.dumps(payload).encode("utf-8"))


class StructuredLLM:
    """Calls a chat model and returns a validated instance of a Pydantic schema."""

    def __init__(
        self,
        model: BaseChatModel,
        *,
        name: str,
        method: str | None = None,
        cache: LLMCache | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.model = model
        self.name = name
        self.method = method
        self.cache = cache
        self.max_attempts = max_attempts

    @classmethod
    def from_settings(cls, settings: Settings) -> "StructuredLLM":
        cache = LLMCache(settings.cache_dir / "llm") if settings.llm_cache else None
        return cls(
            create_chat_model(settings),
            name=model_id(settings),
            method=settings.llm_structured_method,
            cache=cache,
        )

    async def generate[T: BaseModel](
        self, schema: type[T], messages: Sequence[BaseMessage]
    ) -> StructuredResult[T]:
        cache_key = None
        if self.cache is not None:
            cache_key = LLMCache.key(self.name, self.method, schema, messages)
            cached = self.cache.get(cache_key, schema)
            if cached is not None:
                self._record(TokenUsage(), datetime.now(UTC), 0.0, cache_hit=True)
                return StructuredResult(cached, attempts=0, usage=TokenUsage(), from_cache=True)

        kwargs: dict[str, Any] = {"include_raw": True}
        if self.method is not None:
            kwargs["method"] = self.method
        runnable = self.model.with_structured_output(schema, **kwargs)

        conversation = list(messages)
        total = TokenUsage()
        problem = ""
        for attempt in range(1, self.max_attempts + 1):
            active = current_scope()
            if active is not None:
                active.budget.charge_llm_call()
            started_at = datetime.now(UTC)
            started = time.perf_counter()
            try:
                output = await runnable.ainvoke(conversation)
            except Exception as exc:
                self._record(
                    TokenUsage(),
                    started_at,
                    time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
                raise
            if not isinstance(output, dict):
                raise TypeError(f"expected a dict from include_raw, got {type(output).__name__}")
            raw = output["raw"]
            usage = TokenUsage.from_message(raw)
            total += usage
            self._record(usage, started_at, time.perf_counter() - started)

            parsed = output.get("parsed")
            error = output.get("parsing_error")
            if isinstance(parsed, schema) and error is None:
                if self.cache is not None and cache_key is not None:
                    self.cache.put(cache_key, parsed)
                return StructuredResult(parsed, attempts=attempt, usage=total)

            problem = str(error) if error is not None else "no structured output was returned"
            # Feed the problem back as plain text. Replaying the raw reply would break
            # providers that require every tool call to be followed by a tool result.
            reply = raw.content if isinstance(raw, AIMessage) else ""
            conversation = [
                *messages,
                HumanMessage(
                    "Your previous reply could not be used.\n"
                    f"Reply: {str(reply)[:2000]}\n"
                    f"Problem: {problem[:2000]}\n"
                    "Answer again and follow the required output format exactly."
                ),
            ]
        raise StructuredOutputError(
            f"{schema.__name__}: no valid output after {self.max_attempts} attempts: {problem}"
        )

    def _record(
        self,
        usage: TokenUsage,
        started_at: datetime,
        duration_s: float,
        *,
        cache_hit: bool = False,
        error: str | None = None,
    ) -> None:
        active = current_scope()
        if active is None:
            return
        active.record(
            LLMCallRecord(
                run_id=active.run_id,
                task_id=active.task_id,
                model=self.name,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                cache_hit=cache_hit,
                started_at=started_at,
                duration_s=duration_s,
                status=CallStatus.ERROR if error else CallStatus.OK,
                error=error,
            )
        )
