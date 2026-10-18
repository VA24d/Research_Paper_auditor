"""The one place chat models are created and called for structured output (decision D9).

The provider is chosen in config (`AUDITOR_LLM_PROVIDER`, `AUDITOR_LLM_MODEL`), so no
other module depends on which provider the team picks. Structured calls are validated
against a Pydantic model, retried with the validation error when the reply does not fit,
recorded against the current task, and optionally cached on disk.
"""

import asyncio
import contextlib
import functools
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import BaseModel, ValidationError

from auditor.config import Settings
from auditor.infra.files import atomic_write
from auditor.infra.manifest import redact
from auditor.infra.prefix_cache import GenAICacheClient, PrefixCache
from auditor.infra.scope import BudgetExceededError, Scope, current_scope
from auditor.schemas import CallStatus, LLMCallRecord

logger = logging.getLogger(__name__)


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


# How hard the model should think on one call; each provider has its own setting for it.
Effort = Literal["minimal", "low", "medium", "high"]
# Chat model fields by setting name (Gemini's `thinking_level` is the field
# `reasoning_effort`).
_FIELD_NAMES = {"thinking_level": "reasoning_effort"}
# The chat model field that caps the output, by provider.
_OUTPUT_FIELDS = {"google_genai": "max_output_tokens", "openai": "max_tokens"}
# Gemini 2.5 takes a token budget instead of a level.
_GEMINI_25_BUDGETS: dict[str, int] = {"minimal": 0, "low": 1024, "medium": 4096, "high": 12288}


def effort_options(provider: str | None, model: str | None, effort: Effort) -> dict[str, Any]:
    """The chat model settings that make one call think at `effort`, or {} when the
    provider has no such setting."""
    if provider == "google_genai":
        if (model or "").startswith("gemini-2"):
            return {"thinking_budget": _GEMINI_25_BUDGETS[effort]}
        return {"thinking_level": effort}
    if provider == "openai":
        return {"reasoning_effort": effort}
    return {}


def _cache_context(settings: Settings) -> str:
    """Settings that change answers (endpoint, thinking level, temperature), minus secrets."""
    return json.dumps(
        {"base_url": settings.llm_base_url, "options": redact(settings.llm_options)},
        sort_keys=True,
    )


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
        context: str = "",
    ) -> str:
        payload = json.dumps(
            {
                "model": model,
                "context": context,
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


def _text_of(message: AIMessage) -> str:
    """The text of a reply whose content may be a list of parts (Gemini, Anthropic)."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") if isinstance(part, dict) else str(part) for part in content
    )


class CallLimitReachedError(BudgetExceededError):
    """The run has made all the model calls AUDITOR_LLM_MAX_CALLS allows, or spent the
    money AUDITOR_LLM_MAX_COST_USD allows."""


@dataclass
class Spend:
    """What a run's model calls have cost so far, against its budget. The main model and
    the review model share one, so the budget covers both."""

    limit_usd: float | None
    # US dollars per million input and output tokens, by model name; output includes
    # thinking tokens.
    prices: Mapping[str, tuple[float, float]]
    spent_usd: float = 0.0
    # The share of the input price charged for input read from the provider's cache.
    cached_input_share: float = 1.0

    def price(self, model: str) -> tuple[float, float] | None:
        return self.prices.get(model.split(":", 1)[-1])

    def charge(self, model: str, usage: "TokenUsage") -> None:
        price = self.price(model)
        if price is not None:
            cached = min(usage.cached_input_tokens, usage.input_tokens)
            fresh = usage.input_tokens - cached
            input_cost = (fresh + cached * self.cached_input_share) * price[0]
            self.spent_usd += (input_cost + usage.output_tokens * price[1]) / 1e6

    @property
    def exhausted(self) -> bool:
        return self.limit_usd is not None and self.spent_usd >= self.limit_usd


class PromptTooLongError(StructuredOutputError):
    """The prompt does not fit the model's context window; local servers would otherwise
    cut it silently."""


# Rough characters per token for English text and code, used before sending a prompt.
CHARS_PER_TOKEN = 3.5


# What an image in a message costs, in characters' worth of tokens (Gemini counts a few
# hundred tokens per image, not the length of its base64 text).
IMAGE_CHARS = 1_500


def estimated_tokens(messages: Sequence[BaseMessage]) -> int:
    chars = 0
    for message in messages:
        if isinstance(message.content, list):
            for part in message.content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    chars += IMAGE_CHARS
                else:
                    chars += len(str(part.get("text", "")) if isinstance(part, dict) else part)
        else:
            chars += len(str(message.content))
    return int(chars / CHARS_PER_TOKEN)


class StructuredLLM:
    """Calls a chat model and returns a validated instance of a Pydantic schema."""

    def __init__(
        self,
        model: BaseChatModel,
        *,
        name: str,
        method: str | None = None,
        cache: LLMCache | None = None,
        cache_context: str = "",
        max_attempts: int = 3,
        context_tokens: int | None = None,
        max_concurrency: int | None = None,
        max_calls: int | None = None,
        cache_scope: Literal["shared", "run"] = "shared",
        spend: Spend | None = None,
        prefix_cache: PrefixCache | None = None,
        cache_storage_price: float = 0.0,
    ) -> None:
        self.model = model
        self.name = name
        self.cache_context = cache_context
        self.method = method
        self.cache = cache
        self.max_attempts = max_attempts
        self.context_tokens = context_tokens
        self._gate = asyncio.Semaphore(max_concurrency) if max_concurrency else None
        # Model calls this instance may make in total (one per run); cache hits are free.
        self.max_calls = max_calls
        self.calls = 0
        # "run": cached answers are reused only within the run that made them (D10).
        self.cache_scope = cache_scope
        self.spend = spend or Spend(limit_usd=None, prices={})
        # The stronger model for review steps (AUDITOR_LLM_REVIEW_MODEL), if any.
        self.review: StructuredLLM | None = None
        self.provider: str | None = None
        self._variants: dict[tuple[Effort | None, int | None, str | None], BaseChatModel] = {}
        # Explicit context caches for long first messages (Vertex Gemini only), and the
        # storage price charged for each when it is made, in US dollars per million
        # tokens per hour.
        self.prefix_cache = prefix_cache
        self.cache_storage_price = cache_storage_price

    @property
    def limit_reached(self) -> bool:
        calls = self.max_calls is not None and self.calls >= self.max_calls
        return calls or self.spend.exhausted

    @property
    def _bare_name(self) -> str:
        return self.name.split(":", 1)[-1]

    async def _cache_for(self, text: str) -> str | None:
        assert self.prefix_cache is not None
        made = len(self.prefix_cache.created)
        name = await self.prefix_cache.name_for(self._bare_name, text)
        if len(self.prefix_cache.created) > made:
            # Storage for the cache's whole lifetime, charged when it is made.
            hours = self.prefix_cache.ttl_s / 3600
            tokens = len(text) / CHARS_PER_TOKEN
            self.spend.spent_usd += tokens * hours * self.cache_storage_price / 1e6
        return name

    async def aclose(self) -> None:
        """Delete the context caches this model made (the review model shares them)."""
        if self.prefix_cache is not None:
            await self.prefix_cache.close()

    def _check_limits(self) -> None:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise CallLimitReachedError(
                f"the run's {self.max_calls} model calls are used up (AUDITOR_LLM_MAX_CALLS)"
            )
        if self.spend.exhausted:
            raise CallLimitReachedError(
                f"the run's budget of ${self.spend.limit_usd:g} is used up "
                "(AUDITOR_LLM_MAX_COST_USD)"
            )

    @classmethod
    def from_settings(cls, settings: Settings) -> "StructuredLLM":
        """The main model, with the review model attached when one is configured; it has
        its own call limit of the same size."""
        spend = Spend(
            limit_usd=settings.llm_max_cost_usd,
            prices=dict(settings.llm_prices),
            cached_input_share=settings.llm_cached_input_share,
        )
        if spend.limit_usd is not None:
            for model in filter(None, (settings.llm_model, settings.llm_review_model)):
                if spend.price(model) is None:
                    raise ValueError(
                        f"AUDITOR_LLM_MAX_COST_USD needs the price of {model}; add it to "
                        "AUDITOR_LLM_PRICES"
                    )
        prefix_cache = None
        if settings.llm_explicit_cache and settings.llm_provider == "google_genai":
            prefix_cache = PrefixCache(
                GenAICacheClient(dict(settings.llm_options)),
                ttl_s=settings.llm_explicit_cache_ttl_s,
                min_chars=settings.llm_explicit_cache_min_chars,
            )
        llm = cls._one(settings, spend, prefix_cache)
        if settings.llm_review_model:
            options = settings.llm_review_options
            llm.review = cls._one(
                settings.model_copy(
                    update={
                        "llm_model": settings.llm_review_model,
                        "llm_options": settings.llm_options if options is None else options,
                    }
                ),
                spend,
                prefix_cache,
            )
        return llm

    @classmethod
    def _one(
        cls, settings: Settings, spend: Spend, prefix_cache: PrefixCache | None = None
    ) -> "StructuredLLM":
        cache = LLMCache(settings.cache_dir / "llm") if settings.llm_cache else None
        llm = cls(
            create_chat_model(settings),
            name=model_id(settings),
            method=settings.llm_structured_method,
            cache=cache,
            cache_context=_cache_context(settings),
            context_tokens=settings.llm_context_tokens,
            max_concurrency=settings.llm_max_concurrency,
            max_calls=settings.llm_max_calls,
            cache_scope=settings.llm_cache_scope,
            spend=spend,
            prefix_cache=prefix_cache,
            cache_storage_price=settings.llm_cache_storage_price,
        )
        llm.provider = settings.llm_provider
        return llm

    def model_for(
        self,
        effort: Effort | None,
        *,
        max_output_tokens: int | None = None,
        cached_content: str | None = None,
    ) -> BaseChatModel:
        """The chat model with its thinking set to `effort`, its output capped at
        `max_output_tokens` and reading the context cache `cached_content` (the
        configured model for None, or for a provider without such settings)."""
        key = (effort, max_output_tokens, cached_content)
        if key == (None, None, None):
            return self.model
        if key not in self._variants:
            update: dict[str, Any] = {}
            if effort is not None:
                options = effort_options(self.provider, self.name.split(":", 1)[-1], effort)
                update.update({_FIELD_NAMES.get(k, k): v for k, v in options.items()})
                if "reasoning_effort" in update:
                    # A level replaces a configured token budget (Gemini 3 warns about both).
                    update.setdefault("thinking_budget", None)
            if max_output_tokens is not None:
                field = _OUTPUT_FIELDS.get(self.provider or "")
                if field is not None:
                    update[field] = max_output_tokens
            if cached_content is not None:
                update["cached_content"] = cached_content
            self._variants[key] = self.model.model_copy(update=update) if update else self.model
        return self._variants[key]

    async def generate[T: BaseModel](
        self,
        schema: type[T],
        messages: Sequence[BaseMessage],
        *,
        effort: Effort | None = None,
        purpose: str | None = None,
        sample: int = 0,
        max_output_tokens: int | None = None,
        cache_prefix: bool = False,
    ) -> StructuredResult[T]:
        """A validated `schema` value. `effort` sets how hard the model thinks on this call
        (None keeps the configured setting); `purpose` names the step in the usage log;
        `sample` numbers independent tries of the same question, which are cached apart;
        `max_output_tokens` caps this call's answer, so a model stuck repeating itself
        stops early; `cache_prefix` lets the first (system) message come from an explicit
        context cache, when the model has one (AUDITOR_LLM_EXPLICIT_CACHE)."""
        record = functools.partial(self._record, effort=effort, purpose=purpose)
        context = self.cache_context if effort is None else f"{self.cache_context}|{effort}"
        active = current_scope()
        if self.cache_scope == "run" and active is not None:
            context += f"|run={active.run_id}"
        if sample:
            context += f"|sample={sample}"
        if max_output_tokens is not None:
            context += f"|max_output={max_output_tokens}"
        cache_key = None
        if self.cache is not None:
            cache_key = LLMCache.key(self.name, self.method, schema, messages, context)
            cached = self.cache.get(cache_key, schema)
            if cached is not None:
                record(TokenUsage(), datetime.now(UTC), 0.0, cache_hit=True)
                return StructuredResult(cached, attempts=0, usage=TokenUsage(), from_cache=True)

        if self.context_tokens is not None:
            needed = estimated_tokens(messages)
            if needed > self.context_tokens:
                raise PromptTooLongError(
                    f"{schema.__name__}: the prompt needs about {needed} tokens, more than the "
                    f"model's context of {self.context_tokens} (AUDITOR_LLM_CONTEXT_TOKENS)"
                )

        kwargs: dict[str, Any] = {"include_raw": True}
        if self.method is not None:
            kwargs["method"] = self.method
        model = self.model_for(effort, max_output_tokens=max_output_tokens)
        plain = model.with_structured_output(schema, **kwargs)
        head = messages[0] if messages else None
        cache_name = None
        if cache_prefix and self.prefix_cache is not None and isinstance(head, SystemMessage):
            cache_name = await self._cache_for(str(head.content))

        def runnable_for(turns: list[BaseMessage]) -> tuple[Any, list[BaseMessage], bool]:
            # A conversation that starts with the cached message names the cache instead
            # of sending it again.
            if cache_name is not None and turns and turns[0] is head:
                cached = self.model_for(
                    effort, max_output_tokens=max_output_tokens, cached_content=cache_name
                )
                return cached.with_structured_output(schema, **kwargs), turns[1:], True
            return plain, turns, False

        conversation = list(messages)
        total = TokenUsage()
        problem = ""
        for attempt in range(1, self.max_attempts + 1):
            runnable, sent, from_cache = runnable_for(conversation)
            active = current_scope()
            if active is not None:
                active.budget.charge_llm_call()
            self._check_limits()
            async with self._gate or contextlib.nullcontext():
                # Checked again once a slot is free: calls that were waiting for one when
                # the budget ran out are not sent.
                self._check_limits()
                self.calls += 1
                started_at = datetime.now(UTC)
                started = time.perf_counter()
                try:
                    output = await runnable.ainvoke(sent)
                except Exception as exc:
                    record(
                        TokenUsage(),
                        started_at,
                        time.perf_counter() - started,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    if not from_cache or attempt == self.max_attempts:
                        raise
                    # The cache may have expired or been refused: send the message whole.
                    logger.warning("call with a context cache failed (%s); sending it whole", exc)
                    assert self.prefix_cache is not None
                    assert head is not None
                    self.prefix_cache.forget(self._bare_name, str(head.content))
                    cache_name = None
                    continue
            if not isinstance(output, dict):
                raise TypeError(f"expected a dict from include_raw, got {type(output).__name__}")
            raw = output["raw"]
            usage = TokenUsage.from_message(raw)
            total += usage
            self.spend.charge(self.name, usage)
            record(usage, started_at, time.perf_counter() - started)

            parsed = output.get("parsed")
            error = output.get("parsing_error")
            if isinstance(parsed, schema) and error is None:
                if self.cache is not None and cache_key is not None:
                    self.cache.put(cache_key, parsed)
                return StructuredResult(parsed, attempts=attempt, usage=total)

            problem = str(error) if error is not None else "no structured output was returned"
            # Feed the problem back as plain text. Replaying the raw reply would break
            # providers that require every tool call to be followed by a tool result.
            reply = _text_of(raw) if isinstance(raw, AIMessage) else ""
            if reply.strip():
                # The model answered in prose: ask it to restate that answer in the required
                # format, without resending the (possibly long) prompt.
                conversation = [
                    SystemMessage(
                        "Rewrite the answer below in the required output format. Keep its "
                        "content; do not add anything. Fields: "
                        + json.dumps(schema.model_json_schema().get("properties", {}))
                    ),
                    HumanMessage(
                        "Your previous reply could not be used.\n"
                        f"Answer: {reply[:4000]}\nProblem: {problem[:1000]}"
                    ),
                ]
            else:
                conversation = [
                    *messages,
                    HumanMessage(
                        "Your previous reply could not be used.\n"
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
        effort: Effort | None = None,
        purpose: str | None = None,
    ) -> None:
        active = current_scope()
        if active is None:
            return
        active.record(
            LLMCallRecord(
                run_id=active.run_id,
                paper_id=active.paper_id,
                variant=active.variant,
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
                effort=effort,
                purpose=purpose,
            )
        )


class UsageCallback(AsyncCallbackHandler):
    """Records every chat-model call made under it, for code that calls models directly
    (such as the B1 agent) instead of through StructuredLLM."""

    def __init__(self, scope: Scope, model_name: str) -> None:
        self._scope = scope
        self._model_name = model_name
        self._started: dict[UUID, tuple[datetime, float]] = {}

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._started[run_id] = (datetime.now(UTC), time.perf_counter())

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        usage = TokenUsage()
        for generations in response.generations:
            for generation in generations:
                if isinstance(generation, ChatGeneration):
                    usage += TokenUsage.from_message(generation.message)
        self._add(run_id, usage, None)

    async def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._add(run_id, TokenUsage(), f"{type(error).__name__}: {error}")

    def _add(self, run_id: UUID, usage: TokenUsage, error: str | None) -> None:
        started_at, started = self._started.pop(run_id, (datetime.now(UTC), time.perf_counter()))
        self._scope.record(
            LLMCallRecord(
                run_id=self._scope.run_id,
                paper_id=self._scope.paper_id,
                variant=self._scope.variant,
                task_id=self._scope.task_id,
                model=self._model_name,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                reasoning_tokens=usage.reasoning_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                started_at=started_at,
                duration_s=time.perf_counter() - started,
                status=CallStatus.ERROR if error else CallStatus.OK,
                error=error,
            )
        )


@contextlib.asynccontextmanager
async def closing_llm(llm: StructuredLLM | None) -> AsyncIterator[StructuredLLM | None]:
    """`llm`, with its context caches deleted when the block ends."""
    try:
        yield llm
    finally:
        if llm is not None:
            await llm.aclose()
