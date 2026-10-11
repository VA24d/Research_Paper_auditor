from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from auditor.config import Settings
from auditor.infra import llm as llm_module
from auditor.infra.llm import (
    LLMCache,
    LLMNotConfiguredError,
    StructuredLLM,
    StructuredOutputError,
    TokenUsage,
    create_chat_model,
)
from auditor.infra.scope import Budget, BudgetExceededError, Scope, enter_scope
from auditor.infra.usage import UsageLog
from auditor.schemas import CallStatus
from auditor.testing import ScriptedChatModel, tool_reply


class Verdict(BaseModel):
    supported: bool
    reason: str


MESSAGES = [SystemMessage("You check claims."), HumanMessage("Is the claim supported?")]


def structured(model: ScriptedChatModel, cache: LLMCache | None = None) -> StructuredLLM:
    return StructuredLLM(model, name="test:scripted", cache=cache, max_attempts=3)


def test_create_chat_model_requires_configuration() -> None:
    with pytest.raises(LLMNotConfiguredError):
        create_chat_model(Settings())


def test_create_chat_model_passes_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_init(model: str, *, model_provider: str, **kwargs: Any) -> ScriptedChatModel:
        captured.update(model=model, model_provider=model_provider, **kwargs)
        return ScriptedChatModel()

    monkeypatch.setattr(llm_module, "init_chat_model", fake_init)
    settings = Settings(
        llm_provider="openai",
        llm_model="glm-5.3-flash",
        llm_base_url="https://api.example/v4/",
        llm_timeout_s=30,
        llm_max_retries=1,
        llm_options={"temperature": 0},
    )
    create_chat_model(settings)
    assert captured == {
        "model": "glm-5.3-flash",
        "model_provider": "openai",
        "base_url": "https://api.example/v4/",
        "timeout": 30,
        "max_retries": 1,
        "temperature": 0,
    }


@pytest.mark.parametrize(
    ("provider", "model", "class_name"),
    [
        ("google_genai", "gemini-3.5-flash-lite", "ChatGoogleGenerativeAI"),
        ("openai", "gpt-test", "ChatOpenAI"),
    ],
)
def test_create_chat_model_builds_provider_classes_offline(
    provider: str, model: str, class_name: str
) -> None:
    settings = Settings(llm_provider=provider, llm_model=model, llm_options={"api_key": "test"})
    assert type(create_chat_model(settings)).__name__ == class_name


async def test_generate_returns_validated_value_and_records_usage() -> None:
    model = ScriptedChatModel(
        replies=[tool_reply(Verdict, {"supported": True, "reason": "quoted"}, input_tokens=40)]
    )
    usage = UsageLog()
    budget = Budget(max_llm_calls=5)
    with enter_scope(Scope(run_id="r1", task_id="t1", budget=budget, usage=usage)):
        result = await structured(model).generate(Verdict, MESSAGES)
    assert result.value == Verdict(supported=True, reason="quoted")
    assert result.attempts == 1
    assert result.usage == TokenUsage(input_tokens=40, output_tokens=5)
    assert budget.llm_calls == 1
    (record,) = usage.llm_calls
    assert (record.task_id, record.input_tokens, record.status) == ("t1", 40, CallStatus.OK)
    assert model.bound_tools == [Verdict]


async def test_invalid_reply_is_retried_with_feedback() -> None:
    model = ScriptedChatModel(
        replies=[
            tool_reply(Verdict, {"supported": "maybe"}),
            tool_reply(Verdict, {"supported": False, "reason": "not in source"}),
        ]
    )
    result = await structured(model).generate(Verdict, MESSAGES)
    assert result.value.supported is False
    assert result.attempts == 2
    assert result.usage.input_tokens == 20
    retry_messages = model.received[1]
    assert retry_messages[: len(MESSAGES)] == MESSAGES
    feedback = retry_messages[-1]
    assert isinstance(feedback, HumanMessage)
    assert "could not be used" in str(feedback.content)


async def test_reply_without_tool_call_counts_as_failure() -> None:
    model = ScriptedChatModel(
        replies=[
            AIMessage(content="I think it is supported."),
            tool_reply(Verdict, {"supported": True, "reason": "ok"}),
        ]
    )
    result = await structured(model).generate(Verdict, MESSAGES)
    assert result.attempts == 2


async def test_gives_up_after_max_attempts() -> None:
    model = ScriptedChatModel(replies=[AIMessage(content="no") for _ in range(3)])
    with pytest.raises(StructuredOutputError, match="after 3 attempts"):
        await structured(model).generate(Verdict, MESSAGES)
    assert len(model.received) == 3


async def test_cache_serves_repeated_calls(tmp_path: Path) -> None:
    cache = LLMCache(tmp_path / "llm")
    model = ScriptedChatModel(
        replies=[
            tool_reply(Verdict, {"supported": True, "reason": "first"}),
            tool_reply(Verdict, {"supported": False, "reason": "second"}),
        ]
    )
    usage = UsageLog()
    with enter_scope(Scope(run_id="r1", usage=usage)):
        first = await structured(model, cache).generate(Verdict, MESSAGES)
        again = await structured(model, cache).generate(Verdict, MESSAGES)
        other = await structured(model, cache).generate(Verdict, [HumanMessage("Different?")])
    assert again.from_cache
    assert again.value == first.value
    assert other.value.reason == "second"
    assert len(model.received) == 2
    assert [r.cache_hit for r in usage.llm_calls] == [False, True, False]


async def test_model_errors_are_recorded_and_raised() -> None:
    model = ScriptedChatModel(replies=[])
    usage = UsageLog()
    with enter_scope(Scope(run_id="r1", usage=usage)), pytest.raises(RuntimeError):
        await structured(model).generate(Verdict, MESSAGES)
    assert usage.llm_calls[0].status is CallStatus.ERROR


async def test_llm_budget_is_enforced_before_calling() -> None:
    model = ScriptedChatModel(replies=[tool_reply(Verdict, {"supported": True, "reason": "x"})])
    with (
        enter_scope(Scope(run_id="r1", budget=Budget(max_llm_calls=0))),
        pytest.raises(BudgetExceededError),
    ):
        await structured(model).generate(Verdict, MESSAGES)
    assert model.received == []


def test_token_usage_reads_reasoning_and_cache_details() -> None:
    message = AIMessage(
        content="",
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 30,
            "total_tokens": 130,
            "input_token_details": {"cache_read": 60},
            "output_token_details": {"reasoning": 20},
        },
    )
    assert TokenUsage.from_message(message) == TokenUsage(100, 30, 20, 60)


def test_cache_key_changes_with_model_options() -> None:
    from auditor.infra.llm import _cache_context

    low = _cache_context(Settings(llm_options={"thinking_level": "low", "api_key": "a"}))
    high = _cache_context(Settings(llm_options={"thinking_level": "high", "api_key": "a"}))
    rotated_key = _cache_context(Settings(llm_options={"thinking_level": "low", "api_key": "b"}))
    assert LLMCache.key("m", None, Verdict, MESSAGES, low) != LLMCache.key(
        "m", None, Verdict, MESSAGES, high
    )
    assert low == rotated_key
