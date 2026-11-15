import asyncio
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from auditor.config import Settings
from auditor.infra import llm as llm_module
from auditor.infra.llm import (
    CallLimitReachedError,
    Effort,
    LLMCache,
    LLMNotConfiguredError,
    PromptTooLongError,
    Spend,
    StructuredLLM,
    StructuredOutputError,
    TokenUsage,
    create_chat_model,
    effort_options,
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


async def test_calls_of_one_llm_are_capped_but_cache_hits_are_free(tmp_path: Path) -> None:
    model = ScriptedChatModel(replies=[tool_reply(Verdict, {"supported": True, "reason": "x"})] * 3)
    llm = StructuredLLM(model, name="test:scripted", cache=LLMCache(tmp_path / "llm"), max_calls=1)
    await llm.generate(Verdict, MESSAGES)
    again = await llm.generate(Verdict, MESSAGES)
    assert again.from_cache
    with pytest.raises(BudgetExceededError, match="AUDITOR_LLM_MAX_CALLS"):
        await llm.generate(Verdict, [HumanMessage("Different?")])
    assert len(model.received) == 1


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


async def test_a_prose_reply_is_repaired_without_resending_the_prompt() -> None:
    model = ScriptedChatModel(
        replies=[
            AIMessage(content="Not supported: the source never mentions it."),
            tool_reply(Verdict, {"supported": False, "reason": "not in source"}),
        ]
    )
    long_prompt = [SystemMessage("You check claims."), HumanMessage("paper text " * 2000)]
    result = await structured(model).generate(Verdict, long_prompt)
    assert result.value.reason == "not in source"
    repair = model.received[1]
    sent = " ".join(str(m.content) for m in repair)
    assert "paper text paper text" not in sent
    assert "Not supported: the source never mentions it." in sent
    # The schema's fields are named.
    assert "supported" in sent
    assert "reason" in sent


async def test_prompts_longer_than_the_context_fail_loudly() -> None:
    model = ScriptedChatModel(replies=[])
    llm = StructuredLLM(model, name="test:scripted", context_tokens=100)
    with pytest.raises(PromptTooLongError, match="100"):
        await llm.generate(Verdict, [HumanMessage("word " * 1000)])
    assert model.received == []


async def test_concurrent_calls_are_capped() -> None:
    import asyncio

    active = 0
    most = 0

    class Slow(ScriptedChatModel):
        async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
            nonlocal active, most
            active += 1
            most = max(most, active)
            await asyncio.sleep(0.01)
            active -= 1
            return self._generate(*args, **kwargs)

    model = Slow(
        replies=[tool_reply(Verdict, {"supported": True, "reason": "r"}) for _ in range(4)]
    )
    llm = StructuredLLM(model, name="test:scripted", max_concurrency=2)
    await asyncio.gather(*(llm.generate(Verdict, MESSAGES) for _ in range(4)))
    assert most == 2


def test_a_review_model_is_built_next_to_the_main_one(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        cache_dir=tmp_path,
        llm_provider="openai",
        llm_model="small",
        llm_options={"api_key": "x", "temperature": 0},
        llm_review_model="large",
        llm_review_options={"api_key": "x", "reasoning_effort": "high"},
        llm_max_calls=5,
    )
    main = StructuredLLM.from_settings(settings)
    assert main.name == "openai:small"
    assert main.review is not None
    assert main.review.name == "openai:large"
    assert main.review.max_calls == 5
    assert main.review.cache_context != main.cache_context
    alone = StructuredLLM.from_settings(settings.model_copy(update={"llm_review_model": None}))
    assert alone.review is None


@pytest.mark.parametrize(
    ("provider", "model", "effort", "expected"),
    [
        ("google_genai", "gemini-3.5-flash-lite", "minimal", {"thinking_level": "minimal"}),
        ("google_genai", "gemini-3.8-flash", "high", {"thinking_level": "high"}),
        ("google_genai", "gemini-2.5-flash", "minimal", {"thinking_budget": 0}),
        ("google_genai", "gemini-2.5-flash", "high", {"thinking_budget": 12288}),
        ("openai", "gpt-test", "low", {"reasoning_effort": "low"}),
        (
            "vertex_maas",
            "deepseek-ai/deepseek-v4.1-flash-maas",
            "high",
            {"reasoning_effort": "high"},
        ),
        (
            "vertex_maas",
            "deepseek-ai/deepseek-v4.1-flash-maas",
            "minimal",
            {"reasoning_effort": "low"},
        ),
        ("ollama", "qwen3:8b", "low", {}),
    ],
)
def test_thinking_effort_maps_to_each_providers_setting(
    provider: str, model: str, effort: Effort, expected: dict[str, Any]
) -> None:
    assert effort_options(provider, model, effort) == expected


def test_a_call_can_ask_for_less_thinking(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        cache_dir=tmp_path,
        llm_provider="google_genai",
        llm_model="gemini-3.5-flash-lite",
        llm_options={"api_key": "x", "thinking_level": "high"},
    )
    llm = StructuredLLM.from_settings(settings)
    light = llm.model_for("minimal")
    assert getattr(light, "thinking_level", None) == "minimal"
    assert getattr(llm.model, "thinking_level", None) == "high"
    assert llm.model_for(None) is llm.model
    assert llm.model_for("minimal") is light


async def test_effort_is_part_of_the_cache_key_and_the_record(tmp_path: Path) -> None:
    model = ScriptedChatModel(replies=[tool_reply(Verdict, {"supported": True, "reason": "x"})] * 2)
    llm = StructuredLLM(model, name="test:scripted", cache=LLMCache(tmp_path / "llm"))
    usage = UsageLog()
    with enter_scope(Scope(run_id="r1", usage=usage)):
        await llm.generate(Verdict, MESSAGES, effort="low", purpose="screen")
        again = await llm.generate(Verdict, MESSAGES, effort="high", purpose="verify")
    assert not again.from_cache
    assert [(r.effort, r.purpose) for r in usage.llm_calls] == [
        ("low", "screen"),
        ("high", "verify"),
    ]


async def test_a_run_scoped_cache_reuses_answers_within_a_run_only(tmp_path: Path) -> None:
    replies = [tool_reply(Verdict, {"supported": True, "reason": str(i)}) for i in range(3)]
    model = ScriptedChatModel(replies=replies)
    llm = StructuredLLM(
        model, name="test:scripted", cache=LLMCache(tmp_path / "llm"), cache_scope="run"
    )
    with enter_scope(Scope(run_id="exp-full-r1")):
        first = await llm.generate(Verdict, MESSAGES)
        again = await llm.generate(Verdict, MESSAGES)
    with enter_scope(Scope(run_id="exp-full-r2")):
        other_run = await llm.generate(Verdict, MESSAGES)
    assert again.from_cache
    assert again.value == first.value
    assert not other_run.from_cache
    assert len(model.received) == 2


def test_a_call_can_cap_its_output(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        cache_dir=tmp_path,
        llm_provider="google_genai",
        llm_model="gemini-3.5-flash-lite",
        llm_options={"api_key": "x", "max_output_tokens": 32768},
    )
    llm = StructuredLLM.from_settings(settings)
    capped = llm.model_for("low", max_output_tokens=8192)
    assert getattr(capped, "max_output_tokens", None) == 8192
    assert getattr(capped, "thinking_level", None) == "low"
    assert getattr(llm.model, "max_output_tokens", None) == 32768
    assert llm.model_for(None, max_output_tokens=8192) is not llm.model


# A reply that reports 240 input and 5 output tokens costs this at $1 per million, more
# than the worst case reserved for these short prompts with a 5-token answer cap, so one
# call fits a budget of one reply and spends all of it.
ONE_REPLY_USD = 0.000245


def costly_replies(count: int) -> list[AIMessage]:
    return [
        tool_reply(Verdict, {"supported": True, "reason": str(i)}, input_tokens=240)
        for i in range(count)
    ]


async def test_a_run_stops_once_its_dollar_budget_is_spent() -> None:
    model = ScriptedChatModel(replies=costly_replies(3))
    spend = Spend(limit_usd=ONE_REPLY_USD, prices={"scripted": (1.0, 1.0)})
    llm = StructuredLLM(model, name="test:scripted", spend=spend, output_cap=5)
    await llm.generate(Verdict, MESSAGES)
    assert spend.spent_usd == pytest.approx(ONE_REPLY_USD)
    assert llm.limit_reached
    with pytest.raises(CallLimitReachedError, match="AUDITOR_LLM_MAX_COST_USD"):
        await llm.generate(Verdict, [HumanMessage("Another?")])
    assert len(model.received) == 1


def test_the_review_model_shares_the_run_budget(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        cache_dir=tmp_path,
        llm_provider="openai",
        llm_model="small",
        llm_options={"api_key": "x"},
        llm_review_model="large",
        llm_max_cost_usd=2.0,
        llm_prices={"small": (0.1, 0.4), "large": (1.0, 4.0)},
    )
    main = StructuredLLM.from_settings(settings)
    assert main.review is not None
    assert main.spend is main.review.spend


def test_a_budget_needs_the_price_of_every_model(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        cache_dir=tmp_path,
        llm_provider="openai",
        llm_model="unpriced",
        llm_options={"api_key": "x"},
        llm_max_cost_usd=1.0,
    )
    with pytest.raises(ValueError, match="AUDITOR_LLM_PRICES"):
        StructuredLLM.from_settings(settings)


def test_an_image_counts_as_a_few_hundred_tokens_not_its_base64_length() -> None:
    from langchain_core.messages import HumanMessage

    from auditor.infra.llm import IMAGE_CHARS, estimated_tokens

    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 400_000}}
    message = HumanMessage(content=[{"type": "text", "text": "x" * 400}, image])
    assert estimated_tokens([message]) == estimated_tokens(
        [HumanMessage("x" * (400 + IMAGE_CHARS))]
    )


async def test_calls_waiting_for_a_slot_are_not_sent_once_the_budget_is_spent() -> None:
    model = ScriptedChatModel(replies=costly_replies(3))
    spend = Spend(limit_usd=ONE_REPLY_USD, prices={"scripted": (1.0, 1.0)})
    llm = StructuredLLM(model, name="test:scripted", spend=spend, max_concurrency=1, output_cap=5)
    questions = [[HumanMessage(f"Question {i}?")] for i in range(3)]
    results = await asyncio.gather(
        *(llm.generate(Verdict, q) for q in questions), return_exceptions=True
    )
    assert len(model.received) == 1
    assert sum(isinstance(r, CallLimitReachedError) for r in results) == 2


async def test_calls_sent_together_stop_once_their_cost_could_pass_the_budget() -> None:
    # Five calls start together on a budget that holds the worst case of only some of
    # them. Each is reserved at its prompt, schema and answer cap while it runs, and is
    # sent only when that fits what is left, so the run never spends past its limit.
    # Before reservations all five were sent.
    model = ScriptedChatModel(
        replies=[tool_reply(Verdict, {"supported": True, "reason": str(i)}) for i in range(5)]
    )
    spend = Spend(limit_usd=0.00025, prices={"scripted": (1.0, 1.0)})
    llm = StructuredLLM(model, name="test:scripted", spend=spend, output_cap=5)
    questions = [[HumanMessage(f"Question {i}?")] for i in range(5)]
    results = await asyncio.gather(
        *(llm.generate(Verdict, q) for q in questions), return_exceptions=True
    )
    sent = len(model.received)
    assert 1 <= sent < 5
    assert sum(isinstance(r, CallLimitReachedError) for r in results) == 5 - sent
    # Reservations of different sizes added and taken off leave float dust, not money.
    assert spend.reserved_usd == pytest.approx(0, abs=1e-12)
    assert spend.spent_usd <= 0.00025


async def test_a_failed_call_gives_back_its_reservation() -> None:
    spend = Spend(limit_usd=1.0, prices={"scripted": (1.0, 1.0)})
    llm = StructuredLLM(ScriptedChatModel(replies=[]), name="test:scripted", spend=spend)
    with pytest.raises(RuntimeError, match="ran out of replies"):
        await llm.generate(Verdict, MESSAGES)
    assert spend.reserved_usd == 0
    assert spend.spent_usd == 0


def test_the_answer_cap_comes_from_either_option_name(tmp_path: Path) -> None:
    assert llm_module._output_cap({"max_output_tokens": 4096}) == 4096
    assert llm_module._output_cap({"max_tokens": 2048}) == 2048
    assert llm_module._output_cap({"temperature": 0}) is None
    settings = Settings(
        _env_file=None,
        cache_dir=tmp_path,
        llm_provider="openai",
        llm_model="small",
        llm_options={"api_key": "x", "max_tokens": 2048},
    )
    assert StructuredLLM.from_settings(settings).output_cap == 2048


def test_a_ledger_limit_needs_the_price_of_every_model(tmp_path: Path) -> None:
    # Unpriced calls add nothing to the ledger, so its limit would never stop them.
    settings = Settings(
        _env_file=None,
        cache_dir=tmp_path,
        llm_provider="openai",
        llm_model="unpriced",
        llm_options={"api_key": "x"},
        llm_ledger=tmp_path / "ledger.jsonl",
        llm_ledger_limit_usd=1.0,
    )
    with pytest.raises(ValueError, match="AUDITOR_LLM_PRICES"):
        StructuredLLM.from_settings(settings)


def test_runs_writing_one_ledger_at_once_lose_no_lines(tmp_path: Path) -> None:
    import subprocess
    import sys

    from auditor.infra.llm import ledger_total

    ledger = tmp_path / "ledger.jsonl"
    script = (
        "import sys; from pathlib import Path; from auditor.infra.llm import _append_ledger; "
        "[_append_ledger(Path(sys.argv[1]), 'm', 0.001) for _ in range(1000)]"
    )
    runs = [subprocess.Popen([sys.executable, "-c", script, str(ledger)]) for _ in range(4)]
    assert all(run.wait(timeout=120) == 0 for run in runs)
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 4000
    assert ledger_total(ledger) == pytest.approx(4.0)


def test_input_read_from_the_cache_is_charged_at_its_share() -> None:
    spend = Spend(limit_usd=None, prices={"m": (1.0, 2.0)}, cached_input_share=0.25)
    spend.charge(
        "p:m", TokenUsage(input_tokens=1_000_000, output_tokens=0, cached_input_tokens=800_000)
    )
    assert spend.spent_usd == pytest.approx(0.2 + 0.8 * 0.25)


async def test_a_ledger_stops_runs_once_their_total_reaches_its_limit(tmp_path: Path) -> None:
    from auditor.infra.llm import ledger_total

    ledger = tmp_path / "ledger.jsonl"
    prices = {"scripted": (1.0, 1.0)}
    replies = costly_replies(3)
    first = StructuredLLM(
        ScriptedChatModel(replies=replies),
        name="test:scripted",
        spend=Spend(limit_usd=None, prices=prices, ledger=ledger, ledger_limit_usd=ONE_REPLY_USD),
        output_cap=5,
    )
    await first.generate(Verdict, MESSAGES)
    # The reservation and its release cancel out; the reply's cost stays.
    assert ledger_total(ledger) == pytest.approx(ONE_REPLY_USD)
    # Another run sharing the ledger is stopped before it spends anything.
    second = StructuredLLM(
        ScriptedChatModel(replies=replies),
        name="test:scripted",
        spend=Spend(limit_usd=None, prices=prices, ledger=ledger, ledger_limit_usd=ONE_REPLY_USD),
    )
    with pytest.raises(CallLimitReachedError, match="LEDGER"):
        await second.generate(Verdict, [HumanMessage("Another?")])


def test_vertex_partner_models_use_the_openai_compatible_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import auditor.infra.llm as module

    class FakeAuth(module._GoogleBearer):
        def __init__(self) -> None:
            pass

    monkeypatch.setattr(module, "_GoogleBearer", FakeAuth)
    settings = Settings(
        llm_provider="vertex_maas",
        llm_model="deepseek-ai/deepseek-v4.1-flash-maas",
        llm_options={"project": "p", "location": "global", "max_output_tokens": 512},
    )
    model = create_chat_model(settings)
    assert "projects/p/locations/global/endpoints/openapi" in str(
        getattr(model, "openai_api_base", "")
    )
    assert getattr(model, "max_tokens", None) == 512


def test_a_partner_model_call_can_ask_for_thinking_and_an_output_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import auditor.infra.llm as module

    class FakeAuth(module._GoogleBearer):
        def __init__(self) -> None:
            pass

    monkeypatch.setattr(module, "_GoogleBearer", FakeAuth)
    settings = Settings(
        _env_file=None,
        cache_dir=tmp_path,
        llm_provider="vertex_maas",
        llm_model="deepseek-ai/deepseek-v4.1-flash-maas",
        llm_options={"project": "p", "location": "global", "max_output_tokens": 512},
    )
    llm = StructuredLLM.from_settings(settings)
    thinking = llm.model_for("high", max_output_tokens=16_000)
    assert getattr(thinking, "reasoning_effort", None) == "high"
    assert getattr(thinking, "max_tokens", None) == 16_000
    assert getattr(llm.model, "reasoning_effort", None) is None


async def test_a_call_whose_worst_case_does_not_fit_is_not_sent(tmp_path: Path) -> None:
    """The cap holds for the call that would cross it: it is refused before it is sent,
    not charged after. The ledger limit is held the same way."""
    model = ScriptedChatModel(replies=[tool_reply(Verdict, {"supported": True, "reason": "x"})])
    # Up to 16,384 answer tokens at $1 per million may cost more than the $0.001 budget.
    llm = StructuredLLM(
        model, name="test:scripted", spend=Spend(limit_usd=0.001, prices={"scripted": (1.0, 1.0)})
    )
    with pytest.raises(CallLimitReachedError, match="more than is left of the budget"):
        await llm.generate(Verdict, MESSAGES)
    assert model.received == []
    ledger = tmp_path / "ledger.jsonl"
    spend = Spend(
        limit_usd=None, prices={"scripted": (1.0, 1.0)}, ledger=ledger, ledger_limit_usd=0.001
    )
    with pytest.raises(CallLimitReachedError, match="AUDITOR_LLM_LEDGER_LIMIT_USD"):
        await StructuredLLM(model, name="test:scripted", spend=spend).generate(Verdict, MESSAGES)
    assert model.received == []


def test_runs_sharing_a_ledger_cannot_both_take_its_last_dollars(tmp_path: Path) -> None:
    """Two runs with their own budgets reserve on one ledger: the reservation is written
    under the ledger's lock, so the second sees the first's call in flight."""
    ledger = tmp_path / "ledger.jsonl"
    prices = {"m": (1.0, 1.0)}
    first = Spend(limit_usd=None, prices=prices, ledger=ledger, ledger_limit_usd=0.00001)
    second = Spend(limit_usd=None, prices=prices, ledger=ledger, ledger_limit_usd=0.00001)
    refused = pytest.raises(CallLimitReachedError, match="AUDITOR_LLM_LEDGER_LIMIT_USD")
    with first.holding(0.00001, "m"):
        with refused, second.holding(0.00001, "m"):
            pass
        first.charge("p:m", TokenUsage(input_tokens=5, output_tokens=5))
    assert llm_module.ledger_total(ledger) == pytest.approx(0.00001)


async def test_the_reservation_counts_the_cached_prompt_and_the_agent_answer_cap() -> None:
    """A cached system message is still billed, and a tool agent's model may answer at
    more than the configured cap: both are reserved."""
    spend = Spend(limit_usd=None, prices={"scripted": (1.0, 1.0)})
    llm = StructuredLLM(ScriptedChatModel(replies=[]), name="test:scripted", spend=spend)
    long_system = [SystemMessage("x" * 40_000), HumanMessage("Q?")]
    assert llm._most_cost(long_system, 10) > llm._most_cost(long_system[1:], 10) + 0.009
    seen: list[float] = []

    async def call() -> AIMessage:
        seen.append(spend.reserved_usd)
        return AIMessage(
            content="ok", usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
        )

    await llm.direct_call(call, lambda m: TokenUsage.from_message(m), max_output_tokens=8192)
    assert seen[0] >= 8192 / 1e6


def test_a_tool_call_counts_its_arguments() -> None:
    """A model turn sent back with its tool calls carries their arguments, often whole
    files; the estimate must see them."""
    from auditor.infra.llm import estimated_tokens

    call = AIMessage(
        content="",
        tool_calls=[{"name": "write_test", "args": {"code": "x" * 20_000}, "id": "c1"}],
    )
    assert estimated_tokens([call]) >= 20_000 / 4
