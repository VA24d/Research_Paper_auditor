"""The Gemini path end to end, offline: the real ChatGoogleGenerativeAI class with its
HTTP traffic answered by a mock transport. Checks what we send and how we read the reply,
so the switch to a real key only adds the network."""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from auditor.config import Settings
from auditor.infra import llm
from auditor.infra.llm import StructuredLLM
from auditor.infra.scope import Scope, enter_scope
from auditor.infra.usage import UsageLog

MODEL = "gemini-3.5-flash-lite"


class Verdict(BaseModel):
    supported: bool
    reason: str


def gemini_reply(text: str) -> dict[str, Any]:
    return {
        "candidates": [
            {"content": {"role": "model", "parts": [{"text": text}]}, "finishReason": "STOP"}
        ],
        "usageMetadata": {
            "promptTokenCount": 120,
            "candidatesTokenCount": 15,
            "thoughtsTokenCount": 40,
            "totalTokenCount": 175,
        },
        "modelVersion": MODEL,
    }


class FakeGemini:
    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json=gemini_reply(self.texts.pop(0)))


@pytest.fixture
def gemini(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeGemini], StructuredLLM]:
    """Build the model the way a run does, from settings, with HTTP served by `fake`."""

    def build(fake: FakeGemini) -> StructuredLLM:
        def with_transport(*args: Any, **kwargs: Any) -> Any:
            kwargs["client_args"] = {"transport": httpx.MockTransport(fake)}
            return init_chat_model(*args, **kwargs)

        monkeypatch.setattr(llm, "init_chat_model", with_transport)
        settings = Settings(
            llm_provider="google_genai",
            llm_model=MODEL,
            llm_max_retries=0,
            llm_options={"api_key": "test-key", "thinking_level": "low"},
            llm_cache=False,
        )
        return StructuredLLM.from_settings(settings)

    return build


MESSAGES = [SystemMessage("You check claims."), HumanMessage("Is the claim supported?")]


async def test_structured_output_request_and_reply(
    gemini: Callable[[FakeGemini], StructuredLLM],
) -> None:
    fake = FakeGemini(json.dumps({"supported": True, "reason": "quoted"}))
    usage = UsageLog()
    with enter_scope(Scope(run_id="r1", task_id="t1", usage=usage)):
        result = await gemini(fake).generate(Verdict, MESSAGES)

    assert result.value == Verdict(supported=True, reason="quoted")
    (request,) = fake.requests
    assert request.url.path.endswith(f"models/{MODEL}:generateContent")
    body = json.loads(request.content)
    config = body["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    schema = config.get("responseJsonSchema") or config.get("responseSchema")
    assert set(schema["properties"]) == {"supported", "reason"}
    # google-genai 2.x writes this one field in snake case, the rest in camel case.
    thinking = config["thinkingConfig"]
    assert (thinking.get("thinkingLevel") or thinking.get("thinking_level")).lower() == "low"
    assert "You check claims." in json.dumps(body["systemInstruction"])

    (record,) = usage.llm_calls
    assert record.input_tokens == 120
    assert record.output_tokens == 55
    assert record.reasoning_tokens == 40


async def test_invalid_json_is_retried_with_feedback(
    gemini: Callable[[FakeGemini], StructuredLLM],
) -> None:
    fake = FakeGemini("not json at all", json.dumps({"supported": False, "reason": "absent"}))
    result = await gemini(fake).generate(Verdict, MESSAGES)
    assert result.value.supported is False
    assert result.attempts == 2
    retry_body = json.dumps(json.loads(fake.requests[1].content)["contents"])
    assert "could not be used" in retry_body
