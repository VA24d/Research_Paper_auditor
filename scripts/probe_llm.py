"""Check what the configured LLM provider supports before relying on it (decision D9).

Reads the AUDITOR_LLM_* settings from `.env`, then tries structured output with each
method and a plain tool call. Costs a handful of small requests.

    uv run python scripts/probe_llm.py
"""

import asyncio
import sys
import time

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from auditor.config import Settings
from auditor.infra.llm import StructuredLLM, create_chat_model, model_id

METHODS = (None, "json_schema", "function_calling", "json_mode")


class Check(BaseModel):
    supported: bool = Field(description="Whether the quote supports the claim")
    evidence: str = Field(description="The exact words from the quote that decide it")


MESSAGES = [
    SystemMessage(
        "Decide whether the quote supports the claim. "
        'Answer as JSON with the keys "supported" (boolean) and "evidence" (string).'
    ),
    HumanMessage(
        "Claim: The model reaches 51.0% F1.\n"
        "Quote: We build a strong logistic regression model, which achieves an F1 score of "
        "51.0%, a significant improvement over a simple baseline (20%)."
    ),
]


@tool
def lookup_paper(title: str) -> str:
    """Look up a paper by its title."""
    return f"found: {title}"


async def probe_structured(settings: Settings, method: str | None) -> str:
    llm = StructuredLLM(create_chat_model(settings), name=model_id(settings), method=method)
    started = time.perf_counter()
    try:
        result = await llm.generate(Check, MESSAGES)
    except Exception as exc:
        return f"FAILED  {type(exc).__name__}: {str(exc)[:160]}"
    elapsed = time.perf_counter() - started
    return (
        f"ok      {elapsed:5.1f}s attempts={result.attempts} "
        f"in={result.usage.input_tokens} out={result.usage.output_tokens} "
        f"value={result.value.model_dump()}"
    )


async def probe_tool_call(settings: Settings) -> str:
    model = create_chat_model(settings).bind_tools([lookup_paper])
    try:
        reply = await model.ainvoke(
            "Use the tool to look up the paper 'Attention Is All You Need'."
        )
    except Exception as exc:
        return f"FAILED  {type(exc).__name__}: {str(exc)[:160]}"
    calls = getattr(reply, "tool_calls", [])
    return f"ok      tool_calls={calls}" if calls else f"no tool call: {reply.content!r:.160}"


async def main() -> int:
    settings = Settings(llm_cache=False)
    if not settings.llm_provider or not settings.llm_model:
        print("Set AUDITOR_LLM_PROVIDER and AUDITOR_LLM_MODEL in .env first.")
        return 1
    print(f"Provider and model: {model_id(settings)}")
    for method in METHODS:
        print(
            f"structured output, method={method!s:<16} {await probe_structured(settings, method)}"
        )
    print(f"tool calling                              {await probe_tool_call(settings)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
