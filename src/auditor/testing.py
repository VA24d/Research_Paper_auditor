"""Test doubles that other parts can use in their own tests."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class ScriptedChatModel(BaseChatModel):
    """Returns prepared replies in order and remembers what it was sent.

    Supports `bind_tools`, so `with_structured_output` (function calling) and agents work.
    Build tool-call replies with `tool_reply`.
    """

    replies: list[AIMessage] = Field(default_factory=list)
    received: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.received.append(list(messages))
        if not self.replies:
            raise RuntimeError("ScriptedChatModel ran out of replies")
        return ChatResult(generations=[ChatGeneration(message=self.replies.pop(0))])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.bound_tools = list(tools)
        return self


def tool_reply(
    name: str | type[BaseModel],
    args: dict[str, Any],
    *,
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> AIMessage:
    """An AIMessage calling one tool, with token usage attached."""
    tool_name = name if isinstance(name, str) else name.__name__
    return AIMessage(
        content="",
        tool_calls=[{"name": tool_name, "args": args, "id": f"call-{tool_name}"}],
        usage_metadata={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    )
