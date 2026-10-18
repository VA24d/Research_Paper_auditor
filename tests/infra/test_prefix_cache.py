from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from auditor.infra.llm import Spend, StructuredLLM
from auditor.infra.prefix_cache import PrefixCache
from auditor.testing import ScriptedChatModel, tool_reply


class Answer(BaseModel):
    ok: bool


class FakeClient:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.made: list[tuple[str, str, int]] = []
        self.deleted: list[str] = []

    async def create(self, model: str, system_instruction: str, ttl_s: int) -> str:
        if self.fail:
            raise RuntimeError("PERMISSION_DENIED")
        self.made.append((model, system_instruction, ttl_s))
        return f"cachedContents/{len(self.made)}"

    async def delete(self, name: str) -> None:
        self.deleted.append(name)


PAPER = "The paper:\n" + "[s1] A long sentence of the paper. " * 100


def llm(client: FakeClient, replies: int) -> tuple[StructuredLLM, ScriptedChatModel]:
    model = ScriptedChatModel(replies=[tool_reply(Answer, {"ok": True}) for _ in range(replies)])
    cache = PrefixCache(client, ttl_s=600, min_chars=1000)
    spend = Spend(limit_usd=None, prices={"m": (1.0, 1.0)})
    return (
        StructuredLLM(
            model, name="google_genai:m", prefix_cache=cache, spend=spend, cache_storage_price=1.0
        ),
        model,
    )


async def test_a_long_first_message_is_cached_once_and_not_sent_again() -> None:
    client = FakeClient()
    structured, model = llm(client, 2)
    for question in ("First?", "Second?"):
        await structured.generate(
            Answer, [SystemMessage(PAPER), HumanMessage(question)], cache_prefix=True
        )
    assert client.made == [("m", PAPER, 600)]
    # Both calls send only what follows the cached message.
    assert [[m.content for m in call] for call in model.received] == [["First?"], ["Second?"]]
    # Storage for the cache's lifetime is charged when it is made.
    assert structured.spend.spent_usd > 0
    await structured.aclose()
    assert client.deleted == ["cachedContents/1"]


async def test_without_a_cache_the_message_is_sent_whole() -> None:
    client = FakeClient(fail=True)
    structured, model = llm(client, 2)
    await structured.generate(
        Answer, [SystemMessage(PAPER), HumanMessage("First?")], cache_prefix=True
    )
    # Not asked for: a call that does not ask for the cache sends everything.
    await structured.generate(Answer, [SystemMessage(PAPER), HumanMessage("Second?")])
    assert [len(call) for call in model.received] == [2, 2]
    await structured.aclose()
    assert client.deleted == []


async def test_a_short_first_message_is_not_cached() -> None:
    client = FakeClient()
    structured, model = llm(client, 1)
    await structured.generate(
        Answer, [SystemMessage("Short."), HumanMessage("Q?")], cache_prefix=True
    )
    assert client.made == []
    assert len(model.received[0]) == 2
