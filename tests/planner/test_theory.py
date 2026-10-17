import dataclasses
from pathlib import Path
from typing import Any

from auditor.infra.llm import StructuredLLM
from auditor.planner.tasks import plan_paper
from auditor.planner.theory import ClaimInventory, formal_density, paper_listing, theory_tasks
from auditor.schemas import (
    ParsedPaper,
    ParserInfo,
    Section,
    Sentence,
    SentenceAnchor,
    SettingTarget,
    TaskKind,
)
from auditor.testing import ScriptedChatModel, offline_context, tool_reply

PROSE = "Large language models changed how researchers write and read scientific papers today."
FORMAL = "By Lemma 2 we have f(x) = sum_i a_i x^i <= C n^2 for all x in [0, 1]."


def paper() -> ParsedPaper:
    return ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sections=(Section(id="intro", title="Introduction"), Section(id="main", title="Proofs")),
        sentences=tuple(
            [Sentence(id="s0", section_id="intro", text=PROSE)]
            + [Sentence(id=f"s{i}", section_id="main", text=FORMAL) for i in range(1, 7)]
        ),
    )


def claim(ids: list[str], risk: str = "medium", **fields: Any) -> dict[str, Any]:
    return {
        "kind": "lemma",
        "statement": "f is bounded",
        "sentence_ids": ids,
        "depends_on": [],
        "risk": risk,
        "needs_reasoning": True,
        **fields,
    }


def llm(*inventories: dict[str, Any]) -> tuple[StructuredLLM, ScriptedChatModel]:
    model = ScriptedChatModel(replies=[tool_reply(ClaimInventory, i) for i in inventories])
    return StructuredLLM(model, name="test:s"), model


def test_formal_text_is_denser_than_prose() -> None:
    assert formal_density([PROSE]) == 1.0
    assert formal_density([FORMAL]) > 2.0


def test_the_listing_gives_each_sentence_once_with_its_id() -> None:
    listing = paper_listing(paper())
    assert "## Proofs" in listing
    assert f"[s1] {FORMAL}" in listing
    assert listing.count(FORMAL) == 6


async def test_claims_the_model_finds_become_theory_tasks_in_batches() -> None:
    inventory = {"claims": [claim([f"s{i}"]) for i in range(1, 6)]}
    model, scripted = llm(inventory)
    tasks = await theory_tasks(paper(), model, batch_size=2, max_tasks=5)
    assert [len(t.claims or ()) for t in tasks] == [2, 2, 1]
    first = tasks[0]
    assert first.kind is TaskKind.THEORY
    assert first.anchor == SentenceAnchor(sentence_id="s1")
    assert first.target == SettingTarget(key="math")
    assert [c.claim_id for t in tasks for c in t.claims or ()] == ["c1", "c2", "c3", "c4", "c5"]
    (call,) = scripted.received
    assert FORMAL in " ".join(str(m.content) for m in call)


async def test_ids_the_paper_does_not_have_are_dropped() -> None:
    inventory = {
        "claims": [
            claim(["s9", "s2"], depends_on=["s1", "s99"]),
            claim(["s42"]),
        ]
    }
    model, _ = llm(inventory)
    (task,) = await theory_tasks(paper(), model, batch_size=8, max_tasks=5)
    (only,) = task.claims or ()
    assert only.sentence_ids == ("s2",)
    assert only.depends_on == ("s1",)


async def test_over_the_budget_the_least_risky_claims_are_left_out() -> None:
    inventory = {
        "claims": [
            claim(["s1"], risk="low", needs_reasoning=False),
            claim(["s2"], risk="high"),
            claim(["s3"], risk="low"),
            claim(["s4"], risk="medium"),
        ]
    }
    model, _ = llm(inventory)
    tasks = await theory_tasks(paper(), model, batch_size=1, max_tasks=2)
    kept = [c.sentence_ids[0] for t in tasks for c in t.claims or ()]
    assert kept == ["s2", "s4"]


async def test_the_planner_adds_theory_tasks_only_when_enabled(tmp_path: Path) -> None:
    model, _ = llm({"claims": [claim(["s1"])]})
    ctx = offline_context(tmp_path, paper=paper(), llm=model)
    assert not [t for t in await plan_paper(paper(), ctx) if t.kind is TaskKind.THEORY]
    settings = ctx.settings.model_copy(update={"theory_check_enabled": True})
    tasks = await plan_paper(paper(), dataclasses.replace(ctx, settings=settings))
    assert [len(t.claims or ()) for t in tasks if t.kind is TaskKind.THEORY] == [1]
