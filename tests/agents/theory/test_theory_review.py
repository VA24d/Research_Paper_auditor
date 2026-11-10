import dataclasses
from pathlib import Path
from typing import Any

from auditor.agents.theory.review import MathVerdicts, review_math
from auditor.infra.llm import StructuredLLM
from auditor.schemas import (
    CandidateFinding,
    ErrorClass,
    PaperVariant,
    ParsedPaper,
    ParserInfo,
    Section,
    Sentence,
    SentenceAnchor,
    SettingTarget,
    VerificationState,
    VerifiedFinding,
)
from auditor.testing import ScriptedChatModel, offline_context, tool_reply

LEMMA = "Lemma 2. For every n >= 1 we have sum_{k=1}^n k = n^2 / 2 exactly."


def paper() -> ParsedPaper:
    return ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sections=(Section(id="main", title="Sums"),),
        sentences=(Sentence(id="s1", section_id="main", text=LEMMA),),
    )


def decision(number: int, summary: str, error_class: ErrorClass = ErrorClass.MATH_ERROR) -> Any:
    finding = CandidateFinding(
        finding_id=f"theory-{number}-f0",
        task_id=f"theory-{number}",
        paper_id="p",
        variant=PaperVariant.ORIGINAL,
        error_class=error_class,
        anchor=SentenceAnchor(sentence_id="s1"),
        target=SettingTarget(key="math"),
        summary=summary,
    )
    return VerifiedFinding(finding=finding, state=VerificationState.NEEDS_HUMAN_REVIEW, reason="r")


def contexts(tmp_path: Path, replies: list[dict[str, Any]]) -> tuple[Any, ScriptedChatModel]:
    cheap = ScriptedChatModel(replies=[])
    strong = ScriptedChatModel(replies=[tool_reply(MathVerdicts, r) for r in replies])
    ctx = offline_context(tmp_path, paper=paper(), llm=StructuredLLM(cheap, name="cheap"))
    ctx = dataclasses.replace(ctx, review_llm=StructuredLLM(strong, name="strong"))
    return ctx, strong


async def test_the_review_model_drops_false_errors_and_merges_repeats(tmp_path: Path) -> None:
    verdicts = {
        "verdicts": [
            {"number": 1, "verdict": "error", "reason": "n=1 gives 1/2."},
            {"number": 2, "verdict": "same_as", "same_as": 1, "reason": "same lemma"},
            {"number": 3, "verdict": "not_an_error", "reason": "the step is fine"},
        ]
    }
    ctx, strong = contexts(tmp_path, [verdicts])
    found = [decision(1, "Lemma 2 is false"), decision(2, "Lemma 2 sum wrong"), decision(3, "x")]
    reviewed = await review_math(found, ctx)
    first, second, third = reviewed
    assert first.state is VerificationState.NEEDS_HUMAN_REVIEW
    assert first.duplicate_of is None
    assert second.duplicate_of == "theory-1-f0"
    assert third.state is VerificationState.DROPPED
    assert "the step is fine" in third.reason
    (prompt,) = strong.received
    sent = " ".join(str(m.content) for m in prompt)
    assert "1. Lemma 2 is false" in sent
    assert LEMMA in sent


async def test_other_classes_and_unreviewed_numbers_are_left_alone(tmp_path: Path) -> None:
    ctx, _ = contexts(tmp_path, [{"verdicts": [{"number": 7, "verdict": "not_an_error"}]}])
    found = [decision(1, "Lemma 2 is false"), decision(2, "lr", ErrorClass.SETTING_MISMATCH)]
    assert await review_math(found, ctx) == found


async def test_without_math_findings_no_model_is_called(tmp_path: Path) -> None:
    ctx, strong = contexts(tmp_path, [])
    found = [decision(1, "lr", ErrorClass.SETTING_MISMATCH)]
    assert await review_math(found, ctx) == found
    assert strong.received == []


async def test_the_main_model_reviews_when_no_review_model_is_set(tmp_path: Path) -> None:
    model = ScriptedChatModel(
        replies=[tool_reply(MathVerdicts, {"verdicts": [{"number": 1, "verdict": "error"}]})]
    )
    ctx = offline_context(tmp_path, paper=paper(), llm=StructuredLLM(model, name="only"))
    await review_math([decision(1, "Lemma 2 is false")], ctx)
    assert len(model.received) == 1


async def test_an_unusable_answer_sends_flagged_errors_to_review(tmp_path: Path) -> None:
    from langchain_core.messages import AIMessage

    strong = ScriptedChatModel(replies=[AIMessage("Hard to say.")] * 3)
    ctx = offline_context(tmp_path, paper=paper(), llm=StructuredLLM(strong, name="strong"))
    flagged = decision(1, "Lemma 2 is false").model_copy(
        update={"state": VerificationState.VERIFIED_FLAG}
    )
    reviewed = await review_math([flagged, decision(2, "Lemma 2 sum wrong")], ctx)
    assert [d.state for d in reviewed] == [VerificationState.NEEDS_HUMAN_REVIEW] * 2
    assert "challenge step failed" in reviewed[0].reason
    assert reviewed[1].reason == "r"
