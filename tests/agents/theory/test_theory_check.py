import dataclasses
from pathlib import Path
from typing import Any

from auditor.agents.theory import DeepCheck, Screen, anchor_quote, check, locate
from auditor.infra.llm import StructuredLLM
from auditor.infra.scope import Scope, enter_scope
from auditor.infra.usage import UsageLog
from auditor.schemas import (
    AuditTask,
    ErrorClass,
    EvidenceKind,
    ParsedPaper,
    ParserInfo,
    Section,
    Sentence,
    SentenceAnchor,
    SettingTarget,
    SourceKind,
    TaskKind,
    TaskOutcome,
    TheoryClaim,
)
from auditor.testing import ScriptedChatModel, offline_context, tool_reply

DEFINITION = "Let S(n) denote the sum of the first n positive integers."
LEMMA = "Lemma 2. For every n >= 1 we have S(n) = n^2 / 2 exactly."
PROOF = "Proof. By induction on n, using that (n+1)^2 / 2 = n^2 / 2 + n + 1 / 2 holds."
ROUTINE = "Hence S(2) - S(1) = 2, which is the second integer."


def paper() -> ParsedPaper:
    return ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sections=(Section(id="main", title="Sums"),),
        sentences=(
            Sentence(id="s0", section_id="main", text=DEFINITION),
            Sentence(id="s1", section_id="main", text=LEMMA),
            Sentence(id="s2", section_id="main", text=PROOF),
            Sentence(id="s3", section_id="main", text=ROUTINE),
        ),
    )


def task() -> AuditTask:
    claims = [
        TheoryClaim(
            claim_id="c1",
            kind="lemma",
            statement="S(n) = n^2/2",
            sentence_ids=("s1", "s2"),
            depends_on=("s0",),
            risk="high",
        ),
        TheoryClaim(
            claim_id="c2",
            kind="identity",
            sentence_ids=("s3",),
            risk="low",
            needs_reasoning=False,
        ),
    ]
    return AuditTask.create(
        paper_id="p",
        variant=paper().variant,
        kind=TaskKind.THEORY,
        anchor=SentenceAnchor(sentence_id="s1"),
        target=SettingTarget(key="math"),
        claims=claims,
    )


def context(tmp_path: Path, replies: list[Any], **settings: Any) -> tuple[Any, ScriptedChatModel]:
    model = ScriptedChatModel(replies=replies)
    ctx = offline_context(tmp_path, paper=paper(), llm=StructuredLLM(model, name="test:s"))
    # One try per deep check unless a test asks for more.
    update = {"theory_deep_samples": 1, **settings}
    ctx = dataclasses.replace(ctx, settings=ctx.settings.model_copy(update=update))
    return ctx, model


def screen(*verdicts: dict[str, Any]) -> Any:
    return tool_reply(Screen, {"verdicts": list(verdicts)})


ERROR = {
    "has_error": True,
    "quote": "we have S(n) = n^2 / 2 exactly",
    "location": "Lemma 2",
    "explanation": "S(n) is n(n+1)/2.",
    "witness": "For n = 1 the lemma gives 1/2, but S(1) = 1.",
    "confidence": "high",
}


async def test_a_suspicious_claim_gets_a_deep_check_and_becomes_a_finding(
    tmp_path: Path,
) -> None:
    replies = [
        screen(
            {"claim_id": "c1", "verdict": "suspicious", "note": "n=1 fails"},
            {"claim_id": "c2", "verdict": "fine"},
        ),
        tool_reply(DeepCheck, ERROR),
    ]
    ctx, model = context(tmp_path, replies)
    usage = UsageLog()
    with enter_scope(Scope(run_id="r", usage=usage)):
        result = await check(task(), ctx)
    assert result.outcome is TaskOutcome.CHECKED
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.MATH_ERROR
    assert isinstance(finding.anchor, SentenceAnchor)
    assert finding.anchor.sentence_id == "s1"
    assert LEMMA[finding.anchor.start or 0 : finding.anchor.end] == ERROR["quote"]
    assert finding.summary.startswith("Lemma 2:")
    assert "For n = 1" in finding.explanation
    assert not finding.uncertain
    (quote,) = finding.evidence
    assert quote.kind is EvidenceKind.QUOTE
    assert quote.source.kind is SourceKind.SUBMISSION
    # The screen is cheap and shallow; only the deep check thinks hard.
    assert [(r.purpose, r.effort) for r in usage.llm_calls] == [
        ("theory-screen", "low"),
        ("theory-deep", "high"),
    ]
    screen_prompt, deep_prompt = (" ".join(str(m.content) for m in call) for call in model.received)
    assert f"[s0] {DEFINITION}" in screen_prompt  # what the claim uses
    assert "c2" in screen_prompt
    assert ROUTINE not in deep_prompt.split("Claim to check")[-1]


async def test_fine_low_risk_claims_cost_no_deep_check(tmp_path: Path) -> None:
    replies = [screen({"claim_id": "c1", "verdict": "fine"}, {"claim_id": "c2", "verdict": "fine"})]
    ctx, model = context(tmp_path, replies)
    low = task().model_copy(update={"claims": (task().claims or ())[1:]})
    result = await check(low, ctx)
    assert result.findings == ()
    assert len(model.received) == 1


async def test_a_high_risk_claim_is_checked_deeply_even_when_screened_fine(
    tmp_path: Path,
) -> None:
    # A quick read with little thinking cannot clear a delicate argument.
    replies = [
        screen({"claim_id": "c1", "verdict": "fine"}, {"claim_id": "c2", "verdict": "fine"}),
        tool_reply(DeepCheck, ERROR),
    ]
    ctx, model = context(tmp_path, replies)
    (finding,) = (await check(task(), ctx)).findings
    assert finding.anchor.sentence_id == "s1"  # type: ignore[union-attr]
    assert len(model.received) == 2


async def test_risky_claims_that_need_reasoning_are_checked_deeply_too(tmp_path: Path) -> None:
    replies = [
        screen(
            {"claim_id": "c1", "verdict": "needs_reasoning"},
            {"claim_id": "c2", "verdict": "needs_reasoning"},
        ),
        tool_reply(DeepCheck, {"has_error": False}),
    ]
    ctx, model = context(tmp_path, replies)
    assert (await check(task(), ctx)).findings == ()
    # c2 is low risk and only needs reading, so it is not escalated.
    assert len(model.received) == 2


async def test_deep_checks_are_capped_per_task(tmp_path: Path) -> None:
    replies = [
        screen(
            {"claim_id": "c1", "verdict": "suspicious"},
            {"claim_id": "c2", "verdict": "suspicious"},
        ),
        tool_reply(DeepCheck, {"has_error": False}),
    ]
    ctx, model = context(tmp_path, replies, theory_deep_checks=1)
    await check(task(), ctx)
    assert len(model.received) == 2


async def test_an_error_needs_a_witness_and_a_quote_from_the_paper(tmp_path: Path) -> None:
    no_witness = {**ERROR, "witness": ""}
    invented = {**ERROR, "quote": "the sum equals n squared over two for all n"}
    for deep in (no_witness, invented):
        replies = [
            screen(
                {"claim_id": "c1", "verdict": "suspicious"}, {"claim_id": "c2", "verdict": "fine"}
            ),
            tool_reply(DeepCheck, deep),
        ]
        ctx, _ = context(tmp_path, replies)
        assert (await check(task(), ctx)).findings == ()


async def test_without_an_llm_the_task_is_unverifiable(tmp_path: Path) -> None:
    result = await check(task(), offline_context(tmp_path, paper=paper()))
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "LLM" in result.reason


def test_quotes_match_across_latex_delimiters_and_spacing() -> None:
    text = (
        "let \\( A \\to 0 \\) in (3.3) we get\n\\[\ne^{-2T} \\geq e^{-BT},\n\\]\na contradiction."
    )
    quote = "let \\(A \\to 0\\) in (3.3) we get \\( e^{-2T} \\geq e^{-BT} \\)"
    span = locate(quote, text)
    assert span is not None
    start, end = span
    assert text[start:end].startswith("let \\( A")
    assert text[start:end].endswith("e^{-BT}")
    assert locate("we get e^{-3T}", text) is None


def test_quotes_match_across_unicode_minus_signs_and_dashes() -> None:
    text = "Let f be a join-irreducible torsionfree class in \u039b\u2212mod."
    assert locate("Let f be a join-irreducible torsionfree class in \u039b-mod.", text) is not None


async def test_a_quote_from_elsewhere_in_the_paper_still_counts(tmp_path: Path) -> None:
    elsewhere = {**ERROR, "quote": "Hence S(2) - S(1) = 2, which is"}
    replies = [
        screen({"claim_id": "c1", "verdict": "suspicious"}, {"claim_id": "c2", "verdict": "fine"}),
        tool_reply(DeepCheck, elsewhere),
    ]
    ctx, _ = context(tmp_path, replies)
    (finding,) = (await check(task(), ctx)).findings
    assert finding.anchor.sentence_id == "s3"  # type: ignore[union-attr]


async def test_a_high_risk_claim_gets_two_tries_and_either_can_find_the_error(
    tmp_path: Path,
) -> None:
    replies = [
        screen({"claim_id": "c1", "verdict": "fine"}, {"claim_id": "c2", "verdict": "fine"}),
        tool_reply(DeepCheck, {"has_error": False}),
        tool_reply(DeepCheck, ERROR),
    ]
    ctx, model = context(tmp_path, replies, theory_deep_samples=2)
    (finding,) = (await check(task(), ctx)).findings
    assert finding.anchor.sentence_id == "s1"  # type: ignore[union-attr]
    assert len(model.received) == 3


def test_a_quote_the_model_rendered_differently_counts_on_the_sentence_it_names() -> None:
    sentences = paper().sentences
    rendered = "For every n ≥ 1 we have S(n) = n²/2 exactly"
    anchor = anchor_quote(rendered, "s1", sentences)
    assert anchor is not None
    assert anchor.sentence.id == "s1"
    assert anchor.span is None  # the words match, not the characters
    assert anchor_quote("an unrelated remark about graphs and colours", "s1", sentences) is None
    exact = anchor_quote("we have S(n) = n^2 / 2 exactly", "", sentences)
    assert exact is not None
    assert exact.sentence.id == "s1"
    assert exact.span is not None
