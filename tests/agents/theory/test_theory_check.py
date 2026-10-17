from pathlib import Path
from typing import Any

from auditor.agents.theory import MathReview, check
from auditor.infra.llm import StructuredLLM
from auditor.planner.theory import theory_tasks
from auditor.schemas import (
    AuditTask,
    ErrorClass,
    EvidenceKind,
    ParsedPaper,
    ParserInfo,
    Section,
    Sentence,
    SentenceAnchor,
    SourceKind,
    TaskOutcome,
)
from auditor.testing import ScriptedChatModel, offline_context, tool_reply

LEMMA = "Lemma 2. For every n >= 1 we have sum_{k=1}^n k = n^2 / 2 exactly."
PROOF = "Proof. By induction on n, using that (n+1)^2 / 2 = n^2 / 2 + n + 1 / 2 holds."
INTRO = "We study simple sums and their closed forms in this short and friendly note."


def paper() -> ParsedPaper:
    return ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sections=(Section(id="intro", title="Introduction"), Section(id="main", title="Sums")),
        sentences=(
            Sentence(id="s0", section_id="intro", text=INTRO),
            Sentence(id="s1", section_id="main", text=LEMMA),
            Sentence(id="s2", section_id="main", text=PROOF),
        ),
    )


def task() -> AuditTask:
    (only,) = theory_tasks(paper(), 3)
    return only


def run(tmp_path: Path, replies: list[dict[str, Any]]) -> tuple[Any, ScriptedChatModel]:
    model = ScriptedChatModel(replies=[tool_reply(MathReview, reply) for reply in replies])
    ctx = offline_context(tmp_path, paper=paper(), llm=StructuredLLM(model, name="test:s"))
    return ctx, model


ISSUE = {
    "quote": "we have sum_{k=1}^n k = n^2 / 2 exactly",
    "location": "Lemma 2",
    "explanation": "The sum is n(n+1)/2; for n = 1 the lemma gives 1/2 instead of 1.",
    "confidence": "high",
}


async def test_a_quoted_error_becomes_a_math_finding_on_its_sentence(tmp_path: Path) -> None:
    ctx, model = run(tmp_path, [{"errors": [ISSUE]}])
    result = await check(task(), ctx)
    assert result.outcome is TaskOutcome.CHECKED
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.MATH_ERROR
    assert isinstance(finding.anchor, SentenceAnchor)
    assert finding.anchor.sentence_id == "s1"
    assert LEMMA[finding.anchor.start or 0 : finding.anchor.end] == ISSUE["quote"]
    assert finding.summary.startswith("Lemma 2:")
    assert not finding.uncertain
    (quote,) = finding.evidence
    assert quote.kind is EvidenceKind.QUOTE
    assert quote.source.kind is SourceKind.SUBMISSION
    assert quote.text == ISSUE["quote"]
    sent = " ".join(str(m.content) for m in model.received[0])
    assert "[s1]" in sent  # the focus segment carries sentence ids
    assert INTRO in sent  # the whole paper is context


async def test_quotes_that_are_not_in_the_paper_or_too_short_are_dropped(tmp_path: Path) -> None:
    invented = {**ISSUE, "quote": "the sum equals n squared over two for all n"}
    short = {**ISSUE, "quote": "n^2 / 2"}
    ctx, _ = run(tmp_path, [{"errors": [invented, short]}])
    result = await check(task(), ctx)
    assert result.outcome is TaskOutcome.CHECKED
    assert result.findings == ()


async def test_quotes_match_across_spacing_differences(tmp_path: Path) -> None:
    spaced = {**ISSUE, "quote": "we  have sum_{k=1}^n k =\nn^2 / 2 exactly"}
    ctx, _ = run(tmp_path, [{"errors": [spaced]}])
    (finding,) = (await check(task(), ctx)).findings
    assert finding.evidence[0].text == ISSUE["quote"]


async def test_low_confidence_errors_are_uncertain(tmp_path: Path) -> None:
    ctx, _ = run(tmp_path, [{"errors": [{**ISSUE, "confidence": "medium"}]}])
    (finding,) = (await check(task(), ctx)).findings
    assert finding.uncertain


async def test_without_an_llm_the_task_is_unverifiable(tmp_path: Path) -> None:
    result = await check(task(), offline_context(tmp_path, paper=paper()))
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "LLM" in result.reason
