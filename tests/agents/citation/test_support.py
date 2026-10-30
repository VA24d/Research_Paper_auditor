"""Offline tests for the support judgement (P2.5): a scripted model, no network."""

from pathlib import Path

from auditor.agents.citation.support import (
    MIN_QUOTE_CHARS,
    TOP_PASSAGES,
    NumberCheck,
    SupportAnswer,
    check_citation,
    extract_numbers,
    numbers_agree,
)
from auditor.infra.llm import StructuredLLM
from auditor.retrieval.clients import Work
from auditor.retrieval.fulltext import SourceLevel, SourceText
from auditor.retrieval.passages import chunk_text
from auditor.retrieval.resolve import Resolution
from auditor.schemas import (
    AuditTask,
    BibTarget,
    ErrorClass,
    EvidenceKind,
    SentenceAnchor,
    TaskKind,
    TaskOutcome,
)
from auditor.schemas.examples import example_paper
from auditor.testing import ScriptedChatModel, offline_context, tool_reply

SOURCE_TEXT = (
    "The Transformer reaches 28.4 BLEU on WMT 2014 English to German. " * 60
    + "Recurrence and convolution were the dominant sequence models before it. " * 60
)
SUPPORT_QUOTE = "The Transformer reaches 28.4 BLEU on WMT 2014 English to German."
CONTRA_QUOTE = "Recurrence and convolution were the dominant sequence models before it."

WORK = Work(id="arxiv:1706.03762", source="arxiv", title="Attention Is All You Need")


def citation_task(claim: str) -> AuditTask:
    paper = example_paper()
    return AuditTask.create(
        paper_id=paper.paper_id,
        variant=paper.variant,
        kind=TaskKind.CITATION,
        anchor=SentenceAnchor(sentence_id="s2"),
        target=BibTarget(bib_id="b1"),
        claim=claim,
    )


def source(level: SourceLevel = SourceLevel.FULL_HTML) -> SourceText:
    return SourceText(work=WORK, level=level, text=SOURCE_TEXT)


def answer(**overrides: object) -> SupportAnswer:
    fields: dict[str, object] = {
        "verdict": "supported",
        "quote": SUPPORT_QUOTE,
        "reason": "the passage states the number",
        "numbers": [],
    }
    fields.update(overrides)
    return SupportAnswer(**fields)  # type: ignore[arg-type]


def llm_with(reply: SupportAnswer) -> StructuredLLM:
    model = ScriptedChatModel(replies=[tool_reply(SupportAnswer, reply.model_dump())])
    return StructuredLLM(model, name="test:main")


def test_extract_numbers_finds_plain_numbers_only() -> None:
    # Scientific notation stays whole; percentages and units keep the numeric part.
    assert extract_numbers("accuracy of 92.1% on 12,500 examples (3e-4 lr)") == [
        "92.1",
        "12",
        "500",
        "3e-4",
    ]
    assert extract_numbers("no numbers here") == []
    assert extract_numbers("section 3.2 and table 4") == ["3.2", "4"]
    assert extract_numbers("a 1.5e10 parameter model with 1e-3 lr") == ["1.5e10", "1e-3"]


def test_numbers_agree_by_value_with_tolerance() -> None:
    assert numbers_agree("3e-4", "0.0003")
    assert numbers_agree("12.5%", "0.125")
    assert numbers_agree("1,000", "1000")
    assert numbers_agree("92.1", "92.15")  # within 1% relative
    assert not numbers_agree("92.1", "88.4")
    assert not numbers_agree("92.1", "about ninety")
    # A number word is its value, and a suffix says what is counted, not the value.
    assert numbers_agree("eight", "8")
    assert numbers_agree("8x", "eight")
    assert numbers_agree("10-fold", "10")
    assert not numbers_agree("eight", "9")
    assert not numbers_agree("eight", "eighty")


async def test_supported_claim_is_checked_without_findings(tmp_path: Path) -> None:
    task = citation_task("The Transformer reaches 28.4 BLEU on WMT 2014. [1]")
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer())),
        Resolution(work=WORK),
        source(),
    )
    assert result.outcome is TaskOutcome.CHECKED
    assert result.findings == ()


async def test_contradicted_claim_is_a_finding_with_a_verbatim_quote(tmp_path: Path) -> None:
    task = citation_task("The Transformer reaches 28.4 BLEU on WMT 2014. [1]")
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(verdict="contradicted", quote=CONTRA_QUOTE))),
        Resolution(work=WORK),
        source(),
    )
    assert result.outcome is TaskOutcome.CHECKED
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.CLAIM_NOT_SUPPORTED
    kinds = [e.kind for e in finding.evidence]
    assert EvidenceKind.LOOKUP in kinds
    assert EvidenceKind.QUOTE in kinds
    (quote,) = (e for e in finding.evidence if e.kind is EvidenceKind.QUOTE)
    assert quote.text == CONTRA_QUOTE
    assert quote.char_start is not None
    assert quote.char_end is not None
    assert SOURCE_TEXT[quote.char_start : quote.char_end] == CONTRA_QUOTE


async def test_a_number_nowhere_in_a_full_text_is_a_mismatch(tmp_path: Path) -> None:
    """The code scans the full text itself when the model found nothing (known issue 6)."""
    numbers = [NumberCheck(claimed="28.4", found=None, quote="", agrees=True)]
    task = citation_task("The Transformer reaches 28.4 BLEU on WMT 2014. [1]")
    other = SourceText(
        work=WORK,
        level=SourceLevel.FULL_HTML,
        text="The model attains an error rate of 2.7 percent on the translation tasks. " * 40,
    )
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(numbers=numbers))),
        Resolution(work=WORK),
        other,  # a full text that nowhere states 28.4
    )
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.CLAIM_NUMBER_MISMATCH
    assert "nowhere states" in finding.summary


async def test_a_number_stated_by_the_source_makes_no_finding(tmp_path: Path) -> None:
    numbers = [
        NumberCheck(claimed="28.4 BLEU", found=None, quote="", agrees=True),
    ]
    task = citation_task("The Transformer reaches 28.4 BLEU. [1]")
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(numbers=numbers))),
        Resolution(work=WORK),
        source(),  # the full text states 28.4 BLEU
    )
    assert result.findings == ()


async def test_the_source_scan_is_not_run_against_an_abstract(tmp_path: Path) -> None:
    """The number may live in the full text an abstract-only source does not hold."""
    numbers = [NumberCheck(claimed="28.4", found=None, quote="", agrees=True)]
    task = citation_task("The Transformer reaches 28.4 BLEU. [1]")
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(numbers=numbers))),
        Resolution(work=WORK),
        source(level=SourceLevel.ABSTRACT),
    )
    assert result.findings == ()


async def test_a_derived_small_number_is_not_scanned(tmp_path: Path) -> None:
    """`4x fewer parameters` states a derived quantity; scanning would misfire."""
    numbers = [NumberCheck(claimed="4", found=None, quote="", agrees=True)]
    task = citation_task("The model has 4x fewer parameters. [1]")
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(numbers=numbers))),
        Resolution(work=WORK),
        source(),  # the text has no bare "4"
    )
    assert result.findings == ()


async def test_a_quote_not_in_the_source_keeps_no_offsets(tmp_path: Path) -> None:
    task = citation_task("The Transformer reaches 28.4 BLEU on WMT 2014. [1]")
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(verdict="contradicted", quote="x" * 40))),
        Resolution(work=WORK),
        source(),
    )
    (finding,) = result.findings
    (quote,) = (e for e in finding.evidence if e.kind is EvidenceKind.QUOTE)
    assert quote.char_start is None
    assert quote.char_end is None


async def test_a_short_quote_carries_lookup_evidence_only(tmp_path: Path) -> None:
    task = citation_task("The Transformer reaches 28.4 BLEU on WMT 2014. [1]")
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(verdict="contradicted", quote="too short"))),
        Resolution(work=WORK),
        source(),
    )
    (finding,) = result.findings
    assert [e.kind for e in finding.evidence] == [EvidenceKind.LOOKUP]
    assert len("too short") < MIN_QUOTE_CHARS


async def test_insufficient_is_checked_without_findings(tmp_path: Path) -> None:
    task = citation_task("The Transformer reaches 28.4 BLEU on WMT 2014. [1]")
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(verdict="insufficient", quote=""))),
        Resolution(work=WORK),
        source(),
    )
    assert result.outcome is TaskOutcome.CHECKED
    assert result.findings == ()


async def test_an_unknown_verdict_is_treated_as_insufficient(tmp_path: Path) -> None:
    task = citation_task("The Transformer reaches 28.4 BLEU on WMT 2014. [1]")
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(verdict="maybe"))),
        Resolution(work=WORK),
        source(),
    )
    assert result.outcome is TaskOutcome.CHECKED
    assert result.findings == ()


async def test_a_disagreeing_number_is_a_mismatch_finding(tmp_path: Path) -> None:
    numbers = [NumberCheck(claimed="28.4", found="26.4", quote=SUPPORT_QUOTE, agrees=False)]
    task = citation_task("The Transformer reaches 28.4 BLEU on WMT 2014. [1]")
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(numbers=numbers))),
        Resolution(work=WORK),
        source(),
    )
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.CLAIM_NUMBER_MISMATCH
    assert "28.4" in finding.summary
    assert "26.4" in finding.summary


async def test_the_model_saying_agrees_does_not_stop_a_real_mismatch(tmp_path: Path) -> None:
    """The code-side value comparison decides, not the model's hint (known issue 1)."""
    numbers = [NumberCheck(claimed="28.4", found="26.4", quote=SUPPORT_QUOTE, agrees=True)]
    task = citation_task("The Transformer reaches 28.4 BLEU on WMT 2014. [1]")
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(numbers=numbers))),
        Resolution(work=WORK),
        source(),
    )
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.CLAIM_NUMBER_MISMATCH


async def test_a_contradiction_and_a_number_mismatch_both_reported(tmp_path: Path) -> None:
    """A task can yield a claim_not_supported finding AND number mismatches; their
    finding ids must stay unique, or the TaskResult rejects them (live-run failure)."""
    numbers = [NumberCheck(claimed="28.4", found="26.4", quote=SUPPORT_QUOTE, agrees=False)]
    task = citation_task("The Transformer reaches 28.4 BLEU on WMT 2014. [1]")
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(verdict="contradicted", numbers=numbers))),
        Resolution(work=WORK),
        source(),
    )
    assert len(result.findings) == 2
    ids = [finding.finding_id for finding in result.findings]
    assert len(ids) == len(set(ids))
    assert {finding.error_class for finding in result.findings} == {
        ErrorClass.CLAIM_NOT_SUPPORTED,
        ErrorClass.CLAIM_NUMBER_MISMATCH,
    }


async def test_two_number_mismatches_get_unique_finding_ids(tmp_path: Path) -> None:
    numbers = [
        NumberCheck(claimed="28.4", found="26.4", quote=SUPPORT_QUOTE, agrees=False),
        NumberCheck(claimed="12,500", found="12,000", quote=SUPPORT_QUOTE, agrees=False),
    ]
    task = citation_task("The Transformer reaches 28.4 BLEU on 12,500 examples. [1]")
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(numbers=numbers))),
        Resolution(work=WORK),
        source(),
    )
    assert len(result.findings) == 2
    ids = [finding.finding_id for finding in result.findings]
    assert len(ids) == len(set(ids))


async def test_numbers_that_agree_by_value_make_no_finding(tmp_path: Path) -> None:
    numbers = [NumberCheck(claimed="28.4", found="28.41", quote=SUPPORT_QUOTE, agrees=False)]
    task = citation_task("The Transformer reaches 28.4 BLEU on WMT 2014. [1]")
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(numbers=numbers))),
        Resolution(work=WORK),
        source(),
    )
    assert result.findings == ()


async def test_a_number_not_found_in_the_source_makes_no_finding(tmp_path: Path) -> None:
    numbers = [NumberCheck(claimed="28.4", found=None, quote="", agrees=False)]
    task = citation_task("The Transformer reaches 28.4 BLEU on WMT 2014. [1]")
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(numbers=numbers))),
        Resolution(work=WORK),
        source(),
    )
    assert result.findings == ()


async def test_no_llm_is_unverifiable(tmp_path: Path) -> None:
    task = citation_task("The Transformer reaches 28.4 BLEU. [1]")
    result = await check_citation(task, offline_context(tmp_path), Resolution(work=WORK), source())
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "no LLM" in result.reason


async def test_an_empty_source_is_unverifiable(tmp_path: Path) -> None:
    task = citation_task("The Transformer reaches 28.4 BLEU. [1]")
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer())),
        Resolution(work=WORK),
        source(level=SourceLevel.NONE),
    )
    assert result.outcome is TaskOutcome.UNVERIFIABLE


async def test_the_prompt_holds_the_claim_and_numbered_passages(tmp_path: Path) -> None:
    model = ScriptedChatModel(replies=[tool_reply(SupportAnswer, answer().model_dump())])
    task = citation_task("The Transformer reaches 28.4 BLEU on WMT 2014. [1]")
    ctx = offline_context(tmp_path, llm=StructuredLLM(model, name="test:main"))
    await check_citation(task, ctx, Resolution(work=WORK), source())
    (sent,) = model.received
    prompt = sent[0].content
    assert isinstance(prompt, str)
    assert "28.4 BLEU" in prompt
    assert "[1]" in prompt  # the passages are numbered
    assert len(chunk_text(SOURCE_TEXT)) >= TOP_PASSAGES
