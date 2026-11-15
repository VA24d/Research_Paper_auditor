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
    + "The base model reaches 26.4 BLEU after training on 12,000 sentence pairs."
)
SUPPORT_QUOTE = "The Transformer reaches 28.4 BLEU on WMT 2014 English to German."
CONTRA_QUOTE = "Recurrence and convolution were the dominant sequence models before it."
# The sentence that states other values than the claims below do.
MISMATCH_QUOTE = "The base model reaches 26.4 BLEU after training on 12,000 sentence pairs."

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


async def test_insufficient_leaves_the_claim_unverifiable(tmp_path: Path) -> None:
    task = citation_task("The Transformer reaches 28.4 BLEU on WMT 2014. [1]")
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(verdict="insufficient", quote=""))),
        Resolution(work=WORK),
        source(),
    )
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "does not settle the claim" in result.reason
    assert result.findings == ()


async def test_an_unknown_verdict_is_treated_as_insufficient(tmp_path: Path) -> None:
    task = citation_task("The Transformer reaches 28.4 BLEU on WMT 2014. [1]")
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(verdict="maybe"))),
        Resolution(work=WORK),
        source(),
    )
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.findings == ()


async def test_a_disagreeing_number_is_a_mismatch_finding(tmp_path: Path) -> None:
    numbers = [NumberCheck(claimed="28.4", found="26.4", quote=MISMATCH_QUOTE, agrees=False)]
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
    numbers = [NumberCheck(claimed="28.4", found="26.4", quote=MISMATCH_QUOTE, agrees=True)]
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
    numbers = [NumberCheck(claimed="28.4", found="26.4", quote=MISMATCH_QUOTE, agrees=False)]
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
        NumberCheck(claimed="28.4", found="26.4", quote=MISMATCH_QUOTE, agrees=False),
        NumberCheck(claimed="12,500", found="12,000", quote=MISMATCH_QUOTE, agrees=False),
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


async def test_numbers_the_model_made_up_make_no_finding(tmp_path: Path) -> None:
    """A value the model reports must be where it says: `claimed` in the claim, `found` in
    the quoted sentence. A genuine quote with invented values is a misread, and the
    verifier would otherwise accept it on the quote's offsets."""
    task = citation_task("The Transformer reaches 28.4 BLEU on WMT 2014. [1]")
    for check in (
        NumberCheck(claimed="9999", found="8888", quote=MISMATCH_QUOTE, agrees=False),
        NumberCheck(claimed="28.4", found="8888", quote=MISMATCH_QUOTE, agrees=False),
        NumberCheck(claimed="9999", found="26.4", quote=MISMATCH_QUOTE, agrees=False),
    ):
        result = await check_citation(
            task,
            offline_context(tmp_path, llm=llm_with(answer(numbers=[check]))),
            Resolution(work=WORK),
            source(),
        )
        assert result.findings == (), check


async def test_a_value_quoted_from_outside_the_source_makes_no_finding(tmp_path: Path) -> None:
    numbers = [
        NumberCheck(
            claimed="28.4",
            found="26.4",
            quote="An invented sentence that reports 26.4 BLEU for the model.",
            agrees=False,
        )
    ]
    task = citation_task("The Transformer reaches 28.4 BLEU on WMT 2014. [1]")
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(numbers=numbers))),
        Resolution(work=WORK),
        source(),
    )
    assert result.findings == ()


async def test_the_source_scan_is_not_run_against_a_publisher_page(tmp_path: Path) -> None:
    """A landing page may hold the abstract alone: a number it lacks is not a mismatch."""
    numbers = [NumberCheck(claimed="31.7", found=None, quote="", agrees=True)]
    task = citation_task("The Transformer reaches 31.7 BLEU. [1]")
    page = SourceText(work=WORK, level=SourceLevel.FULL_HTML, text=SOURCE_TEXT, partial=True)
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(numbers=numbers))),
        Resolution(work=WORK),
        page,
    )
    assert result.findings == ()
    # The same text as a complete full text is scanned, and the number is missing.
    result = await check_citation(
        task,
        offline_context(tmp_path, llm=llm_with(answer(numbers=numbers))),
        Resolution(work=WORK),
        source(),
    )
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.CLAIM_NUMBER_MISMATCH


async def test_a_value_must_be_a_whole_number_of_its_text(tmp_path: Path) -> None:
    """Part of a number, a rounded neighbour or a word inside a word is not the value:
    "2.1" is not in "92.1", "500" not in "12,500", "one" not in "someone"."""
    text = "Prior work reports 92.1% exact match with 12,500 examples, 13e-4 lr and someone."
    task = citation_task(text + " [1]")
    page = SourceText(work=WORK, level=SourceLevel.FULL_HTML, text=SOURCE_TEXT + " " + text)
    for claimed, found in (
        ("2.1", "92.1"),
        ("88,500", "12,500"),
        ("3e-4", "13e-4"),
        ("92.9", "92.1"),
        ("one", "13e-4"),
    ):
        numbers = [NumberCheck(claimed=claimed, found=found, quote=text, agrees=False)]
        result = await check_citation(
            task,
            offline_context(tmp_path, llm=llm_with(answer(numbers=numbers))),
            Resolution(work=WORK),
            page,
        )
        assert result.findings == (), (claimed, found)


async def test_values_are_compared_as_their_texts_write_them(tmp_path: Path) -> None:
    """A percentage keeps both readings, a leading dot and a sign are part of the value,
    and an exponent's digits are not a number of their own."""
    cases = (
        # claim value, quote, claimed, found, a mismatch?
        ("3e-4", "The learning rate is 3e-4 in this experiment.", "4", "3e-4", False),
        ("92.1%", "Accuracy is 92.1% on this benchmark.", "92.1%", "92.1", False),
        ("92.1%", "Accuracy is 92.1% on this benchmark.", "0.921", "92.1", False),
        (".5", "The weight decay is 0.5 in all runs.", ".5", "0.5", False),
        ("-0.5", "The correlation is 0.5 on held-out data.", "-0.5", "0.5", True),
    )
    for value, quote, claimed, found, mismatch in cases:
        task = citation_task(f"The cited work reports {value} for this. [1]")
        page = SourceText(work=WORK, level=SourceLevel.FULL_HTML, text=SOURCE_TEXT + " " + quote)
        numbers = [NumberCheck(claimed=claimed, found=found, quote=quote, agrees=False)]
        result = await check_citation(
            task,
            offline_context(tmp_path, llm=llm_with(answer(numbers=numbers))),
            Resolution(work=WORK),
            page,
        )
        assert bool(result.findings) is mismatch, (value, claimed, found)
