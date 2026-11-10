from pathlib import Path
from typing import Any

from auditor.agents.integrity import RULE_ID, ScreenFindings, check, integrity_tasks
from auditor.infra.llm import StructuredLLM
from auditor.schemas import (
    AuditTask,
    DocumentAnchor,
    ErrorClass,
    EvidenceKind,
    ParsedPaper,
    ParserInfo,
    RuleTarget,
    Section,
    Sentence,
    SentenceAnchor,
    SentenceKind,
    TaskKind,
    TaskOutcome,
)
from auditor.testing import ScriptedChatModel, offline_context, tool_reply

CAPTION = (
    "Table 2: Cross-dataset transfer. Numbers are simulated drafting results for manuscript "
    "preparation and should be replaced by actual experimental results before submission."
)
BODY = "As shown in Table 2, our method consistently achieves the best performance."
TODO = "TODO: add the ablation on the second dataset here."
UNRESOLVED = "The bound follows from Lemma ?? and the argument in Section ??."
CLEAN = "We report the mean and standard deviation over five seeds."


def paper(*extra: Sentence) -> ParsedPaper:
    return ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sections=(Section(id="exp", title="Experiments"),),
        sentences=(
            Sentence(id="c1", section_id="exp", kind=SentenceKind.CAPTION, text=CAPTION),
            Sentence(id="s1", section_id="exp", text=BODY),
            Sentence(id="s2", section_id="exp", text=CLEAN),
            *extra,
        ),
    )


def run(tmp_path: Path, found: list[dict[str, Any]], p: ParsedPaper) -> Any:
    model = ScriptedChatModel(replies=[tool_reply(ScreenFindings, {"findings": found})])
    ctx = offline_context(tmp_path, paper=p, llm=StructuredLLM(model, name="test:s"))
    return ctx, model


def screen_task(p: ParsedPaper) -> AuditTask:
    (task,) = [t for t in integrity_tasks(p) if t.target == RuleTarget(rule_id=RULE_ID)]
    return task


def test_every_paper_gets_the_integrity_tasks() -> None:
    tasks = integrity_tasks(paper())
    assert {t.kind for t in tasks} == {TaskKind.INTEGRITY}
    assert {t.anchor for t in tasks} == {DocumentAnchor()}
    assert RuleTarget(rule_id=RULE_ID) in {t.target for t in tasks}


async def test_a_caption_saying_the_results_are_simulated_is_flagged(tmp_path: Path) -> None:
    found = [
        {
            "sentence_id": "c1",
            "quote": "Numbers are simulated drafting results for manuscript preparation",
            "kind": "placeholder_results",
            "explanation": "The caption says the reported numbers are simulated.",
        }
    ]
    ctx, model = run(tmp_path, found, paper())
    result = await check(screen_task(ctx.paper), ctx)
    assert result.outcome is TaskOutcome.CHECKED
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.UNFINISHED_CONTENT
    assert isinstance(finding.anchor, SentenceAnchor)
    assert finding.anchor.sentence_id == "c1"
    assert not finding.uncertain
    (quote,) = finding.evidence
    assert quote.kind is EvidenceKind.QUOTE
    assert quote.text in CAPTION
    sent = " ".join(str(m.content) for m in model.received[0])
    assert f"[c1] {CAPTION}" in sent  # captions are read, not only the body


async def test_text_that_looks_cut_off_goes_to_review(tmp_path: Path) -> None:
    # A GROBID split from set C: "The global parsing map is shown in Figs. 5(b-c), ..."
    cut = Sentence(id="s3", section_id="exp", text="The global parsing map is shown in Figs.")
    found = [
        {"sentence_id": "s3", "quote": "The global parsing map is shown", "kind": "unfinished_text"}
    ]
    ctx, model = run(tmp_path, found, paper(cut))
    (finding,) = (await check(screen_task(ctx.paper), ctx)).findings
    assert finding.uncertain
    assert "extracted from the PDF" in str(model.received[0][0].content)


async def test_quotes_that_are_not_verbatim_are_dropped(tmp_path: Path) -> None:
    found = [
        {"sentence_id": "s1", "quote": "results are placeholders", "kind": "placeholder_results"}
    ]
    ctx, _ = run(tmp_path, found, paper())
    assert (await check(screen_task(ctx.paper), ctx)).findings == ()


async def test_leftover_notes_and_unresolved_references_need_no_model(tmp_path: Path) -> None:
    extra = (
        Sentence(id="s3", section_id="exp", text=TODO),
        Sentence(id="s4", section_id="exp", text=UNRESOLVED),
    )
    p = paper(*extra)
    result = await check(screen_task(p), offline_context(tmp_path, paper=p))
    assert result.outcome is TaskOutcome.CHECKED
    by_sentence = {f.anchor.sentence_id: f for f in result.findings}  # type: ignore[union-attr]
    assert set(by_sentence) == {"s3", "s4"}
    assert "TODO" in by_sentence["s3"].evidence[0].text
    assert "??" in by_sentence["s4"].evidence[0].text


async def test_without_a_model_the_screen_says_what_it_skipped(tmp_path: Path) -> None:
    result = await check(screen_task(paper()), offline_context(tmp_path, paper=paper()))
    assert result.outcome is TaskOutcome.CHECKED
    assert result.findings == ()


async def test_variable_names_and_pseudocode_are_not_leftover_markers(tmp_path: Path) -> None:
    # Seen in real papers: a stock called XXX, a rate of X%, an empty list in pseudocode.
    extra = (
        Sentence(id="s3", section_id="exp", text="One stock (call it XXX) is uncorrelated."),
        Sentence(id="s4", section_id="exp", text="A true positive rate of X%, i.e., recall."),
        Sentence(id="s5", section_id="exp", text="2: Initialize: history C = [] 3: repeat"),
    )
    p = paper(*extra)
    result = await check(screen_task(p), offline_context(tmp_path, paper=p))
    assert result.findings == ()


async def test_more_placeholder_spellings_seen_in_real_reviews(tmp_path: Path) -> None:
    # From ICLR reviews: "TODOXXX" in the abstract, "Section X" and "Figure X",
    # "[insert performance difference]".
    extra = (
        Sentence(id="s3", section_id="exp", text="Our method reaches TODOXXX accuracy."),
        Sentence(id="s4", section_id="exp", text="As discussed in Section X, the loss is convex."),
        Sentence(id="s5", section_id="exp", text="It improves by [insert performance difference]."),
        Sentence(id="s6", section_id="exp", text="Let X be a random variable in Section 2."),
    )
    p = paper(*extra)
    result = await check(screen_task(p), offline_context(tmp_path, paper=p))
    found = {f.anchor.sentence_id for f in result.findings}  # type: ignore[union-attr]
    assert found == {"s3", "s4", "s5"}


async def test_table_x_is_table_ten_in_papers_with_roman_numerals(tmp_path: Path) -> None:
    # IEEE-style papers number tables I, II, ..., X; "Table X" there is the tenth table.
    extra = (
        Sentence(id="s3", section_id="exp", text="TABLE IX: Ablation results."),
        Sentence(id="s4", section_id="exp", text="Table X lists the range of the learned values."),
    )
    p = paper(*extra)
    result = await check(screen_task(p), offline_context(tmp_path, paper=p))
    assert result.findings == ()
