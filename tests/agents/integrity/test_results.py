import dataclasses
from pathlib import Path
from typing import Any

import pytest

from auditor.agents.integrity import check
from auditor.agents.integrity.results import (
    DataClaims,
    ResultVerdicts,
    TableImages,
    TableNumbers,
    cell_findings,
    grim_findings,
    number_candidates,
    quoted_cells,
    repeated_spreads,
    table_findings,
    table_text,
    text_comparisons,
    wrong_aggregates,
)
from auditor.infra.context import RunContext
from auditor.infra.llm import StructuredLLM
from auditor.planner.integrity import RESULTS_RULE_ID, integrity_tasks
from auditor.schemas import (
    AuditTask,
    ErrorClass,
    EvidenceKind,
    ParsedPaper,
    ParserInfo,
    RuleTarget,
    Section,
    Sentence,
    SentenceAnchor,
    SentenceKind,
    Table,
    TaskOutcome,
)
from auditor.testing import ScriptedChatModel, offline_context, tool_reply

ROWS = [
    "Method | B2a Acc. (%) | PMI Acc. (%)",
    "Scratch | 86.1 ± 1.4 | 73.8 ± 0.7",
    "Contrastive-SSL | 88.8 ± 1.1 | 76.1 ± 0.6",
    "Ours | 90.5 ± 1.0 | 75.5 ± 0.5",
]
GAIN = "On B2a, Ours improves over Contrastive-SSL by 2.7 points."
BEST = "Ours achieves the best accuracy on both benchmarks."
RIGHT = "On B2a, Ours improves over Scratch by 4.4 points."


def paper(rows: list[str] = ROWS) -> ParsedPaper:
    table_rows = [
        Sentence(id=f"t{i}", section_id="exp", kind=SentenceKind.TABLE, text=row)
        for i, row in enumerate(rows)
    ]
    return ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sections=(Section(id="exp", title="Experiments"),),
        sentences=(
            Sentence(
                id="c1", section_id="exp", kind=SentenceKind.CAPTION, text="Table 1: Accuracy."
            ),
            *table_rows,
            Sentence(id="s1", section_id="exp", text=GAIN),
            Sentence(id="s2", section_id="exp", text=BEST),
            Sentence(id="s3", section_id="exp", text=RIGHT),
        ),
        tables=(Table(id="tab1", label="Table 1", row_ids=tuple(s.id for s in table_rows)),),
    )


def results_task(p: ParsedPaper) -> AuditTask:
    (task,) = [t for t in integrity_tasks(p) if t.target == RuleTarget(rule_id=RESULTS_RULE_ID)]
    return task


def claim(sentence: str, quote: str, kind: str, **fields: Any) -> dict[str, Any]:
    return {"sentence_id": sentence, "quote": quote, "kind": kind, "table_id": "tab1", **fields}


GAIN_CLAIM = claim(
    "s1",
    "Ours improves over Contrastive-SSL by 2.7 points",
    "difference",
    stated="2.7",
    subject={"row_label": "Ours", "value": "90.5"},
    others=[{"row_label": "Contrastive-SSL", "value": "88.8"}],
)
BEST_CLAIM = claim(
    "s2",
    "Ours achieves the best accuracy on both benchmarks",
    "best",
    subject={"row_label": "Ours", "value": "75.5"},
    others=[
        {"row_label": "Scratch", "value": "73.8"},
        {"row_label": "Contrastive-SSL", "value": "76.1"},
    ],
)
RIGHT_CLAIM = claim(
    "s3",
    "Ours improves over Scratch by 4.4 points",
    "difference",
    stated="4.4",
    subject={"row_label": "Ours", "value": "90.5"},
    others=[{"row_label": "Scratch", "value": "86.1"}],
)


def without_challenge(ctx: RunContext) -> RunContext:
    settings = ctx.settings.model_copy(update={"results_challenge_enabled": False})
    return dataclasses.replace(ctx, settings=settings)


def without_number_pass(ctx: RunContext) -> RunContext:
    settings = ctx.settings.model_copy(update={"results_number_pass_enabled": False})
    return dataclasses.replace(ctx, settings=settings)


async def run(
    tmp_path: Path,
    claims: list[dict[str, Any]],
    p: ParsedPaper | None = None,
    verdicts: list[dict[str, Any]] | None = None,
    numbers: list[list[dict[str, Any]]] | None = None,
    **settings: Any,
) -> Any:
    """The check on `p` with the model giving `claims`, then one reply of the number pass
    per list in `numbers` (the pass is off when `numbers` is None), and then `verdicts` in
    the challenge step (which is skipped when `verdicts` is None)."""
    p = p or paper()
    replies = [tool_reply(DataClaims, {"claims": claims})]
    for reply in numbers or []:
        replies.append(tool_reply(TableNumbers, {"numbers": reply}))
    if verdicts is not None:
        replies.append(tool_reply(ResultVerdicts, {"verdicts": verdicts}))
    model = ScriptedChatModel(replies=replies)
    ctx = offline_context(tmp_path, paper=p, llm=StructuredLLM(model, name="test:s"))
    if verdicts is None:
        ctx = without_challenge(ctx)
    if numbers is None:
        ctx = without_number_pass(ctx)
    else:
        ctx = dataclasses.replace(
            ctx, settings=ctx.settings.model_copy(update={"results_number_pass_enabled": True})
        )
    # The cell rule is off unless a test asks for it, so each test sees one step's findings.
    settings = {"results_cell_rule_enabled": False, **settings}
    ctx = dataclasses.replace(ctx, settings=ctx.settings.model_copy(update=settings))
    return await check(results_task(p), ctx), model


async def test_the_challenge_step_drops_a_misread_candidate(tmp_path: Path) -> None:
    verdicts = [
        {"number": 1, "verdict": "inconsistent"},
        {"number": 2, "verdict": "misread", "reason": "the claim is about PMI"},
    ]
    result, model = await run(tmp_path, [GAIN_CLAIM, BEST_CLAIM], verdicts=verdicts)
    (finding,) = result.findings
    assert finding.anchor.sentence_id == "s1"
    challenge = " ".join(str(m.content) for m in model.received[1])
    assert "Contrastive-SSL | 88.8" in challenge
    assert GAIN in challenge


async def test_a_wrong_difference_and_a_wrong_best_claim_are_found(tmp_path: Path) -> None:
    result, model = await run(tmp_path, [GAIN_CLAIM, BEST_CLAIM, RIGHT_CLAIM])
    assert result.outcome is TaskOutcome.CHECKED
    by_sentence = {f.anchor.sentence_id: f for f in result.findings}
    assert set(by_sentence) == {"s1", "s2"}  # the 4.4-point claim is right
    gain = by_sentence["s1"]
    assert gain.error_class is ErrorClass.RESULT_INCONSISTENCY
    assert "1.7" in gain.summary
    assert "2.7" in gain.summary
    kinds = [e.kind for e in gain.evidence]
    assert kinds == [EvidenceKind.QUOTE, EvidenceKind.LOOKUP]
    assert "76.1" in by_sentence["s2"].summary
    prompt = " ".join(str(m.content) for m in model.received[0])
    assert "[tab1] Table 1" in prompt
    assert "Contrastive-SSL | 88.8" in prompt


async def test_numbers_the_model_misreads_from_the_table_are_ignored(tmp_path: Path) -> None:
    wrong_row = {**GAIN_CLAIM, "others": [{"row_label": "Contrastive-SSL", "value": "87.8"}]}
    result, _ = await run(tmp_path, [wrong_row])
    assert result.findings == ()


def test_a_negative_spread_in_a_table_needs_no_model() -> None:
    # Over 100% can be a relative measure (a gain, a normalised speed), so it is not flagged.
    rows = [*ROWS[:2], "Ours | 101.5% ± 1.0 | 75.5 ± -0.5", "Gain | +150% | (120%)"]
    task = results_task(paper(rows))
    found = table_findings(task, paper(rows))
    summaries = " ".join(f.summary for f in found)
    assert "-0.5" in summaries
    assert "101.5" not in summaries
    assert len(found) == 1
    assert all(isinstance(f.anchor, SentenceAnchor) for f in found)


async def test_without_a_model_only_the_table_rules_run(tmp_path: Path) -> None:
    p = paper()
    result = await check(results_task(p), offline_context(tmp_path, paper=p))
    assert result.outcome is TaskOutcome.CHECKED
    assert result.findings == ()


def text_paper(sentence: str) -> ParsedPaper:
    return ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sentences=(Sentence(id="s1", text=sentence),),
    )


COMPARISON = "PlaNet-S reached an IoU of 0.73, significantly higher than U-Net with 0.78."


async def test_a_comparison_in_the_text_alone_is_checked(tmp_path: Path) -> None:
    # A real case (arXiv 2312.11580v1): no table is needed to see that 0.73 < 0.78.
    found = claim(
        "s1",
        "significantly higher than U-Net with 0.78",
        "comparison",
        table_id="",
        relation="higher",
        subject={"row_label": "PlaNet-S", "value": "0.73"},
        others=[{"row_label": "U-Net", "value": "0.78"}],
    )
    result, _ = await run(tmp_path, [found], text_paper(COMPARISON))
    (finding,) = result.findings
    assert isinstance(finding.anchor, SentenceAnchor)
    assert finding.anchor.sentence_id == "s1"
    assert "0.73" in finding.summary
    assert "0.78" in finding.summary


async def test_a_correct_comparison_and_numbers_not_in_the_sentence_pass(tmp_path: Path) -> None:
    right = claim(
        "s1",
        "significantly higher than U-Net with 0.78",
        "comparison",
        table_id="",
        relation="lower",
        subject={"row_label": "PlaNet-S", "value": "0.73"},
        others=[{"row_label": "U-Net", "value": "0.78"}],
    )
    # 0.70 is not in the sentence, so this claim is not judged.
    invented = {
        **right,
        "relation": "higher",
        "others": [{"row_label": "U-Net", "value": "0.70"}],
    }
    result, _ = await run(tmp_path, [right, invented], text_paper(COMPARISON))
    assert result.findings == ()


async def test_a_gain_in_percent_may_mean_percentage_points(tmp_path: Path) -> None:
    # Seen on a real paper: "improves by 4.8%" for 71.5 against 66.7, a 4.8-point gain.
    points = claim(
        "s1",
        "Ours improves over Contrastive-SSL by 2.7 points",
        "relative_percent",
        stated="1.7",
        subject={"row_label": "Ours", "value": "90.5"},
        others=[{"row_label": "Contrastive-SSL", "value": "88.8"}],
    )
    result, _ = await run(tmp_path, [points])
    assert result.findings == ()


def two_table_paper() -> ParsedPaper:
    rows1 = ["Method | AUC", "No Fusion | 0.754", "Ours | 0.801"]
    rows8 = ["Variant | AUC", "Without A-Att | 0.774", "Full | 0.801"]
    sentences = [
        Sentence(id=f"a{i}", section_id="exp", kind=SentenceKind.TABLE, text=r)
        for i, r in enumerate(rows1)
    ] + [
        Sentence(id=f"b{i}", section_id="exp", kind=SentenceKind.TABLE, text=r)
        for i, r in enumerate(rows8)
    ]
    sentences.append(
        Sentence(
            id="s1",
            section_id="exp",
            text="Removing the attention (Table 8) gives the No Fusion model of Table 1.",
        )
    )
    return ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sections=(Section(id="exp", title="Experiments"),),
        sentences=tuple(sentences),
        tables=(
            Table(id="tab1", label="Table 1", row_ids=("a0", "a1", "a2")),
            Table(id="tab8", label="Table 8", row_ids=("b0", "b1", "b2")),
        ),
    )


async def test_one_result_reported_differently_in_two_tables_is_found(tmp_path: Path) -> None:
    # A real ICLR review (arXiv 2407.11546): "Why does the No Fusion value (0.754) in
    # Table 1 differ from the Without A-Att value (0.774) in Table 8?"
    same = claim(
        "s1",
        "gives the No Fusion model of Table 1",
        "same_value",
        table_id="tab1",
        subject={"row_label": "No Fusion", "value": "0.754"},
        others=[{"row_label": "Without A-Att", "value": "0.774", "table_id": "tab8"}],
    )
    result, _ = await run(tmp_path, [same], two_table_paper())
    (finding,) = result.findings
    assert finding.uncertain  # which results are the same is the model's reading
    assert "Table 1" in finding.summary
    assert "Table 8" in finding.summary
    matching = {**same, "others": [{"row_label": "Full", "value": "0.801", "table_id": "tab8"}]}
    matching["subject"] = {"row_label": "Ours", "value": "0.801"}
    result, _ = await run(tmp_path, [matching], two_table_paper())
    assert result.findings == ()


async def test_a_drop_written_with_a_minus_sign_is_a_difference_of_that_size(
    tmp_path: Path,
) -> None:
    # Seen on a real paper: "-8.6 points" for 30.0 against 38.6.
    drop = claim(
        "s3",
        "Ours improves over Scratch by 4.4 points",
        "difference",
        stated="-4.4",
        subject={"row_label": "Scratch", "value": "86.1"},
        others=[{"row_label": "Ours", "value": "90.5"}],
    )
    result, _ = await run(tmp_path, [drop])
    assert result.findings == ()


async def test_a_table_the_parser_could_not_read_is_read_from_its_page_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("auditor.agents.integrity.results.render_page", lambda pdf, page: b"png")
    p = ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sections=(Section(id="exp", title="Experiments"),),
        sentences=(Sentence(id="s1", section_id="exp", text=GAIN),),
        tables=(Table(id="layout-p3", label="Table 1", page=3),),
    )
    transcription = {
        "tables": [
            {
                "label": "Table 1",
                "rows": [["Method", "B2a"], ["Contrastive-SSL", "88.8"], ["Ours", "90.5"]],
            }
        ]
    }
    gain = {**GAIN_CLAIM, "table_id": "layout-p3-img0"}
    model = ScriptedChatModel(
        replies=[
            tool_reply(TableImages, transcription),
            tool_reply(DataClaims, {"claims": [gain]}),
        ]
    )
    ctx = offline_context(tmp_path, paper=p, llm=StructuredLLM(model, name="test:s"))
    ctx = without_number_pass(without_challenge(ctx))
    ctx = dataclasses.replace(ctx, source_pdf=tmp_path / "paper.pdf")
    result = await check(results_task(p), ctx)
    (finding,) = result.findings
    assert "1.7" in finding.summary
    image_call = model.received[0]
    assert any(
        isinstance(m.content, list)
        and any(isinstance(part, dict) and part.get("type") == "image_url" for part in m.content)
        for m in image_call
    )
    plan = " ".join(str(m.content) for m in model.received[1])
    assert "Contrastive-SSL | 88.8" in plan


def test_a_table_read_from_the_layout_replaces_the_parsers_copy() -> None:
    p = ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sentences=(
            Sentence(id="g1", kind=SentenceKind.TABLE, text="Ours 90.5 ± -0.5 merged"),
            Sentence(id="l1", kind=SentenceKind.TABLE, text="Ours | 90.5 ± -0.5"),
        ),
        tables=(
            Table(id="t0", label="Table 1", row_ids=("g1",)),
            Table(id="layout-p3-0", label="Table 1", row_ids=("l1",), page=3),
        ),
    )
    (finding,) = table_findings(results_task(p), p)
    assert finding.anchor.sentence_id == "l1"  # type: ignore[union-attr]


async def test_a_claim_without_a_table_id_is_checked_against_the_table_holding_its_numbers(
    tmp_path: Path,
) -> None:
    result, _ = await run(tmp_path, [{**GAIN_CLAIM, "table_id": ""}])
    (finding,) = result.findings
    assert "1.7" in finding.summary


async def test_a_stated_drop_in_percent_is_compared_by_size(tmp_path: Path) -> None:
    drop = "Without contrast, accuracy falls by 1.9% from Ours to Contrastive-SSL."
    p = paper()
    p = p.model_copy(
        update={"sentences": (*p.sentences, Sentence(id="s4", section_id="exp", text=drop))}
    )
    relative = claim(
        "s4",
        "accuracy falls by 1.9% from Ours",
        "relative_percent",
        stated="1.9%",
        subject={"row_label": "Contrastive-SSL", "value": "88.8"},
        others=[{"row_label": "Ours", "value": "90.5"}],
    )
    result, _ = await run(tmp_path, [relative], p)
    assert not result.findings


async def test_a_relative_change_may_be_taken_against_either_number(tmp_path: Path) -> None:
    # Seen on a real paper: "a relative degradation of 58% (0.0788 cf. 0.0335)".
    rows = ["System | min-tDCF", "Full | 0.0335", "w/o pooling | 0.0788"]

    def degradation(stated: str) -> dict[str, Any]:
        return claim(
            "s1",
            "Ours improves over Contrastive-SSL by 2.7 points",
            "relative_percent",
            stated=stated,
            subject={"row_label": "w/o pooling", "value": "0.0788"},
            others=[{"row_label": "Full", "value": "0.0335"}],
        )

    result, _ = await run(tmp_path, [degradation("58%")], paper(rows))
    assert not result.findings
    result, _ = await run(tmp_path, [degradation("80%")], paper(rows))
    assert len(result.findings) == 1


async def test_a_change_another_column_of_the_same_rows_gives_is_not_reported(
    tmp_path: Path,
) -> None:
    # Ours against Scratch is 4.4 points on B2a and 1.7 on PMI; the model read B2a.
    def gain(stated: str) -> dict[str, Any]:
        return {**RIGHT_CLAIM, "stated": stated}

    result, _ = await run(tmp_path, [gain("1.7")])
    assert not result.findings
    # 0.4 is the difference of the spreads (1.0 against 1.4), not of two results.
    result, _ = await run(tmp_path, [gain("0.4")])
    assert len(result.findings) == 1


async def test_a_number_the_text_gives_for_a_row_is_checked_against_that_row(
    tmp_path: Path,
) -> None:
    # A real paper (arXiv 2512.21706): the text gives a mean rating of 8.93 for the
    # simulation data, its Table 6 gives 6.30.
    rows = ["Audio Source | Mean Rating", "GPT-4 (full-duplex) | 7.07", "Simulation | 6.30"]
    text = "Simulation achieves the highest mean score (8.93) > GPT-4 (7.06)."
    p = paper(rows)
    p = p.model_copy(
        update={"sentences": (*p.sentences, Sentence(id="s4", section_id="exp", text=text))}
    )
    best = claim(
        "s4",
        "Simulation achieves the highest mean score",
        "best",
        subject={"row_label": "Simulation", "value": "8.93"},
        others=[{"row_label": "GPT-4 (full-duplex)", "value": "7.06"}],
    )
    result, _ = await run(tmp_path, [best], p)
    (finding,) = result.findings
    assert "8.93" in finding.summary
    assert "6.30" in finding.summary


async def test_a_page_image_reading_that_runs_numbers_together_is_not_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("auditor.agents.integrity.results.render_page", lambda pdf, page: b"png")
    p = ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sections=(Section(id="exp", title="Experiments"),),
        sentences=(Sentence(id="s1", section_id="exp", text="Ours reaches 64.9 in Table 4."),),
        tables=(Table(id="layout-p7", label="Table 4", page=7),),
    )
    garbled = {"tables": [{"label": "Table 4", "rows": [["Model", "GSM8k"], ["Ours", "64.91.32"]]}]}
    model = ScriptedChatModel(replies=[tool_reply(TableImages, garbled)])
    ctx = offline_context(tmp_path, paper=p, llm=StructuredLLM(model, name="test:s"))
    ctx = dataclasses.replace(without_challenge(ctx), source_pdf=tmp_path / "paper.pdf")
    result = await check(results_task(p), ctx)
    assert not result.findings
    assert len(model.received) == 1  # no table to plan against


async def test_a_value_claim_on_a_row_named_only_by_a_number_is_not_judged(
    tmp_path: Path,
) -> None:
    # A real false alarm (arXiv 2306.00160): "CV = 128 ... resulting in 0.35 M parameters"
    # is about one branch, the row "128 | 128 | 1.01" about the whole model.
    rows = ["CA | CV | Params (M)", "128 | 128 | 1.01", "512 | 128 | 5.75"]
    text = "The block uses CV = 128, resulting in 0.35 M parameters."
    p = paper(rows)
    p = p.model_copy(
        update={"sentences": (*p.sentences, Sentence(id="s4", section_id="exp", text=text))}
    )
    value = claim(
        "s4",
        "resulting in 0.35 M parameters",
        "value",
        stated="0.35",
        subject={"row_label": "128", "value": "1.01"},
    )
    result, _ = await run(tmp_path, [value], p)
    assert not result.findings


def comparison_paper(text: str) -> ParsedPaper:
    return ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sections=(Section(id="exp", title="Experiments"),),
        sentences=(Sentence(id="s1", section_id="exp", text=text),),
    )


async def test_a_reversed_comparison_in_the_text_is_found_without_a_model(
    tmp_path: Path,
) -> None:
    # A real error (arXiv 2312.11580v1).
    text = "PlaNet-S had significantly higher IoU (0.73 ± 0.13) than U-Net (0.78 ± 0.010) (p<0.01)."
    p = comparison_paper(text)
    result = await check(results_task(p), offline_context(tmp_path, paper=p))
    (finding,) = result.findings
    assert finding.anchor.sentence_id == "s1"  # type: ignore[union-attr]
    assert "0.73" in finding.summary
    assert "0.78" in finding.summary


async def test_a_number_just_before_the_comparative_word_is_compared(tmp_path: Path) -> None:
    p = comparison_paper("In Table 6, Source-view reaches 54.81, higher than Merged (56.06).")
    result = await check(results_task(p), offline_context(tmp_path, paper=p))
    (finding,) = result.findings
    assert "54.81" in finding.summary


@pytest.mark.parametrize(
    "text",
    [
        "PlaNet-S had higher IoU (0.83 ± 0.13) than U-Net (0.78 ± 0.01).",
        "Our accuracy is 3.16% higher than the benchmark of 73.47%.",
        "In Table 6, Source-view reaches 57.81, higher than Merged (56.06).",
        "Our model uses more than 2.5 times fewer parameters than the baseline.",
        "The p-value is lower (p<0.01) than the threshold of 0.05 used before.",
        "Loss is lower with 0.5 dropout than with 0.1 dropout on this task.",
    ],
)
async def test_comparisons_that_agree_or_are_not_two_values_are_not_reported(
    tmp_path: Path, text: str
) -> None:
    p = comparison_paper(text)
    result = await check(results_task(p), offline_context(tmp_path, paper=p))
    if "dropout" in text:
        # Settings, not results: the rule cannot tell. With a model, the challenge step
        # sees the sentence (next test); without one, it is reported.
        assert len(result.findings) == 1
    else:
        assert not result.findings


async def test_the_challenge_step_also_reviews_the_comparisons_the_rule_finds(
    tmp_path: Path,
) -> None:
    p = comparison_paper("Loss is lower with 0.5 dropout than with 0.1 dropout on this task.")
    verdicts = [{"number": 1, "verdict": "other_reading", "reason": "settings"}]
    result, model = await run(tmp_path, [], p, verdicts=verdicts)
    assert not result.findings
    assert "0.5 dropout" in " ".join(str(m.content) for m in model.received[-1])


def test_one_mean_and_spread_in_two_rows_is_flagged_as_uncertain() -> None:
    rows = [
        *ROWS[:2],
        "Contrastive-SSL | 88.8 ± 1.1 | 76.1 ± 0.6",
        "Ours | 88.8 ± 1.1 | 75.5 ± 0.5",
    ]
    p = paper(rows)
    (finding,) = repeated_spreads(results_task(p), p)
    assert finding.uncertain
    assert "88.8 ± 1.1" in finding.summary
    # Different spreads, or a mean too short to be telling, are left alone.
    assert repeated_spreads(results_task(paper()), paper()) == []
    # Rows that share another cell repeat a setting by design.
    same = [*ROWS[:2], "A | 88.8 ± 1.1 | 76.1 ± 0.6", "B | 88.8 ± 1.1 | 76.1 ± 0.6"]
    assert repeated_spreads(results_task(paper(same)), paper(same)) == []


def test_an_average_column_that_does_not_average_its_row_is_flagged() -> None:
    rows = [
        "Method | A | B | C | Avg",
        "Base | 70.0 | 80.0 | 90.0 | 80.0",
        "Small | 60.0 | 70.0 | 80.0 | 70.0",
        "Large | 72.0 | 81.0 | 90.0 | 81.0",
        "Ours | 71.0 | 82.0 | 93.0 | 84.0",
    ]
    p = paper(rows)
    (finding,) = wrong_aggregates(results_task(p), p)
    assert finding.anchor.sentence_id == "t4"  # type: ignore[union-attr]
    assert "82.0" in finding.summary  # the mean the row's cells give
    assert finding.uncertain
    # A last column no row averages is another quantity, not a wrong average.
    other = [
        "Method | A | B | Mean",
        "X | 1.0 | 2.0 | 9.0",
        "Y | 3.0 | 4.0 | 8.0",
        "Z | 5.0 | 6.0 | 7.0",
    ]
    assert wrong_aggregates(results_task(paper(other)), paper(other)) == []
    totals = ["Split | Train | Val | Test | Total", "Images | 800 | 100 | 100 | 1000"]
    assert wrong_aggregates(results_task(paper(totals)), paper(totals)) == []


def test_the_grim_test_flags_an_impossible_mean_of_whole_numbers() -> None:
    def with_text(text: str) -> ParsedPaper:
        p = paper()
        extra = Sentence(id="s9", section_id="exp", text=text)
        return p.model_copy(update={"sentences": (*p.sentences, extra)})

    bad = with_text("Participants rated the tool highly (M = 3.47, SD = 0.81, N = 28).")
    (finding,) = grim_findings(results_task(bad), bad)
    assert "3.47" in finding.summary
    assert finding.uncertain
    # 3.46 is 97/28, a possible mean; large samples say nothing.
    for text in ("Ratings were high (M = 3.46, N = 28).", "Ratings were high (M = 3.47, N = 300)."):
        ok = with_text(text)
        assert grim_findings(results_task(ok), ok) == []


async def test_a_gain_whose_sentence_gives_both_numbers_is_not_excused_by_another_column(
    tmp_path: Path,
) -> None:
    # Ours against Scratch is 4.4 on B2a and 1.7 on PMI; the sentence pins B2a.
    text = "Ours improves over Scratch by 1.7 points (90.5 against 86.1)."
    p = paper()
    extra = Sentence(id="s5", section_id="exp", text=text)
    p = p.model_copy(update={"sentences": (*p.sentences, extra)})
    pinned = {**RIGHT_CLAIM, "sentence_id": "s5", "stated": "1.7", "quote": text[:-1]}
    result, _ = await run(tmp_path, [pinned], p)
    assert len(result.findings) == 1


async def test_a_gain_without_its_compared_result_is_completed_from_the_sentence(
    tmp_path: Path,
) -> None:
    text = "SAINT-i improves over Logistic Reg. by 3.6 points (92.83 against 90.73)."
    p = paper()
    p = p.model_copy(
        update={"sentences": (*p.sentences, Sentence(id="s7", section_id="exp", text=text))}
    )
    gain = claim(
        "s7",
        "SAINT-i improves over Logistic Reg. by 3.6 points",
        "difference",
        stated="3.6",
        subject={"row_label": "SAINT-i", "value": "92.83"},
        others=[],
    )
    result, _ = await run(tmp_path, [gain], p)
    (finding,) = result.findings
    assert "2.1" in finding.summary


SCORES = ["Model | Avg. ACC | Avg. MF1", "BASE | 68.76 | 63.34", "Ours | 75.10 | 70.02"]
# A set D plant: the text gives 74.77 for BASE's MF1, which Table 1 gives as 63.34.
WORST = "In Table 1, BASE performs the worst (68.76% in average ACC and 74.77% in average MF1)."


def score_paper(*texts: str, rows: list[str] = SCORES) -> ParsedPaper:
    table_rows = [
        Sentence(id=f"t{i}", section_id="exp", kind=SentenceKind.TABLE, text=row)
        for i, row in enumerate(rows)
    ]
    body = [Sentence(id=f"s{i}", section_id="exp", text=t) for i, t in enumerate(texts, start=1)]
    return ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sections=(Section(id="exp", title="Experiments"),),
        sentences=(*table_rows, *body),
        tables=(Table(id="tab1", label="Table 1", row_ids=tuple(s.id for s in table_rows)),),
    )


def mf1(table_value: str, **fields: Any) -> dict[str, Any]:
    return {
        "sentence_id": "s1",
        "number": "74.77",
        "quote": "and 74.77% in average MF1",
        "table_id": "tab1",
        "row_label": "BASE",
        "column": "Avg. MF1",
        "table_value": table_value,
        **fields,
    }


ACC = {
    "sentence_id": "s1",
    "number": "68.76",
    "quote": "68.76% in average ACC",
    "table_id": "tab1",
    "row_label": "BASE",
    "column": "Avg. ACC",
    "table_value": "68.76",
}


async def test_a_number_that_differs_from_the_row_it_quotes_is_found(tmp_path: Path) -> None:
    verdicts = [{"number": 1, "verdict": "inconsistent"}]
    p = score_paper(WORST)
    result, model = await run(tmp_path, [], p, verdicts=verdicts, numbers=[[ACC, mf1("63.34")]])
    (finding,) = result.findings
    assert finding.anchor.sentence_id == "s1"
    assert "74.77" in finding.summary
    assert "63.34" in finding.summary
    assert "Avg. MF1" in finding.summary
    assert not finding.uncertain
    numbers = " ".join(str(m.content) for m in model.received[1])
    assert f"[s1] {WORST}" in numbers
    assert "BASE | 68.76 | 63.34" in numbers
    # The candidate went through the challenge step with its table.
    challenge = " ".join(str(m.content) for m in model.received[2])
    assert "BASE | 68.76 | 63.34" in challenge
    assert WORST in challenge


async def test_a_number_that_matches_its_cell_gives_nothing(tmp_path: Path) -> None:
    right = WORST.replace("74.77", "63.3")  # 63.3 is 63.34 rounded
    p = score_paper(right)
    result, _ = await run(tmp_path, [], p, numbers=[[ACC, mf1("63.34", number="63.3")]])
    assert not result.findings
    # A percentage the table gives as a fraction is the same number.
    fractions = ["Model | Avg. ACC | Avg. MF1", "BASE | 0.6876 | 0.6334", "Ours | 0.75 | 0.70"]
    p = score_paper(right, rows=fractions)
    result, _ = await run(tmp_path, [], p, numbers=[[mf1("0.6334", number="63.3")]])
    assert not result.findings


async def test_a_number_the_model_says_is_not_a_table_value_gives_nothing(
    tmp_path: Path,
) -> None:
    p = score_paper(WORST)
    unread = mf1("", table_id="", row_label="", column="")
    result, _ = await run(tmp_path, [], p, numbers=[[unread]])
    assert not result.findings


async def test_a_table_value_that_is_not_in_the_named_row_is_ignored(tmp_path: Path) -> None:
    p = score_paper(WORST)
    # 70.02 is Ours's MF1, not BASE's; 64.34 is in no row; 74.7 is not in the sentence.
    replies = [mf1("70.02"), mf1("64.34"), mf1("63.34", number="74.7")]
    result, _ = await run(tmp_path, [], p, numbers=[replies])
    assert not result.findings


def test_the_number_pass_reads_sentences_naming_a_table_first_up_to_the_cap() -> None:
    p = score_paper(
        "Ours reaches 75.10 in average accuracy, the best of all models.",
        "We train for 10 epochs with the settings of Table 1.",  # no decimal number
        "Table 1 lists the scores, where BASE gets 68.76 on average.",
        "Table 1 shows that Ours reaches 70.02 in average MF1.",
        "A plain sentence with 3.5 points of gain and no table or row.",
    )
    picked = number_candidates(p, set(), 10)
    assert [s.id for s, _ in picked] == ["s3", "s4", "s1"]
    assert all([t.id for t in tables] == ["tab1"] for _, tables in picked)
    assert [s.id for s, _ in number_candidates(p, {"s3"}, 2)] == ["s4", "s1"]
    assert [s.id for s, _ in number_candidates(p, set(), 1)] == ["s3"]


async def test_the_number_pass_respects_its_cap(tmp_path: Path) -> None:
    texts = [f"Table 1 shows that BASE reaches {60 + i}.5 in setting {i}." for i in range(6)]
    p = score_paper(*texts)
    result, model = await run(tmp_path, [], p, numbers=[[]], results_number_max_sentences=2)
    assert not result.findings
    assert len(model.received) == 2  # the plan, then one call of the number pass
    numbers = " ".join(str(m.content) for m in model.received[1])
    assert "[s1]" in numbers
    assert "[s2]" in numbers
    assert "[s3]" not in numbers


async def test_with_the_number_pass_off_there_is_no_extra_call(tmp_path: Path) -> None:
    p = score_paper(WORST)
    result, model = await run(tmp_path, [], p, numbers=[], results_number_pass_enabled=False)
    assert not result.findings
    assert len(model.received) == 1  # only the plan


CELL_ROWS = [
    "Model | Avg. ACC | Avg. MF1",
    "BASE | 68.76 | 63.34",
    "SleepDG | 75.03 ± 0.41 | 69.64 ± 0.52",
]
SECOND = ["Model | ACC", "BASE | 55.12", "SleepDG | 61.40"]
# A set D plant, with the table named: BASE's MF1 is 63.34 in Table 1.
CELL_WRONG = "As Table 1 shows, BASE performs the worst (68.76% in average ACC and 74.77% in MF1)."


def cell_paper(*texts: str, rows: list[str] = CELL_ROWS) -> ParsedPaper:
    """A paper whose body sentences are `texts`, with Table 1 (`rows`) and Table 2."""
    first = [
        Sentence(id=f"t{i}", section_id="exp", kind=SentenceKind.TABLE, text=row)
        for i, row in enumerate(rows)
    ]
    second = [
        Sentence(id=f"u{i}", section_id="exp", kind=SentenceKind.TABLE, text=row)
        for i, row in enumerate(SECOND)
    ]
    body = [Sentence(id=f"s{i}", section_id="exp", text=t) for i, t in enumerate(texts, start=1)]
    return ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sections=(Section(id="exp", title="Experiments"),),
        sentences=(*first, *second, *body),
        tables=(
            Table(id="tab1", label="Table 1", row_ids=tuple(s.id for s in first)),
            Table(id="tab2", label="Table 2", row_ids=tuple(s.id for s in second)),
        ),
    )


def cells(*texts: str, rows: list[str] = CELL_ROWS) -> list[str]:
    p = cell_paper(*texts, rows=rows)
    return [f.summary for f in cell_findings(results_task(p), p)]


def test_the_cell_rule_flags_a_number_no_cell_of_its_row_holds() -> None:
    p = cell_paper(CELL_WRONG)
    (finding,) = cell_findings(results_task(p), p)
    assert finding.uncertain
    start = CELL_WRONG.index("74.77")
    assert finding.anchor == SentenceAnchor(sentence_id="s1", start=start, end=start + 5)
    assert "74.77" in finding.summary
    assert "BASE" in finding.summary
    assert "63.34" in finding.summary
    assert "BASE | 68.76 | 63.34" in finding.evidence[-1].text


def test_the_cell_rule_accepts_numbers_that_agree_with_the_row() -> None:
    assert not cells(
        # The cell itself, rounded, and cut to fewer decimals.
        "In Table 1, BASE reaches 63.34 in average MF1, which is the lowest score.",
        "In Table 1, BASE reaches 63.3 in average MF1, which is the lowest score.",
        "In Table 1, SleepDG reaches 75.0 in average accuracy and is the best of the models.",
        # A mean with its spread.
        "In Table 1, SleepDG reaches 69.64 ± 0.52 in MF1, which is the best of all.",
    )
    # A percentage the table gives as a fraction.
    fractions = ["Model | Avg. ACC | Avg. MF1", "BASE | 0.6876 | 0.6334"]
    assert not cells("In Table 1, BASE reaches 63.34% in average MF1 on the data.", rows=fractions)


def test_the_cell_rule_leaves_changes_and_other_rows_numbers_alone() -> None:
    assert not cells(
        # Changes between two rows, written in several ways.
        "In Table 1, SleepDG improves the MF1 of BASE by 6.30 points on all of the data.",
        "In Table 1, SleepDG gets an MF1 that is 6.30 points above the score of BASE.",
        "In Table 1, SleepDG reaches +6.30 in MF1 and +6.27 in accuracy over the baseline.",
        "Table 1 shows a relative gain for SleepDG of 9.94% in MF1 over BASE on the data.",
        "In Table 1, SleepDG reaches 6.30 in MF1 above the baseline, the largest of the gains.",
        # Each number belongs to the nearest label before it.
        "In Table 1, BASE gets 63.34 and SleepDG 69.64 in average MF1 on the data.",
        "In Table 1, BASE is far behind SleepDG (69.64) in average MF1 on the data.",
        # A number of another table, and ranges.
        "In Table 1, BASE reaches 55.12 in accuracy on the second set, as in the other table.",
        "In Table 1, BASE scores between 63.34 and 64.50 in MF1 over the five domains.",
        "In Table 1, BASE reaches 60.12-64.50 in MF1 over the five domains in the study.",
        # Values for several rows, read in order.
        "In Table 1, BASE and SleepDG reach 64.50 and 70.10 in MF1, respectively.",
    )


def test_the_cell_rule_needs_the_table_named_and_a_prose_sentence() -> None:
    worst = "BASE performs the worst (68.76% in average ACC and 74.77% in MF1)."
    assert not cells(worst)
    assert not cells("We compare five models on the data.", worst)
    # The sentence before it may name the table.
    (summary,) = cells("Table 1 compares five models on the data.", worst)
    assert "74.77" in summary
    # A table the parser ran into the body text.
    assert not cells("Table 1 Model ACC MF1 BASE 74.77 63.34 SleepDG 75.03 69.64 MF1 ACC")
    # A spread written without a space.
    spread = "In Table 1, SleepDG reaches an MF1 of 69.64±0.71 on all of the data."
    assert not cells(spread, rows=["Model | AUC | MF1", "SleepDG | 0.75 | 69.64"])


async def test_the_cell_rule_is_on_by_default_and_can_be_turned_off(tmp_path: Path) -> None:
    p = cell_paper(CELL_WRONG)
    ctx = offline_context(tmp_path, paper=p)
    (finding,) = (await check(results_task(p), ctx)).findings
    assert "74.77" in finding.summary
    off = ctx.settings.model_copy(update={"results_cell_rule_enabled": False})
    assert not (await check(results_task(p), dataclasses.replace(ctx, settings=off))).findings


async def test_the_challenge_step_reviews_the_cell_rule_findings(tmp_path: Path) -> None:
    p = cell_paper(CELL_WRONG)
    verdicts = [{"number": 1, "verdict": "misread"}]
    result, model = await run(tmp_path, [], p, verdicts=verdicts, results_cell_rule_enabled=True)
    assert not result.findings
    challenge = " ".join(str(m.content) for m in model.received[1])
    assert "74.77" in challenge
    assert "BASE | 68.76 | 63.34" in challenge
    assert CELL_WRONG in challenge


WORDED_ROWS = [
    "Model | Data | Labels | MF1",
    "BASE | all of the data | yes | 63.34",
    "SleepDG | the source domains | no | 69.64",
]
# Table 1 run into the body by the parser, with words enough to pass for prose.
RUN_IN = (
    "Table 1 Model Data Labels MF1 BASE all of the data yes 74.77 SleepDG the source "
    "domains no 69.64"
)


def test_table_text_in_the_body_is_found_by_its_words() -> None:
    near = "In Table 1, BASE is trained on less data than SleepDG and reaches 63.34 in MF1."
    p = cell_paper(RUN_IN, near, CELL_WRONG, rows=WORDED_ROWS)
    assert table_text(p) == {"s1"}
    # Short sentences are left alone.
    assert not table_text(cell_paper("BASE 63.34 SleepDG 69.64", rows=WORDED_ROWS))
    # The last cells of a table run together with its caption.
    caption = "31.36 25.73 Table 4: Performance of the networks with fewer latent layers."
    assert table_text(cell_paper(caption, near, rows=WORDED_ROWS)) == {"s1"}


async def test_no_rule_reads_table_text_in_the_body(tmp_path: Path) -> None:
    p = cell_paper(RUN_IN, rows=WORDED_ROWS)
    task = results_task(p)
    # Read as prose, the run-in table gives BASE a number its row does not hold.
    assert cell_findings(task, p)
    assert not (await check(task, offline_context(tmp_path, paper=p))).findings
    reversed_ = cell_paper("In Table 1, BASE reaches 68.76, higher than SleepDG (75.03).")
    assert text_comparisons(results_task(reversed_), reversed_)
    assert not text_comparisons(results_task(reversed_), reversed_, skip={"s1"})


def test_the_cell_rule_reads_labels_in_any_case_and_without_generic_tails() -> None:
    rows = [
        "Audio Source | Mean Rating",
        "GPT-4 (full-duplex) | 7.07",
        "Moshi (full-duplex) | 6.85",
        "Simulation dataset audio | 6.30",
    ]
    # arXiv 2512.21706: Table 6 gives 6.30 for simulated audio, the text 8.93.
    text = "As Table 1 shows, rationales on simulation were rated highest (mean 8.93)."
    (summary,) = cells(text, rows=rows)
    assert "8.93" in summary
    assert "6.30" in summary
    assert not cells("As Table 1 shows, rationales on simulation were rated 6.30.", rows=rows)


def test_the_cell_rule_reads_a_unique_first_part_of_a_label() -> None:
    rows = [
        "Model | Score",
        "GPT-4o mini 2024-07 | 7.07",
        "GPT-4o large 2024-05 | 7.52",
        "Ours w/o A | 6.85",
        "Ours w/o B | 6.12",
    ]
    (summary,) = cells("In Table 1, GPT-4o mini reaches a mean score of 7.31.", rows=rows)
    assert "7.31" in summary
    # One word is not enough, and a first part two labels share names neither.
    assert not cells("In Table 1, GPT-4o reaches a mean score of 7.31.", rows=rows)
    assert not cells("In Table 1, Ours w/o reaches a mean score of 6.47.", rows=rows)


def test_quoted_cells_are_the_numbers_prose_quotes_from_a_named_row() -> None:
    rows = ["Audio Source | Mean Rating", "Moshi | 6.85", "Simulation dataset audio | 6.30"]
    p = cell_paper(
        "Rationales on simulation were rated 6.30 on average by the five raters.",
        "Both Moshi and the other source were rated 6.85 by the raters.",
        RUN_IN,
        rows=rows,
    )
    found = [(s.id, s.text[a:b], t.id, label) for s, a, b, t, label in quoted_cells(p)]
    assert found == [
        ("s1", "6.30", "tab1", "Simulation"),
        ("s2", "6.85", "tab1", "Moshi"),
    ]
    assert [s.id for s, *_ in quoted_cells(p, {"s1"})] == ["s2"]
    # A change that happens to equal a cell is not a quoted value.
    changes = cell_paper(
        "On new sources, Moshi exceeds the simulation by 6.30 on the mean rating.",
        "Moshi is rated 6.85 points higher than the simulation by the raters.",
        rows=rows,
    )
    assert not quoted_cells(changes)
