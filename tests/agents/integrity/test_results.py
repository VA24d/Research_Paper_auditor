import dataclasses
from pathlib import Path
from typing import Any

import pytest

from auditor.agents.integrity import check
from auditor.agents.integrity.results import (
    DataClaims,
    ResultVerdicts,
    TableImages,
    table_findings,
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


async def run(
    tmp_path: Path,
    claims: list[dict[str, Any]],
    p: ParsedPaper | None = None,
    verdicts: list[dict[str, Any]] | None = None,
) -> Any:
    """The check on `p` with the model giving `claims`, and then `verdicts` in the
    challenge step (which is skipped when `verdicts` is None)."""
    p = p or paper()
    replies = [tool_reply(DataClaims, {"claims": claims})]
    if verdicts is not None:
        replies.append(tool_reply(ResultVerdicts, {"verdicts": verdicts}))
    model = ScriptedChatModel(replies=replies)
    ctx = offline_context(tmp_path, paper=p, llm=StructuredLLM(model, name="test:s"))
    if verdicts is None:
        ctx = without_challenge(ctx)
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
    ctx = dataclasses.replace(without_challenge(ctx), source_pdf=tmp_path / "paper.pdf")
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


@pytest.mark.parametrize(
    "text",
    [
        "PlaNet-S had higher IoU (0.83 ± 0.13) than U-Net (0.78 ± 0.01).",
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
