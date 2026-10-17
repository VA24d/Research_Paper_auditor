from pathlib import Path
from typing import Any

from auditor.agents.integrity import check
from auditor.agents.integrity.results import DataClaims, table_findings
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


async def run(tmp_path: Path, claims: list[dict[str, Any]], p: ParsedPaper | None = None) -> Any:
    p = p or paper()
    model = ScriptedChatModel(replies=[tool_reply(DataClaims, {"claims": claims})])
    ctx = offline_context(tmp_path, paper=p, llm=StructuredLLM(model, name="test:s"))
    return await check(results_task(p), ctx), model


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
            id="s1", section_id="exp", text="Removing the attention gives the No Fusion model."
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
        "Removing the attention gives the No Fusion model",
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
