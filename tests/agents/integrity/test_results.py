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
