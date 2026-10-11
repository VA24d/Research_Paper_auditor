from typing import Any

from langchain_core.tools import tool

from auditor.baselines.b0 import run_b0
from auditor.baselines.b1 import run_b1
from auditor.baselines.common import ReportedErrors, render_paper, to_predictions
from auditor.infra.llm import StructuredLLM
from auditor.infra.tools import tracked_tool
from auditor.infra.usage import UsageLog
from auditor.schemas import (
    BibliographyAnchor,
    BibTarget,
    DocumentAnchor,
    ErrorClass,
    MarkerTarget,
    PageAnchor,
    RuleTarget,
    SentenceAnchor,
    SettingTarget,
    SystemName,
)
from auditor.schemas.examples import REPOSITORY, example_paper, example_venue
from auditor.testing import ScriptedChatModel, tool_reply


def error(**fields: Any) -> dict[str, Any]:
    base = {
        "error_class": "claim_number_mismatch",
        "anchor_kind": "sentence",
        "anchor_id": "s2",
        "target_kind": "bibliography",
        "target": "b1",
        "summary": "The number is not in the cited work.",
    }
    return {**base, **fields}


def test_render_paper_shows_ids_pages_references_and_rules() -> None:
    text = render_paper(example_paper(), venue=example_venue(), repository=REPOSITORY)
    assert "Pages: 2" in text
    assert "Code repository: https://github.com/example/auditor-demo" in text
    assert "## Introduction" in text
    assert "[s2] (p. 1) Prior work reports 92.1% exact match on SQuAD [2, 3]." in text
    assert "[b0] Ashish Vaswani, Noam Shazeer. Attention Is All You Need. NeurIPS. 2017" in text
    assert "[page-limit] The main text is at most 9 pages." in text


def test_to_predictions_maps_every_anchor_and_target_kind() -> None:
    reported = ReportedErrors.model_validate(
        {
            "errors": [
                error(),
                error(
                    error_class="reference_not_found",
                    anchor_kind="bibliography",
                    anchor_id="b2",
                    target="b2",
                ),
                error(anchor_id="s3", target_kind="marker", target="[7]"),
                error(
                    error_class="setting_mismatch",
                    anchor_id="s4",
                    target_kind="setting",
                    target="learning_rate",
                ),
                error(
                    error_class="template_violation",
                    anchor_kind="page",
                    anchor_id="2",
                    target_kind="rule",
                    target="page-limit",
                ),
                error(
                    error_class="template_violation",
                    anchor_kind="document",
                    anchor_id=None,
                    target_kind="rule",
                    target="anonymous",
                ),
            ]
        }
    )
    predictions, dropped = to_predictions(
        reported, example_paper(), run_id="r1", system=SystemName.B0
    )
    assert dropped == []
    assert [(p.anchor, p.target) for p in predictions] == [
        (SentenceAnchor(sentence_id="s2"), BibTarget(bib_id="b1")),
        (BibliographyAnchor(bib_id="b2"), BibTarget(bib_id="b2")),
        (SentenceAnchor(sentence_id="s3"), MarkerTarget(marker="[7]")),
        (SentenceAnchor(sentence_id="s4"), SettingTarget(key="learning_rate")),
        (PageAnchor(pages=(2,)), RuleTarget(rule_id="page-limit")),
        (DocumentAnchor(), RuleTarget(rule_id="anonymous")),
    ]
    assert all(p.state is None and p.is_positive for p in predictions)


def test_to_predictions_drops_errors_that_point_outside_the_paper() -> None:
    reported = ReportedErrors.model_validate(
        {
            "errors": [
                error(anchor_id="s99"),
                error(target="b99"),
                error(anchor_kind="page", anchor_id="7"),
                error(anchor_kind="page", anchor_id="two"),
                error(target_kind="setting", target="2nd rate"),
                error(target_kind="setting", target="Learning Rate"),
                error(),
            ]
        }
    )
    predictions, dropped = to_predictions(
        reported, example_paper(), run_id="r1", system=SystemName.B0
    )
    assert [p.target for p in predictions] == [
        SettingTarget(key="learning_rate"),
        BibTarget(bib_id="b1"),
    ]
    assert len(dropped) == 5


async def test_b0_asks_once_and_records_usage() -> None:
    model = ScriptedChatModel(replies=[tool_reply(ReportedErrors, {"errors": [error()]})])
    usage = UsageLog()
    result = await run_b0(
        example_paper(), StructuredLLM(model, name="test:scripted"), run_id="r1", usage=usage
    )
    assert [p.error_class for p in result.predictions] == [ErrorClass.CLAIM_NUMBER_MISMATCH]
    assert result.predictions[0].system is SystemName.B0
    assert len(model.received) == 1
    assert "[s2]" in str(model.received[0][-1].content)
    assert len(usage.llm_calls) == 1
    assert usage.llm_calls[0].task_id is None


@tracked_tool("lookup_reference")
async def _lookup(title: str) -> str:
    return f"Found: {title}"


@tool
async def lookup_reference(title: str) -> str:
    """Look up a cited work by its title."""
    return await _lookup(title)


async def test_b1_uses_tools_then_answers() -> None:
    model = ScriptedChatModel(
        replies=[
            tool_reply("lookup_reference", {"title": "SQuAD"}, input_tokens=100),
            tool_reply(ReportedErrors, {"errors": [error()]}, input_tokens=150),
        ]
    )
    usage = UsageLog()
    result = await run_b1(
        example_paper(),
        model,
        [lookup_reference],
        run_id="r1",
        model_name="test:scripted",
        usage=usage,
    )
    assert [p.system for p in result.predictions] == [SystemName.B1]
    assert result.dropped == []
    assert [r.tool for r in usage.tool_calls] == ["lookup_reference"]
    assert [r.input_tokens for r in usage.llm_calls] == [100, 150]
    tool_result = model.received[1][-1]
    assert "Found: SQuAD" in str(tool_result.content)


async def test_b1_stops_at_its_model_call_limit() -> None:
    model = ScriptedChatModel(
        replies=[tool_reply("lookup_reference", {"title": f"t{i}"}) for i in range(5)]
    )
    result = await run_b1(
        example_paper(),
        model,
        [lookup_reference],
        run_id="r1",
        model_name="test:scripted",
        max_model_calls=2,
    )
    assert result.predictions == []
    assert result.dropped == ["the agent stopped without a final answer"]
    assert len(model.received) == 2
