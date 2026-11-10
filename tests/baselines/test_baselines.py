from typing import Any

import pytest
from langchain_core.tools import tool

from auditor.baselines.b0 import run_b0
from auditor.baselines.b1 import run_b1
from auditor.baselines.common import (
    IncompleteAnswerError,
    ReportedErrors,
    render_paper,
    to_predictions,
)
from auditor.infra.llm import CallLimitReachedError, Spend, StructuredLLM
from auditor.infra.tools import tracked_tool
from auditor.infra.usage import UsageLog
from auditor.schemas import (
    BibliographyAnchor,
    BibTarget,
    DocumentAnchor,
    ErrorClass,
    MarkerTarget,
    PageAnchor,
    PaperVariant,
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


def test_render_paper_shows_every_author_and_identifier() -> None:
    paper = example_paper()
    entry = paper.bibliography[0]
    changed = entry.model_copy(
        update={"authors": (*entry.authors, "A Third", "B Fourth"), "doi": "10.1/x"}
    )
    paper = paper.model_copy(update={"bibliography": (changed, *paper.bibliography[1:])})
    text = render_paper(paper)
    assert "Noam Shazeer, A Third, B Fourth. Attention Is All You Need" in text
    assert "DOI 10.1/x" in text


def test_render_paper_shows_layout_measurements_and_rule_parameters() -> None:
    paper = example_paper()
    assert paper.layout is not None
    page = paper.layout.pages[1].model_copy(update={"text_left": 1.0})
    font = paper.layout.fonts[0].model_copy(update={"size": 4.0})
    layout = paper.layout.model_copy(
        update={"pages": (paper.layout.pages[0], page), "fonts": (font, *paper.layout.fonts[1:])}
    )
    text = render_paper(paper.model_copy(update={"layout": layout}), venue=example_venue())
    assert "left 1.0 (p. 2)" in text
    assert "4 pt" in text
    assert "(check: max_pages; max_pages=9)" in text


def test_the_answer_guide_gives_the_anchors_and_targets_the_specialists_use() -> None:
    # The full system's checks and the gold labels use these; baselines that were not told
    # them could not match a method or metric label however right they were.
    from auditor.agents.code.data_checks import ERROR_CLASSES
    from auditor.baselines.common import ANSWER_GUIDE
    from auditor.planner.integrity import RESULTS_RULE_ID, RULE_ID
    from auditor.planner.tasks import DEFECTS_KEY, METHOD_KEY
    from auditor.planner.theory import THEORY_KEY

    keys = {METHOD_KEY, DEFECTS_KEY, THEORY_KEY, RULE_ID, RESULTS_RULE_ID, "eval_metric"}
    for key in keys | set(ERROR_CLASSES):
        assert f'"{key}"' in ANSWER_GUIDE, key
    for error_class in set(ERROR_CLASSES.values()):
        assert f"- {error_class.value}:" in ANSWER_GUIDE, error_class

    from auditor.schemas import Evidence, EvidenceKind, GoldLabel, SourceKind, SourceRef
    from auditor.schemas.evaluation import matches

    label = GoldLabel(
        label_id="m1",
        paper_id=example_paper().paper_id,
        variant=PaperVariant.ORIGINAL,
        error_class=ErrorClass.METHOD_MISMATCH,
        anchor=DocumentAnchor(),
        target=SettingTarget(key=METHOD_KEY),
        rationale="the loss differs",
        evidence=(
            Evidence(
                kind=EvidenceKind.QUOTE,
                source=SourceRef(kind=SourceKind.SUBMISSION, id="p"),
                text="We use a focal loss.",
            ),
        ),
    )
    answer = error(
        error_class="method_mismatch",
        anchor_kind="document",
        anchor_id=None,
        target_kind="setting",
        target="method",
        summary="The loss differs: the paper specifies a focal loss.",
    )
    (prediction,), _ = to_predictions(
        ReportedErrors.model_validate({"errors": [answer]}),
        example_paper(),
        run_id="r1",
        system=SystemName.B0,
    )
    assert matches(prediction, label)


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


@pytest.mark.parametrize("reply", [{}, {"error": "Unable to inspect the paper"}])
async def test_b0_retries_a_reply_without_an_errors_list(reply: dict[str, Any]) -> None:
    model = ScriptedChatModel(
        replies=[tool_reply(ReportedErrors, reply), tool_reply(ReportedErrors, {"errors": []})]
    )
    result = await run_b0(example_paper(), StructuredLLM(model, name="test:scripted"), run_id="r1")
    assert result.predictions == []
    assert len(model.received) == 2


async def test_b1_retries_a_reply_without_an_errors_list() -> None:
    model = ScriptedChatModel(
        replies=[tool_reply(ReportedErrors, {}), tool_reply(ReportedErrors, {"errors": [error()]})]
    )
    result = await run_b1(
        example_paper(), StructuredLLM(model, name="test:scripted"), [lookup_reference], run_id="r1"
    )
    assert [p.error_class for p in result.predictions] == [ErrorClass.CLAIM_NUMBER_MISMATCH]
    assert len(model.received) == 2


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
    (record,) = usage.llm_calls
    assert (record.paper_id, record.variant, record.task_id) == (
        example_paper().paper_id,
        PaperVariant.ORIGINAL,
        None,
    )


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
        StructuredLLM(model, name="test:scripted"),
        [lookup_reference],
        run_id="r1",
        usage=usage,
    )
    assert [p.system for p in result.predictions] == [SystemName.B1]
    assert result.dropped == []
    assert [r.tool for r in usage.tool_calls] == ["lookup_reference"]
    assert [r.input_tokens for r in usage.llm_calls] == [100, 150]
    expected = (example_paper().paper_id, PaperVariant.ORIGINAL)
    assert {(r.paper_id, r.variant) for r in usage.tool_calls} == {expected}
    assert {(r.paper_id, r.variant) for r in usage.llm_calls} == {expected}
    tool_result = model.received[1][-1]
    assert "Found: SQuAD" in str(tool_result.content)


async def test_b1_stops_at_its_model_call_limit() -> None:
    model = ScriptedChatModel(
        replies=[tool_reply("lookup_reference", {"title": f"t{i}"}) for i in range(5)]
    )
    with pytest.raises(IncompleteAnswerError, match="without a final answer"):
        await run_b1(
            example_paper(),
            StructuredLLM(model, name="test:scripted"),
            [lookup_reference],
            run_id="r1",
            max_model_calls=2,
        )
    assert len(model.received) == 2


async def test_b1_answers_on_its_last_model_call() -> None:
    # Twenty turns used to pass the graph's step limit before the call limit was reached.
    replies = [tool_reply("lookup_reference", {"title": f"t{i}"}) for i in range(19)]
    model = ScriptedChatModel(replies=[*replies, tool_reply(ReportedErrors, {"errors": []})])
    result = await run_b1(
        example_paper(), StructuredLLM(model, name="test:scripted"), [lookup_reference], run_id="r1"
    )
    assert result.predictions == []
    assert len(model.received) == 20
    assert [t.name for t in model.bound_tools] == ["ReportedErrors"]


async def test_b1_calls_count_against_the_spend_limit() -> None:
    model = ScriptedChatModel(
        replies=[
            tool_reply("lookup_reference", {"title": "SQuAD"}, input_tokens=1_000_000),
            tool_reply(ReportedErrors, {"errors": []}, input_tokens=1_000_000),
        ]
    )
    spend = Spend(limit_usd=0.5, prices={"scripted": (1.0, 0.0)})
    llm = StructuredLLM(model, name="test:scripted", spend=spend)
    with pytest.raises(CallLimitReachedError):
        await run_b1(example_paper(), llm, [lookup_reference], run_id="r1")
    assert spend.spent_usd == pytest.approx(1.0)


def test_baselines_are_told_every_error_class() -> None:
    from auditor.baselines.common import ERROR_CLASS_GUIDE
    from auditor.schemas import ErrorClass

    for error_class in ErrorClass:
        assert f"- {error_class.value}:" in ERROR_CLASS_GUIDE


@tool
async def fetch_repository_file(path: str) -> str:
    """Read a file of the paper's repository."""
    raise RuntimeError("HTTP 403 for https://api.github.com/repos/o/r/contents/train.py")


async def test_b1_goes_on_after_a_tool_fails() -> None:
    # A rate-limited GitHub call used to end the whole paper's B1 run.
    model = ScriptedChatModel(
        replies=[
            tool_reply("fetch_repository_file", {"path": "train.py"}),
            tool_reply(ReportedErrors, {"errors": [error()]}),
        ]
    )
    result = await run_b1(
        example_paper(),
        StructuredLLM(model, name="test:scripted"),
        [fetch_repository_file],
        run_id="r1",
    )
    assert [p.error_class for p in result.predictions] == [ErrorClass.CLAIM_NUMBER_MISMATCH]
    told = str(model.received[1][-1].content)
    assert "HTTP 403" in told


@tool
async def over_budget_tool(path: str) -> str:
    """A tool whose call is past the run's budget."""
    from auditor.infra.scope import BudgetExceededError

    raise BudgetExceededError("tool calls used up")


async def test_b1_still_stops_when_a_tool_is_over_budget() -> None:
    from auditor.infra.scope import BudgetExceededError

    model = ScriptedChatModel(replies=[tool_reply("over_budget_tool", {"path": "x"})])
    with pytest.raises(BudgetExceededError):
        await run_b1(
            example_paper(),
            StructuredLLM(model, name="test:scripted"),
            [over_budget_tool],
            run_id="r1",
        )
