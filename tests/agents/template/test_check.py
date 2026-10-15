from pathlib import Path

import pytest

from auditor.agents.template import check
from auditor.agents.template.fixtures import (
    TemplateCase,
    build_paper,
    judge,
    rule,
    template_cases,
)
from auditor.infra.scope import Scope, enter_scope
from auditor.infra.usage import UsageLog
from auditor.orchestration import registry
from auditor.planner.tasks import template_tasks
from auditor.schemas import (
    AuditTask,
    DocumentAnchor,
    ErrorClass,
    EvidenceKind,
    PageAnchor,
    ParsedPaper,
    SourceKind,
    TaskOutcome,
    VenueProfile,
    VenueRule,
)
from auditor.schemas.examples import example_paper, example_venue
from auditor.testing import offline_context

CASES = template_cases()


def task_for(paper: ParsedPaper, venue_rule: VenueRule) -> AuditTask:
    [task] = template_tasks(paper, VenueProfile(id="v", name="Venue", rules=(venue_rule,)))
    return task


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
async def test_fixture_cases(case: TemplateCase, tmp_path: Path) -> None:
    task = task_for(case.paper, case.rule)
    result = await check(task, offline_context(tmp_path, paper=case.paper))
    assert judge(case, result)
    match case.expected:
        case "pass":
            assert result.outcome is TaskOutcome.CHECKED
            assert not result.findings
        case "unverifiable":
            assert result.outcome is TaskOutcome.UNVERIFIABLE
            assert result.reason
        case "violation":
            assert result.outcome is TaskOutcome.CHECKED
            [finding] = result.findings
            assert finding.error_class is ErrorClass.TEMPLATE_VIOLATION
            assert finding.target == task.target
            expected_anchor = PageAnchor(pages=case.pages) if case.pages else DocumentAnchor()
            assert finding.anchor == expected_anchor
            case.paper.validate_anchor(finding.anchor)
            [evidence] = finding.evidence
            assert evidence.kind is EvidenceKind.RULE
            assert evidence.source.kind is SourceKind.VENUE_RULE
            assert evidence.source.id == case.rule.id
            assert case.rule.id in evidence.text
            assert "Measured:" in evidence.text


async def test_visual_rules_are_unverifiable_with_suspicions_kept(tmp_path: Path) -> None:
    paper = build_paper()
    visual = rule("max_pages", {"max_pages": 8}).model_copy(
        update={"needs_visual_inspection": True}
    )
    result = await check(task_for(paper, visual), offline_context(tmp_path, paper=paper))
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "visual inspection" in result.reason
    assert result.findings
    assert all(finding.uncertain for finding in result.findings)


async def test_unknown_checks_are_unverifiable(tmp_path: Path) -> None:
    paper = build_paper()
    result = await check(
        task_for(paper, rule("line_spacing")), offline_context(tmp_path, paper=paper)
    )
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "line_spacing" in result.reason


async def test_layout_rules_without_layout_are_unverifiable(tmp_path: Path) -> None:
    paper = build_paper(with_layout=False)
    task = task_for(paper, rule("margins", {"min_left": 108}))
    result = await check(task, offline_context(tmp_path, paper=paper))
    assert result.outcome is TaskOutcome.UNVERIFIABLE


async def test_example_venue_runs_on_the_example_paper(tmp_path: Path) -> None:
    paper = example_paper()
    ctx = offline_context(tmp_path, paper=paper)
    results = [await check(task, ctx) for task in template_tasks(paper, example_venue())]
    outcomes = {r.task_id: r.outcome for r in results}
    assert len(outcomes) == 3
    assert TaskOutcome.UNVERIFIABLE in outcomes.values()


async def test_each_rule_check_is_a_counted_tool_call(tmp_path: Path) -> None:
    paper = build_paper()
    usage = UsageLog()
    with enter_scope(Scope.for_paper("run", paper, usage)):
        await check(
            task_for(paper, rule("max_pages", {"max_pages": 9})),
            offline_context(tmp_path, paper=paper),
        )
    tools = [record.tool for record in usage.tool_calls]
    assert tools == ["check_template_rule"]


def test_the_pipeline_uses_the_real_template_checker() -> None:
    specialists, mocked = registry.load_specialists()
    assert "template" not in mocked
    assert specialists.template is check
    assert "template" not in registry.missing_tools()
