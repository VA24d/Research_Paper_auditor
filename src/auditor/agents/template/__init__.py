"""Template checker: deterministic venue formatting rules (P3.6). No LLM calls.

`check(task, ctx)` runs the check named by `task.rule.check` (see `rules.RULE_CHECKS`) on
`ctx.paper`. Every violation becomes a `template_violation` finding with rule evidence,
anchored on the pages involved when they are known and on the document otherwise
(decision D3). Unknown checks, rules that need visual inspection and layout rules without
layout measurements are `unverifiable` with a reason (decision D6).
"""

from auditor.agents.template.rules import RuleReport, Violation
from auditor.agents.template.shared import check_template_rule
from auditor.infra.context import RunContext
from auditor.schemas import (
    AuditTask,
    CandidateFinding,
    ErrorClass,
    Evidence,
    EvidenceKind,
    PageAnchor,
    SourceKind,
    SourceRef,
    TaskOutcome,
    TaskResult,
    VenueRule,
)

__all__ = ["check"]


async def check(task: AuditTask, ctx: RunContext) -> TaskResult:
    rule = task.rule
    if rule is None:
        return TaskResult.for_task(
            task, TaskOutcome.UNVERIFIABLE, reason="The task carries no venue rule."
        )
    report = await check_template_rule(ctx.paper, rule.check, rule.params)
    visual = rule.needs_visual_inspection
    findings = [
        _finding(task, rule, number, violation, report, uncertain=visual)
        for number, violation in enumerate(report.violations, start=1)
    ]
    if visual:
        reason = f"Rule '{rule.id}' needs visual inspection."
        if report.unverifiable is not None:
            reason += f" {report.unverifiable}"
        elif report.measured:
            reason += f" Measured automatically: {report.measured}."
        return TaskResult.for_task(task, TaskOutcome.UNVERIFIABLE, reason=reason, findings=findings)
    if report.unverifiable is not None:
        return TaskResult.for_task(
            task, TaskOutcome.UNVERIFIABLE, reason=report.unverifiable, findings=findings
        )
    return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=findings)


def _finding(
    task: AuditTask,
    rule: VenueRule,
    number: int,
    violation: Violation,
    report: RuleReport,
    *,
    uncertain: bool,
) -> CandidateFinding:
    params = ", ".join(f"{key}={value}" for key, value in rule.params.items())
    definition = f"{rule.check}{'; ' + params if params else ''}"
    evidence = Evidence(
        kind=EvidenceKind.RULE,
        source=SourceRef(kind=SourceKind.VENUE_RULE, id=rule.id, title=rule.description),
        text=f"Rule {rule.id} ({definition}): {rule.description} "
        f"Measured: {violation.measured or report.measured}.",
    )
    return CandidateFinding.for_task(
        task,
        number,
        error_class=ErrorClass.TEMPLATE_VIOLATION,
        summary=violation.summary,
        anchor=PageAnchor(pages=violation.pages) if violation.pages else None,
        explanation=violation.explanation,
        evidence=[evidence],
        uncertain=uncertain or violation.uncertain,
    )
