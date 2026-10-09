"""Stand-ins for modules that are not merged yet, so the pipeline runs end to end."""

from collections.abc import Sequence

from auditor.infra.context import RunContext
from auditor.schemas import (
    AuditTask,
    TaskOutcome,
    TaskResult,
    VerificationState,
    VerifiedFinding,
)


async def check_nothing(task: AuditTask, ctx: RunContext) -> TaskResult:
    """A specialist that runs and finds nothing."""
    return TaskResult(
        task_id=task.task_id,
        paper_id=task.paper_id,
        variant=task.variant,
        kind=task.kind,
        outcome=TaskOutcome.CHECKED,
    )


async def send_all_to_review(
    results: Sequence[TaskResult], ctx: RunContext
) -> list[VerifiedFinding]:
    """A verifier that settles nothing: every finding goes to a human."""
    return [
        VerifiedFinding(
            finding=finding,
            state=VerificationState.NEEDS_HUMAN_REVIEW,
            reason="The verifier is not connected yet.",
        )
        for result in results
        for finding in result.findings
    ]
