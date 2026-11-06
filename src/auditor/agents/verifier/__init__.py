"""The verifier (part 4): every candidate finding is confirmed, dropped or sent to a human,
with a reason (decisions D6 and D19, team plan P4.1 and P4.2).

Each piece of evidence is first found again in the source it names (`grounding`): a quote
in the paper's text or the cited work, code at its file and lines in the repository
snapshot. Then the class's rule (`policy`) decides whether both sides of the reported
disagreement are shown. With AUDITOR_VERIFIER_RECHECK=true, a second model reads the flags
that rest on a model's judgement (`recheck`).
"""

import logging
from collections.abc import Sequence

from auditor.agents.verifier.decide import decide
from auditor.agents.verifier.grounding import PaperText
from auditor.agents.verifier.recheck import recheck
from auditor.infra.context import RunContext
from auditor.infra.snapshots import (
    Snapshot,
    SnapshotUnavailableError,
    SourceError,
    fetch_snapshot,
)
from auditor.schemas import EvidenceKind, SourceKind, TaskResult, VerifiedFinding

logger = logging.getLogger(__name__)


def _uses_repository(results: Sequence[TaskResult]) -> bool:
    return any(
        e.source.kind is SourceKind.REPOSITORY and (e.kind is EvidenceKind.CODE or e.path)
        for r in results
        for f in r.findings
        for e in f.evidence
    )


async def _snapshots(results: Sequence[TaskResult], ctx: RunContext) -> dict[str, Snapshot]:
    """The paper's repository snapshot, by source id, when some evidence needs it. Evidence
    from another snapshot (a linked dataset) cannot be re-read and stays unchecked."""
    if ctx.repository is None or not _uses_repository(results):
        return {}
    try:
        snapshot = await fetch_snapshot(ctx.repository, ctx.http, ctx.settings)
    except (SnapshotUnavailableError, SourceError) as exc:
        logger.warning("the verifier could not load the repository: %s", exc)
        return {}
    return {snapshot.source_id: snapshot}


async def verify(results: Sequence[TaskResult], ctx: RunContext) -> list[VerifiedFinding]:
    text = PaperText(ctx.paper)
    snapshots = await _snapshots(results, ctx)
    verified = [
        decide(finding, result.outcome, ctx.paper, text, snapshots)
        for result in results
        for finding in result.findings
    ]
    if ctx.settings.verifier_recheck:
        verified = await recheck(verified, ctx)
    return verified


__all__ = ["verify"]
