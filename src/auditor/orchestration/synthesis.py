"""The step after the verifier: merge duplicate findings and rank them (decision D20).

Specialists check one task at a time, so the same problem can come back several times: a
paper that states its metric in two sentences gets two metric mismatches with the same
repository evidence. Here such findings are grouped, the best one of each group stays, and
the others point at it with `duplicate_of`. Nothing is deleted, so scoring (decision D3)
is unchanged; the report and the review load count each group once.

Every finding also gets a severity, and the list is ordered for a reader: verified flags
first, then findings for human review, then dropped ones; within each, by severity, with
certain findings before uncertain ones.
"""

from collections.abc import Sequence

from auditor.schemas import (
    CandidateFinding,
    ErrorClass,
    RuleTarget,
    Severity,
    SourceKind,
    VerificationState,
    VerifiedFinding,
)

SEVERITY = {
    ErrorClass.REFERENCE_NOT_FOUND: Severity.HIGH,
    ErrorClass.REFERENCE_METADATA_MISMATCH: Severity.LOW,
    ErrorClass.CLAIM_NOT_SUPPORTED: Severity.HIGH,
    ErrorClass.CLAIM_NUMBER_MISMATCH: Severity.HIGH,
    ErrorClass.SETTING_MISMATCH: Severity.MEDIUM,
    ErrorClass.DATA_SPLIT_MISMATCH: Severity.MEDIUM,
    ErrorClass.METRIC_MISMATCH: Severity.HIGH,
    ErrorClass.DATASET_SIZE_MISMATCH: Severity.MEDIUM,
    ErrorClass.LABEL_DISTRIBUTION_MISMATCH: Severity.MEDIUM,
    ErrorClass.DATA_LEAKAGE: Severity.HIGH,
    ErrorClass.METHOD_MISMATCH: Severity.HIGH,
    ErrorClass.MATH_ERROR: Severity.HIGH,
    ErrorClass.TEMPLATE_VIOLATION: Severity.LOW,
}
# Template rules whose violation gets a submission rejected without review.
DESK_REJECT_RULES = frozenset({"max_pages", "anonymity"})

_STATE_RANK = {
    VerificationState.VERIFIED_FLAG: 0,
    VerificationState.NEEDS_HUMAN_REVIEW: 1,
    VerificationState.DROPPED: 2,
}
_SEVERITY_RANK = {Severity.HIGH: 0, Severity.MEDIUM: 1, Severity.LOW: 2}


def severity_of(finding: CandidateFinding) -> Severity:
    if (
        finding.error_class is ErrorClass.TEMPLATE_VIOLATION
        and isinstance(finding.target, RuleTarget)
        and finding.target.rule_id in DESK_REJECT_RULES
    ):
        return Severity.HIGH
    return SEVERITY[finding.error_class]


def _same_problem(finding: CandidateFinding) -> tuple[object, ...] | None:
    """What two findings must share to report the same problem, or None for a finding
    that is never merged. The quotes from the paper may differ (the claim is stated in two places);
    the claim (summary) and every other piece of evidence must be the same. A finding with
    no evidence beyond the paper's own text is never merged, since nothing shows that two
    such findings are about the same thing."""
    outside = tuple(
        sorted(
            (e.kind.value, e.source.id, e.path or "", e.line_start or 0, e.line_end or 0, e.text)
            for e in finding.evidence
            if e.source.kind is not SourceKind.SUBMISSION
        )
    )
    if not outside:
        return None
    return (
        finding.paper_id,
        finding.variant,
        finding.error_class,
        finding.target.model_dump_json(),
        finding.summary,
        outside,
    )


def synthesise(verified: Sequence[VerifiedFinding]) -> list[VerifiedFinding]:
    rated = [d.model_copy(update={"severity": severity_of(d.finding)}) for d in verified]

    def rank(item: tuple[int, VerifiedFinding]) -> tuple[int, int, bool, int]:
        index, decision = item
        assert decision.severity is not None
        return (
            _STATE_RANK[decision.state],
            _SEVERITY_RANK[decision.severity],
            decision.finding.uncertain,
            index,
        )

    ordered = [d for _, d in sorted(enumerate(rated), key=rank)]
    kept: dict[tuple[object, ...], str] = {}
    out: list[VerifiedFinding] = []
    for decision in ordered:
        key = _same_problem(decision.finding)
        if key is None or decision.state is VerificationState.DROPPED:
            out.append(decision)
        elif key in kept:
            out.append(decision.model_copy(update={"duplicate_of": kept[key]}))
        else:
            kept[key] = decision.finding.finding_id
            out.append(decision)
    return out
