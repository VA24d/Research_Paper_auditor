"""The decision on one candidate finding, from its grounded evidence (team plan P4.2)."""

from collections.abc import Mapping, Sequence

from auditor.agents.verifier.grounding import Check, Grounding, PaperText, ground
from auditor.agents.verifier.policy import NEEDS, PaperSide
from auditor.infra.snapshots.snapshot import Snapshot
from auditor.schemas import (
    BibliographyAnchor,
    CandidateFinding,
    EvidenceKind,
    ParsedPaper,
    SentenceAnchor,
    SourceKind,
    TaskOutcome,
    VerificationState,
    VerifiedFinding,
)

_SHOWN = (Grounding.FOUND, Grounding.RECORD)


def _anchor_exists(finding: CandidateFinding, paper: ParsedPaper, text: PaperText) -> bool:
    match finding.anchor:
        case SentenceAnchor(sentence_id=sentence_id):
            return sentence_id in text.ids
        case BibliographyAnchor(bib_id=bib_id):
            return any(entry.id == bib_id for entry in paper.bibliography)
        case _:
            return True


def _notes(checks: Sequence[Check]) -> str:
    return "; ".join(dict.fromkeys(c.note for c in checks if c.note))


def _side(checks: Sequence[Check], *, records: bool) -> Grounding | None:
    """How well one side is shown: FOUND (or RECORD, when records count), else NOT_FOUND
    when all its evidence failed, UNCHECKED when its source was not available, and None
    when no evidence was given for it."""
    if not checks:
        return None
    for check in checks:
        if check.grounding is Grounding.FOUND or (records and check.grounding is Grounding.RECORD):
            return Grounding.FOUND
    if any(c.grounding is Grounding.UNCHECKED for c in checks):
        return Grounding.UNCHECKED
    if all(c.grounding is Grounding.NOT_FOUND for c in checks):
        return Grounding.NOT_FOUND
    return None


def decide(
    finding: CandidateFinding,
    outcome: TaskOutcome,
    paper: ParsedPaper,
    text: PaperText,
    snapshots: Mapping[str, Snapshot],
) -> VerifiedFinding:
    anchor = finding.anchor
    sentence_id = anchor.sentence_id if isinstance(anchor, SentenceAnchor) else None
    checks = [ground(e, text, snapshots, sentence_id) for e in finding.evidence]
    accepted = tuple(c.evidence for c in checks if c.grounding in _SHOWN)
    finished = outcome is TaskOutcome.CHECKED

    def decision(state: VerificationState, reason: str) -> VerifiedFinding:
        kept = accepted if state is not VerificationState.DROPPED else ()
        if state is VerificationState.VERIFIED_FLAG and not kept:
            state, reason = VerificationState.NEEDS_HUMAN_REVIEW, "No evidence to show."
        return VerifiedFinding(finding=finding, state=state, reason=reason, accepted_evidence=kept)

    if not checks:
        if finished:
            return decision(VerificationState.DROPPED, "The finding carries no evidence.")
        return decision(
            VerificationState.NEEDS_HUMAN_REVIEW,
            f"A suspicion from a check that did not finish ({outcome.value}), with no evidence.",
        )

    need = NEEDS[finding.error_class]
    quotes = [c for c in checks if c.evidence.source.kind is SourceKind.SUBMISSION]
    paper_quotes = [c for c in quotes if c.evidence.kind is EvidenceKind.QUOTE]
    others = [c for c in checks if c.evidence.source.kind in need.other]

    # 1. Evidence that is not in its source, when it was a side's only support.
    paper_side = _side(paper_quotes, records=False)
    other_side = _side(others, records=need.records) if need.other else Grounding.FOUND
    failed = [c for c in checks if c.grounding is Grounding.NOT_FOUND]
    if paper_side is Grounding.NOT_FOUND and need.paper is not PaperSide.NONE:
        return decision(VerificationState.DROPPED, f"Not in the paper: {_notes(failed)}.")
    if other_side is Grounding.NOT_FOUND:
        return decision(VerificationState.DROPPED, f"Not in its source: {_notes(failed)}.")

    # 3. A side that is not shown.
    missing: list[str] = []
    if need.paper is PaperSide.QUOTE and paper_side is not Grounding.FOUND:
        missing.append("no quote of the paper")
    if (
        need.paper is PaperSide.ANCHOR
        and paper_side is None
        and not _anchor_exists(finding, paper, text)
    ):
        missing.append("the flagged place is not in the paper")
    if other_side is None:
        missing.append(f"no evidence from the {' or '.join(sorted(need.other))}")
    if other_side is Grounding.UNCHECKED or paper_side is Grounding.UNCHECKED:
        missing.append(_notes(checks) or "a source could not be checked")
    if missing:
        return decision(VerificationState.NEEDS_HUMAN_REVIEW, f"Not settled: {'; '.join(missing)}.")

    # 4. and 5. Grounded, but not something to report as settled.
    review: list[str] = []
    if need.absence:
        review.append("it rests on a record not being found, which needs a human to confirm")
    if finding.uncertain:
        review.append("the specialist marked it uncertain")
    if not finished:
        review.append(f"its check did not finish ({outcome.value})")
    if failed:
        review.append(f"some evidence is not in its source ({_notes(failed)})")
    if isinstance(finding.anchor, SentenceAnchor) and paper_side is Grounding.FOUND:
        spanned = {sid for c in paper_quotes for sid in c.sentence_ids}
        if finding.anchor.sentence_id not in spanned:
            review.append("its quotes are from another place than the flagged sentence")
    if review:
        return decision(
            VerificationState.NEEDS_HUMAN_REVIEW,
            f"The evidence is in its sources, but {'; '.join(review)}.",
        )
    return decision(
        VerificationState.VERIFIED_FLAG,
        "Every side of the finding is shown by evidence found again in its source.",
    )
