"""Verifier decisions on hand-built findings: true errors, false accusations, uncertain
cases (team plan P4.3)."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from auditor.agents.verifier import verify
from auditor.agents.verifier.grounding import PaperText, ground, normalise
from auditor.infra.context import RunContext
from auditor.infra.llm import StructuredLLM
from auditor.infra.snapshots import Snapshot, SnapshotFile, SnapshotManifest
from auditor.infra.snapshots.sources import SourceKind as HostKind
from auditor.schemas import (
    AuditTask,
    BibliographyAnchor,
    CandidateFinding,
    DocumentAnchor,
    ErrorClass,
    Evidence,
    EvidenceKind,
    PaperVariant,
    RuleTarget,
    Sentence,
    SentenceAnchor,
    SourceKind,
    SourceRef,
    TaskKind,
    TaskOutcome,
    TaskResult,
    VerificationState,
)
from auditor.schemas.examples import example_paper
from auditor.testing import ScriptedChatModel, offline_context, tool_reply

PAPER_ID = example_paper().paper_id
REPO_ID = "https://github.com/example/auditor-demo@abc"
S4 = "We train with a learning rate of 3e-4 and a batch size of 256."


def task() -> AuditTask:
    # The verifier does not look at the task; the finding carries its own anchor.
    return AuditTask.create(
        paper_id=PAPER_ID,
        variant=PaperVariant.ORIGINAL,
        kind=TaskKind.INTEGRITY,
        anchor=DocumentAnchor(),
        target=RuleTarget(rule_id="results"),
    )


def paper_quote(text: str) -> Evidence:
    return Evidence(
        kind=EvidenceKind.QUOTE,
        source=SourceRef(kind=SourceKind.SUBMISSION, id=PAPER_ID),
        text=text,
    )


def code(text: str, line: int = 2, path: str = "train.py") -> Evidence:
    return Evidence(
        kind=EvidenceKind.CODE,
        source=SourceRef(kind=SourceKind.REPOSITORY, id=REPO_ID),
        text=text,
        path=path,
        line_start=line,
    )


def cited(text: str, *, located: bool) -> Evidence:
    return Evidence(
        kind=EvidenceKind.QUOTE,
        source=SourceRef(kind=SourceKind.CITED_WORK, id="doi:10.1/x"),
        text=text,
        char_start=10 if located else None,
        char_end=10 + len(text) if located else None,
    )


def lookup(kind: SourceKind = SourceKind.SCHOLARLY_INDEX) -> Evidence:
    return Evidence(
        kind=EvidenceKind.LOOKUP, source=SourceRef(kind=kind, id="openalex"), text="no match"
    )


def finding(
    error_class: ErrorClass,
    *evidence: Evidence,
    sentence_id: str = "s4",
    uncertain: bool = False,
    anchor: SentenceAnchor | BibliographyAnchor | None = None,
    number: int = 0,
) -> CandidateFinding:
    return CandidateFinding.for_task(
        task(),
        number,
        error_class=error_class,
        summary="something is wrong",
        anchor=anchor or SentenceAnchor(sentence_id=sentence_id),
        evidence=evidence,
        uncertain=uncertain,
    )


def result(item: CandidateFinding, outcome: TaskOutcome = TaskOutcome.CHECKED) -> TaskResult:
    reason = "the source was not available" if outcome is TaskOutcome.UNVERIFIABLE else None
    return TaskResult(
        task_id=item.task_id,
        paper_id=item.paper_id,
        variant=item.variant,
        kind=TaskKind.INTEGRITY,
        outcome=outcome,
        reason=reason,
        findings=(item,),
    )


def snapshot(root: Path) -> Snapshot:
    files = root / "files"
    files.mkdir()
    (files / "train.py").write_text("import torch\nlr = 1e-3\nbatch_size = 256\n", "utf-8")
    manifest = SnapshotManifest(
        source=HostKind.GITHUB,
        url="https://github.com/example/auditor-demo",
        identity="example/auditor-demo",
        requested_ref=None,
        revision="abc",
        fetched_at=datetime.now(UTC),
        limits={},
        files=(SnapshotFile(path="train.py", size=40, sha256="0" * 64),),
        skipped=(),
        skipped_count=0,
        incomplete=(),
        metadata={},
    )
    return Snapshot(files, manifest)


@pytest.fixture
def ctx(tmp_path: Path) -> RunContext:
    return offline_context(tmp_path)


async def decide_one(
    ctx: RunContext,
    item: CandidateFinding,
    outcome: TaskOutcome = TaskOutcome.CHECKED,
    snap: Snapshot | None = None,
) -> tuple[VerificationState, str]:
    from auditor.agents.verifier.decide import decide

    snapshots = {REPO_ID: snap} if snap is not None else {}
    decision = decide(item, outcome, ctx.paper, PaperText(ctx.paper), snapshots)
    return decision.state, decision.reason


# Grounding


def test_normalisation_joins_ligatures_hyphens_and_dashes() -> None:
    ligature, en_dash, left, right = chr(0xFB03), chr(0x2013), chr(0x201C), chr(0x201D)
    text = f"e{ligature}cient  multi-\n task {en_dash} {left}best{right}"
    assert normalise(text) == 'efficient multitask - "best"'


def test_a_quote_across_two_sentences_is_found(ctx: RunContext) -> None:
    text = PaperText(ctx.paper)
    quote = "most sequence tasks. Prior work reports 92.1%"
    check = ground(paper_quote(quote), text, {})
    assert check.sentence_ids == ("s1", "s2")


def test_a_hyphenated_line_break_in_the_paper_still_matches(tmp_path: Path) -> None:
    paper = example_paper()
    broken = Sentence(id="s4", section_id="sec2", text="We train with a learn- ing rate of 3e-4.")
    paper = paper.model_copy(
        update={"sentences": tuple(broken if s.id == "s4" else s for s in paper.sentences)}
    )
    check = ground(paper_quote("with a learning rate of 3e-4"), PaperText(paper), {})
    assert check.sentence_ids == ("s4",)


def test_a_short_span_given_by_offsets_is_checked_exactly(ctx: RunContext) -> None:
    text = PaperText(ctx.paper)
    span = Evidence(
        kind=EvidenceKind.QUOTE,
        source=SourceRef(kind=SourceKind.SUBMISSION, id=PAPER_ID),
        text="256",
        char_start=S4.index("256"),
        char_end=S4.index("256") + 3,
    )
    assert ground(span, text, {}, "s4").sentence_ids == ("s4",)
    assert ground(span, text, {}, "s3").grounding.value == "not_found"


# Decisions


async def test_a_grounded_setting_mismatch_is_verified(ctx: RunContext, tmp_path: Path) -> None:
    item = finding(ErrorClass.SETTING_MISMATCH, paper_quote(S4), code("lr = 1e-3"))
    state, _ = await decide_one(ctx, item, snap=snapshot(tmp_path))
    assert state is VerificationState.VERIFIED_FLAG


async def test_code_that_is_not_at_its_line_is_dropped(ctx: RunContext, tmp_path: Path) -> None:
    item = finding(ErrorClass.SETTING_MISMATCH, paper_quote(S4), code("lr = 1e-3", line=3))
    state, reason = await decide_one(ctx, item, snap=snapshot(tmp_path))
    assert state is VerificationState.DROPPED
    assert "train.py:3" in reason


async def test_a_quote_the_paper_does_not_contain_is_dropped(
    ctx: RunContext, tmp_path: Path
) -> None:
    made_up = paper_quote("We train with a learning rate of 1e-3 for 90 epochs.")
    item = finding(ErrorClass.SETTING_MISMATCH, made_up, code("lr = 1e-3"))
    state, reason = await decide_one(ctx, item, snap=snapshot(tmp_path))
    assert state is VerificationState.DROPPED
    assert "not in the paper" in reason


async def test_without_the_snapshot_the_code_side_goes_to_review(ctx: RunContext) -> None:
    item = finding(ErrorClass.SETTING_MISMATCH, paper_quote(S4), code("lr = 1e-3"))
    state, reason = await decide_one(ctx, item)
    assert state is VerificationState.NEEDS_HUMAN_REVIEW
    assert "snapshot" in reason


async def test_a_mismatch_with_only_the_paper_side_goes_to_review(ctx: RunContext) -> None:
    state, reason = await decide_one(ctx, finding(ErrorClass.METRIC_MISMATCH, paper_quote(S4)))
    assert state is VerificationState.NEEDS_HUMAN_REVIEW
    assert "repository" in reason


async def test_a_cited_quote_the_code_located_verifies_the_claim(ctx: RunContext) -> None:
    item = finding(
        ErrorClass.CLAIM_NOT_SUPPORTED,
        lookup(),
        cited("F1 of 51.0%", located=True),
        sentence_id="s2",
    )
    state, _ = await decide_one(ctx, item)
    assert state is VerificationState.VERIFIED_FLAG


async def test_a_cited_quote_that_was_not_located_is_dropped(ctx: RunContext) -> None:
    item = finding(
        ErrorClass.CLAIM_NOT_SUPPORTED,
        lookup(),
        cited("F1 of 51.0%", located=False),
        sentence_id="s2",
    )
    state, _ = await decide_one(ctx, item)
    assert state is VerificationState.DROPPED


async def test_a_number_the_cited_work_never_states_goes_to_review(ctx: RunContext) -> None:
    # Only the lookup: the finding rests on the number not being found.
    item = finding(ErrorClass.CLAIM_NUMBER_MISMATCH, lookup(), sentence_id="s2")
    state, _ = await decide_one(ctx, item)
    assert state is VerificationState.NEEDS_HUMAN_REVIEW


async def test_a_reference_no_index_lists_goes_to_review(ctx: RunContext) -> None:
    item = finding(ErrorClass.REFERENCE_NOT_FOUND, lookup(), anchor=BibliographyAnchor(bib_id="b2"))
    state, reason = await decide_one(ctx, item)
    assert state is VerificationState.NEEDS_HUMAN_REVIEW
    assert "not being found" in reason


async def test_a_metadata_mismatch_from_the_index_is_verified(ctx: RunContext) -> None:
    item = finding(
        ErrorClass.REFERENCE_METADATA_MISMATCH, lookup(), anchor=BibliographyAnchor(bib_id="b2")
    )
    state, _ = await decide_one(ctx, item)
    assert state is VerificationState.VERIFIED_FLAG


async def test_a_bibliography_entry_the_paper_lacks_goes_to_review(ctx: RunContext) -> None:
    item = finding(
        ErrorClass.REFERENCE_METADATA_MISMATCH, lookup(), anchor=BibliographyAnchor(bib_id="b9")
    )
    state, _ = await decide_one(ctx, item)
    assert state is VerificationState.NEEDS_HUMAN_REVIEW


async def test_a_template_finding_that_names_its_rule_is_verified(ctx: RunContext) -> None:
    rule = Evidence(
        kind=EvidenceKind.RULE,
        source=SourceRef(kind=SourceKind.VENUE_RULE, id="max_pages"),
        text="Rule max_pages: at most 9 pages. Measured: 11.",
    )
    state, _ = await decide_one(ctx, finding(ErrorClass.TEMPLATE_VIOLATION, rule))
    assert state is VerificationState.VERIFIED_FLAG


async def test_a_paper_internal_error_needs_a_quote(ctx: RunContext) -> None:
    quoted = finding(ErrorClass.RESULT_INCONSISTENCY, paper_quote("a batch size of 256"))
    assert (await decide_one(ctx, quoted))[0] is VerificationState.VERIFIED_FLAG
    bare = finding(ErrorClass.MATH_ERROR, lookup(SourceKind.SUBMISSION))
    assert (await decide_one(ctx, bare))[0] is VerificationState.NEEDS_HUMAN_REVIEW


async def test_a_quote_from_another_sentence_than_the_flagged_one_goes_to_review(
    ctx: RunContext,
) -> None:
    item = finding(ErrorClass.MATH_ERROR, paper_quote("Retrieval also improves factual accuracy"))
    state, reason = await decide_one(ctx, item)
    assert state is VerificationState.NEEDS_HUMAN_REVIEW
    assert "another place" in reason


async def test_uncertain_and_unfinished_findings_go_to_review(ctx: RunContext) -> None:
    quote = paper_quote("a batch size of 256")
    uncertain = finding(ErrorClass.RESULT_INCONSISTENCY, quote, uncertain=True)
    assert (await decide_one(ctx, uncertain))[0] is VerificationState.NEEDS_HUMAN_REVIEW
    item = finding(ErrorClass.RESULT_INCONSISTENCY, quote)
    state, _ = await decide_one(ctx, item, TaskOutcome.UNVERIFIABLE)
    assert state is VerificationState.NEEDS_HUMAN_REVIEW


async def test_a_finding_without_evidence(ctx: RunContext) -> None:
    item = finding(ErrorClass.METHOD_MISMATCH)
    assert (await decide_one(ctx, item))[0] is VerificationState.DROPPED
    state, _ = await decide_one(ctx, item, TaskOutcome.UNVERIFIABLE)
    assert state is VerificationState.NEEDS_HUMAN_REVIEW


async def test_verify_decides_every_finding_in_order(ctx: RunContext) -> None:
    first = finding(ErrorClass.RESULT_INCONSISTENCY, paper_quote("a batch size of 256"))
    second = finding(ErrorClass.METHOD_MISMATCH, sentence_id="s3", number=1)
    decisions = await verify([result(first), result(second)], ctx)
    assert [d.finding.finding_id for d in decisions] == [first.finding_id, second.finding_id]
    assert [d.state for d in decisions] == [
        VerificationState.VERIFIED_FLAG,
        VerificationState.DROPPED,
    ]
    assert decisions[0].accepted_evidence == first.evidence
    assert decisions[1].accepted_evidence == ()


# The optional second reading


def claim(sentence_id: str, number: int = 0) -> CandidateFinding:
    return finding(
        ErrorClass.CLAIM_NOT_SUPPORTED,
        lookup(),
        cited("F1 of 51.0%", located=True),
        sentence_id=sentence_id,
        number=number,
    )


def recheck_ctx(ctx: RunContext, *verdicts: dict[str, object]) -> RunContext:
    from auditor.agents.verifier.recheck import Rechecks

    model = ScriptedChatModel(replies=[tool_reply(Rechecks, {"verdicts": list(verdicts)})])
    settings = ctx.settings.model_copy(update={"verifier_recheck": True})
    return replace(ctx, settings=settings, llm=StructuredLLM(model, name="test:r"))


async def test_the_second_reading_drops_and_escalates(ctx: RunContext) -> None:
    items = [claim("s1"), claim("s2", 1), claim("s3", 2)]
    checked = recheck_ctx(
        ctx,
        {"number": 1, "verdict": "shown", "reason": "ok"},
        {"number": 2, "verdict": "not_shown", "reason": "the passage agrees"},
    )
    decisions = await verify([result(i) for i in items], checked)
    assert [d.state for d in decisions] == [
        VerificationState.VERIFIED_FLAG,
        VerificationState.DROPPED,
        # No verdict for the third: not confirmed.
        VerificationState.NEEDS_HUMAN_REVIEW,
    ]
    assert "the passage agrees" in decisions[1].reason


async def test_without_the_setting_no_model_is_called(ctx: RunContext) -> None:
    model = ScriptedChatModel(replies=[])
    quiet = replace(ctx, llm=StructuredLLM(model, name="test:r"))
    decisions = await verify([result(claim("s1"))], quiet)
    assert decisions[0].state is VerificationState.VERIFIED_FLAG


def test_every_error_class_has_a_rule() -> None:
    from auditor.agents.verifier.policy import NEEDS

    assert set(NEEDS) == set(ErrorClass)
