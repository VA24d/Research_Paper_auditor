from datetime import datetime

import pytest
from pydantic import ValidationError

from auditor.schemas import (
    NO_ERROR,
    BibTarget,
    CandidateFinding,
    ErrorClass,
    Evidence,
    EvidenceKind,
    GoldLabel,
    PaperVariant,
    Prediction,
    RunManifest,
    SentenceAnchor,
    SourceKind,
    SourceRef,
    SystemName,
    TaskOutcome,
    TaskResult,
    VerificationState,
    VerifiedFinding,
    matches,
)
from auditor.schemas.examples import (
    example_finding,
    example_gold_labels,
    example_predictions,
    example_task_results,
    example_tasks,
)

SOURCE = SourceRef(kind=SourceKind.REPOSITORY, id="https://github.com/x/y@abc")


def test_error_classes_match_decision_d4() -> None:
    assert {c.value for c in ErrorClass} == {
        "reference_not_found",
        "reference_metadata_mismatch",
        "claim_not_supported",
        "claim_number_mismatch",
        "setting_mismatch",
        "data_split_mismatch",
        "metric_mismatch",
        "dataset_size_mismatch",
        "label_distribution_mismatch",
        "data_leakage",
        "method_mismatch",
        "math_error",
        "unfinished_or_placeholder_content",
        "template_violation",
    }


def test_code_evidence_needs_path_and_line() -> None:
    with pytest.raises(ValidationError, match="path and line_start"):
        Evidence(kind=EvidenceKind.CODE, source=SOURCE, text="lr = 1e-3")


def test_line_end_before_line_start_is_rejected() -> None:
    with pytest.raises(ValidationError, match="line_end"):
        Evidence(
            kind=EvidenceKind.CODE,
            source=SOURCE,
            text="lr = 1e-3",
            path="train.py",
            line_start=10,
            line_end=9,
        )


def test_char_offsets_come_in_pairs() -> None:
    with pytest.raises(ValidationError, match="together"):
        Evidence(kind=EvidenceKind.QUOTE, source=SOURCE, text="x", char_start=3)


def result(**fields: object) -> TaskResult:
    task = example_tasks()[1]
    base: dict[str, object] = {
        "task_id": task.task_id,
        "paper_id": task.paper_id,
        "variant": task.variant,
        "kind": task.kind,
    }
    return TaskResult.model_validate({**base, **fields})


def test_unverifiable_result_needs_reason() -> None:
    with pytest.raises(ValidationError, match="needs a reason"):
        result(outcome=TaskOutcome.UNVERIFIABLE)


def test_failed_result_needs_error() -> None:
    with pytest.raises(ValidationError, match="needs the error"):
        result(outcome=TaskOutcome.FAILED)


def test_only_failed_results_carry_an_error() -> None:
    with pytest.raises(ValidationError, match="only failed"):
        result(outcome=TaskOutcome.CHECKED, error="boom")


def test_result_rejects_findings_from_another_task() -> None:
    other = example_finding().model_copy(update={"task_id": "citation-0000000000000000"})
    with pytest.raises(ValidationError, match="different task"):
        result(outcome=TaskOutcome.CHECKED, findings=(other,))


def test_result_rejects_duplicate_finding_ids() -> None:
    finding = example_finding()
    with pytest.raises(ValidationError, match="unique"):
        result(outcome=TaskOutcome.CHECKED, findings=(finding, finding))


def test_example_results_cover_every_outcome() -> None:
    assert {r.outcome for r in example_task_results()} == set(TaskOutcome)


def test_verified_flag_needs_accepted_evidence() -> None:
    with pytest.raises(ValidationError, match="accepted evidence"):
        VerifiedFinding(
            finding=example_finding(),
            state=VerificationState.VERIFIED_FLAG,
            reason="looks wrong",
        )


def test_review_and_dropped_states_need_no_evidence() -> None:
    for state in (VerificationState.NEEDS_HUMAN_REVIEW, VerificationState.DROPPED):
        VerifiedFinding(finding=example_finding(), state=state, reason="unclear")


def test_prediction_positivity() -> None:
    full, baseline = example_predictions()
    assert full.is_positive
    assert baseline.is_positive
    review = full.model_copy(update={"state": VerificationState.NEEDS_HUMAN_REVIEW})
    assert not review.is_positive


@pytest.mark.parametrize(
    ("system", "state"),
    [
        (SystemName.FULL, None),
        (SystemName.FULL_ABSTRACT_ONLY, None),
        (SystemName.B0, VerificationState.VERIFIED_FLAG),
        (SystemName.FULL_NO_VERIFIER, VerificationState.NEEDS_HUMAN_REVIEW),
    ],
)
def test_state_must_match_whether_the_system_has_a_verifier(
    system: SystemName, state: VerificationState | None
) -> None:
    data = example_predictions()[0].model_dump()
    data.update(system=system, state=state)
    with pytest.raises(ValidationError, match="state"):
        Prediction.model_validate(data)


def test_unverified_ablation_counts_every_flag() -> None:
    data = example_predictions()[0].model_dump()
    data.update(system=SystemName.FULL_NO_VERIFIER, state=None)
    assert Prediction.model_validate(data).is_positive


def test_prediction_matches_gold_label() -> None:
    full_prediction = example_predictions()[0]
    injected_label, clean_label = example_gold_labels()
    assert matches(full_prediction, injected_label)
    assert not matches(full_prediction, clean_label)


def test_matching_requires_same_variant() -> None:
    prediction = example_predictions()[0].model_copy(update={"variant": PaperVariant.INJECTED})
    assert not matches(prediction, example_gold_labels()[0])


def test_gold_label_needs_evidence() -> None:
    with pytest.raises(ValidationError):
        GoldLabel(
            label_id="l9",
            paper_id="p1",
            variant=PaperVariant.ORIGINAL,
            error_class=NO_ERROR,
            anchor=SentenceAnchor(sentence_id="s1"),
            target=BibTarget(bib_id="b0"),
            rationale="clean",
            evidence=(),
        )


def test_gold_label_accepts_none_class() -> None:
    clean = example_gold_labels()[1]
    assert clean.error_class == NO_ERROR
    assert not clean.is_error


def test_prediction_cannot_use_none_class() -> None:
    data = example_predictions()[1].model_dump()
    data["error_class"] = NO_ERROR
    with pytest.raises(ValidationError):
        Prediction.model_validate(data)


def test_run_manifest_needs_timezone() -> None:
    with pytest.raises(ValidationError):
        RunManifest(
            run_id="r1",
            system=SystemName.B0,
            started_at=datetime(2026, 10, 10, 12, 0),  # noqa: DTZ001
            python_version="3.13",
            platform="x",
        )


def test_results_and_findings_built_for_a_task_carry_its_ids() -> None:
    task = example_tasks()[1]
    finding = CandidateFinding.for_task(
        task, 2, error_class=ErrorClass.CLAIM_NOT_SUPPORTED, summary="Not in the cited work."
    )
    assert finding.finding_id == f"{task.task_id}-f2"
    assert (finding.anchor, finding.target) == (task.anchor, task.target)
    result = TaskResult.for_task(task, TaskOutcome.CHECKED, findings=[finding])
    assert (result.task_id, result.paper_id, result.variant, result.kind) == (
        task.task_id,
        task.paper_id,
        task.variant,
        task.kind,
    )
    with pytest.raises(ValidationError, match="needs a reason"):
        TaskResult.for_task(task, TaskOutcome.UNVERIFIABLE)
