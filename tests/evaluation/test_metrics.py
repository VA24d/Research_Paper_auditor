"""D3 matching and the scores built on it (team plan P4.5)."""

from auditor.evaluation.metrics import AREAS, Truth, score, verifier_decisions
from auditor.schemas import (
    BibTarget,
    ErrorClass,
    Evidence,
    EvidenceKind,
    GoldLabel,
    PaperVariant,
    Prediction,
    SentenceAnchor,
    SettingTarget,
    SourceKind,
    SourceRef,
    SystemName,
    Target,
    VerificationState,
)

PAPER = "p1"
EVIDENCE = Evidence(
    kind=EvidenceKind.QUOTE, source=SourceRef(kind=SourceKind.SUBMISSION, id=PAPER), text="x"
)


def label(
    label_id: str,
    error_class: ErrorClass | str,
    sentence: str,
    target: Target,
    variant: PaperVariant = PaperVariant.INJECTED,
) -> GoldLabel:
    return GoldLabel(
        label_id=label_id,
        paper_id=PAPER,
        variant=variant,
        error_class=error_class,  # type: ignore[arg-type]
        anchor=SentenceAnchor(sentence_id=sentence),
        target=target,
        rationale="hand-made",
        evidence=(EVIDENCE,),
    )


def flag(
    error_class: ErrorClass,
    sentence: str,
    target: Target,
    *,
    system: SystemName = SystemName.FULL,
    state: VerificationState | None = VerificationState.VERIFIED_FLAG,
    variant: PaperVariant = PaperVariant.INJECTED,
    duplicate_of: str | None = None,
) -> Prediction:
    return Prediction(
        run_id="r",
        system=system,
        paper_id=PAPER,
        variant=variant,
        error_class=error_class,
        anchor=SentenceAnchor(sentence_id=sentence),
        target=target,
        state=state if system is not SystemName.B0 else None,
        summary="flag",
        duplicate_of=duplicate_of,
    )


LR = SettingTarget(key="learning_rate")
BS = SettingTarget(key="batch_size")
B1 = BibTarget(bib_id="b1")
LABELS = [
    label("l-lr", ErrorClass.SETTING_MISMATCH, "s4", LR),
    label("l-bs", ErrorClass.SETTING_MISMATCH, "s5", BS),
    label("l-cite", ErrorClass.CLAIM_NOT_SUPPORTED, "s2", B1),
    label("l-clean", "none", "s7", LR),
]


def test_true_positive_miss_clean_hit_and_unlabelled() -> None:
    scores = score(
        LABELS,
        [
            flag(ErrorClass.SETTING_MISMATCH, "s4", LR),  # true positive
            flag(ErrorClass.SETTING_MISMATCH, "s7", LR),  # clean control: false positive
            flag(ErrorClass.SETTING_MISMATCH, "s9", LR),  # no label: unlabelled
        ],
    )
    setting = scores.per_class[ErrorClass.SETTING_MISMATCH]
    assert (setting.true_positives, setting.false_positives, setting.false_negatives) == (1, 1, 1)
    assert scores.per_class[ErrorClass.CLAIM_NOT_SUPPORTED].false_negatives == 1
    assert scores.clean_flagged == 1
    assert scores.clean_controls == 1
    assert scores.false_positive_rate == 1.0
    assert scores.unlabelled == 1
    assert scores.outcomes == {
        "l-lr": "found",
        "l-clean": "flagged",
        "l-bs": "missed",
        "l-cite": "missed",
    }
    # Macro-F1 over the two labelled classes: setting F1 0.5, claim 0.
    assert scores.macro_f1 == 0.25


def test_wrong_class_is_a_false_positive_and_a_miss() -> None:
    scores = score(LABELS, [flag(ErrorClass.METRIC_MISMATCH, "s4", LR)])
    assert scores.per_class[ErrorClass.METRIC_MISMATCH].false_positives == 1
    assert scores.per_class[ErrorClass.SETTING_MISMATCH].false_negatives == 2
    # The metric class has no labels, so it does not count towards Macro-F1.
    assert not scores.per_class[ErrorClass.METRIC_MISMATCH].labelled


def test_only_positive_predictions_are_scored() -> None:
    review = flag(ErrorClass.SETTING_MISMATCH, "s4", LR, state=VerificationState.NEEDS_HUMAN_REVIEW)
    b0 = flag(ErrorClass.SETTING_MISMATCH, "s4", LR, system=SystemName.B0)
    assert score(LABELS, [review]).total.true_positives == 0
    assert score(LABELS, [b0]).total.true_positives == 1


def test_matching_is_one_to_one() -> None:
    twice = [flag(ErrorClass.SETTING_MISMATCH, "s4", LR)] * 2
    scores = score(LABELS, twice)
    assert scores.total.true_positives == 1
    assert scores.unlabelled == 1


def test_the_same_unlabelled_flag_on_both_copies_counts_once_as_unique() -> None:
    scores = score(
        LABELS,
        [
            flag(ErrorClass.SETTING_MISMATCH, "s9", LR, variant=PaperVariant.INJECTED),
            flag(ErrorClass.SETTING_MISMATCH, "s9", LR, variant=PaperVariant.ORIGINAL),
        ],
    )
    assert (scores.unlabelled, scores.unlabelled_unique) == (2, 1)


def test_an_area_scores_only_its_classes_and_the_clean_controls() -> None:
    scores = score(
        LABELS,
        [
            flag(ErrorClass.CLAIM_NOT_SUPPORTED, "s2", B1),
            flag(ErrorClass.SETTING_MISMATCH, "s7", LR),
        ],
        classes=AREAS["code"],
    )
    assert set(scores.per_class) == {ErrorClass.SETTING_MISMATCH}
    assert scores.clean_flagged == 1


def test_verifier_decisions_by_truth_count_duplicates_once() -> None:
    table = verifier_decisions(
        LABELS,
        [
            flag(ErrorClass.SETTING_MISMATCH, "s4", LR),
            flag(ErrorClass.SETTING_MISMATCH, "s4", LR, duplicate_of="x"),
            flag(ErrorClass.SETTING_MISMATCH, "s7", LR, state=VerificationState.DROPPED),
            flag(ErrorClass.METRIC_MISMATCH, "s5", BS, state=VerificationState.NEEDS_HUMAN_REVIEW),
            flag(ErrorClass.SETTING_MISMATCH, "s9", LR, state=VerificationState.NEEDS_HUMAN_REVIEW),
            flag(ErrorClass.SETTING_MISMATCH, "s4", LR, system=SystemName.B0),
        ],
    )
    assert table[Truth.REAL_ERROR] == {VerificationState.VERIFIED_FLAG: 1}
    assert table[Truth.CLEAN] == {VerificationState.DROPPED: 1}
    assert table[Truth.WRONG_CLASS] == {VerificationState.NEEDS_HUMAN_REVIEW: 1}
    assert table[Truth.UNLABELLED] == {VerificationState.NEEDS_HUMAN_REVIEW: 1}
