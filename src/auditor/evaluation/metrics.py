"""Scores of one run's predictions against gold labels (decisions D3, D4 and D14).

Matching follows D3: a positive prediction matches a gold label about the same paper,
variant, target and anchor, one to one, in the order the predictions come. The error class
is compared after matching:

- same class as an error label: a true positive of that class;
- another class than the error label: a false positive of the predicted class and a miss
  of the label's class;
- a clean control (class "none"): a false positive, and a hit for the false-positive rate;
- no label at all: an unlabelled flag. It may be a false alarm or a real error already in
  the paper, so it is counted apart and not scored (D14 adjudicates a sample by hand).

Only positive predictions are scored: VERIFIED_FLAG for systems with a verifier, every flag
for B0, B1 and the full system without its verifier (`Prediction.is_positive`).
"""

from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field

from auditor.schemas import ErrorClass, GoldLabel, Prediction, VerificationState, matches
from auditor.schemas.evaluation import is_method_comparison, method_overlap

# The areas of decision D4, for scoring one specialist's classes at a time.
AREAS: dict[str, frozenset[ErrorClass]] = {
    "citation": frozenset(
        {
            ErrorClass.REFERENCE_NOT_FOUND,
            ErrorClass.REFERENCE_METADATA_MISMATCH,
            ErrorClass.CLAIM_NOT_SUPPORTED,
            ErrorClass.CLAIM_NUMBER_MISMATCH,
        }
    ),
    "code": frozenset(
        {
            ErrorClass.SETTING_MISMATCH,
            ErrorClass.DATA_SPLIT_MISMATCH,
            ErrorClass.METRIC_MISMATCH,
            ErrorClass.METHOD_MISMATCH,
            ErrorClass.CODE_DEFECT,
        }
    ),
    "data": frozenset(
        {
            ErrorClass.DATASET_SIZE_MISMATCH,
            ErrorClass.LABEL_DISTRIBUTION_MISMATCH,
            ErrorClass.DATA_LEAKAGE,
        }
    ),
    "theory": frozenset({ErrorClass.MATH_ERROR}),
    "integrity": frozenset({ErrorClass.UNFINISHED_CONTENT, ErrorClass.RESULT_INCONSISTENCY}),
    "template": frozenset({ErrorClass.TEMPLATE_VIOLATION}),
}


@dataclass
class ClassScore:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        found = self.true_positives + self.false_positives
        return self.true_positives / found if found else 0.0

    @property
    def recall(self) -> float:
        wanted = self.true_positives + self.false_negatives
        return self.true_positives / wanted if wanted else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    @property
    def labelled(self) -> bool:
        """Whether the labels contain this class, so it counts towards Macro-F1."""
        return self.true_positives + self.false_negatives > 0


@dataclass
class Scores:
    per_class: dict[ErrorClass, ClassScore] = field(default_factory=dict)
    clean_controls: int = 0
    clean_flagged: int = 0
    unlabelled: int = 0
    # The same flag raised on both copies of a paper counts once here.
    unlabelled_unique: int = 0
    # What happened to each label in scope, by label id, for failure analysis: "found",
    # "wrong_class", "missed" for error labels; "flagged" or "clean" for clean controls.
    outcomes: dict[str, str] = field(default_factory=dict)

    @property
    def macro_f1(self) -> float:
        """Over the classes the labels contain (decision D4)."""
        scores = [s for s in self.per_class.values() if s.labelled]
        return sum(s.f1 for s in scores) / len(scores) if scores else 0.0

    @property
    def total(self) -> ClassScore:
        """All classes pooled, for micro precision and recall."""
        return ClassScore(
            sum(s.true_positives for s in self.per_class.values()),
            sum(s.false_positives for s in self.per_class.values()),
            sum(s.false_negatives for s in self.per_class.values()),
        )

    @property
    def false_positive_rate(self) -> float:
        """Share of clean control items that got a positive flag (D14)."""
        return self.clean_flagged / self.clean_controls if self.clean_controls else 0.0


def _unlabelled_key(prediction: Prediction) -> tuple[str, str, str, str]:
    return (
        prediction.paper_id,
        prediction.error_class.value,
        prediction.anchor.model_dump_json(include={"kind", "sentence_id", "bib_id", "pages"}),
        prediction.target.model_dump_json(),
    )


def best_match_assignment(
    predictions: Sequence[Prediction],
    labels: Sequence[GoldLabel],
) -> list[tuple[Prediction, GoldLabel | None]]:
    """Assign predictions to labels 1-to-1, maximizing match strength.

    Duplicate predictions claiming the same label are deduplicated: only the
    best match is assigned; subsequent duplicates cannot claim other labels.
    """
    candidates: list[tuple[float, bool, int, int]] = []
    for p_idx, pred in enumerate(predictions):
        for l_idx, lbl in enumerate(labels):
            if matches(pred, lbl):
                score_val = method_overlap(pred, lbl) if is_method_comparison(pred, lbl) else 1.0
                same_class = pred.error_class == lbl.error_class
                candidates.append((score_val, same_class, p_idx, l_idx))

    candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)

    assigned_preds: dict[int, int] = {}
    assigned_labels: set[int] = set()
    for _, _, p_idx, l_idx in candidates:
        if p_idx not in assigned_preds and l_idx not in assigned_labels:
            assigned_preds[p_idx] = l_idx
            assigned_labels.add(l_idx)

    result: list[tuple[Prediction, GoldLabel | None]] = []
    for p_idx, pred in enumerate(predictions):
        matched_idx = assigned_preds.get(p_idx)
        result.append((pred, labels[matched_idx] if matched_idx is not None else None))
    return result


def score(
    labels: Sequence[GoldLabel],
    predictions: Sequence[Prediction],
    *,
    classes: Collection[ErrorClass] | None = None,
) -> Scores:
    """Scores of the positive predictions (D3 matching). With `classes`, only those
    classes and the clean controls are scored, as for one specialist's module metrics."""
    scores = Scores()
    in_scope = [
        label
        for label in labels
        if not label.is_error or classes is None or label.error_class in classes
    ]
    for label in in_scope:
        if label.is_error:
            scores.per_class.setdefault(ErrorClass(label.error_class), ClassScore())
    scores.clean_controls = sum(1 for label in in_scope if not label.is_error)

    active_preds = [
        p for p in predictions if p.is_positive and (classes is None or p.error_class in classes)
    ]
    assignments = best_match_assignment(active_preds, in_scope)
    unmatched = list(in_scope)
    unlabelled: set[tuple[str, str, str, str]] = set()

    for prediction, matched in assignments:
        if matched is None:
            scores.unlabelled += 1
            unlabelled.add(_unlabelled_key(prediction))
            continue
        unmatched.remove(matched)
        predicted = scores.per_class.setdefault(prediction.error_class, ClassScore())
        if not matched.is_error:
            scores.clean_flagged += 1
            predicted.false_positives += 1
            scores.outcomes[matched.label_id] = "flagged"
        elif matched.error_class == prediction.error_class:
            predicted.true_positives += 1
            scores.outcomes[matched.label_id] = "found"
        else:
            predicted.false_positives += 1
            scores.per_class[ErrorClass(matched.error_class)].false_negatives += 1
            scores.outcomes[matched.label_id] = "wrong_class"
    for label in unmatched:
        if label.is_error:
            scores.per_class[ErrorClass(label.error_class)].false_negatives += 1
        scores.outcomes[label.label_id] = "missed" if label.is_error else "clean"
    scores.unlabelled_unique = len(unlabelled)
    return scores


class Truth:
    """What a finding turned out to be, by its gold label."""

    REAL_ERROR = "real_error"  # matches an error label of the same class
    WRONG_CLASS = "wrong_class"  # matches an error label of another class
    CLEAN = "clean_control"  # matches a clean control: a false accusation
    UNLABELLED = "unlabelled"  # matches no label


TRUTHS = (Truth.REAL_ERROR, Truth.WRONG_CLASS, Truth.CLEAN, Truth.UNLABELLED)


def verifier_decisions(
    labels: Sequence[GoldLabel], predictions: Sequence[Prediction]
) -> dict[str, Counter[VerificationState]]:
    """The verifier's states, split by what each finding turned out to be (team plan P4.3).

    Every decision counts, not only the positive ones, so this shows what the verifier did
    with real errors (ideally verified) and with false accusations (ideally dropped).
    Duplicates (D20) count once, as in the report."""
    table: dict[str, Counter[VerificationState]] = {truth: Counter() for truth in TRUTHS}
    valid_preds = [p for p in predictions if p.state is not None and p.duplicate_of is None]
    assignments = best_match_assignment(valid_preds, list(labels))
    for prediction, matched in assignments:
        if matched is None:
            truth = Truth.UNLABELLED
        else:
            if not matched.is_error:
                truth = Truth.CLEAN
            elif matched.error_class == prediction.error_class:
                truth = Truth.REAL_ERROR
            else:
                truth = Truth.WRONG_CLASS
        assert prediction.state is not None
        table[truth][prediction.state] += 1
    return table
