"""The shared evaluation runner (team plan P4.5): every run of every system against the same
gold labels, with the same rules.

For each run it computes the D3 scores (Macro-F1, precision, recall, false-positive rate on
clean controls, unlabelled flags), the D14 numbers on unchanged originals (flags per
original paper, share of originals flagged), the review load, coverage, grounding and cost.
A full-system run is also scored with its verifier taken out (`full_no_verifier`), which
is the no-verifier ablation (P4.7) at no extra model cost. Runs of the same system are then
summarised as mean and sample standard deviation.
"""

import statistics
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from auditor.config import Settings
from auditor.evaluation.grounding import GroundingStats, PaperInput, PaperKey, grounding
from auditor.evaluation.metrics import TRUTHS, Scores, score, verifier_decisions
from auditor.evaluation.runs import RunData, Usage
from auditor.schemas import (
    ErrorClass,
    GoldLabel,
    PaperVariant,
    Prediction,
    SystemName,
    TaskOutcome,
    VerificationState,
)


@dataclass
class RunScore:
    run_id: str
    system: str
    papers: int
    scores: Scores
    usage: Usage
    incomplete: int = 0
    failed: list[str] = field(default_factory=list)
    mocked: list[str] = field(default_factory=list)
    # Labels on paper versions the run has no complete predictions for; scored as misses.
    labels_without_run: int = 0
    original_papers: int = 0
    original_flagged: int = 0
    original_flags: int = 0
    review_load: int | None = None
    outcomes: dict[str, int] | None = None
    grounding: GroundingStats | None = None
    decisions: dict[str, dict[str, int]] | None = None

    @property
    def coverage(self) -> float | None:
        """Share of the full system's tasks that were checked (D6)."""
        if not self.outcomes:
            return None
        total = sum(self.outcomes.values())
        return self.outcomes[TaskOutcome.CHECKED.value] / total if total else None

    @property
    def flags_per_original(self) -> float | None:
        return self.original_flags / self.original_papers if self.original_papers else None

    @property
    def originals_flagged_share(self) -> float | None:
        return self.original_flagged / self.original_papers if self.original_papers else None

    @property
    def review_per_paper(self) -> float | None:
        if self.review_load is None or not self.papers:
            return None
        return self.review_load / self.papers

    @property
    def cost_per_paper(self) -> float | None:
        if self.usage.cost_usd is None or not self.papers:
            return None
        return self.usage.cost_usd / self.papers

    @property
    def llm_calls_per_paper(self) -> float | None:
        if not self.usage.logged or not self.papers:
            return None
        return self.usage.llm_calls / self.papers


def _original_flags(
    predictions: Sequence[Prediction], originals: Collection[str]
) -> tuple[int, int]:
    """Positive flags on unchanged originals, each duplicate group once (D14, D20): the
    number of flags and the number of originals with at least one."""
    flagged: Counter[str] = Counter(
        p.paper_id
        for p in predictions
        if p.is_positive
        and p.duplicate_of is None
        and p.variant is PaperVariant.ORIGINAL
        and p.paper_id in originals
    )
    return sum(flagged.values()), len(flagged)


async def score_run(
    run: RunData,
    labels: Sequence[GoldLabel],
    *,
    without_verifier: bool = False,
    classes: Collection[ErrorClass] | None = None,
    inputs: Mapping[PaperKey, PaperInput] | None = None,
    settings: Settings | None = None,
) -> RunScore:
    predictions = run.predictions_without_verifier() if without_verifier else run.predictions()
    if classes is not None:
        predictions = [p for p in predictions if p.error_class in classes]
    complete = run.complete_papers
    done = {(p.paper_id, p.variant.value) for p in complete}
    originals = {p.paper_id for p in complete if p.variant is PaperVariant.ORIGINAL}
    flags, flagged = _original_flags(predictions, originals)
    system = SystemName.FULL_NO_VERIFIER if without_verifier else run.system
    result = RunScore(
        run_id=run.manifest.run_id,
        system=system.value,
        papers=len(complete),
        scores=score(labels, predictions, classes=classes),
        usage=run.usage,
        incomplete=len(run.papers) - len(complete),
        failed=run.failed,
        mocked=run.mocked_parts,
        labels_without_run=sum(
            1
            for label in labels
            if (label.paper_id, label.variant.value) not in done
            and (not label.is_error or classes is None or label.error_class in classes)
        ),
        original_papers=len(originals),
        original_flagged=flagged,
        original_flags=flags,
    )
    if run.system is SystemName.FULL:
        outcomes = {k.value: v for k, v in run.task_outcomes().items()}
        result.outcomes = outcomes if any(outcomes.values()) else None
    if system is SystemName.FULL:
        result.review_load = sum(
            1
            for p in predictions
            if p.state is VerificationState.NEEDS_HUMAN_REVIEW and p.duplicate_of is None
        )
        result.decisions = {
            truth: {state.value: counts[state] for state in VerificationState}
            for truth, counts in verifier_decisions(labels, predictions).items()
        }
    if inputs is not None:
        result.grounding = await grounding(predictions, inputs, settings or Settings())
    return result


async def evaluate(
    runs: Sequence[RunData],
    labels: Sequence[GoldLabel],
    *,
    classes: Collection[ErrorClass] | None = None,
    inputs: Mapping[PaperKey, PaperInput] | None = None,
    settings: Settings | None = None,
    ablate_verifier: bool = True,
) -> list[RunScore]:
    """Scores of every run; with `ablate_verifier`, each full-system run is also scored
    without its verifier."""
    out = []
    for run in runs:
        out.append(await score_run(run, labels, classes=classes, inputs=inputs, settings=settings))
        if ablate_verifier and run.system is SystemName.FULL:
            out.append(
                await score_run(
                    run,
                    labels,
                    without_verifier=True,
                    classes=classes,
                    inputs=inputs,
                    settings=settings,
                )
            )
    return out


# --- summaries ------------------------------------------------------------------------

SYSTEM_ORDER = [s.value for s in SystemName]


@dataclass
class Spread:
    mean: float
    sd: float | None  # None for a single run
    n: int

    def format(self, spec: str = ".3f") -> str:
        mean = format(self.mean, spec)
        return mean if self.sd is None else f"{mean} ± {format(self.sd, spec)}"

    def __str__(self) -> str:
        return self.format()


def spread(values: Sequence[float | None]) -> Spread | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    sd = statistics.stdev(present) if len(present) > 1 else None
    return Spread(statistics.fmean(present), sd, len(present))


def by_system(scores: Sequence[RunScore]) -> dict[str, list[RunScore]]:
    groups: dict[str, list[RunScore]] = defaultdict(list)
    for item in scores:
        groups[item.system].append(item)
    return dict(sorted(groups.items(), key=lambda kv: SYSTEM_ORDER.index(kv[0])))


# Name, value of one run, and how to print it.
HEADLINE: list[tuple[str, Any, str]] = [
    ("Macro-F1", lambda r: r.scores.macro_f1, ".3f"),
    ("Precision", lambda r: r.scores.total.precision, ".3f"),
    ("Recall", lambda r: r.scores.total.recall, ".3f"),
    ("FP rate (clean)", lambda r: r.scores.false_positive_rate, ".3f"),
    ("Flags / original", lambda r: r.flags_per_original, ".2f"),
    ("Originals flagged", lambda r: r.originals_flagged_share, ".2f"),
    ("Unlabelled", lambda r: float(r.scores.unlabelled), ".1f"),
    ("Review / paper", lambda r: r.review_per_paper, ".2f"),
    ("Coverage", lambda r: r.coverage, ".2f"),
    ("Flags backed", lambda r: r.grounding.backed_share if r.grounding else None, ".2f"),
    ("Quotes grounded", lambda r: r.grounding.rate if r.grounding else None, ".2f"),
    ("LLM calls / paper", lambda r: r.llm_calls_per_paper, ".1f"),
    ("Cost / paper ($)", lambda r: r.cost_per_paper, ".3f"),
]


def _cell(value: Spread | None, spec: str = ".3f") -> str:
    return "-" if value is None else value.format(spec)


def render_markdown(scores: Sequence[RunScore], *, title: str = "Evaluation") -> str:
    groups = by_system(scores)
    lines = [f"# {title}", ""]
    lines += [
        "Only verified flags count for systems with a verifier; every flag counts for B0, "
        "B1 and `full_no_verifier` (D3). Values are the mean ± sample standard deviation "
        "over runs. Unlabelled flags match no gold label and are not scored (D14).",
        "",
        "| System | Runs | " + " | ".join(name for name, _, _ in HEADLINE) + " |",
        "|---|---|" + "---|" * len(HEADLINE),
    ]
    for system, runs in groups.items():
        cells = [_cell(spread([get(r) for r in runs]), spec) for _, get, spec in HEADLINE]
        lines.append(f"| {system} | {len(runs)} | " + " | ".join(cells) + " |")

    classes = sorted({c for r in scores for c, s in r.scores.per_class.items() if s.labelled})
    if classes:
        lines += ["", "## F1 per class", ""]
        lines += [
            "| System | " + " | ".join(c.value for c in classes) + " |",
            "|---|" + "---|" * len(classes),
        ]
        for system, runs in groups.items():
            cells = [
                _cell(
                    spread(
                        [r.scores.per_class[c].f1 if c in r.scores.per_class else 0.0 for r in runs]
                    )
                )
                for c in classes
            ]
            lines.append(f"| {system} | " + " | ".join(cells) + " |")

    decided = [r for r in scores if r.decisions]
    if decided:
        lines += [
            "",
            "## Verifier decisions",
            "",
            "Every full-system finding, by what its gold label says it is, summed over runs. "
            "Real errors should be verified; findings on clean controls should be dropped.",
            "",
            "| Finding | Verified | Human review | Dropped |",
            "|---|---|---|---|",
        ]
        for truth in TRUTHS:
            totals = Counter[str]()
            for r in decided:
                totals.update((r.decisions or {}).get(truth, {}))
            lines.append(
                f"| {truth} | {totals[VerificationState.VERIFIED_FLAG.value]} | "
                f"{totals[VerificationState.NEEDS_HUMAN_REVIEW.value]} | "
                f"{totals[VerificationState.DROPPED.value]} |"
            )

    lines += ["", "## Runs", ""]
    lines += [
        "| Run | System | Papers | Macro-F1 | TP | FP | FN | Clean flagged | Unlabelled "
        "(unique) | LLM calls | Tokens in / out | Cost ($) | Notes |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in scores:
        total = r.scores.total
        notes = []
        if r.mocked:
            notes.append("mocked: " + ", ".join(r.mocked))
        if r.incomplete:
            notes.append(f"{r.incomplete} incomplete")
        if r.failed:
            notes.append(f"{len(r.failed)} failed")
        if r.labels_without_run:
            notes.append(f"{r.labels_without_run} labels on papers not run")
        cost = "-" if r.usage.cost_usd is None else f"{r.usage.cost_usd:.2f}"
        lines.append(
            f"| {r.run_id} | {r.system} | {r.papers} | {r.scores.macro_f1:.3f} | "
            f"{total.true_positives} | {total.false_positives} | {total.false_negatives} | "
            f"{r.scores.clean_flagged} of {r.scores.clean_controls} | "
            f"{r.scores.unlabelled} ({r.scores.unlabelled_unique}) | {r.usage.llm_calls} | "
            f"{r.usage.input_tokens:,} / {r.usage.output_tokens:,} | {cost} | "
            f"{'; '.join(notes)} |"
        )
    return "\n".join(lines) + "\n"


def to_json(scores: Sequence[RunScore]) -> dict[str, Any]:
    """Every run's numbers, and the per-system summary, as plain JSON."""

    def run(r: RunScore) -> dict[str, Any]:
        return {
            "run_id": r.run_id,
            "system": r.system,
            "papers": r.papers,
            "incomplete": r.incomplete,
            "failed": r.failed,
            "mocked": r.mocked,
            "labels_without_run": r.labels_without_run,
            "macro_f1": r.scores.macro_f1,
            "precision": r.scores.total.precision,
            "recall": r.scores.total.recall,
            "false_positive_rate": r.scores.false_positive_rate,
            "clean_controls": r.scores.clean_controls,
            "clean_flagged": r.scores.clean_flagged,
            "unlabelled": r.scores.unlabelled,
            "unlabelled_unique": r.scores.unlabelled_unique,
            "per_class": {
                c.value: {
                    "tp": s.true_positives,
                    "fp": s.false_positives,
                    "fn": s.false_negatives,
                    "precision": s.precision,
                    "recall": s.recall,
                    "f1": s.f1,
                }
                for c, s in sorted(r.scores.per_class.items())
            },
            "original_papers": r.original_papers,
            "original_flagged": r.original_flagged,
            "original_flags": r.original_flags,
            "review_load": r.review_load,
            "task_outcomes": r.outcomes,
            "coverage": r.coverage,
            "grounding": None
            if r.grounding is None
            else {
                **vars(r.grounding),
                "backed_share": r.grounding.backed_share,
                "rate": r.grounding.rate,
            },
            "verifier_decisions": r.decisions,
            "label_outcomes": r.scores.outcomes,
            "usage": vars(r.usage),
        }

    summary = {
        system: {
            name: None if (s := spread([get(r) for r in runs])) is None else vars(s)
            for name, get, _ in HEADLINE
        }
        for system, runs in by_system(scores).items()
    }
    return {"runs": [run(r) for r in scores], "summary": summary}
