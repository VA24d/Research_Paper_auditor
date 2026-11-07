"""Failure analysis of finished runs against gold labels (team plan P4.8).

For every error label, and every run of the full system, this finds where the label was
lost, in pipeline order:

- planned: no task points at the label's place and target, so nothing checked it;
- unverifiable / failed: a task did, but could not finish (the reason is counted);
- no finding: the task finished and raised nothing that matches;
- human review: a finding matches, but the verifier sent it to review (with its reason);
- dropped: a finding matches, but the verifier or the math review dropped it;
- wrong class: a verified finding matches the place but names another class;
- found: a verified finding of the right class.

It also counts, per system, how many of the runs found each label (stability across
repetitions), and what the full system would score if findings held back only because
their check marked them uncertain counted as flags (the open question in D23).

    uv run python scripts/p4_analysis.py datasets/set_c/labels.jsonl runs/p4-setc-* \
        --markdown results/p4/set_c-failures.md
"""

import argparse
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from auditor.evaluation.metrics import score
from auditor.evaluation.runs import RunData, find_runs, load_run, read_jsonl
from auditor.schemas import (
    AuditTask,
    DocumentAnchor,
    GoldLabel,
    Prediction,
    SystemName,
    TaskOutcome,
    VerificationState,
    anchors_match,
)

UNCERTAIN_ONLY = "The evidence is in its sources, but the specialist marked it uncertain."
STAGES = (
    "found",
    "wrong class",
    "human review",
    "dropped",
    "no finding",
    "unverifiable",
    "failed",
    "not planned",
    "paper not run",
)


def _about(label: GoldLabel, anchor: object, target: object) -> bool:
    return target == label.target and anchors_match(anchor, label.anchor)  # type: ignore[arg-type]


def _short(reason: str) -> str:
    """A reason without its specifics (file names, numbers), so similar ones count together."""
    words = []
    for word in reason.split():
        if any(c.isdigit() for c in word) or "/" in word or "." in word.strip(".,;:"):
            words.append("<x>")
        else:
            words.append(word)
    return " ".join(words)[:140]


def stage_of(label: GoldLabel, run: RunData) -> tuple[str, str]:
    """Where `label` was lost in one full-system run, and the reason when there is one."""
    paper = next(
        (
            p
            for p in run.complete_papers
            if (p.paper_id, p.variant) == (label.paper_id, label.variant)
        ),
        None,
    )
    if paper is None:
        return "paper not run", ""
    decisions = [d for d in paper.verified if _about(label, d.finding.anchor, d.finding.target)]
    right = [d for d in decisions if d.finding.error_class == label.error_class]
    for wanted, stage in (
        (VerificationState.VERIFIED_FLAG, "found"),
        (VerificationState.NEEDS_HUMAN_REVIEW, "human review"),
        (VerificationState.DROPPED, "dropped"),
    ):
        hit = next((d for d in right if d.state is wanted), None)
        if hit is not None:
            return stage, "" if stage == "found" else _short(hit.reason)
    if any(d.state is VerificationState.VERIFIED_FLAG for d in decisions):
        return "wrong class", ""
    tasks = read_jsonl(paper.folder / "tasks.jsonl", AuditTask)
    # A document-level label (a method discrepancy) is checked by a task anchored on one
    # sentence of the method section, so only its target has to agree.
    whole = isinstance(label.anchor, DocumentAnchor)
    ids = {
        t.task_id
        for t in tasks
        if (t.target == label.target if whole else _about(label, t.anchor, t.target))
    }
    if not ids:
        return "not planned", ""
    results = [r for r in paper.results if r.task_id in ids]
    if results and all(r.outcome is TaskOutcome.UNVERIFIABLE for r in results):
        return "unverifiable", _short(results[0].reason or "")
    if results and all(r.outcome is not TaskOutcome.CHECKED for r in results):
        return "failed", _short(results[0].error or "")
    return "no finding", ""


def uncertain_counted(run: RunData) -> list[Prediction]:
    """The full system's predictions with findings held back only as uncertain counted
    as verified flags."""
    out = []
    for paper in run.complete_papers:
        for decision in paper.verified:
            state = decision.state
            if state is VerificationState.NEEDS_HUMAN_REVIEW and decision.reason == UNCERTAIN_ONLY:
                state = VerificationState.VERIFIED_FLAG
            finding = decision.finding
            out.append(
                Prediction(
                    run_id=run.manifest.run_id,
                    system=SystemName.FULL,
                    paper_id=finding.paper_id,
                    variant=finding.variant,
                    error_class=finding.error_class,
                    anchor=finding.anchor,
                    target=finding.target,
                    state=state,
                    summary=finding.summary,
                    evidence=finding.evidence,
                    finding_id=finding.finding_id,
                    duplicate_of=decision.duplicate_of,
                )
            )
    return out


def analyse(labels: Sequence[GoldLabel], runs: Sequence[RunData]) -> str:
    errors = [label for label in labels if label.is_error]
    full = [run for run in runs if run.system is SystemName.FULL]
    lines = ["# Failure analysis", ""]

    lines += [
        "## Where the full system loses each planted error",
        "",
        "Each planted error in each full-system run, by the first stage it did not pass.",
        "",
    ]
    classes = sorted({str(label.error_class) for label in errors})
    stages: dict[str, Counter[str]] = defaultdict(Counter)
    reasons: dict[str, Counter[str]] = defaultdict(Counter)
    for run in full:
        for label in errors:
            stage, reason = stage_of(label, run)
            stages[str(label.error_class)][stage] += 1
            if reason:
                reasons[stage][reason] += 1
    lines += [
        "| Stage | " + " | ".join(classes) + " | All |",
        "|---|" + "---|" * (len(classes) + 1),
    ]
    for stage in STAGES:
        counts = [stages[c][stage] for c in classes]
        if any(counts):
            lines.append(f"| {stage} | " + " | ".join(map(str, counts)) + f" | {sum(counts)} |")
    totals = [sum(stages[c].values()) for c in classes]
    lines.append("| total | " + " | ".join(map(str, totals)) + f" | {sum(totals)} |")
    for stage in ("human review", "dropped", "unverifiable", "failed"):
        if reasons[stage]:
            lines += ["", f"Most common reasons, {stage}:", ""]
            lines += [f"- {n} x {reason}" for reason, n in reasons[stage].most_common(6)]

    lines += [
        "",
        "## Stability across runs",
        "",
        "How many runs of each system found each planted error (verified flags only for the "
        "full system).",
        "",
    ]
    by_system: dict[SystemName, list[RunData]] = defaultdict(list)
    for run in runs:
        by_system[run.system].append(run)
    for system, group in sorted(by_system.items(), key=lambda kv: kv[0].value):
        found: Counter[str] = Counter()
        for run in group:
            outcomes = score(labels, run.predictions()).outcomes
            found.update(lid for lid, outcome in outcomes.items() if outcome == "found")
        histogram = Counter(found.get(label.label_id, 0) for label in errors)
        cells = ", ".join(f"{k} of {len(group)}: {histogram[k]}" for k in range(len(group), -1, -1))
        lines.append(f"- {system.value} ({len(group)} runs): {cells}")

    if full:
        lines += [
            "",
            "## If uncertain findings counted (D23)",
            "",
            "| Run | Macro-F1 as run | Macro-F1 with uncertain counted | Unlabelled as run | "
            "Unlabelled with uncertain counted |",
            "|---|---|---|---|---|",
        ]
        for run in full:
            now = score(labels, run.predictions())
            alt = score(labels, uncertain_counted(run))
            lines.append(
                f"| {run.manifest.run_id} | {now.macro_f1:.3f} | {alt.macro_f1:.3f} | "
                f"{now.unlabelled} | {alt.unlabelled} |"
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("labels", type=Path)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    labels = read_jsonl(args.labels, GoldLabel)
    runs = [load_run(folder) for folder in find_runs(args.runs)]
    text = analyse(labels, runs)
    if args.markdown is not None:
        args.markdown.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
