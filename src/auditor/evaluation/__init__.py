"""Evaluation (part 4, team plan P4.5): scoring runs against gold labels.

`auditor evaluate LABELS RUN...` scores every run with the same rules and writes the
tables as Markdown and the numbers as JSON:

    uv run auditor evaluate datasets/set_c/labels.jsonl runs/main-full-r1 runs/main-b1-r1 \
        --dataset datasets/set_c/dataset.jsonl --markdown results/main.md --json results/main.json

A path may also be a folder of run folders. `--area code` scores one specialist's classes.
`--dataset` loads the papers for the grounding rate; without it grounding is skipped.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

from auditor.config import Settings
from auditor.evaluation.grounding import load_inputs
from auditor.evaluation.metrics import AREAS
from auditor.evaluation.runner import evaluate, render_markdown, to_json
from auditor.evaluation.runs import find_runs, load_run, read_jsonl
from auditor.schemas import ErrorClass, GoldLabel

logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    pass


async def run_evaluation(
    labels_path: Path,
    run_paths: Sequence[Path],
    *,
    settings: Settings,
    dataset: Path | None = None,
    remap: tuple[str, str] | None = None,
    areas: Sequence[str] = (),
    strict: bool = False,
    ablate_verifier: bool = True,
    title: str = "Evaluation",
) -> tuple[str, dict[str, object]]:
    """The Markdown tables and the JSON numbers for `run_paths` against `labels_path`."""
    labels = read_jsonl(labels_path, GoldLabel)
    if not labels:
        raise EvaluationError(f"{labels_path}: no gold labels")
    folders = find_runs(run_paths)
    if not folders:
        raise EvaluationError("no run folders (folders with a manifest.json) found")
    runs = [load_run(folder, remap) for folder in folders]
    mocked = [
        f"{run.manifest.run_id} ({', '.join(run.mocked_parts)})" for run in runs if run.mocked_parts
    ]
    if mocked and strict:
        raise EvaluationError("runs with placeholder parts (D19): " + "; ".join(mocked))
    for item in mocked:
        logger.warning("run with placeholder parts: %s", item)
    classes: set[ErrorClass] | None = None
    if areas:
        unknown = set(areas) - set(AREAS)
        if unknown:
            raise EvaluationError(f"unknown area(s) {sorted(unknown)}; use {sorted(AREAS)}")
        classes = {c for area in areas for c in AREAS[area]}
    inputs = load_inputs(dataset, remap) if dataset is not None else None
    scores = await evaluate(
        runs,
        labels,
        classes=classes,
        inputs=inputs,
        settings=settings,
        ablate_verifier=ablate_verifier,
    )
    report = to_json(scores)
    report["labels"] = str(labels_path)
    report["areas"] = list(areas)
    return render_markdown(scores, title=title), report


__all__ = ["EvaluationError", "run_evaluation"]
