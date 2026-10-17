"""Fetch and score SPOT's math and CS proof subset, the theory checker's outside benchmark (D21).

    uv run python scripts/spot.py fetch --data data/spot --out datasets/spot
    AUDITOR_THEORY_CHECK_ENABLED=true uv run auditor batch data/spot/dataset.jsonl \
        --experiment spot --systems full,b0 --runs 1
    uv run python scripts/spot.py score runs/spot-full-r1 --errors datasets/spot/errors.jsonl \
        --data data/spot [--before-verifier] [--any-class]

`fetch` downloads SPOT at a pinned revision (needs `uv sync --extra data` for parquet), and
writes the papers and the batch dataset file to `--data` (not committed, D13) and the
annotated errors to `--out`. `score` asks an LLM judge, once per annotated error with
reports, whether the run reported it, and writes the grades next to the run
(`spot_grades.jsonl`). Existing grades are reused, so a second `score` makes no model
calls. A person checks every grade (`human_agrees`) before numbers are reported.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from pydantic import BaseModel

from auditor.agents.theory.spot import (
    SPOT_DATASET,
    SPOT_REVISION,
    Grade,
    SpotError,
    judge_paper,
    paper_from_pages,
    score_grades,
    spot_subset,
)
from auditor.config import Settings
from auditor.infra.http import HttpClient
from auditor.infra.llm import StructuredLLM
from auditor.schemas import (
    DatasetEntry,
    ErrorClass,
    ParsedPaper,
    Prediction,
    VerificationState,
)

PARQUET = "data/train-00000-of-00001.parquet"
ERRORS = "errors.jsonl"
GRADES = "spot_grades.jsonl"


def _read_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [model.model_validate_json(line) for line in lines if line.strip()]


def _write_jsonl(path: Path, records: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(record.model_dump_json() + "\n" for record in records)
    path.write_text(text, encoding="utf-8", newline="\n")


async def _fetch(args: argparse.Namespace) -> int:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("reading SPOT needs pyarrow: uv sync --extra data", file=sys.stderr)
        return 2
    settings = Settings()
    data: Path = args.data
    parquet = data / "spot.parquet"
    if not parquet.is_file():
        url = f"https://huggingface.co/datasets/{SPOT_DATASET}/resolve/{SPOT_REVISION}/{PARQUET}"
        async with HttpClient(settings) as http:
            await http.download_to(url, parquet, max_bytes=200_000_000)
    errors, pages = spot_subset(pq.read_table(parquet).to_pylist())
    titles = {e.paper_id: e.title for e in errors}
    entries = []
    for paper_id, paper_pages in pages.items():
        paper = paper_from_pages(paper_id, titles[paper_id], paper_pages, SPOT_REVISION)
        path = data / "papers" / f"{paper_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(paper.model_dump_json(indent=1), encoding="utf-8", newline="\n")
        entries.append(DatasetEntry(paper=str(path.resolve())))
    _write_jsonl(data / "dataset.jsonl", list(entries))
    _write_jsonl(args.out / ERRORS, list(errors))
    print(f"{len(pages)} papers, {len(errors)} errors")
    return 0


def _reports(
    run: Path, before_verifier: bool, any_class: bool
) -> tuple[dict[str, list[Prediction]], int]:
    """The reports of each paper of the run that count, and how many there are. Repeats
    of an earlier finding (D20) do not count twice."""
    reports: dict[str, list[Prediction]] = {}
    for path in sorted(run.rglob("predictions.jsonl")):
        for prediction in _read_jsonl(path, Prediction):
            if prediction.duplicate_of is not None:
                continue
            if not any_class and prediction.error_class is not ErrorClass.MATH_ERROR:
                continue
            if before_verifier:
                if prediction.state is VerificationState.DROPPED:
                    continue
            elif not prediction.is_positive:
                continue
            reports.setdefault(prediction.paper_id, []).append(prediction)
    return reports, sum(len(r) for r in reports.values())


async def _score(args: argparse.Namespace) -> int:
    errors = _read_jsonl(args.errors, SpotError)
    reports, total = _reports(args.run, args.before_verifier, args.any_class)
    grades_path: Path = args.grades or args.run / GRADES
    kept = {g.error_id: g for g in _read_jsonl(grades_path, Grade)}
    todo: dict[str, list[SpotError]] = {}
    for error in errors:
        if error.error_id not in kept:
            todo.setdefault(error.paper_id, []).append(error)
    calls = sum(len(e) for paper_id, e in todo.items() if reports.get(paper_id))
    print(f"{total} reports; {calls} judge calls needed ({len(kept)} grades reused)")
    if calls > args.max_calls:
        print(
            f"more than --max-calls {args.max_calls}; rerun with a higher limit to go on",
            file=sys.stderr,
        )
        return 2
    llm = StructuredLLM.from_settings(Settings()) if calls else None
    for paper_id, paper_errors in todo.items():
        paper_reports = reports.get(paper_id, [])
        if not paper_reports or llm is None:
            new = [Grade(error_id=e.error_id, paper_id=paper_id, matched=()) for e in paper_errors]
        else:
            path = args.data / "papers" / f"{paper_id}.json"
            paper = ParsedPaper.model_validate_json(path.read_bytes())
            texts = {s.id: s.text for s in paper.sentences}
            new = await judge_paper(llm, paper_errors, paper_reports, texts)
        for grade in new:
            kept[grade.error_id] = grade
    grades = [kept[e.error_id] for e in errors if e.error_id in kept]
    _write_jsonl(grades_path, list(grades))
    scores = score_grades(grades, predictions=total)
    print(
        json.dumps(
            {
                "errors": scores.errors,
                "found": scores.found,
                "recall": round(scores.recall, 3),
                "reports": scores.predictions,
                "matched_reports": scores.matched,
                "precision": round(scores.precision, 3),
                "grades_not_checked_by_a_person": scores.unchecked,
            }
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch")
    fetch.add_argument("--data", type=Path, required=True)
    fetch.add_argument("--out", type=Path, required=True)

    score = commands.add_parser("score")
    score.add_argument("run", type=Path)
    score.add_argument("--errors", type=Path, required=True)
    score.add_argument("--data", type=Path, required=True)
    score.add_argument("--grades", type=Path)
    score.add_argument("--before-verifier", action="store_true")
    score.add_argument("--any-class", action="store_true")
    score.add_argument("--max-calls", type=int, default=50)

    args = parser.parse_args()
    match args.command:
        case "fetch":
            return asyncio.run(_fetch(args))
        case "score":
            return asyncio.run(_score(args))
    return 2


if __name__ == "__main__":
    sys.exit(main())
