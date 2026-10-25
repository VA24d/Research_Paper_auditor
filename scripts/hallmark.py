"""Run the citation agent's reference check on a HALLMARK split (set A, P2.7).

    uv run python scripts/hallmark.py download --split test_public --out data/hallmark
    uv run python scripts/hallmark.py run --file data/hallmark/test_public.jsonl \
        --out runs/hallmark/results.jsonl [--limit 200]
    uv run python scripts/hallmark.py score --results runs/hallmark/results.jsonl

`download` fetches the split from the benchmark's repository (network). `run` turns every
entry into a one-entry paper and runs the agent's reference check on it (network: the
scholarly APIs). `score` reports accuracy, precision, recall and F1 of flagging, the
false-positive rate on the valid entries, and the per-type breakdown; it is offline.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, cast

from auditor.agents.citation.hallmark import (
    DATA_URL,
    SPLITS,
    ExistenceResult,
    Flag,
    check_entry,
    load_split,
    read_results,
    score,
    write_results,
)
from auditor.config import Settings
from auditor.infra.context import RunContext
from auditor.infra.http import HttpClient


async def _download(args: argparse.Namespace) -> int:
    if args.split not in SPLITS:
        args.error(f"unknown split {args.split}; choose from {', '.join(SPLITS)}")
    settings = Settings()
    async with HttpClient(settings) as http:
        response = await http.get(DATA_URL.format(split=args.split))
    response.raise_for_status()
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.split}.jsonl"
    path.write_bytes(response.content)
    print(f"wrote {path} ({len(response.content)} bytes)")
    return 0


async def _run(args: argparse.Namespace) -> int:
    settings = Settings()
    entries = load_split(args.file)
    if args.limit:
        entries = entries[: args.limit]
    done: dict[str, ExistenceResult] = {}
    if args.out.exists():  # resumable: entries already checked are skipped
        for result in read_results(args.out):
            done[result.bibtex_key] = result
    results: list[ExistenceResult] = []
    total = len([e for e in entries if e.bibtex_key not in done])
    async with HttpClient(settings) as http:
        ctx = RunContext(settings=settings, paper=None, http=http, llm=None, logger=None)  # type: ignore[arg-type]
        for number, entry in enumerate(
            [e for e in entries if e.bibtex_key not in done], start=1
        ):
            try:
                result = await check_entry(ctx, entry)
            except Exception as exc:
                print(f"{entry.bibtex_key}: {type(exc).__name__}", file=sys.stderr)
                result = ExistenceResult(
                    bibtex_key=entry.bibtex_key,
                    label=entry.label,
                    hallucination_type=entry.hallucination_type,
                    difficulty_tier=entry.difficulty_tier,
                    flag=Flag.UNVERIFIABLE,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            results.append(result)
            if number % 25 == 0 or number == total:
                # Periodic checkpoint: a crash loses at most the last 25 entries.
                write_results([*done.values(), *results], args.out)
                print(f"{number}/{total} checked", file=sys.stderr)
    write_results([*done.values(), *results], args.out)
    print(f"wrote {args.out} ({len(results)} new, {len(done)} skipped)")
    return 0


def _score(args: argparse.Namespace) -> int:
    scores = score(read_results(args.results))
    print(f"accuracy {scores.accuracy:.3f}")
    print(
        f"flagging precision {scores.overall.precision:.3f}  "
        f"recall {scores.overall.recall:.3f}  F1 {scores.overall.f1:.3f}"
    )
    print(
        f"false-positive rate on valid entries {scores.false_positive_rate:.3f} "
        f"({scores.valid_flagged} of {scores.valid_entries})"
    )
    if scores.unverifiable:
        print(f"unverifiable (not scored) {scores.unverifiable}")
    for name, type_score in sorted(scores.per_type.items()):
        print(
            f"{name:24} P {type_score.precision:.3f}  R {type_score.recall:.3f}  "
            f"F1 {type_score.f1:.3f}  (tp {type_score.true_positives}, "
            f"fp {type_score.false_positives}, fn {type_score.false_negatives})"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="fetch a split from the benchmark")
    download.add_argument("--split", default="test_public", choices=SPLITS)
    download.add_argument("--out", type=Path, default=Path("data/hallmark"))
    download.set_defaults(func=_download)

    run = commands.add_parser("run", help="run the reference check over a split")
    run.add_argument("--file", type=Path, required=True)
    run.add_argument("--out", type=Path, default=Path("runs/hallmark/results.jsonl"))
    run.add_argument("--limit", type=int, default=0, help="check only the first N entries")
    run.set_defaults(func=_run)

    scoring = commands.add_parser("score", help="score a results file (offline)")
    scoring.add_argument("--results", type=Path, required=True)
    scoring.set_defaults(func=_score)

    args = parser.parse_args()
    function: Any = args.func
    if asyncio.iscoroutinefunction(function):
        return cast(int, asyncio.run(function(args)))
    return cast(int, function(args))


if __name__ == "__main__":
    sys.exit(main())
