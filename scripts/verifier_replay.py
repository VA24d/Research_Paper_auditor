"""Run the verifier again on a finished run, with no model calls and no network.

Each paper's `results.jsonl` goes through `auditor.agents.verifier.verify` (without the
paid recheck), and the run's own review decisions are kept: a finding the run had dropped
(the math review, decision D21) stays dropped, and its duplicates stay merged. The output
is a run folder with fresh `verified.jsonl` and `predictions.jsonl` files, which the
dataset scripts score as usual (`scripts/set_c.py score`, `scripts/spot.py score`, ...),
to measure precision before and after the verifier.

    uv run python scripts/verifier_replay.py data/set_c/runs/rq1-c5-full-r1 \
        --out data/scratch/replay/rq1-c5-full-r1
"""

import argparse
import asyncio
import json
import logging
import shutil
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from auditor.agents.verifier import verify
from auditor.config import Settings
from auditor.experiments import InputError, load_paper, read_dataset, resolve_repository
from auditor.infra.context import RunContext
from auditor.orchestration.outputs import predictions_from_verified, write_jsonl
from auditor.orchestration.synthesis import synthesise
from auditor.schemas import (
    DatasetEntry,
    ParsedPaper,
    RunManifest,
    TaskResult,
    VerificationState,
    VerifiedFinding,
)
from auditor.testing import offline_http

logger = logging.getLogger("verifier_replay")


def _read[T: BaseModel](path: Path, schema: type[T]) -> list[T]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [schema.model_validate_json(line) for line in lines if line]


def _entries(manifest: RunManifest, root: Path) -> list[DatasetEntry]:
    dataset = manifest.config.get("dataset")
    if not isinstance(dataset, str):
        return []
    path = Path(dataset)
    path = path if path.is_absolute() else root / path
    return read_dataset(path) if path.is_file() else []


def _carry_over(fresh: list[VerifiedFinding], old: list[VerifiedFinding]) -> list[VerifiedFinding]:
    """The run's review decisions on top of the fresh ones: what it dropped and merged."""
    before = {d.finding.finding_id: d for d in old}
    out = []
    for decision in fresh:
        earlier = before.get(decision.finding.finding_id)
        if earlier is not None and earlier.state is VerificationState.DROPPED:
            decision = decision.model_copy(
                update={"state": earlier.state, "reason": earlier.reason, "accepted_evidence": ()}
            )
        if earlier is not None and earlier.duplicate_of is not None:
            decision = decision.model_copy(update={"duplicate_of": earlier.duplicate_of})
        out.append(decision)
    return out


def _prepare(run: Path, out: Path) -> list[Path]:
    """The run's paper folders; the output folder gets the run's manifest."""
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "manifest.json", out / "manifest.json")
    return sorted(p for p in run.iterdir() if (p / "results.jsonl").is_file())


async def _replay(run: Path, out: Path, settings: Settings) -> Counter[tuple[str, str]]:
    manifest = RunManifest.model_validate_json((run / "manifest.json").read_text("utf-8"))
    entries = _entries(manifest, Path.cwd())
    http = offline_http(settings.cache_dir)
    counts: Counter[tuple[str, str]] = Counter()
    folders = _prepare(run, out)
    papers: dict[tuple[str, str], tuple[ParsedPaper, DatasetEntry]] = {}
    for entry in entries:
        try:
            paper = await load_paper(Path(entry.paper), settings, http)
        except (InputError, AssertionError) as exc:
            logger.warning("%s: %s", entry.paper, exc)
            continue
        papers.setdefault((paper.paper_id, paper.variant.value), (paper, entry))
    for folder in folders:
        results = _read(folder / "results.jsonl", TaskResult)
        if not results:
            continue
        key = (results[0].paper_id, results[0].variant.value)
        if key not in papers and entries:
            # A dataset file rewritten since the run: look for the paper beside the others.
            sibling = Path(entries[0].paper).parent / f"{folder.name.rsplit('-', 1)[0]}.json"
            if sibling.is_file():
                paper = await load_paper(sibling, settings, http)
                papers[key] = (paper, DatasetEntry(paper=str(sibling)))
        if key not in papers:
            logger.warning("%s: its paper is not in the run's dataset", folder.name)
            continue
        paper, entry = papers[key]
        try:
            repository = resolve_repository(entry.repository, paper, entry.repository_commit)
        except InputError as exc:
            logger.warning("%s: %s", folder.name, exc)
            continue
        ctx = RunContext(
            settings=settings.model_copy(update={"verifier_recheck": False}),
            paper=paper,
            http=http,
            llm=None,
            logger=logger,
            repository=repository,
        )
        fresh = await verify(results, ctx)
        old = _read(folder / "verified.jsonl", VerifiedFinding)
        verified = synthesise(_carry_over(fresh, old))
        for decision in verified:
            if decision.duplicate_of is None:
                counts[(decision.finding.error_class.value, decision.state.value)] += 1
        target = out / folder.name
        write_jsonl(target / "verified.jsonl", verified)
        write_jsonl(
            target / "predictions.jsonl",
            predictions_from_verified(verified, run_id=manifest.run_id, system=manifest.system),
        )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    counts = asyncio.run(_replay(args.run, args.out, Settings()))
    table: dict[str, dict[str, int]] = {}
    for (error_class, state), n in counts.items():
        table.setdefault(error_class, {})[state] = n
    print(json.dumps(table, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
