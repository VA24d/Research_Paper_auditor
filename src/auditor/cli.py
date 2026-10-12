"""Command-line entry point: `uv run auditor --help`."""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from auditor.baselines.common import IncompleteAnswerError
from auditor.config import Settings
from auditor.experiments import (
    BATCH_SYSTEMS,
    InputError,
    load_paper,
    load_venue,
    missing_parts,
    optional_llm,
    required_llm,
    resolve_repository,
    run_baseline,
    run_batch,
    run_full,
)
from auditor.infra.files import atomic_write
from auditor.infra.http import HttpClient
from auditor.infra.log import configure_logging
from auditor.infra.manifest import build_manifest, new_run_id, write_manifest
from auditor.infra.usage import UsageLog
from auditor.orchestration.graph import NothingToResumeError, ThreadInUseError
from auditor.orchestration.outputs import RunDirectory
from auditor.schemas import (
    LinkMethod,
    ParsedPaper,
    SystemName,
    TaskOutcome,
    VerificationState,
)

logger = logging.getLogger("auditor.cli")


def _summary(paper: ParsedPaper) -> str:
    linked = sum(1 for c in paper.citations if c.bib_ids)
    heuristic = sum(1 for c in paper.citations if c.link_method is LinkMethod.HEURISTIC)
    pages = paper.layout.page_count if paper.layout else "?"
    return (
        f"{paper.paper_id}: {pages} pages, {len(paper.sections)} sections, "
        f"{len(paper.sentences)} sentences, {len(paper.bibliography)} references, "
        f"{linked}/{len(paper.citations)} citation markers linked ({heuristic} by heuristic), "
        f"{len(paper.tables)} tables, {len(paper.links)} links"
    )


async def _parse(args: argparse.Namespace, settings: Settings) -> int:
    async with HttpClient(settings) as http:
        paper = await load_paper(args.pdf, settings, http, refresh=args.refresh)
    if args.output is not None:
        atomic_write(args.output, (paper.model_dump_json(indent=1) + "\n").encode("utf-8"))
    print(_summary(paper))
    return 0


async def _audit(args: argparse.Namespace, settings: Settings) -> int:
    if args.resume and not args.run_id:
        raise InputError("--resume needs the --run-id of the run to continue")
    run_id = args.run_id or new_run_id(SystemName.FULL)
    run = RunDirectory(settings.runs_dir, run_id)
    configure_logging(settings.log_level, run.path / "run.log")
    venue = load_venue(args.venue)

    async with HttpClient(settings) as http:
        paper = await load_paper(args.input, settings, http)
        repository = resolve_repository(args.repo, paper, args.repo_commit)
        llm = optional_llm(settings, not args.no_llm)
        manifest = build_manifest(
            run_id=run_id,
            system=SystemName.FULL,
            settings=settings,
            model=llm.name if llm else None,
            extra_config={
                "paper_id": paper.paper_id,
                "variant": paper.variant.value,
                "repository": repository.url if repository else None,
                "repository_commit": repository.commit if repository else None,
                "venue": venue.id if venue else None,
                "mocked_parts": list(missing_parts(SystemName.FULL)),
            },
        )
        write_manifest(manifest, run.path)
        async with AsyncSqliteSaver.from_conn_string(str(run.path / "checkpoints.sqlite")) as saver:
            try:
                result = await run_full(
                    paper,
                    settings=settings,
                    http=http,
                    llm=llm,
                    run=run,
                    run_id=run_id,
                    usage=UsageLog(run.usage_dir),
                    checkpointer=saver,
                    repository=repository,
                    venue=venue,
                    max_concurrency=args.max_concurrency,
                    resume=args.resume,
                )
            except (ThreadInUseError, NothingToResumeError) as exc:
                raise InputError(str(exc)) from exc
    write_manifest(manifest.model_copy(update={"finished_at": datetime.now(UTC)}), run.path)

    outcome = result.outcome
    outcomes = {o: sum(1 for r in outcome.results if r.outcome is o) for o in TaskOutcome}
    states = {s: sum(1 for v in outcome.verified if v.state is s) for s in VerificationState}
    print(
        f"{run_id}: {len(outcome.tasks)} tasks "
        f"({outcomes[TaskOutcome.CHECKED]} checked, {outcomes[TaskOutcome.UNVERIFIABLE]} "
        f"unverifiable, {outcomes[TaskOutcome.FAILED]} failed); "
        f"{states[VerificationState.VERIFIED_FLAG]} verified flags, "
        f"{states[VerificationState.NEEDS_HUMAN_REVIEW]} for human review, "
        f"{states[VerificationState.DROPPED]} dropped. Output: {result.folder}"
    )
    return 0


async def _baseline(args: argparse.Namespace, settings: Settings) -> int:
    system = SystemName(args.system)
    run_id = args.run_id or new_run_id(system)
    run = RunDirectory(settings.runs_dir, run_id)
    configure_logging(settings.log_level, run.path / "run.log")
    venue = load_venue(args.venue)
    llm = required_llm(settings)

    async with HttpClient(settings) as http:
        paper = await load_paper(args.input, settings, http)
        repository = resolve_repository(args.repo, paper, args.repo_commit)
        manifest = build_manifest(
            run_id=run_id,
            system=system,
            settings=settings,
            model=llm.name,
            extra_config={
                "paper_id": paper.paper_id,
                "variant": paper.variant.value,
                "mocked_parts": list(missing_parts(system)),
            },
        )
        write_manifest(manifest, run.path)
        try:
            result, path = await run_baseline(
                system,
                paper,
                settings=settings,
                http=http,
                llm=llm,
                run=run,
                run_id=run_id,
                usage=UsageLog(run.usage_dir),
                repository=repository,
                venue=venue,
            )
        except IncompleteAnswerError as exc:
            print(f"{run_id}: {exc}; no predictions written.", file=sys.stderr)
            return 1
    write_manifest(manifest.model_copy(update={"finished_at": datetime.now(UTC)}), run.path)
    print(
        f"{run_id}: {len(result.predictions)} predictions "
        f"({len(result.dropped)} answers dropped as invalid). Output: {path}"
    )
    return 0


async def _batch(args: argparse.Namespace, settings: Settings) -> int:
    if not args.dataset.is_file():
        raise InputError(f"No such dataset file: {args.dataset}")
    if not args.llm_cache:
        # Reported runs must not reuse cached answers (decision D10).
        settings = settings.model_copy(update={"llm_cache": False})
    summaries = await run_batch(
        args.dataset,
        experiment=args.experiment,
        systems=[SystemName(s) for s in args.systems],
        runs=args.runs,
        settings=settings,
        strict=args.strict,
        max_concurrency=args.max_concurrency,
    )
    failed = 0
    for summary in summaries:
        print(
            f"{summary.run_id}: {summary.done} done, {summary.skipped} already done, "
            f"{len(summary.failed)} failed"
        )
        for message in summary.failed:
            print(f"    {message}")
        failed += len(summary.failed)
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auditor", description="Audit research papers.")
    commands = parser.add_subparsers(dest="command", required=True)

    parse = commands.add_parser("parse", help="Parse a PDF into ParsedPaper JSON (needs GROBID)")
    parse.add_argument("pdf", type=Path)
    parse.add_argument("-o", "--output", type=Path, help="Write the ParsedPaper JSON here")
    parse.add_argument("--refresh", action="store_true", help="Ignore the parse cache")

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("input", type=Path, help="A PDF, or ParsedPaper JSON")
        command.add_argument("--repo", help="Code repository URL (default: found in the paper)")
        command.add_argument(
            "--repo-commit", help="Commit, tag or branch of the repository to check against"
        )
        command.add_argument("--venue", type=Path, help="Venue profile JSON")
        command.add_argument("--run-id", help="Run id (default: generated)")

    audit = commands.add_parser("audit", help="Run the full audit on one paper")
    add_common(audit)
    audit.add_argument("--resume", action="store_true", help="Continue an interrupted run")
    audit.add_argument("--no-llm", action="store_true", help="Skip the planner's LLM pass")
    audit.add_argument("--max-concurrency", type=int, default=8)

    baseline = commands.add_parser("baseline", help="Run baseline B0 or B1 on one paper")
    baseline.add_argument("system", choices=[SystemName.B0.value, SystemName.B1.value])
    add_common(baseline)

    batch = commands.add_parser(
        "batch", help="Run systems over a dataset several times (resumable)"
    )
    batch.add_argument("dataset", type=Path, help="JSON Lines file of DatasetEntry records")
    batch.add_argument("--experiment", required=True, help="Name used in the run ids")
    batch.add_argument(
        "--systems",
        nargs="+",
        default=[s.value for s in BATCH_SYSTEMS],
        choices=[s.value for s in BATCH_SYSTEMS],
    )
    batch.add_argument("--runs", type=int, default=3, help="Repetitions of each system")
    batch.add_argument(
        "--llm-cache", action="store_true", help="Reuse cached LLM answers (development only)"
    )
    batch.add_argument(
        "--strict", action="store_true", help="Refuse to run while any part is still a mock"
    )
    batch.add_argument("--max-concurrency", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    configure_logging(settings.log_level)
    commands = {"parse": _parse, "audit": _audit, "baseline": _baseline, "batch": _batch}
    try:
        return asyncio.run(commands[args.command](args, settings))
    except InputError as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
