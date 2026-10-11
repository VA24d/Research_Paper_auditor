"""Command-line entry point: `uv run auditor --help`."""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from auditor.baselines.b0 import BaselineResult, run_b0
from auditor.baselines.b1 import run_b1
from auditor.config import Settings
from auditor.infra.context import RunContext
from auditor.infra.files import atomic_write
from auditor.infra.http import HttpClient
from auditor.infra.llm import (
    LLMNotConfiguredError,
    StructuredLLM,
    create_chat_model,
    model_id,
)
from auditor.infra.log import configure_logging
from auditor.infra.manifest import build_manifest, new_run_id, write_manifest
from auditor.infra.usage import UsageLog
from auditor.orchestration.graph import (
    AuditPipeline,
    NothingToResumeError,
    ThreadInUseError,
    run_audit,
)
from auditor.orchestration.outputs import RunDirectory, predictions_from_verified
from auditor.orchestration.registry import load_specialists, load_tools, load_verifier
from auditor.parser.grobid import GrobidClient
from auditor.parser.pipeline import ParseError, parse_pdf
from auditor.planner.repository import find_repository, repository_url
from auditor.planner.tasks import PlannerOptions, make_planner
from auditor.schemas import (
    LinkMethod,
    ParsedPaper,
    RepositoryRef,
    SystemName,
    TaskOutcome,
    VenueProfile,
    VerificationState,
)

logger = logging.getLogger("auditor.cli")


class UsageError(Exception):
    """A problem with the command's input, reported without a traceback."""


def _summary(paper: ParsedPaper) -> str:
    linked = sum(1 for c in paper.citations if c.bib_ids)
    heuristic = sum(1 for c in paper.citations if c.link_method is LinkMethod.HEURISTIC)
    pages = paper.layout.page_count if paper.layout else "?"
    return (
        f"{paper.paper_id}: {pages} pages, {len(paper.sections)} sections, "
        f"{len(paper.sentences)} sentences, {len(paper.bibliography)} references, "
        f"{linked}/{len(paper.citations)} citation markers linked ({heuristic} by heuristic), "
        f"{len(paper.links)} links"
    )


async def _load_paper(
    source: Path, settings: Settings, http: HttpClient, *, refresh: bool = False
) -> ParsedPaper:
    if not await asyncio.to_thread(source.is_file):
        raise UsageError(f"No such file: {source}")
    if source.suffix.lower() == ".json":
        return ParsedPaper.model_validate_json(await asyncio.to_thread(source.read_bytes))
    grobid = GrobidClient(http, settings.grobid_url)
    if not await grobid.is_alive():
        raise UsageError(
            f"GROBID is not reachable at {settings.grobid_url}. Start it with: docker compose up -d"
        )
    try:
        return await parse_pdf(source, grobid=grobid, cache_dir=settings.cache_dir, refresh=refresh)
    except ParseError as exc:
        raise UsageError(f"Could not parse: {exc}") from exc


def _load_venue(path: Path | None) -> VenueProfile | None:
    if path is None:
        return None
    if not path.is_file():
        raise UsageError(f"No such venue profile: {path}")
    return VenueProfile.model_validate_json(path.read_bytes())


def _repository(value: str | None, paper: ParsedPaper) -> RepositoryRef | None:
    if value is None:
        return find_repository(paper)
    url = repository_url(value)
    if url is None:
        raise UsageError(f"Not a GitHub or GitLab repository URL: {value}")
    return RepositoryRef(url=url)


def _optional_llm(settings: Settings, enabled: bool) -> StructuredLLM | None:
    if not enabled:
        return None
    try:
        return StructuredLLM.from_settings(settings)
    except LLMNotConfiguredError:
        logger.warning("no LLM configured; the planner runs without its LLM pass")
        return None


async def _parse(args: argparse.Namespace, settings: Settings) -> int:
    async with HttpClient(settings) as http:
        paper = await _load_paper(args.pdf, settings, http, refresh=args.refresh)
    if args.output is not None:
        atomic_write(args.output, (paper.model_dump_json(indent=1) + "\n").encode("utf-8"))
    print(_summary(paper))
    return 0


async def _audit(args: argparse.Namespace, settings: Settings) -> int:
    if args.resume and not args.run_id:
        raise UsageError("--resume needs the --run-id of the run to continue")
    run_id = args.run_id or new_run_id(SystemName.FULL)
    run = RunDirectory(settings.runs_dir, run_id)
    configure_logging(settings.log_level, run.path / "run.log")
    usage = UsageLog(run.usage_dir)
    venue = _load_venue(args.venue)

    async with HttpClient(settings) as http:
        paper = await _load_paper(args.input, settings, http)
        repository = _repository(args.repo, paper)
        llm = _optional_llm(settings, not args.no_llm)
        ctx = RunContext(settings=settings, paper=paper, http=http, llm=llm, logger=logger)
        specialists, mocked = load_specialists()
        verifier, mock_verifier = load_verifier()
        pipeline = AuditPipeline(
            ctx=ctx,
            planner=make_planner(
                PlannerOptions(repository=repository, venue=venue, use_llm=llm is not None)
            ),
            specialists=specialists,
            verifier=verifier,
            max_concurrency=args.max_concurrency,
        )
        manifest = build_manifest(
            run_id=run_id,
            system=SystemName.FULL,
            settings=settings,
            model=llm.name if llm else None,
            extra_config={
                "paper_id": paper.paper_id,
                "variant": paper.variant.value,
                "repository": repository.url if repository else None,
                "venue": venue.id if venue else None,
                "mocked_specialists": list(mocked),
                "mocked_verifier": mock_verifier,
            },
        )
        write_manifest(manifest, run.path)
        async with AsyncSqliteSaver.from_conn_string(str(run.path / "checkpoints.sqlite")) as saver:
            try:
                outcome = await run_audit(
                    pipeline,
                    paper,
                    run_id=run_id,
                    system=SystemName.FULL,
                    usage=usage,
                    checkpointer=saver,
                    resume=args.resume,
                )
            except (ThreadInUseError, NothingToResumeError) as exc:
                raise UsageError(str(exc)) from exc

    predictions = predictions_from_verified(outcome.verified, run_id=run_id, system=SystemName.FULL)
    folder = run.save_audit(paper, outcome, predictions)
    write_manifest(manifest.model_copy(update={"finished_at": datetime.now(UTC)}), run.path)

    outcomes = {o: sum(1 for r in outcome.results if r.outcome is o) for o in TaskOutcome}
    states = {s: sum(1 for v in outcome.verified if v.state is s) for s in VerificationState}
    print(
        f"{run_id}: {len(outcome.tasks)} tasks "
        f"({outcomes[TaskOutcome.CHECKED]} checked, {outcomes[TaskOutcome.UNVERIFIABLE]} "
        f"unverifiable, {outcomes[TaskOutcome.FAILED]} failed); "
        f"{states[VerificationState.VERIFIED_FLAG]} verified flags, "
        f"{states[VerificationState.NEEDS_HUMAN_REVIEW]} for human review, "
        f"{states[VerificationState.DROPPED]} dropped. Output: {folder}"
    )
    return 0


async def _baseline(args: argparse.Namespace, settings: Settings) -> int:
    system = SystemName(args.system)
    run_id = args.run_id or new_run_id(system)
    run = RunDirectory(settings.runs_dir, run_id)
    configure_logging(settings.log_level, run.path / "run.log")
    usage = UsageLog(run.usage_dir)
    venue = _load_venue(args.venue)
    try:
        llm = StructuredLLM.from_settings(settings)
    except LLMNotConfiguredError as exc:
        raise UsageError(f"Baselines need an LLM. {exc}") from exc

    async with HttpClient(settings) as http:
        paper = await _load_paper(args.input, settings, http)
        repository = _repository(args.repo, paper)
        manifest = build_manifest(
            run_id=run_id,
            system=system,
            settings=settings,
            model=llm.name,
            extra_config={"paper_id": paper.paper_id, "variant": paper.variant.value},
        )
        write_manifest(manifest, run.path)
        result: BaselineResult
        if system is SystemName.B0:
            result = await run_b0(
                paper, llm, run_id=run_id, usage=usage, venue=venue, repository=repository
            )
        else:
            ctx = RunContext(settings=settings, paper=paper, http=http, llm=llm, logger=logger)
            result = await run_b1(
                paper,
                create_chat_model(settings),
                load_tools(ctx),
                run_id=run_id,
                model_name=model_id(settings),
                usage=usage,
                venue=venue,
                repository=repository,
            )

    path = run.save_predictions(paper, result.predictions)
    write_manifest(manifest.model_copy(update={"finished_at": datetime.now(UTC)}), run.path)
    print(
        f"{run_id}: {len(result.predictions)} predictions "
        f"({len(result.dropped)} answers dropped as invalid). Output: {path}"
    )
    return 0


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    configure_logging(settings.log_level)
    commands = {"parse": _parse, "audit": _audit, "baseline": _baseline}
    try:
        return asyncio.run(commands[args.command](args, settings))
    except UsageError as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
