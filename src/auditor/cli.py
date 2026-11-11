"""Command-line entry point: `uv run auditor --help`."""

import argparse
import asyncio
import importlib.util
import json
import logging
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from auditor.baselines.common import IncompleteAnswerError
from auditor.config import Settings
from auditor.evaluation import EvaluationError, run_evaluation
from auditor.evaluation.runs import read_jsonl
from auditor.experiments import (
    BATCH_SYSTEMS,
    InputError,
    PaperInput,
    check_same_inputs,
    code_sha256,
    limit_failures,
    load_paper,
    load_venue,
    missing_parts,
    optional_llm,
    required_llm,
    resolve_repository,
    run_baseline,
    run_batch,
    run_fingerprint,
    run_full,
    start_manifest,
)
from auditor.infra.files import atomic_write
from auditor.infra.http import HttpClient
from auditor.infra.llm import closing_llm
from auditor.infra.log import configure_logging
from auditor.infra.manifest import build_manifest, new_run_id, write_manifest
from auditor.infra.usage import UsageLog
from auditor.orchestration.graph import NothingToResumeError, ThreadInUseError
from auditor.orchestration.outputs import RunDirectory
from auditor.reporting import build_report, write_report
from auditor.schemas import (
    AuditTask,
    LinkMethod,
    ParsedPaper,
    SystemName,
    TaskOutcome,
    TaskResult,
    VerificationState,
    VerifiedFinding,
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
    started = (run.path / "manifest.json").is_file()
    if args.resume and not started:
        raise InputError(f"{run_id} has no run to resume")
    if started and not args.resume:
        raise InputError(
            f"{run_id} already exists. Pass --resume to continue it, or choose another --run-id."
        )
    venue = load_venue(args.venue)

    async with HttpClient(settings) as http:
        paper = await load_paper(args.input, settings, http)
        repository = resolve_repository(args.repo, paper, args.repo_commit)
        llm = optional_llm(settings, not args.no_llm)
        missing = missing_parts(SystemName.FULL, settings.skip_parts)
        inputs = run_fingerprint(
            SystemName.FULL,
            settings,
            llm,
            [PaperInput(paper=paper, repository=repository, venue=venue)],
            missing,
            code_sha256(),
        )
        # Checked before anything is written, so a refused resume leaves the run as it was.
        check_same_inputs(run, run_id, inputs, "Use a new --run-id")
        configure_logging(settings.log_level, run.path / "run.log")
        manifest = start_manifest(
            run,
            run_id,
            SystemName.FULL,
            settings,
            llm.name if llm else None,
            {
                "paper_id": paper.paper_id,
                "variant": paper.variant.value,
                "repository": repository.url if repository else None,
                "repository_commit": repository.commit if repository else None,
                "venue": venue.id if venue else None,
                "mocked_parts": list(missing),
                "inputs": inputs,
            },
        )
        async with (
            AsyncSqliteSaver.from_conn_string(str(run.path / "checkpoints.sqlite")) as saver,
            closing_llm(llm),
        ):
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
                    source_pdf=args.input if args.input.suffix.lower() == ".pdf" else None,
                )
            except (ThreadInUseError, NothingToResumeError) as exc:
                raise InputError(str(exc)) from exc
    write_manifest(manifest.model_copy(update={"finished_at": datetime.now(UTC)}), run.path)

    outcome = result.outcome
    outcomes = {o: sum(1 for r in outcome.results if r.outcome is o) for o in TaskOutcome}
    states = {s: sum(1 for v in outcome.verified if v.state is s) for s in VerificationState}
    duplicates = sum(1 for v in outcome.verified if v.duplicate_of is not None)
    print(
        f"{run_id}: {len(outcome.tasks)} tasks "
        f"({outcomes[TaskOutcome.CHECKED]} checked, {outcomes[TaskOutcome.UNVERIFIABLE]} "
        f"unverifiable, {outcomes[TaskOutcome.FAILED]} failed); "
        f"{states[VerificationState.VERIFIED_FLAG]} verified flags, "
        f"{states[VerificationState.NEEDS_HUMAN_REVIEW]} for human review, "
        f"{states[VerificationState.DROPPED]} dropped"
        + (f" ({duplicates} repeat an earlier finding)" if duplicates else "")
        + f". Report: {result.folder / 'report.md'}"
    )
    hit = limit_failures(outcome.results)
    if hit:
        print(
            f"{hit} tasks failed because the run's model calls were used up "
            "(AUDITOR_LLM_MAX_CALLS); the results are incomplete",
            file=sys.stderr,
        )
        return 1
    return 0


async def _baseline(args: argparse.Namespace, settings: Settings) -> int:
    system = SystemName(args.system)
    run_id = args.run_id or new_run_id(system)
    run = RunDirectory(settings.runs_dir, run_id)
    if (run.path / "manifest.json").is_file():
        raise InputError(f"{run_id} already exists; choose another --run-id.")
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
                "mocked_parts": list(missing_parts(system, settings.skip_parts)),
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
        # Reported runs reuse an answer only within the run that made it (decision D10).
        settings = settings.model_copy(update={"llm_cache": True, "llm_cache_scope": "run"})
    if args.no_llm:
        if set(args.systems) - {SystemName.FULL.value}:
            raise InputError("--no-llm: baselines need an LLM; run them separately")
        settings = settings.model_copy(update={"llm_provider": None})
    summaries = await run_batch(
        args.dataset,
        experiment=args.experiment,
        systems=[SystemName(s) for s in args.systems],
        runs=args.runs,
        settings=settings,
        strict=args.strict,
        max_concurrency=args.max_concurrency,
        paper_concurrency=args.paper_concurrency,
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


async def _evaluate(args: argparse.Namespace, settings: Settings) -> int:
    remap = None
    if args.remap:
        old, sep, new = args.remap.partition("=")
        if not sep or not old:
            raise InputError("--remap takes OLD=NEW")
        remap = (old, new)
    try:
        markdown, report = await run_evaluation(
            args.labels,
            args.runs,
            settings=settings,
            dataset=args.dataset,
            remap=remap,
            areas=args.area or (),
            strict=args.strict,
            ablate_verifier=not args.no_ablation,
            title=args.title,
        )
    except EvaluationError as exc:
        raise InputError(str(exc)) from exc
    if args.markdown is not None:
        atomic_write(args.markdown, markdown.encode("utf-8"))
    if args.json is not None:
        atomic_write(args.json, (json.dumps(report, indent=1) + "\n").encode("utf-8"))
    print(markdown)
    return 0


async def _report(args: argparse.Namespace, settings: Settings) -> int:
    folder: Path = args.folder
    verified = read_jsonl(folder / "verified.jsonl", VerifiedFinding)
    results = read_jsonl(folder / "results.jsonl", TaskResult)
    if not verified and not results:
        raise InputError(f"{folder}: no verified.jsonl or results.jsonl (a full-system run?)")
    async with HttpClient(settings) as http:
        paper = await load_paper(args.paper, settings, http)
    run_id = folder.parent.name if (folder.parent / "manifest.json").is_file() else None
    report = build_report(
        paper,
        read_jsonl(folder / "tasks.jsonl", AuditTask),
        results,
        verified,
        run_id=run_id,
    )
    path = write_report(args.output or folder, report)
    print(
        f"{len(report.verified)} verified flags, {len(report.review)} for human review, "
        f"{len(report.not_checked)} not checked, {len(report.dropped)} set aside. Report: {path}"
    )
    return 0


async def _ui(args: argparse.Namespace, settings: Settings) -> int:
    if importlib.util.find_spec("streamlit") is None:
        raise InputError("The UI needs Streamlit: run `uv sync --extra ui` first.")
    app = Path(__file__).parent / "ui" / "app.py"
    # Only this computer can reach it (an audit spends the team's credits), and Streamlit's
    # usage statistics are off.
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address",
        "localhost",
        "--server.port",
        str(args.port),
        "--browser.gatherUsageStats",
        "false",
    ]
    if not args.browser:
        command += ["--server.headless", "true"]
    # Streamlit runs until the user stops it; nothing else runs in this event loop.
    return await asyncio.to_thread(subprocess.call, command)


_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _name(value: str) -> str:
    """A run id or experiment name, used as a folder name on every operating system."""
    if not _NAME.fullmatch(value):
        raise argparse.ArgumentTypeError(
            f"{value!r}: use letters, digits, '.', '_' and '-', starting with a letter or digit"
        )
    return value


def _positive(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number < 1:
        raise argparse.ArgumentTypeError(f"{value!r}: must be a whole number of at least 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auditor", description="Audit research papers.")
    commands = parser.add_subparsers(dest="command", required=True)

    parse = commands.add_parser("parse", help="Parse a PDF into ParsedPaper JSON (needs GROBID)")
    parse.add_argument("pdf", type=Path)
    parse.add_argument("-o", "--output", type=Path, help="Write the ParsedPaper JSON here")
    parse.add_argument("--refresh", action="store_true", help="Ignore the parse cache")

    def add_common(command: argparse.ArgumentParser) -> None:
        command.add_argument("input", type=Path, help="A PDF, or ParsedPaper JSON")
        command.add_argument(
            "--repo",
            help=(
                "Code or data to check against: a GitHub, GitLab, Hugging Face, Kaggle dataset, "
                "Anonymous GitHub or Zenodo link, or a local folder (default: found in the paper)"
            ),
        )
        command.add_argument(
            "--repo-commit",
            help="Commit, tag or branch (Kaggle: version number) to check against",
        )
        command.add_argument(
            "--analyse-data",
            action="store_true",
            help=(
                "Download the dataset's data files to count rows, read labels and look for "
                "test rows that also occur in training data (slow; part of the run settings)"
            ),
        )
        command.add_argument("--venue", type=Path, help="Venue profile JSON")
        command.add_argument("--run-id", type=_name, help="Run id (default: generated)")

    audit = commands.add_parser("audit", help="Run the full audit on one paper")
    add_common(audit)
    audit.add_argument("--resume", action="store_true", help="Continue an interrupted run")
    audit.add_argument(
        "--no-llm",
        action="store_true",
        help="Run without the LLM, even when one is configured; checks that need it are "
        "reported unverifiable",
    )
    audit.add_argument("--max-concurrency", type=_positive, default=8)

    baseline = commands.add_parser("baseline", help="Run baseline B0 or B1 on one paper")
    baseline.add_argument("system", choices=[SystemName.B0.value, SystemName.B1.value])
    add_common(baseline)

    batch = commands.add_parser(
        "batch", help="Run systems over a dataset several times (resumable)"
    )
    batch.add_argument("dataset", type=Path, help="JSON Lines file of DatasetEntry records")
    batch.add_argument("--experiment", required=True, type=_name, help="Name used in the run ids")
    batch.add_argument(
        "--systems",
        nargs="+",
        default=[s.value for s in BATCH_SYSTEMS],
        choices=[s.value for s in BATCH_SYSTEMS],
    )
    batch.add_argument(
        "--runs", type=_positive, default=3, help="Repetitions of each system (default 3)"
    )
    batch.add_argument(
        "--llm-cache",
        action="store_true",
        help="Reuse cached LLM answers across runs (development only; by default an answer "
        "is reused only within the run that made it)",
    )
    batch.add_argument(
        "--strict", action="store_true", help="Refuse to run while any part is still a mock"
    )
    batch.add_argument("--max-concurrency", type=_positive, default=8)
    batch.add_argument(
        "--paper-concurrency",
        type=_positive,
        default=1,
        help="Papers processed at the same time (outputs do not depend on it)",
    )
    batch.add_argument(
        "--analyse-data",
        action="store_true",
        help="Run data analysis mode for every system (see audit --help)",
    )
    batch.add_argument(
        "--no-llm",
        action="store_true",
        help="Run the full system without the LLM, even when one is configured; checks that "
        "need it are reported unverifiable (part of the run settings)",
    )

    evaluate = commands.add_parser(
        "evaluate", help="Score runs against gold labels (Macro-F1, false alarms, grounding, cost)"
    )
    evaluate.add_argument("labels", type=Path, help="JSON Lines file of GoldLabel records")
    evaluate.add_argument(
        "runs", type=Path, nargs="+", help="Run folders, or folders that hold run folders"
    )
    evaluate.add_argument(
        "--dataset", type=Path, help="The runs' dataset file, to load papers for grounding"
    )
    evaluate.add_argument(
        "--remap", metavar="OLD=NEW", help="Rewrite a path prefix in runs made on another machine"
    )
    evaluate.add_argument(
        "--area",
        action="append",
        choices=["citation", "code", "data", "theory", "integrity", "template"],
        help="Score only this area's classes (repeatable)",
    )
    evaluate.add_argument(
        "--strict", action="store_true", help="Refuse runs in which a part was a placeholder"
    )
    evaluate.add_argument(
        "--no-ablation",
        action="store_true",
        help="Do not also score full-system runs without their verifier",
    )
    evaluate.add_argument("--title", default="Evaluation", help="Heading of the Markdown")
    evaluate.add_argument("--markdown", type=Path, help="Write the tables here")
    evaluate.add_argument("--json", type=Path, help="Write the numbers here")

    report = commands.add_parser(
        "report", help="Write the audit report for one paper of a finished full-system run"
    )
    report.add_argument("folder", type=Path, help="The paper's folder in the run")
    report.add_argument("--paper", type=Path, required=True, help="The paper: PDF or JSON")
    report.add_argument("-o", "--output", type=Path, help="Folder to write into (default: folder)")

    ui = commands.add_parser("ui", help="Open the web UI to run audits and read reports")
    ui.add_argument("--port", type=_positive, default=8501)
    ui.add_argument(
        "--no-browser", dest="browser", action="store_false", help="Do not open a browser tab"
    )
    return parser


def settings_for_args(args: argparse.Namespace, settings: Settings) -> Settings:
    """Settings with the command's switches applied. Switches that change results go
    through Settings, so the run fingerprint covers them (decision D10)."""
    if getattr(args, "analyse_data", False):
        settings = settings.model_copy(update={"data_analysis_enabled": True})
    return settings


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = settings_for_args(args, Settings())
    configure_logging(settings.log_level)
    commands = {
        "parse": _parse,
        "audit": _audit,
        "baseline": _baseline,
        "batch": _batch,
        "evaluate": _evaluate,
        "report": _report,
        "ui": _ui,
    }
    try:
        return asyncio.run(commands[args.command](args, settings))
    except InputError as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
