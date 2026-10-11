"""Running the systems on papers: one paper (`auditor audit`, `auditor baseline`) or a whole
dataset several times (`auditor batch`).

A batch is resumable. Run ids are `<experiment>-<system>-r<n>`, a paper with
`predictions.jsonl` in its run directory is skipped, and a full-system paper that was
interrupted continues from its checkpoint. A paper that fails is recorded in
`failures.jsonl` and the batch moves on.
"""

import asyncio
import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import JsonValue, ValidationError

from auditor.baselines.b0 import BaselineResult, run_b0
from auditor.baselines.b1 import run_b1
from auditor.config import Settings
from auditor.infra.context import RunContext
from auditor.infra.http import HttpClient
from auditor.infra.llm import LLMNotConfiguredError, StructuredLLM, create_chat_model, model_id
from auditor.infra.log import configure_logging
from auditor.infra.manifest import build_manifest, write_manifest
from auditor.infra.usage import UsageLog
from auditor.orchestration.graph import AuditOutcome, AuditPipeline, ThreadInUseError, run_audit
from auditor.orchestration.outputs import RunDirectory, predictions_from_verified
from auditor.orchestration.registry import load_specialists, load_tools, load_verifier
from auditor.parser.grobid import GrobidClient
from auditor.parser.pipeline import ParseError, parse_pdf
from auditor.planner.repository import find_repository, repository_url, snapshot_ref
from auditor.planner.tasks import PlannerOptions, make_planner
from auditor.schemas import (
    ParsedPaper,
    Prediction,
    RepositoryRef,
    RunManifest,
    SystemName,
    VenueProfile,
)
from auditor.schemas.evaluation import DatasetEntry

logger = logging.getLogger(__name__)

BATCH_SYSTEMS = (SystemName.FULL, SystemName.B0, SystemName.B1)


class InputError(Exception):
    """A problem with an input (missing file, GROBID down, bad URL), reported plainly."""


async def load_paper(
    source: Path, settings: Settings, http: HttpClient, *, refresh: bool = False
) -> ParsedPaper:
    """A ParsedPaper from its JSON file, or by parsing a PDF with GROBID."""
    if not await asyncio.to_thread(source.is_file):
        raise InputError(f"No such file: {source}")
    if source.suffix.lower() == ".json":
        try:
            return ParsedPaper.model_validate_json(await asyncio.to_thread(source.read_bytes))
        except ValidationError as exc:
            raise InputError(f"{source} is not a valid ParsedPaper: {exc}") from exc
    grobid = GrobidClient(http, settings.grobid_url)
    if not await grobid.is_alive():
        raise InputError(
            f"GROBID is not reachable at {settings.grobid_url}. Start it with: docker compose up -d"
        )
    try:
        return await parse_pdf(source, grobid=grobid, cache_dir=settings.cache_dir, refresh=refresh)
    except ParseError as exc:
        raise InputError(f"Could not parse: {exc}") from exc


def load_venue(path: Path | None) -> VenueProfile | None:
    if path is None:
        return None
    if not path.is_file():
        raise InputError(f"No such venue profile: {path}")
    try:
        return VenueProfile.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        raise InputError(f"{path} is not a valid venue profile: {exc}") from exc


def resolve_repository(
    value: str | None, paper: ParsedPaper, commit: str | None = None
) -> RepositoryRef | None:
    """The repository to check against: the given URL, or the one the paper announces.
    The commit comes from `commit` or from a /tree/ or /commit/ part of the URL."""
    if value is None:
        found = find_repository(paper)
        if commit is None or found is None:
            if commit is not None:
                raise InputError("A repository commit was given, but no repository.")
            return found
        return found.model_copy(update={"commit": commit})
    url = repository_url(value)
    if url is None:
        raise InputError(f"Not a GitHub or GitLab repository URL: {value}")
    in_url = snapshot_ref(value)
    if commit is not None and in_url is not None and commit != in_url:
        raise InputError(f"{value} names {in_url}, but the commit given is {commit}")
    return RepositoryRef(url=url, commit=commit or in_url)


def optional_llm(settings: Settings, enabled: bool = True) -> StructuredLLM | None:
    if not enabled:
        return None
    try:
        return StructuredLLM.from_settings(settings)
    except LLMNotConfiguredError:
        logger.warning("no LLM configured; the planner runs without its LLM pass")
        return None


def required_llm(settings: Settings) -> StructuredLLM:
    try:
        return StructuredLLM.from_settings(settings)
    except LLMNotConfiguredError as exc:
        raise InputError(f"Baselines need an LLM. {exc}") from exc


@dataclass(frozen=True)
class FullRun:
    outcome: AuditOutcome
    predictions: list[Prediction]
    folder: Path


async def run_full(
    paper: ParsedPaper,
    *,
    settings: Settings,
    http: HttpClient,
    llm: StructuredLLM | None,
    run: RunDirectory,
    run_id: str,
    usage: UsageLog | None,
    checkpointer: BaseCheckpointSaver[Any] | None,
    repository: RepositoryRef | None,
    venue: VenueProfile | None,
    max_concurrency: int = 8,
    resume: bool = False,
) -> FullRun:
    ctx = RunContext(
        settings=settings,
        paper=paper,
        http=http,
        llm=llm,
        logger=logger,
        repository=repository,
    )
    specialists, _ = load_specialists()
    verifier, _ = load_verifier()
    pipeline = AuditPipeline(
        ctx=ctx,
        planner=make_planner(
            PlannerOptions(repository=repository, venue=venue, use_llm=llm is not None)
        ),
        specialists=specialists,
        verifier=verifier,
        max_concurrency=max_concurrency,
    )
    outcome = await run_audit(
        pipeline,
        paper,
        run_id=run_id,
        system=SystemName.FULL,
        usage=usage,
        checkpointer=checkpointer,
        resume=resume,
    )
    predictions = predictions_from_verified(outcome.verified, run_id=run_id, system=SystemName.FULL)
    folder = run.save_audit(paper, outcome, predictions)
    return FullRun(outcome=outcome, predictions=predictions, folder=folder)


async def run_baseline(
    system: SystemName,
    paper: ParsedPaper,
    *,
    settings: Settings,
    http: HttpClient,
    llm: StructuredLLM,
    run: RunDirectory,
    run_id: str,
    usage: UsageLog | None,
    repository: RepositoryRef | None,
    venue: VenueProfile | None,
) -> tuple[BaselineResult, Path]:
    result: BaselineResult
    if system is SystemName.B0:
        result = await run_b0(
            paper, llm, run_id=run_id, usage=usage, venue=venue, repository=repository
        )
    elif system is SystemName.B1:
        ctx = RunContext(
            settings=settings,
            paper=paper,
            http=http,
            llm=llm,
            logger=logger,
            repository=repository,
        )
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
    else:
        raise ValueError(f"{system.value} is not a baseline")
    return result, run.save_predictions(paper, result.predictions)


def mocked_parts() -> list[str]:
    _, mocked = load_specialists()
    _, verifier_mocked = load_verifier()
    return [*mocked, *(["verifier"] if verifier_mocked else [])]


def read_dataset(path: Path) -> list[DatasetEntry]:
    """JSON Lines, one entry per paper; relative paths are relative to the dataset file."""
    entries = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = DatasetEntry.model_validate_json(line)
        except ValidationError as exc:
            raise InputError(f"{path}:{number}: {exc}") from exc
        base = path.parent
        entries.append(
            entry.model_copy(
                update={
                    "paper": str(base / entry.paper),
                    "venue": str(base / entry.venue) if entry.venue else None,
                }
            )
        )
    if not entries:
        raise InputError(f"{path} has no entries")
    return entries


@dataclass
class RunSummary:
    run_id: str
    done: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Prepared:
    paper: ParsedPaper
    repository: RepositoryRef | None
    venue: VenueProfile | None


def _record_failure(run: RunDirectory, paper: ParsedPaper, error: BaseException) -> str:
    message = f"{paper.paper_id}-{paper.variant.value}: {type(error).__name__}: {error}"
    record = {
        "paper_id": paper.paper_id,
        "variant": paper.variant.value,
        "error": f"{type(error).__name__}: {error}"[:2000],
        "at": datetime.now(UTC).isoformat(),
    }
    with (run.path / "failures.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record) + "\n")
    return message


def _start_manifest(
    run: RunDirectory,
    run_id: str,
    system: SystemName,
    settings: Settings,
    model: str | None,
    extra: dict[str, JsonValue],
) -> RunManifest:
    existing = run.path / "manifest.json"
    if existing.is_file():
        # A resumed run keeps its original start time and configuration.
        return RunManifest.model_validate_json(existing.read_bytes()).model_copy(
            update={"finished_at": None}
        )
    manifest = build_manifest(
        run_id=run_id,
        system=system,
        settings=settings,
        model=model,
        extra_config=extra,
    )
    write_manifest(manifest, run.path)
    return manifest


async def _run_one(
    system: SystemName,
    item: _Prepared,
    *,
    settings: Settings,
    http: HttpClient,
    llm: StructuredLLM | None,
    run: RunDirectory,
    run_id: str,
    usage: UsageLog,
    checkpointer: BaseCheckpointSaver[Any] | None,
    max_concurrency: int,
) -> None:
    if system is SystemName.FULL:
        # A paper whose thread already has state was interrupted earlier; continue it.
        for resume in (False, True):
            try:
                await run_full(
                    item.paper,
                    settings=settings,
                    http=http,
                    llm=llm,
                    run=run,
                    run_id=run_id,
                    usage=usage,
                    checkpointer=checkpointer,
                    repository=item.repository,
                    venue=item.venue,
                    max_concurrency=max_concurrency,
                    resume=resume,
                )
                return
            except ThreadInUseError:
                if resume:
                    raise
        return
    if llm is None:
        raise InputError("Baselines need an LLM.")
    await run_baseline(
        system,
        item.paper,
        settings=settings,
        http=http,
        llm=llm,
        run=run,
        run_id=run_id,
        usage=usage,
        repository=item.repository,
        venue=item.venue,
    )


async def run_batch(
    dataset: Path,
    *,
    experiment: str,
    systems: Sequence[SystemName],
    runs: int,
    settings: Settings,
    strict: bool = False,
    max_concurrency: int = 8,
) -> list[RunSummary]:
    entries = read_dataset(dataset)
    mocked = mocked_parts()
    if strict and mocked and set(systems) & {SystemName.FULL, SystemName.B1}:
        raise InputError(f"--strict: these parts are still mocks: {', '.join(mocked)}")
    needs_llm = bool(set(systems) & {SystemName.B0, SystemName.B1})
    llm = required_llm(settings) if needs_llm else optional_llm(settings)
    dataset_hash = hashlib.sha256(await asyncio.to_thread(dataset.read_bytes)).hexdigest()

    summaries = []
    async with HttpClient(settings) as http:
        prepared = []
        seen = set()
        for entry in entries:
            paper = await load_paper(Path(entry.paper), settings, http)
            key = (paper.paper_id, paper.variant)
            if key in seen:
                raise InputError(
                    f"{dataset}: {paper.paper_id} ({paper.variant.value}) is listed twice"
                )
            seen.add(key)
            prepared.append(
                _Prepared(
                    paper=paper,
                    repository=resolve_repository(entry.repository, paper, entry.repository_commit),
                    venue=load_venue(Path(entry.venue) if entry.venue else None),
                )
            )

        for system in systems:
            for index in range(1, runs + 1):
                run_id = f"{experiment}-{system.value}-r{index}"
                run = RunDirectory(settings.runs_dir, run_id)
                configure_logging(settings.log_level, run.path / "run.log")
                usage = UsageLog(run.usage_dir)
                manifest = _start_manifest(
                    run,
                    run_id,
                    system,
                    settings,
                    llm.name if llm else None,
                    {
                        "experiment": experiment,
                        "dataset": str(dataset),
                        "dataset_sha256": dataset_hash,
                        "run_index": index,
                        "papers": len(prepared),
                        "mocked_parts": list(mocked),
                    },
                )
                summary = RunSummary(run_id=run_id)
                async with AsyncSqliteSaver.from_conn_string(
                    str(run.path / "checkpoints.sqlite")
                ) as saver:
                    for item in prepared:
                        if (run.paper_dir(item.paper) / "predictions.jsonl").is_file():
                            summary.skipped += 1
                            continue
                        try:
                            await _run_one(
                                system,
                                item,
                                settings=settings,
                                http=http,
                                llm=llm,
                                run=run,
                                run_id=run_id,
                                usage=usage,
                                checkpointer=saver if system is SystemName.FULL else None,
                                max_concurrency=max_concurrency,
                            )
                        except Exception as exc:
                            logger.exception("%s failed on %s", run_id, item.paper.paper_id)
                            summary.failed.append(_record_failure(run, item.paper, exc))
                            continue
                        summary.done += 1
                write_manifest(
                    manifest.model_copy(update={"finished_at": datetime.now(UTC)}), run.path
                )
                summaries.append(summary)
    return summaries
