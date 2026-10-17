"""Running the systems on papers: one paper (`auditor audit`, `auditor baseline`) or a whole
dataset several times (`auditor batch`).

A batch is resumable. Run ids are `<experiment>-<system>-r<n>`, a paper with
`predictions.jsonl` in its run directory is skipped, and a full-system paper that was
interrupted continues from its checkpoint. A paper that fails is recorded in
`failures.jsonl` and the batch moves on; running the batch again retries it.

Each run records a fingerprint of what produced it: model and LLM settings, a hash of
the auditor's source code, package versions, and every paper, repository and venue.
Continuing a run with a different fingerprint is refused, so one run never mixes
configurations.
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

import auditor
from auditor.agents.theory.review import review_math
from auditor.baselines.b0 import BaselineResult, run_b0
from auditor.baselines.b1 import run_b1
from auditor.config import Settings
from auditor.infra.context import RunContext
from auditor.infra.http import HttpClient
from auditor.infra.llm import (
    CallLimitReachedError,
    LLMNotConfiguredError,
    StructuredLLM,
    create_chat_model,
    model_id,
)
from auditor.infra.log import configure_logging
from auditor.infra.manifest import build_manifest, package_versions, redact, write_manifest
from auditor.infra.snapshots import SourceError, parse_source
from auditor.infra.usage import UsageLog
from auditor.orchestration.graph import AuditOutcome, AuditPipeline, ThreadInUseError, run_audit
from auditor.orchestration.outputs import RunDirectory, predictions_from_verified
from auditor.orchestration.registry import (
    load_specialists,
    load_tools,
    load_verifier,
    missing_tools,
)
from auditor.parser.grobid import GrobidClient
from auditor.parser.pipeline import ParseError, parse_pdf
from auditor.planner.repository import find_repository
from auditor.planner.tasks import PlannerOptions, make_planner
from auditor.schemas import (
    ParsedPaper,
    Prediction,
    RepositoryRef,
    RunManifest,
    SystemName,
    TaskResult,
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
    """The code or data to check against: the given source (a GitHub, GitLab, Hugging Face,
    Kaggle, Anonymous GitHub or Zenodo link, or a local folder), or the repository the paper
    announces. The commit comes from `commit` or from a /tree/ or /commit/ part of the URL."""
    if value is None:
        found = find_repository(paper)
        if commit is None or found is None:
            if commit is not None:
                raise InputError("A repository commit was given, but no repository.")
            return found
        return found.model_copy(update={"commit": commit})
    try:
        spec = parse_source(value, commit)
    except SourceError as exc:
        raise InputError(str(exc)) from exc
    return RepositoryRef(url=spec.url, commit=spec.ref)


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
        review_llm=llm.review if llm is not None else None,
    )
    specialists, _ = load_specialists()
    verifier, _ = load_verifier()
    reviews = (review_math,) if settings.theory_check_enabled else ()
    pipeline = AuditPipeline(
        ctx=ctx,
        planner=make_planner(
            PlannerOptions(repository=repository, venue=venue, use_llm=llm is not None)
        ),
        specialists=specialists,
        verifier=verifier,
        reviews=reviews,
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


def missing_parts(system: SystemName) -> list[str]:
    """Parts a system needs that are still placeholders: the specialists and verifier for
    the full system, the specialists' tools for B1 (decision D7). B0 needs none."""
    if system is SystemName.FULL:
        _, mocked = load_specialists()
        _, verifier_mocked = load_verifier()
        return [*mocked, *(["verifier"] if verifier_mocked else [])]
    if system is SystemName.B1:
        return [f"{name} tools" for name in missing_tools()]
    return []


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
        repository = entry.repository
        # A local repository folder, such as a patched copy, is relative to the dataset too.
        if repository and "://" not in repository and (base / repository).is_dir():
            repository = str(base / repository)
        entries.append(
            entry.model_copy(
                update={
                    "paper": str(base / entry.paper),
                    "venue": str(base / entry.venue) if entry.venue else None,
                    "repository": repository,
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
class PaperInput:
    """A paper with the repository and venue it is checked against."""

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


def _sha256(data: str | bytes) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def code_sha256() -> str:
    """Hash of the auditor package's source files, including the specialists' code."""
    root = Path(auditor.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


# Settings that only decide where files go and what is logged. Every other setting is
# part of a run's fingerprint, including ones added later. The GROBID address is left
# out because the parsed paper itself is hashed; host tokens only authenticate requests.
_OPERATIONAL_SETTINGS = frozenset(
    {
        "data_dir",
        "cache_dir",
        "runs_dir",
        "scratch_dir",
        "grobid_url",
        "log_level",
        "contact_email",
        "github_token",
        "hf_token",
        # Limits on how fast or how much a run calls the model; raising them to resume a
        # run that hit them does not change any result.
        "llm_max_concurrency",
        "llm_max_calls",
    }
)


def run_fingerprint(
    system: SystemName,
    settings: Settings,
    llm: StructuredLLM | None,
    prepared: Sequence[PaperInput],
    missing: list[str],
    code: str,
) -> dict[str, JsonValue]:
    """Everything that can change a run's answers, compared when a run is continued.
    `missing` is missing_parts(system) and `code` is code_sha256(), passed in so a batch
    computes them once."""
    papers: list[JsonValue] = [
        {
            "paper_id": item.paper.paper_id,
            "variant": item.paper.variant.value,
            "paper_sha256": _sha256(item.paper.model_dump_json()),
            "repository": item.repository.model_dump(mode="json") if item.repository else None,
            "venue_sha256": _sha256(item.venue.model_dump_json()) if item.venue else None,
        }
        for item in prepared
    ]
    fingerprint: dict[str, JsonValue] = {
        "system": system.value,
        "model": llm.name if llm else None,
        "settings": settings.model_dump(mode="json", exclude=set(_OPERATIONAL_SETTINGS)),
        "code_sha256": code,
        "packages": dict(package_versions()),
        "missing_parts": list(missing),
        "papers": papers,
    }
    # Stored manifests are redacted JSON; compare in the same form.
    stored: dict[str, JsonValue] = redact(json.loads(json.dumps(fingerprint)))
    return stored


def check_same_inputs(
    run: RunDirectory, run_id: str, fingerprint: dict[str, JsonValue], advice: str
) -> None:
    """Refuse to continue a run that was started with different inputs."""
    existing = run.path / "manifest.json"
    if not existing.is_file():
        return
    before = RunManifest.model_validate_json(existing.read_bytes()).config.get("inputs")
    if before == fingerprint:
        return
    if isinstance(before, dict):
        changed = sorted(
            k for k in fingerprint.keys() | before.keys() if before.get(k) != fingerprint.get(k)
        )
    else:
        changed = ["(not recorded)"]
    raise InputError(
        f"{run_id} was started with different {', '.join(changed)}. {advice}, or delete "
        f"{run.path} to start that run again."
    )


def start_manifest(
    run: RunDirectory,
    run_id: str,
    system: SystemName,
    settings: Settings,
    model: str | None,
    extra: dict[str, JsonValue],
) -> RunManifest:
    existing = run.path / "manifest.json"
    if existing.is_file():
        # A continued run keeps its original start time; check_same_inputs has already
        # confirmed that its configuration is unchanged.
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
    item: PaperInput,
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
                full = await run_full(
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
            except ThreadInUseError:
                if resume:
                    raise
                continue
            _refuse_incomplete(full.outcome.results, llm)
            return
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


def limit_failures(results: Sequence[TaskResult]) -> int:
    """Tasks that failed because the run's model calls were used up."""
    prefix = f"{CallLimitReachedError.__name__}:"
    return sum(1 for r in results if r.error is not None and r.error.startswith(prefix))


def _refuse_incomplete(results: Sequence[TaskResult], llm: StructuredLLM | None) -> None:
    """A paper whose checks ran out of model calls must not pass as done: its missing
    findings would read as clean results."""
    hit = limit_failures(results)
    if hit:
        limit = llm.max_calls if llm is not None else None
        raise CallLimitReachedError(
            f"incomplete: {hit} of {len(results)} tasks hit the run's limit of {limit} model "
            "calls (AUDITOR_LLM_MAX_CALLS); rerun under a new --experiment with a higher limit"
        )


async def _run_papers(
    prepared: Sequence[PaperInput],
    system: SystemName,
    summary: RunSummary,
    *,
    settings: Settings,
    http: HttpClient,
    llm: StructuredLLM | None,
    run: RunDirectory,
    run_id: str,
    usage: UsageLog,
    checkpointer: BaseCheckpointSaver[Any] | None,
    max_concurrency: int,
    paper_concurrency: int,
) -> None:
    """Runs the papers of one run, `paper_concurrency` at a time, recording the outcome of
    each in `summary`. A failure is kept and the others go on."""
    gate = asyncio.Semaphore(paper_concurrency)

    async def one(item: PaperInput) -> None:
        if (run.paper_dir(item.paper) / "predictions.jsonl").is_file():
            summary.skipped += 1
            return
        async with gate:
            if llm is not None and llm.limit_reached:
                error = CallLimitReachedError(
                    f"not started: the run's {llm.max_calls} model calls are used up "
                    "(AUDITOR_LLM_MAX_CALLS)"
                )
                summary.failed.append(_record_failure(run, item.paper, error))
                return
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
                    checkpointer=checkpointer,
                    max_concurrency=max_concurrency,
                )
            except Exception as exc:
                logger.exception("%s failed on %s", run_id, item.paper.paper_id)
                summary.failed.append(_record_failure(run, item.paper, exc))
                return
        summary.done += 1

    await asyncio.gather(*(one(item) for item in prepared))


async def run_batch(
    dataset: Path,
    *,
    experiment: str,
    systems: Sequence[SystemName],
    runs: int,
    settings: Settings,
    strict: bool = False,
    max_concurrency: int = 8,
    paper_concurrency: int = 1,
) -> list[RunSummary]:
    """`paper_concurrency` papers run at the same time; each paper's results are written to
    its own folder, so the outputs do not depend on it."""
    entries = read_dataset(dataset)
    missing = {system: missing_parts(system) for system in systems}
    if strict and any(missing.values()):
        problems = "; ".join(f"{s.value}: {', '.join(p)}" for s, p in missing.items() if p)
        raise InputError(f"--strict: these parts are still mocks or missing: {problems}")
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
                PaperInput(
                    paper=paper,
                    repository=resolve_repository(entry.repository, paper, entry.repository_commit),
                    venue=load_venue(Path(entry.venue) if entry.venue else None),
                )
            )

        code = code_sha256()
        fingerprints = {
            system: run_fingerprint(system, settings, llm, prepared, missing[system], code)
            for system in systems
        }
        # Refuse before doing any work if an existing run was made differently.
        for system in systems:
            for index in range(1, runs + 1):
                run_id = f"{experiment}-{system.value}-r{index}"
                check_same_inputs(
                    RunDirectory(settings.runs_dir, run_id),
                    run_id,
                    fingerprints[system],
                    "Use a new --experiment name",
                )

        for system in systems:
            for index in range(1, runs + 1):
                run_id = f"{experiment}-{system.value}-r{index}"
                run = RunDirectory(settings.runs_dir, run_id)
                configure_logging(settings.log_level, run.path / "run.log")
                usage = UsageLog(run.usage_dir)
                manifest = start_manifest(
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
                        "mocked_parts": list(missing[system]),
                        "inputs": fingerprints[system],
                    },
                )
                summary = RunSummary(run_id=run_id)
                async with AsyncSqliteSaver.from_conn_string(
                    str(run.path / "checkpoints.sqlite")
                ) as saver:
                    await _run_papers(
                        prepared,
                        system,
                        summary,
                        settings=settings,
                        http=http,
                        llm=llm,
                        run=run,
                        run_id=run_id,
                        usage=usage,
                        checkpointer=saver if system is SystemName.FULL else None,
                        max_concurrency=max_concurrency,
                        paper_concurrency=paper_concurrency,
                    )
                    summary.failed.sort()
                write_manifest(
                    manifest.model_copy(update={"finished_at": datetime.now(UTC)}), run.path
                )
                summaries.append(summary)
    return summaries
