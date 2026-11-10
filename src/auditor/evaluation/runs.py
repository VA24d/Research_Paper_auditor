"""Reading a finished run folder for the evaluation (layout in `auditor.orchestration.outputs`).

A run folder holds the manifest, the usage logs and one folder per paper version. A paper
whose checks ran out of model calls has `predictions.incomplete.jsonl` instead of
`predictions.jsonl`; it is read but not scored, so its missing findings do not count as
clean results.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from auditor.agents.theory.review import NO_ERROR_REASON
from auditor.config import Settings
from auditor.orchestration.outputs import INCOMPLETE_PREDICTIONS, PREDICTIONS
from auditor.schemas import (
    LLMCallRecord,
    PaperVariant,
    Prediction,
    RunManifest,
    SystemName,
    TaskOutcome,
    TaskResult,
    VerificationState,
    VerifiedFinding,
)

logger = logging.getLogger(__name__)


def read_jsonl[T: BaseModel](
    path: Path, schema: type[T], remap: tuple[str, str] | None = None
) -> list[T]:
    """The records of a JSON Lines file, or none when it is missing. `remap` rewrites a
    path prefix first, for a run made on another machine."""
    if not path.is_file():
        return []
    text = path.read_text(encoding="utf-8")
    if remap is not None:
        text = text.replace(*remap)
    return [schema.model_validate_json(line) for line in text.splitlines() if line.strip()]


@dataclass
class PaperRun:
    """One paper version in a run."""

    paper_id: str
    variant: PaperVariant
    folder: Path
    complete: bool
    predictions: list[Prediction]
    verified: list[VerifiedFinding] = field(default_factory=list)
    results: list[TaskResult] = field(default_factory=list)


@dataclass
class Usage:
    """What a run spent (decision D8). `cost_usd` is None when a model has no price or the
    run kept no model-call log."""

    llm_calls: int = 0
    cached_llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = 0.0
    tool_calls: int = 0
    http_requests: int = 0
    # Whether the run kept a model-call log; without one the calls are unknown, not zero.
    logged: bool = True


@dataclass
class RunData:
    path: Path
    manifest: RunManifest
    papers: list[PaperRun]
    usage: Usage
    # Paper versions that failed and never got predictions, as "<paper_id>-<variant>".
    failed: list[str] = field(default_factory=list)

    @property
    def system(self) -> SystemName:
        return self.manifest.system

    @property
    def mocked_parts(self) -> list[str]:
        """Parts that ran as placeholders (decision D19)."""
        parts = self.manifest.config.get("mocked_parts")
        return [str(p) for p in parts] if isinstance(parts, list) else []

    @property
    def complete_papers(self) -> list[PaperRun]:
        return [p for p in self.papers if p.complete]

    def predictions(self) -> list[Prediction]:
        """The predictions of the complete papers, as the system made them."""
        return [p for paper in self.complete_papers for p in paper.predictions]

    def predictions_without_verifier(self) -> list[Prediction]:
        """The full system's findings with the verifier taken out (ablation, team plan
        P4.7): every candidate finding is a flag, as for B0 and B1. Only the drops of the
        theory checker's own math review (D21) still count, since that review is part of
        the specialist, not of the verifier."""
        if self.system is not SystemName.FULL:
            raise ValueError(f"{self.system.value} has no verifier to take out")
        out = []
        for paper in self.complete_papers:
            for decision in paper.verified:
                if decision.state is VerificationState.DROPPED and decision.reason.startswith(
                    NO_ERROR_REASON
                ):
                    continue
                finding = decision.finding
                out.append(
                    Prediction(
                        run_id=self.manifest.run_id,
                        system=SystemName.FULL_NO_VERIFIER,
                        paper_id=finding.paper_id,
                        variant=finding.variant,
                        error_class=finding.error_class,
                        anchor=finding.anchor,
                        target=finding.target,
                        summary=finding.summary,
                        evidence=finding.evidence,
                        finding_id=finding.finding_id,
                        severity=decision.severity,
                        duplicate_of=decision.duplicate_of,
                    )
                )
        return out

    def task_outcomes(self) -> dict[TaskOutcome, int]:
        """How the full system's tasks ended, over the complete papers (coverage)."""
        counts = dict.fromkeys(TaskOutcome, 0)
        for paper in self.complete_papers:
            for result in paper.results:
                counts[result.outcome] += 1
        return counts


def _usage(folder: Path, manifest: RunManifest) -> Usage:
    usage = Usage()
    config = manifest.config
    # The prices the run was charged at, else today's defaults.
    prices = dict(Settings.model_fields["llm_prices"].get_default(call_default_factory=True))
    prices_config = config.get("llm_prices")
    if isinstance(prices_config, dict):
        prices = {
            str(model): (float(price[0]), float(price[1]))
            for model, price in prices_config.items()
            if isinstance(price, list) and len(price) == 2
            if isinstance(price[0], int | float) and isinstance(price[1], int | float)
        }
    share = config.get("llm_cached_input_share")
    cached_share = float(share) if isinstance(share, int | float) else 1.0
    cost = 0.0
    log = folder / "usage" / "llm_calls.jsonl"
    priced = usage.logged = log.is_file()
    for call in read_jsonl(log, LLMCallRecord):
        if call.cache_hit:
            usage.cached_llm_calls += 1
            continue
        usage.llm_calls += 1
        usage.input_tokens += call.input_tokens
        usage.output_tokens += call.output_tokens
        usage.reasoning_tokens += call.reasoning_tokens
        price = prices.get(call.model.split(":", 1)[-1])
        if price is None:
            priced = False
            continue
        cached = min(call.cached_input_tokens, call.input_tokens)
        fresh = call.input_tokens - cached
        cost += ((fresh + cached * cached_share) * price[0] + call.output_tokens * price[1]) / 1e6
    usage.cost_usd = cost if priced else None
    tool_calls = folder / "usage" / "tool_calls.jsonl"
    if tool_calls.is_file():
        with tool_calls.open(encoding="utf-8") as lines:
            usage.tool_calls = sum(1 for line in lines if line.strip())
    http = folder / "usage" / "http_requests.jsonl"
    if http.is_file():
        with http.open(encoding="utf-8") as lines:
            for line in lines:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not record.get("cache_hit"):
                    usage.http_requests += int(record.get("attempts", 1)) + int(
                        record.get("redirects", 0)
                    )
    return usage


def _paper(folder: Path, remap: tuple[str, str] | None) -> PaperRun | None:
    complete = (folder / PREDICTIONS).is_file()
    if not complete and not (folder / INCOMPLETE_PREDICTIONS).is_file():
        return None
    predictions = read_jsonl(
        folder / (PREDICTIONS if complete else INCOMPLETE_PREDICTIONS), Prediction
    )
    verified = read_jsonl(folder / "verified.jsonl", VerifiedFinding, remap)
    results = read_jsonl(folder / "results.jsonl", TaskResult, remap)
    paper_id, _, variant = folder.name.rpartition("-")
    if predictions:
        paper_id, variant = predictions[0].paper_id, predictions[0].variant.value
    elif results:
        paper_id, variant = results[0].paper_id, results[0].variant.value
    return PaperRun(
        paper_id=paper_id,
        variant=PaperVariant(variant),
        folder=folder,
        complete=complete,
        predictions=predictions,
        verified=verified,
        results=results,
    )


def load_run(folder: Path, remap: tuple[str, str] | None = None) -> RunData:
    """A run folder as written by `auditor audit`, `auditor baseline` or `auditor batch`."""
    manifest = RunManifest.model_validate_json((folder / "manifest.json").read_text("utf-8"))
    papers = [
        paper
        for child in sorted(folder.iterdir())
        if child.is_dir() and child.name != "usage"
        if (paper := _paper(child, remap)) is not None
    ]
    done = {f"{p.paper_id}-{p.variant.value}" for p in papers if p.complete}
    failed: list[str] = []
    failures = folder / "failures.jsonl"
    if failures.is_file():
        with failures.open(encoding="utf-8") as lines:
            for line in lines:
                if line.strip():
                    record = json.loads(line)
                    key = f"{record.get('paper_id')}-{record.get('variant')}"
                    if key not in done and key not in failed:
                        failed.append(key)
    return RunData(
        path=folder,
        manifest=manifest,
        papers=papers,
        usage=_usage(folder, manifest),
        failed=failed,
    )


def find_runs(paths: Sequence[Path]) -> list[Path]:
    """Run folders among `paths`: each path is a run folder, or a folder of run folders."""
    found: list[Path] = []
    for path in paths:
        if (path / "manifest.json").is_file():
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(p for p in path.iterdir() if (p / "manifest.json").is_file()))
    return found
