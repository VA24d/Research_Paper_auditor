"""Grounding rate: whether a system's flags rest on evidence that is in its sources (team
plan P4.5).

The evidence of every positive flag is looked up again with the verifier's own check
(`auditor.agents.verifier.grounding`), so all systems are measured the same way. B0 and B1
attach no evidence, so none of their flags is backed. Two numbers come out:

- the share of flags backed by at least one piece of evidence found in its source, or by
  a tool record (an index lookup, a venue rule);
- the share of quotes and code that are found, over those that could be checked. A record
  (an index lookup, a venue rule) has nothing to find and is counted apart, as is evidence
  whose source is not available offline (a repository snapshot that is not cached).
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

from auditor.agents.verifier.grounding import Grounding, PaperText, ground
from auditor.config import Settings
from auditor.experiments import InputError, read_dataset, resolve_repository
from auditor.infra.http import HttpClient
from auditor.infra.snapshots import Snapshot, fetch_snapshot
from auditor.schemas import ParsedPaper, Prediction, RepositoryRef, SentenceAnchor, SourceKind

logger = logging.getLogger(__name__)

PaperKey = tuple[str, str]


@dataclass
class PaperInput:
    paper: ParsedPaper
    repository: RepositoryRef | None


@dataclass
class GroundingStats:
    flags: int = 0
    backed: int = 0
    found: int = 0
    not_found: int = 0
    unchecked: int = 0
    records: int = 0
    # Flags on papers whose ParsedPaper could not be loaded; not counted above.
    skipped: int = 0

    @property
    def backed_share(self) -> float | None:
        return self.backed / self.flags if self.flags else None

    @property
    def rate(self) -> float | None:
        checked = self.found + self.not_found
        return self.found / checked if checked else None


def load_inputs(dataset: Path, remap: tuple[str, str] | None = None) -> dict[PaperKey, PaperInput]:
    """The papers and repositories of a dataset file, by paper id and variant. Only
    ParsedPaper JSON files are read; PDFs would need GROBID and are skipped."""
    inputs: dict[PaperKey, PaperInput] = {}
    for entry in read_dataset(dataset):
        paper_path, repository = entry.paper, entry.repository
        if remap is not None:
            paper_path = paper_path.replace(*remap)
            repository = repository.replace(*remap) if repository else None
        path = Path(paper_path)
        if path.suffix != ".json" or not path.is_file():
            logger.warning("%s: not a ParsedPaper JSON file here; its grounding is skipped", path)
            continue
        paper = ParsedPaper.model_validate_json(path.read_text(encoding="utf-8"))
        try:
            ref = resolve_repository(repository, paper, entry.repository_commit)
        except InputError as exc:
            logger.warning("%s: %s", path, exc)
            ref = None
        inputs.setdefault((paper.paper_id, paper.variant.value), PaperInput(paper, ref))
    return inputs


def _offline(settings: Settings) -> HttpClient:
    """A client that serves only the cache: snapshots already on disk are reused, and
    nothing is downloaded."""

    def refuse(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, request=request)

    return HttpClient(
        settings.model_copy(update={"http_max_attempts": 1}),
        transport=httpx.MockTransport(refuse),
    )


async def _snapshot(
    ref: RepositoryRef | None,
    http: HttpClient,
    settings: Settings,
    cache: dict[str, Snapshot | None],
) -> dict[str, Snapshot]:
    if ref is None:
        return {}
    key = f"{ref.url}@{ref.commit}"
    if key not in cache:
        try:
            cache[key] = await fetch_snapshot(ref, http, settings)
        except Exception as exc:  # any failure leaves the code unchecked
            logger.info("%s is not available offline: %s", ref.url, exc)
            cache[key] = None
    snapshot = cache[key]
    return {snapshot.source_id: snapshot} if snapshot is not None else {}


async def grounding(
    predictions: Sequence[Prediction],
    inputs: Mapping[PaperKey, PaperInput],
    settings: Settings,
) -> GroundingStats:
    """Grounding of the positive flags among `predictions`; duplicates (D20) count once."""
    stats = GroundingStats()
    texts: dict[PaperKey, PaperText] = {}
    snapshots: dict[str, Snapshot | None] = {}
    async with _offline(settings) as http:
        for prediction in predictions:
            if not prediction.is_positive or prediction.duplicate_of is not None:
                continue
            key = (prediction.paper_id, prediction.variant.value)
            source = inputs.get(key)
            if source is None:
                stats.skipped += 1
                continue
            stats.flags += 1
            if key not in texts:
                texts[key] = PaperText(source.paper)
            needs_code = any(e.source.kind is SourceKind.REPOSITORY for e in prediction.evidence)
            found_snapshots = (
                await _snapshot(source.repository, http, settings, snapshots) if needs_code else {}
            )
            anchor = prediction.anchor
            sentence_id = anchor.sentence_id if isinstance(anchor, SentenceAnchor) else None
            backed = False
            for evidence in prediction.evidence:
                check = ground(evidence, texts[key], found_snapshots, sentence_id)
                match check.grounding:
                    case Grounding.FOUND:
                        stats.found += 1
                        backed = True
                    case Grounding.NOT_FOUND:
                        stats.not_found += 1
                    case Grounding.UNCHECKED:
                        stats.unchecked += 1
                    case Grounding.RECORD:
                        stats.records += 1
                        backed = True
            stats.backed += backed
    return stats
