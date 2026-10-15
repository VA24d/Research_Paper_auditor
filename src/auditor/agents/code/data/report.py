"""The data report of a snapshot: which data files are fetched, and the cached analysis.

The report is written next to the snapshot as `data_report-<key>.json`, where the key
covers the analysed files' SHA-256s, the analysis limits, the label hint and
ANALYSIS_VERSION, so a rerun with the same inputs reads the same report.
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass

from auditor.agents.code.data.analysis import ANALYSIS_VERSION, AnalysisLimits, DataReport, analyse
from auditor.agents.code.data.splits import map_splits
from auditor.agents.code.extract import parse_card
from auditor.agents.code.extract.normalise import split_from_parts
from auditor.infra.context import RunContext
from auditor.infra.files import atomic_write
from auditor.infra.snapshots import Snapshot, SourceKind
from auditor.infra.snapshots.data import fetch_data_files
from auditor.schemas import ExtractedSetting, RepositoryRef, RepositorySource

NO_SPLIT_FILES = (
    "no data files named after a train, validation or test split (or listed in the dataset card)"
)
_ORDER = ("train", "validation", "test")
_MAX_LABEL_NAMES = 20


@dataclass(frozen=True)
class SnapshotData:
    snapshot: Snapshot
    report: DataReport | None
    reason: str | None = None


_cache: dict[tuple[str, str | None], SnapshotData] = {}
_locks: dict[tuple[str, str | None], asyncio.Lock] = {}


async def snapshot_data(
    snapshot: Snapshot, ctx: RunContext, config: str | None = None
) -> SnapshotData:
    key = (str(snapshot.root), config)
    async with _locks.setdefault(key, asyncio.Lock()):
        if key not in _cache:
            _cache[key] = await _build(snapshot, ctx, config)
        return _cache[key]


async def _build(snapshot: Snapshot, ctx: RunContext, config: str | None) -> SnapshotData:
    manifest = snapshot.manifest
    card_files: dict[str, tuple[str, ...]] | None = None
    label_hint = None
    if manifest.source is SourceKind.HUGGINGFACE and "README.md" in {
        f.path for f in manifest.files
    }:
        card = parse_card(snapshot.read_text("README.md"))
        chosen = card.config(config) if card is not None else None
        if chosen is not None:
            card_files = chosen.data_files or None
            label_hint = chosen.label_column
    mapping, ambiguous = map_splits([f.path for f in manifest.data_files], card_files)
    # Extra splits such as IMDB's "unsupervised" are not what a paper's split refers to.
    mapping = {split: paths for split, paths in mapping.items() if split in _ORDER}
    if not mapping:
        reason = NO_SPLIT_FILES
        if ambiguous:
            reason += f"; ambiguous: {', '.join(sorted(ambiguous))}"
        return SnapshotData(snapshot, None, reason)

    paths = [path for split in sorted(mapping) for path in mapping[split]]
    fetched = await fetch_data_files(snapshot, paths, ctx.http, ctx.settings)
    files = {
        split: [fetched.blobs[p] for p in split_paths if p in fetched.blobs]
        for split, split_paths in mapping.items()
    }
    files = {split: blobs for split, blobs in files.items() if blobs}
    not_read = {**fetched.skipped, **ambiguous}
    if not files:
        details = "; ".join(f"{p}: {r}" for p, r in sorted(not_read.items()))
        return SnapshotData(snapshot, None, f"no data files could be read ({details})")

    limits = AnalysisLimits(max_hashed_rows=ctx.settings.data_analysis_max_hashed_rows)
    key_payload = json.dumps(
        {
            "files": {s: [b.sha256 for b in blobs] for s, blobs in sorted(files.items())},
            "limits": limits.max_hashed_rows,
            "label": label_hint,
            "version": ANALYSIS_VERSION,
        },
        sort_keys=True,
    )
    key = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()[:16]
    path = snapshot.root.parent / f"data_report-{key}.json"
    if path.is_file():
        report = DataReport.model_validate_json(path.read_bytes())
    else:
        report = await asyncio.to_thread(
            analyse,
            {split: [b.local for b in blobs] for split, blobs in files.items()},
            limits,
            label_hint=label_hint,
            names={b.local: b.path for blobs in files.values() for b in blobs},
        )
        report = report.model_copy(update={"not_read": dict(sorted(not_read.items()))})
        atomic_write(path, report.model_dump_json(indent=2).encode("utf-8"))
    return SnapshotData(snapshot, report)


def report_settings(data: SnapshotData) -> list[ExtractedSetting]:
    """Row counts, split shares and the number of classes as settings, for comparing with
    what the paper states."""
    report = data.report
    if report is None:
        return []
    manifest = data.snapshot.manifest
    ref = RepositoryRef(url=manifest.url, commit=manifest.revision)
    rows = {
        s: report.splits[s].rows for s in _ORDER if s in report.splits and report.splits[s].rows
    }

    def setting(key: str, raw: str, value: int | str, split: str | None) -> ExtractedSetting:
        files = report.splits[split].files if split else next(iter(report.splits.values())).files
        return ExtractedSetting(
            key=key,
            raw_name="data files",
            raw_value=raw,
            value=value,
            source=RepositorySource(repository=ref, path=files[0], line=1),
        )

    found = []
    if "train" in rows and "test" in rows:
        ordered = [s for s in _ORDER if s in rows]
        value = split_from_parts([float(rows[s]) for s in ordered])
        if value is not None:
            raw = " / ".join(f"{s} {rows[s]}" for s in ordered) + " rows"
            found.append(setting("data_split", raw, value, "test"))
    for split, count in rows.items():
        found.append(setting(f"{split}_examples", f"{split} {count} rows", count, split))
    if rows:
        total = sum(rows.values())
        raw = " + ".join(f"{s} {n}" for s, n in rows.items()) + " rows"
        found.append(setting("total_examples", raw, total, next(iter(rows))))
    labels = sorted(
        {label for stats in report.splits.values() for label in (stats.label_counts or {})}
    )
    if labels:
        shown = ", ".join(labels[:_MAX_LABEL_NAMES]) + (
            ", ..." if len(labels) > _MAX_LABEL_NAMES else ""
        )
        found.append(setting("num_classes", f"{len(labels)} classes: {shown}", len(labels), None))
    return found
