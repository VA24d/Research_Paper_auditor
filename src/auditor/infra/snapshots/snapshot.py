"""Downloads the core files of a repository or dataset into the scratch folder.

A snapshot lives in `<scratch_dir>/snapshots/<host>/<name>/<revision>-<limits>/`:
`files/` holds the downloaded files and `snapshot.json` the manifest, listing every file
kept (with its SHA-256) and every file left out with the reason. Pinned snapshots are
reused across runs; the scratch folder can be deleted at any time.
"""

import asyncio
import hashlib
import json
import logging
import re
import secrets
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, JsonValue, ValidationError

from auditor.config import Settings
from auditor.infra.files import atomic_write
from auditor.infra.http import HttpClient, ResponseTooLargeError
from auditor.infra.snapshots.hosts import FileUnavailableError, Listing, make_host
from auditor.infra.snapshots.policy import (
    DATA_SUFFIXES,
    TOO_LARGE,
    TOTAL_LIMIT,
    DownloadLimits,
    RemoteFile,
    Skipped,
    safe_relative_path,
    select_files,
)
from auditor.infra.snapshots.sources import SourceKind, SourceSpec, parse_source
from auditor.schemas import RepositoryRef

logger = logging.getLogger(__name__)

FILES_DIR = "files"
MANIFEST_NAME = "snapshot.json"
# Keeps the manifest readable for hosts that list thousands of data files.
MAX_SKIPPED_RECORDS = 1_000
MAX_DATA_FILES = 5_000
_DOWNLOAD_CONCURRENCY = 8


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SnapshotFile(_Model):
    path: str
    size: int
    sha256: str


class SkippedFile(_Model):
    path: str
    size: int | None = None
    reason: str


class DataFile(_Model):
    """A data file the host lists, downloaded only in data analysis mode."""

    path: str
    size: int | None = None
    sha256: str | None = None
    url: str


class SnapshotManifest(_Model):
    format_version: Literal[1] = 1
    source: SourceKind
    url: str
    identity: str
    requested_ref: str | None
    revision: str
    fetched_at: AwareDatetime
    limits: dict[str, int]
    files: tuple[SnapshotFile, ...]
    skipped: tuple[SkippedFile, ...]
    skipped_count: int
    # Why the snapshot may be missing core files; empty when nothing was cut.
    incomplete: tuple[str, ...]
    metadata: dict[str, JsonValue]
    data_files: tuple[DataFile, ...] = ()


@dataclass(frozen=True)
class Snapshot:
    root: Path
    manifest: SnapshotManifest

    @property
    def source_id(self) -> str:
        """`url@revision`, the id to put in evidence (`SourceRef.id`)."""
        return f"{self.manifest.url}@{self.manifest.revision}"

    def path(self, relative: str) -> Path:
        """The local path of a downloaded file; KeyError for anything not in the manifest."""
        if relative not in {f.path for f in self.manifest.files}:
            raise KeyError(relative)
        return self.root.joinpath(*PurePosixPath(relative).parts)

    def read_text(self, relative: str) -> str:
        return self.path(relative).read_text(encoding="utf-8", errors="replace")


_locks: dict[tuple[str, ...], asyncio.Lock] = {}


async def fetch_snapshot(
    source: RepositoryRef | SourceSpec | str,
    http: HttpClient,
    settings: Settings,
    *,
    limits: DownloadLimits | None = None,
) -> Snapshot:
    """The core files of `source`, downloaded within `limits` (from settings by default).

    Raises SourceError for a source that is not supported and SnapshotUnavailableError when
    it cannot be fetched. Repository code is only stored, never executed (decision D11).
    """
    match source:
        case RepositoryRef():
            spec = parse_source(source.url, source.commit)
        case SourceSpec():
            spec = source
        case str():
            spec = parse_source(source)
    limits = limits or DownloadLimits.from_settings(settings)
    host = make_host(spec, http, settings, limits)
    base = settings.scratch_dir / "snapshots" / spec.kind.value / _name(spec.identity)
    key = (spec.kind.value, spec.identity, spec.ref or "", limits.key(), str(base))
    async with _locks.setdefault(key, asyncio.Lock()):
        revision = await host.resolve()
        if revision is not None and (found := _existing(base, revision, limits)):
            return found
        listing = await host.listing(revision)
        revision = revision or listing.revision
        if revision is not None and (found := _existing(base, revision, limits)):
            return found

        selection = select_files(listing.files, limits)
        staging = base / f".partial-{secrets.token_hex(6)}"
        try:
            files, skipped, incomplete = await _download(
                selection.chosen, staging / FILES_DIR, limits
            )
            revision = revision or _content_revision(files)
            all_skipped = [*listing.skipped, *selection.skipped, *skipped]
            manifest = SnapshotManifest(
                source=spec.kind,
                url=spec.url,
                identity=spec.identity,
                requested_ref=spec.ref,
                revision=revision,
                fetched_at=datetime.now(UTC),
                limits={
                    "max_file_bytes": limits.max_file_bytes,
                    "max_total_bytes": limits.max_total_bytes,
                    "max_files": limits.max_files,
                    "max_archive_bytes": limits.max_archive_bytes,
                },
                files=tuple(files),
                skipped=tuple(
                    SkippedFile(path=s.path, size=s.size, reason=s.reason)
                    for s in all_skipped[:MAX_SKIPPED_RECORDS]
                ),
                skipped_count=len(all_skipped),
                incomplete=_unique([*listing.incomplete, *selection.incomplete, *incomplete]),
                metadata=listing.metadata,
                data_files=_data_files(listing),
            )
            atomic_write(
                staging / MANIFEST_NAME, manifest.model_dump_json(indent=2).encode("utf-8")
            )
            target = base / _folder(revision, limits)
            _publish(staging, target)
            logger.info(
                "snapshot of %s at %s: %d files, %d skipped",
                spec.url,
                revision,
                len(files),
                len(all_skipped),
            )
            return _load(target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)


async def _download(
    chosen: Sequence[RemoteFile], folder: Path, limits: DownloadLimits
) -> tuple[list[SnapshotFile], list[Skipped], list[str]]:
    """Downloads `chosen` a window at a time, in parallel within a window, and spends the
    total budget in priority order, so the result does not depend on which download
    finishes first. At most one window is downloaded past the budget."""
    files: list[SnapshotFile] = []
    skipped: list[Skipped] = []
    incomplete: list[str] = []
    used = 0
    for start in range(0, len(chosen), _DOWNLOAD_CONCURRENCY):
        window = chosen[start : start + _DOWNLOAD_CONCURRENCY]
        if used >= limits.max_total_bytes:
            skipped += [Skipped(f.path, f.size, TOTAL_LIMIT) for f in window]
            incomplete.append(TOTAL_LIMIT)
            continue
        loaded = await asyncio.gather(*(_load_file(f, limits.max_file_bytes) for f in window))
        for file, data in zip(window, loaded, strict=True):
            if isinstance(data, Skipped):
                skipped.append(data)
            elif used + len(data) > limits.max_total_bytes:
                skipped.append(Skipped(file.path, file.size, TOTAL_LIMIT))
                incomplete.append(TOTAL_LIMIT)
            else:
                destination = folder.joinpath(*PurePosixPath(file.path).parts)
                await asyncio.to_thread(atomic_write, destination, data)
                used += len(data)
                files.append(
                    SnapshotFile(
                        path=file.path, size=len(data), sha256=hashlib.sha256(data).hexdigest()
                    )
                )
    return files, skipped, list(_unique(incomplete))


async def _load_file(file: RemoteFile, max_bytes: int) -> bytes | Skipped:
    try:
        return await file.load(max_bytes)
    except ResponseTooLargeError:
        return Skipped(file.path, file.size, TOO_LARGE)
    except FileUnavailableError as exc:
        return Skipped(file.path, file.size, str(exc))


def _data_files(listing: Listing) -> tuple[DataFile, ...]:
    found: dict[str, DataFile] = {}
    candidates = [
        *(f for f in listing.files if PurePosixPath(f.path).suffix.lower() in DATA_SUFFIXES),
        *listing.data_files,
    ]
    for file in candidates:
        path = safe_relative_path(file.path)
        if path is not None and file.url is not None and path not in found:
            found[path] = DataFile(path=path, size=file.size, sha256=file.sha256, url=file.url)
    return tuple(found[path] for path in sorted(found)[:MAX_DATA_FILES])


def _content_revision(files: Sequence[SnapshotFile]) -> str:
    payload = json.dumps(sorted((f.path, f.sha256) for f in files)).encode("utf-8")
    return f"content-{hashlib.sha256(payload).hexdigest()[:16]}"


def _name(text: str) -> str:
    """A folder name for `text` that is safe on every OS and unique per `text`."""
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")[-48:] or "source"
    return f"{readable}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]}"


def _folder(revision: str, limits: DownloadLimits) -> str:
    return f"{re.sub(r'[^A-Za-z0-9._-]+', '_', revision)[:64]}-{limits.key()}"


def _existing(base: Path, revision: str, limits: DownloadLimits) -> Snapshot | None:
    target = base / _folder(revision, limits)
    if not (target / MANIFEST_NAME).is_file():
        return None
    try:
        return _load(target)
    except (OSError, ValidationError):
        logger.warning("discarding unreadable snapshot %s", target)
        shutil.rmtree(target, ignore_errors=True)
        return None


def _load(target: Path) -> Snapshot:
    manifest = SnapshotManifest.model_validate_json((target / MANIFEST_NAME).read_bytes())
    return Snapshot(root=target / FILES_DIR, manifest=manifest)


def _publish(staging: Path, target: Path) -> None:
    """Moves a finished snapshot into place. Another process may have finished the same
    snapshot first; then its copy is kept."""
    (staging / FILES_DIR).mkdir(parents=True, exist_ok=True)
    try:
        staging.rename(target)
    except OSError:
        if not (target / MANIFEST_NAME).is_file():
            raise


def _unique(items: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(items))
