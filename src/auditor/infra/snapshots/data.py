"""Downloads a snapshot's data files for data analysis mode.

Files are stored by content in `<scratch_dir>/data-blobs/<sha[:2]>/<sha256><suffix>`, so
revisions and repositories that share a file share one copy and a file whose SHA-256 the
host publishes (Hugging Face LFS files) is never downloaded twice. Local folders are read
in place. Only formats `auditor.agents.code.data.readers` can read safely are ever
fetched; nothing is executed or unpickled.
"""

import asyncio
import hashlib
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from urllib.request import url2pathname

import httpx

from auditor.config import Settings
from auditor.infra.http import HttpClient, HttpError, HttpStatusError, ResponseTooLargeError
from auditor.infra.snapshots.policy import DATA_SUFFIXES
from auditor.infra.snapshots.snapshot import DataFile, Snapshot

logger = logging.getLogger(__name__)

LFS_POINTER = b"version https://git-lfs.github.com/spec/v1"
TOO_LARGE = "larger than the data file limit"
TOTAL_LIMIT = "data download limit reached"
NOT_LISTED = "not a data file of this snapshot"
LFS_ONLY = "stored in Git LFS; the repository only holds a pointer"
NO_SPACE = "not enough free disk space"


@dataclass(frozen=True)
class DataBlob:
    path: str
    local: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class DataFetch:
    blobs: dict[str, DataBlob] = field(default_factory=dict)
    # Path -> why it was not fetched.
    skipped: dict[str, str] = field(default_factory=dict)


async def fetch_data_files(
    snapshot: Snapshot, paths: list[str], http: HttpClient, settings: Settings
) -> DataFetch:
    """Fetch the named data files of `snapshot`, in order, within the data limits."""
    listed = {f.path: f for f in snapshot.manifest.data_files}
    store = settings.scratch_dir / "data-blobs"
    blobs: dict[str, DataBlob] = {}
    skipped: dict[str, str] = {}
    used = 0
    for path in paths:
        entry = listed.get(path)
        if entry is None or PurePosixPath(path).suffix.lower() not in DATA_SUFFIXES:
            skipped[path] = NOT_LISTED
            continue
        if entry.size is not None and entry.size > settings.data_analysis_max_file_bytes:
            skipped[path] = TOO_LARGE
            continue
        left = settings.data_analysis_max_total_bytes - used
        if left <= 0 or (entry.size is not None and entry.size > left):
            skipped[path] = TOTAL_LIMIT
            continue
        try:
            blob = await _fetch(entry, store, http, settings, left)
        except ResponseTooLargeError:
            skipped[path] = (
                TOO_LARGE if left >= settings.data_analysis_max_file_bytes else TOTAL_LIMIT
            )
            continue
        except HttpStatusError as exc:
            skipped[path] = f"not available from the host (HTTP {exc.response.status_code})"
            continue
        except _SkipError as exc:
            skipped[path] = str(exc)
            continue
        blobs[path] = blob
        used += blob.size
    return DataFetch(blobs, skipped)


class _SkipError(Exception):
    pass


async def _fetch(
    entry: DataFile, store: Path, http: HttpClient, settings: Settings, left: int
) -> DataBlob:
    suffix = PurePosixPath(entry.path).suffix.lower()
    if entry.url.startswith("file:"):
        local, size, digest = await asyncio.to_thread(_local_file, entry.url)
        if size > settings.data_analysis_max_file_bytes:
            raise ResponseTooLargeError(f"{local} is larger than the data file limit")
        return DataBlob(entry.path, local, size, digest)

    if entry.sha256 is not None:
        known = _blob_path(store, entry.sha256, suffix)
        cached = _size_if_present(known)
        if cached is not None:
            return DataBlob(entry.path, known, cached, entry.sha256)
    if entry.size is not None and shutil.disk_usage(_existing_parent(store)).free < 2 * entry.size:
        raise _SkipError(NO_SPACE)

    incoming = store / "incoming" / f"{hashlib.sha256(entry.url.encode()).hexdigest()}{suffix}"
    try:
        download = await http.download_to(
            entry.url,
            incoming,
            max_bytes=min(settings.data_analysis_max_file_bytes, left),
            headers=_headers(entry.url, settings),
            sha256=entry.sha256,
        )
    except HttpError as exc:
        if isinstance(exc, ResponseTooLargeError | HttpStatusError):
            raise
        raise _SkipError(f"download failed: {exc}") from exc
    final = _blob_path(store, download.sha256, suffix)
    if not _store(incoming, final):
        raise _SkipError(LFS_ONLY)
    return DataBlob(entry.path, final, download.size, download.sha256)


def _size_if_present(path: Path) -> int | None:
    return path.stat().st_size if path.is_file() else None


def _store(incoming: Path, final: Path) -> bool:
    """Moves a download into the blob store, unless it is only a Git LFS pointer."""
    if incoming.stat().st_size < 200 and incoming.read_bytes().startswith(LFS_POINTER):
        incoming.unlink(missing_ok=True)
        return False
    final.parent.mkdir(parents=True, exist_ok=True)
    incoming.replace(final)
    return True


def _blob_path(store: Path, sha256: str, suffix: str) -> Path:
    return store / sha256[:2] / f"{sha256}{suffix}"


def _existing_parent(path: Path) -> Path:
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def _local_file(url: str) -> tuple[Path, int, str]:
    """A local data file, read in place: its path, size and SHA-256."""
    path = Path(url2pathname(urlsplit(url).path)).resolve()
    return (path, *_hash_file(path))


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def _headers(url: str, settings: Settings) -> dict[str, str] | None:
    """Hugging Face needs the token for gated datasets; httpx drops it on the redirect to
    the CDN."""
    if settings.hf_token is not None and httpx.URL(url).host == "huggingface.co":
        return {"Authorization": f"Bearer {settings.hf_token.get_secret_value()}"}
    return None
