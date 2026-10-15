"""One adapter per host: how to pin a revision, list files and download one file.

Every request goes through the shared HttpClient (decision D8). Adapters only list and
download; choosing files and enforcing limits is done by `policy` and `snapshot`.
"""

import asyncio
import hashlib
import io
import json
import os
import re
import zipfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol
from urllib.parse import quote

from pydantic import JsonValue

from auditor.config import Settings
from auditor.infra.http import HttpClient, HttpResponse, ResponseTooLargeError
from auditor.infra.snapshots.policy import (
    EXCLUDED_DIR,
    SYMLINK,
    UNSAFE_PATH,
    DownloadLimits,
    RemoteFile,
    Skipped,
    excluded_dir,
    safe_relative_path,
)
from auditor.infra.snapshots.sources import SourceKind, SourceSpec


class SnapshotUnavailableError(Exception):
    """The source cannot be fetched (missing, private, gated or withdrawn). The code agent
    reports such a task as unverifiable with this reason, never as a mismatch."""


class FileUnavailableError(Exception):
    """One listed file could not be downloaded; the rest of the snapshot is still usable."""


LISTING_TRUNCATED = "the host truncated the file listing"
LISTING_TOO_LONG = "the file listing was too long to read in full"
SUBMODULE = "git submodule"
LFS = "stored in large file storage"

# A listing longer than this many pages (or folders, for Anonymous GitHub) is cut short.
MAX_LISTING_PAGES = 100
MAX_CRAWLED_FOLDERS = 500
_UNAVAILABLE_STATUSES = frozenset({401, 403, 404, 410, 422, 451})
_FULL_SHA = re.compile(r"[0-9a-fA-F]{40}")
_NEXT_LINK = re.compile(r'<([^>]+)>\s*;\s*rel="?next"?')


@dataclass
class Listing:
    files: list[RemoteFile] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    incomplete: list[str] = field(default_factory=list)
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    # Set by hosts whose revision is only known from the listing itself.
    revision: str | None = None


class Host(Protocol):
    async def resolve(self) -> str | None:
        """The pinned revision, or None when only the listing or the content can tell."""
        ...

    async def listing(self, revision: str | None) -> Listing: ...


def make_host(
    spec: SourceSpec, http: HttpClient, settings: Settings, limits: DownloadLimits
) -> Host:
    match spec.kind:
        case SourceKind.GITHUB:
            return GitHub(spec, http, settings)
        case SourceKind.GITLAB:
            return GitLab(spec, http)
        case SourceKind.HUGGINGFACE:
            return HuggingFace(spec, http, settings)
        case SourceKind.KAGGLE:
            return Kaggle(spec, http)
        case SourceKind.ANONYMOUS_GITHUB:
            return AnonymousGitHub(spec, http)
        case SourceKind.ZENODO:
            return Zenodo(spec, http, limits)
        case SourceKind.LOCAL:
            return LocalFolder(spec)


async def _get(
    http: HttpClient,
    url: str,
    what: str,
    *,
    headers: Mapping[str, str] | None = None,
    use_cache: bool = True,
    max_bytes: int | None = None,
) -> HttpResponse:
    """GET that turns "not there or not allowed" into SnapshotUnavailableError. Rate limits
    and server errors still raise, so the orchestrator can retry the task."""
    response = await http.get(url, headers=headers, use_cache=use_cache, max_bytes=max_bytes)
    rate_limited = response.headers.get("x-ratelimit-remaining") == "0"
    if response.status_code in _UNAVAILABLE_STATUSES and not rate_limited:
        raise SnapshotUnavailableError(f"{what}: HTTP {response.status_code} from {url}")
    response.raise_for_status()
    return response


def _loader(
    http: HttpClient, url: str, headers: Mapping[str, str] | None = None
) -> Callable[[int], Awaitable[bytes]]:
    async def load(max_bytes: int) -> bytes:
        response = await http.get(url, headers=headers, use_cache=False, max_bytes=max_bytes)
        if response.status_code in _UNAVAILABLE_STATUSES:
            raise FileUnavailableError(f"not available from the host (HTTP {response.status_code})")
        response.raise_for_status()
        return response.content

    return load


def _json_dict(response: HttpResponse) -> dict[str, Any]:
    data = response.json()
    if not isinstance(data, dict):
        raise SnapshotUnavailableError(f"unexpected answer from {response.url}")
    return data


def _json_list(response: HttpResponse) -> list[dict[str, Any]]:
    data = response.json()
    if not isinstance(data, list):
        raise SnapshotUnavailableError(f"unexpected answer from {response.url}")
    return [item for item in data if isinstance(item, dict)]


def _pick(data: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, JsonValue]:
    return {key: data[key] for key in keys if key in data}


class GitHub:
    def __init__(self, spec: SourceSpec, http: HttpClient, settings: Settings) -> None:
        self._spec = spec
        self._http = http
        self._api = f"https://api.github.com/repos/{spec.identity}"
        self._headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if settings.github_token is not None:
            self._headers["Authorization"] = f"Bearer {settings.github_token.get_secret_value()}"

    async def resolve(self) -> str:
        ref = self._spec.ref or "HEAD"
        if _FULL_SHA.fullmatch(ref):
            return ref.lower()
        response = await _get(
            self._http,
            f"{self._api}/commits/{quote(ref, safe='')}",
            f"no commit, branch or tag {ref!r} in {self._spec.url}",
            headers=self._headers,
            use_cache=False,
        )
        return str(_json_dict(response)["sha"])

    async def listing(self, revision: str | None) -> Listing:
        assert revision is not None
        response = await _get(
            self._http,
            f"{self._api}/git/trees/{revision}?recursive=1",
            f"cannot list {self._spec.url}",
            headers=self._headers,
        )
        data = _json_dict(response)
        listing = Listing()
        if data.get("truncated"):
            listing.incomplete.append(LISTING_TRUNCATED)
        base = f"https://raw.githubusercontent.com/{self._spec.identity}/{revision}"
        for entry in data.get("tree", []):
            path = str(entry.get("path", ""))
            size = entry.get("size")
            if entry.get("type") == "commit":
                listing.skipped.append(Skipped(path, None, SUBMODULE))
            elif entry.get("type") == "blob" and entry.get("mode") == "120000":
                listing.skipped.append(Skipped(path, size, SYMLINK))
            elif entry.get("type") == "blob":
                listing.files.append(
                    RemoteFile(path, size, _loader(self._http, f"{base}/{quote(path)}"))
                )
        return listing


class GitLab:
    def __init__(self, spec: SourceSpec, http: HttpClient) -> None:
        self._spec = spec
        self._http = http
        self._api = f"https://gitlab.com/api/v4/projects/{quote(spec.identity, safe='')}"

    async def resolve(self) -> str:
        ref = self._spec.ref
        if ref is not None and _FULL_SHA.fullmatch(ref):
            return ref.lower()
        if ref is None:
            project = await _get(
                self._http, self._api, f"no project {self._spec.url}", use_cache=False
            )
            ref = str(_json_dict(project)["default_branch"])
        response = await _get(
            self._http,
            f"{self._api}/repository/commits/{quote(ref, safe='')}",
            f"no commit, branch or tag {ref!r} in {self._spec.url}",
            use_cache=False,
        )
        return str(_json_dict(response)["id"])

    async def listing(self, revision: str | None) -> Listing:
        assert revision is not None
        listing = Listing()
        page = "1"
        for _ in range(MAX_LISTING_PAGES):
            response = await _get(
                self._http,
                f"{self._api}/repository/tree?ref={revision}&recursive=true&per_page=100"
                f"&page={page}",
                f"cannot list {self._spec.url}",
            )
            for entry in _json_list(response):
                path = str(entry.get("path", ""))
                if entry.get("type") == "commit":
                    listing.skipped.append(Skipped(path, None, SUBMODULE))
                elif entry.get("type") == "blob" and entry.get("mode") == "120000":
                    listing.skipped.append(Skipped(path, None, SYMLINK))
                elif entry.get("type") == "blob":
                    url = f"{self._api}/repository/files/{quote(path, safe='')}/raw?ref={revision}"
                    # GitLab's tree gives no sizes; the download limit catches large files.
                    listing.files.append(RemoteFile(path, None, _loader(self._http, url)))
            page = response.headers.get("x-next-page", "")
            if not page:
                return listing
        listing.incomplete.append(LISTING_TOO_LONG)
        return listing


class HuggingFace:
    def __init__(self, spec: SourceSpec, http: HttpClient, settings: Settings) -> None:
        self._spec = spec
        self._http = http
        self._type = spec.repo_type or "model"
        self._api = f"https://huggingface.co/api/{self._type}s/{spec.identity}"
        prefix = "" if self._type == "model" else f"{self._type}s/"
        self._files = f"https://huggingface.co/{prefix}{spec.identity}/resolve"
        self._headers: dict[str, str] = {}
        if settings.hf_token is not None:
            self._headers["Authorization"] = f"Bearer {settings.hf_token.get_secret_value()}"

    async def resolve(self) -> str:
        ref = self._spec.ref or "main"
        if _FULL_SHA.fullmatch(ref):
            return ref.lower()
        response = await _get(
            self._http,
            f"{self._api}/revision/{quote(ref, safe='')}",
            f"no revision {ref!r} of {self._spec.url}",
            headers=self._headers,
            use_cache=False,
        )
        return str(_json_dict(response)["sha"])

    async def listing(self, revision: str | None) -> Listing:
        assert revision is not None
        listing = Listing()
        url: str | None = f"{self._api}/tree/{revision}?recursive=true"
        for _ in range(MAX_LISTING_PAGES):
            if url is None:
                break
            response = await _get(
                self._http, url, f"cannot list {self._spec.url}", headers=self._headers
            )
            for entry in _json_list(response):
                if entry.get("type") != "file":
                    continue
                path = str(entry.get("path", ""))
                size = entry.get("size")
                if "lfs" in entry:
                    listing.skipped.append(Skipped(path, size, LFS))
                    continue
                file_url = f"{self._files}/{revision}/{quote(path)}"
                listing.files.append(
                    RemoteFile(path, size, _loader(self._http, file_url, self._headers))
                )
            found = _NEXT_LINK.search(response.headers.get("link", ""))
            url = found.group(1) if found else None
        if url is not None:
            listing.incomplete.append(LISTING_TOO_LONG)
        if self._type == "dataset":
            await self._dataset_size(listing)
        return listing

    async def _dataset_size(self, listing: Listing) -> None:
        """Row counts per split, for checking a paper's split sizes against the data."""
        url = (
            "https://datasets-server.huggingface.co/size?dataset="
            f"{quote(self._spec.identity, safe='')}"
        )
        response = await self._http.get(url, headers=self._headers)
        if response.ok:
            listing.metadata["dataset_size"] = response.json()
            listing.metadata["dataset_size_note"] = (
                "computed by Hugging Face for the dataset's current main branch, "
                "which can differ from the pinned revision"
            )
        else:
            listing.metadata["dataset_size_unavailable"] = f"HTTP {response.status_code}"


class Kaggle:
    _API = "https://www.kaggle.com/api/v1/datasets"
    _VIEW_KEYS = (
        "title",
        "subtitle",
        "licenseName",
        "totalBytes",
        "currentVersionNumber",
        "lastUpdated",
    )

    def __init__(self, spec: SourceSpec, http: HttpClient) -> None:
        self._spec = spec
        self._http = http
        self._view: dict[str, Any] = {}

    async def resolve(self) -> str:
        response = await _get(
            self._http,
            f"{self._API}/view/{self._spec.identity}",
            f"no public Kaggle dataset {self._spec.url}",
            use_cache=False,
        )
        self._view = _json_dict(response)
        return self._spec.ref or str(self._view["currentVersionNumber"])

    async def listing(self, revision: str | None) -> Listing:
        assert revision is not None
        listing = Listing(metadata=_pick(self._view, self._VIEW_KEYS))
        described: list[JsonValue] = []
        base = f"{self._API}/list/{self._spec.identity}?datasetVersionNumber={revision}"
        token = ""
        for _ in range(MAX_LISTING_PAGES):
            url = base + (f"&pageToken={quote(token, safe='')}" if token else "")
            data = _json_dict(await _get(self._http, url, f"cannot list {self._spec.url}"))
            for entry in data.get("datasetFiles", []):
                name = str(entry.get("name", ""))
                size = entry.get("totalBytes")
                columns = [str(c.get("name")) for c in entry.get("columns", []) or []]
                described.append({"name": name, "bytes": size, "columns": list(columns)})
                download = (
                    f"{self._API}/download/{self._spec.identity}/{quote(name)}"
                    f"?datasetVersionNumber={revision}"
                )
                listing.files.append(RemoteFile(name, size, _loader(self._http, download)))
            token = str(data.get("nextPageToken") or "")
            if not token:
                break
        else:
            listing.incomplete.append(LISTING_TOO_LONG)
        # Data files are not downloaded, but their names, sizes and columns are evidence for
        # what a paper says about its data.
        listing.metadata["files"] = described
        return listing


class AnonymousGitHub:
    def __init__(self, spec: SourceSpec, http: HttpClient) -> None:
        self._spec = spec
        self._http = http
        self._api = f"https://anonymous.4open.science/api/repo/{spec.identity}"
        self._options: dict[str, Any] = {}

    async def resolve(self) -> None:
        response = await _get(
            self._http,
            f"{self._api}/options",
            f"no anonymised repository {self._spec.url}",
            use_cache=False,
        )
        self._options = _json_dict(response)
        if self._options.get("download") is False:
            raise SnapshotUnavailableError(f"the owner of {self._spec.url} disabled downloads")
        # The host keeps no commit id, so the revision comes from the listing.
        return None

    async def listing(self, revision: str | None) -> Listing:
        listing = Listing(
            metadata={
                **_pick(
                    self._options,
                    ("anonymizedAt", "sourceCommitDate", "lastUpdateDate", "truncatedFolders"),
                ),
                "note": "Content is anonymised by the host; some words are replaced by XXXX.",
            }
        )
        truncated = [str(f) for f in self._options.get("truncatedFolders") or []]
        if truncated:
            listing.incomplete.append(f"the host truncated these folders: {', '.join(truncated)}")
        identity: list[tuple[str, str, int]] = []
        folders = [""]
        crawled = 0
        while folders:
            if crawled >= MAX_CRAWLED_FOLDERS:
                listing.incomplete.append(LISTING_TOO_LONG)
                break
            folder = folders.pop(0)
            crawled += 1
            url = f"{self._api}/files/" + (f"?path={quote(folder)}" if folder else "")
            response = await _get(self._http, url, f"cannot list {self._spec.url}", use_cache=False)
            for entry in _json_list(response):
                name = str(entry.get("name", ""))
                path = f"{folder}/{name}" if folder else name
                if "size" not in entry:
                    if excluded_dir(name):
                        listing.skipped.append(Skipped(f"{path}/", None, EXCLUDED_DIR))
                    else:
                        folders.append(path)
                    continue
                size = int(entry["size"])
                identity.append((path, str(entry.get("sha", "")), size))
                url = f"{self._api}/file/{quote(path)}"
                listing.files.append(RemoteFile(path, size, _loader(self._http, url)))
        digest = hashlib.sha256(json.dumps(sorted(identity)).encode("utf-8")).hexdigest()
        listing.revision = f"listing-{digest[:16]}"
        return listing


class Zenodo:
    def __init__(self, spec: SourceSpec, http: HttpClient, limits: DownloadLimits) -> None:
        self._spec = spec
        self._http = http
        self._limits = limits
        self._record: dict[str, Any] = {}

    async def resolve(self) -> str:
        response = await _get(
            self._http,
            f"https://zenodo.org/api/records/{self._spec.identity}",
            f"no Zenodo record {self._spec.url}",
            use_cache=False,
        )
        self._record = _json_dict(response)
        return str(self._record["id"])

    async def listing(self, revision: str | None) -> Listing:
        metadata = self._record.get("metadata") or {}
        listing = Listing(
            metadata={
                **_pick(self._record, ("doi",)),
                **_pick(metadata, ("title", "version", "publication_date")),
            }
        )
        entries = list(self._record.get("files") or [])
        archives = [e for e in entries if str(e.get("key", "")).lower().endswith(".zip")]
        for entry in entries:
            key = str(entry.get("key", ""))
            size = entry.get("size")
            links = entry.get("links") or {}
            url = str(links.get("content") or links.get("self") or "")
            if entry not in archives:
                listing.files.append(RemoteFile(key, size, _loader(self._http, url)))
                continue
            if size is not None and size > self._limits.max_archive_bytes:
                listing.skipped.append(Skipped(key, size, "archive larger than the limit"))
                listing.incomplete.append(f"archive {key} was not downloaded")
                continue
            response = await _get(
                self._http,
                url,
                f"cannot download {key} from {self._spec.url}",
                use_cache=False,
                max_bytes=self._limits.max_archive_bytes,
            )
            _check_md5(key, response.content, str(entry.get("checksum", "")))
            prefix = "" if len(archives) == 1 else f"{PurePosixPath(key).stem}/"
            _add_zip_members(listing, response.content, prefix)
        return listing


def _check_md5(key: str, content: bytes, checksum: str) -> None:
    algorithm, _, expected = checksum.partition(":")
    if algorithm == "md5" and hashlib.md5(content, usedforsecurity=False).hexdigest() != expected:
        raise SnapshotUnavailableError(f"{key} does not match its published checksum")


def _add_zip_members(listing: Listing, content: bytes, prefix: str) -> None:
    """Members of a zip archive as files, without the single top folder release archives
    wrap everything in. Symbolic links and encrypted members are skipped."""
    archive = zipfile.ZipFile(io.BytesIO(content))
    members = [info for info in archive.infolist() if not info.is_dir()]
    tops = {info.filename.split("/", 1)[0] for info in members}
    strip = len(tops) == 1 and all("/" in info.filename for info in members)
    for info in members:
        name = info.filename.split("/", 1)[1] if strip else info.filename
        path = prefix + name
        if (info.external_attr >> 16) & 0o170000 == 0o120000:
            listing.skipped.append(Skipped(path, info.file_size, SYMLINK))
        elif info.flag_bits & 0x1:
            listing.skipped.append(Skipped(path, info.file_size, "encrypted archive member"))
        else:
            listing.files.append(RemoteFile(path, info.file_size, _zip_loader(archive, info)))


def _zip_loader(
    archive: zipfile.ZipFile, info: zipfile.ZipInfo
) -> Callable[[int], Awaitable[bytes]]:
    async def load(max_bytes: int) -> bytes:
        too_large = ResponseTooLargeError(f"{info.filename} is larger than {max_bytes} bytes")
        if info.file_size > max_bytes:
            raise too_large
        with archive.open(info) as member:
            # The header's size can lie; never read more than the limit.
            data = member.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise too_large
        return data

    return load


class LocalFolder:
    """A folder on disk, such as a pinned copy patched to inject a code-side error."""

    def __init__(self, spec: SourceSpec) -> None:
        assert spec.local_path is not None
        self._root = spec.local_path

    async def resolve(self) -> None:
        # The revision is the hash of the copied files, known after the copy.
        return None

    async def listing(self, revision: str | None) -> Listing:
        return await asyncio.to_thread(self._walk)

    def _walk(self) -> Listing:
        listing = Listing()
        for folder, dirnames, filenames in os.walk(self._root, followlinks=False):
            here = Path(folder)
            relative = here.relative_to(self._root).as_posix()
            kept = []
            for name in sorted(dirnames):
                path = name if relative == "." else f"{relative}/{name}"
                if (here / name).is_symlink():
                    listing.skipped.append(Skipped(path, None, SYMLINK))
                elif excluded_dir(name):
                    listing.skipped.append(Skipped(f"{path}/", None, EXCLUDED_DIR))
                else:
                    kept.append(name)
            dirnames[:] = kept
            for name in sorted(filenames):
                file = here / name
                path = name if relative == "." else f"{relative}/{name}"
                if file.is_symlink():
                    listing.skipped.append(Skipped(path, None, SYMLINK))
                elif safe_relative_path(path) is None:
                    listing.skipped.append(Skipped(path, None, UNSAFE_PATH))
                else:
                    size = file.stat().st_size
                    listing.files.append(RemoteFile(path, size, _file_loader(file)))
        return listing


def _file_loader(file: Path) -> Callable[[int], Awaitable[bytes]]:
    def read(max_bytes: int) -> bytes:
        with file.open("rb") as handle:
            data = handle.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ResponseTooLargeError(f"{file} is larger than {max_bytes} bytes")
        return data

    async def load(max_bytes: int) -> bytes:
        return await asyncio.to_thread(read, max_bytes)

    return load
