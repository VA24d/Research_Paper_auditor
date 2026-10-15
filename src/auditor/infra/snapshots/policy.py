"""Which files of a repository or dataset are worth downloading, and how much.

Only core files are fetched: source code, configuration, scripts and documentation, where
settings, splits and metrics are written down. Data, weights, media and vendored or
generated folders are listed in the manifest with the reason they were skipped, never
downloaded.
"""

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Self

from auditor.config import Settings

# Files that state settings directly come first when a budget runs out.
SETTINGS_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".yaml",
        ".yml",
        ".json",
        ".jsonnet",
        ".toml",
        ".cfg",
        ".ini",
        ".conf",
        ".gin",
        ".sh",
        ".bash",
    }
)
OTHER_SUFFIXES = frozenset({".ipynb", ".md", ".rst", ".txt"})
CORE_NAMES = frozenset({"makefile", "dockerfile"})
EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".github",
        ".gitlab",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        "site-packages",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".ipynb_checkpoints",
        ".idea",
        ".vscode",
        "wandb",
        "mlruns",
        "lightning_logs",
        "dist",
        "build",
    }
)

# Core files that rarely state an experiment's settings; they are kept only when the
# budget allows after everything else.
DEMOTED_DIRS = frozenset({"tests", "test", "docs", "doc", "templates"})

# Files that hold a dataset's rows. They are listed in every snapshot and downloaded only
# in data analysis mode (auditor.infra.snapshots.data).
DATA_SUFFIXES = frozenset({".csv", ".tsv", ".jsonl", ".json", ".parquet"})

NOT_CORE = "not a core file type"
EXCLUDED_DIR = "excluded directory"
TOO_LARGE = "larger than the per-file limit"
TOTAL_LIMIT = "total download limit reached"
COUNT_LIMIT = "file count limit reached"
UNSAFE_PATH = "unsafe path"
CASE_DUPLICATE = "same path as another file when case is ignored"
SYMLINK = "symbolic link"

_WINDOWS_RESERVED = re.compile(r"^(con|prn|aux|nul|com\d|lpt\d)(\..*)?$", re.IGNORECASE)
_BAD_CHARS = re.compile(r'[<>:"|?*\\\x00-\x1f]')
_MAX_PATH_CHARS = 200


@dataclass(frozen=True)
class DownloadLimits:
    max_file_bytes: int = 1_000_000
    max_total_bytes: int = 10_000_000
    max_files: int = 500
    max_archive_bytes: int = 50_000_000

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(
            max_file_bytes=settings.snapshot_max_file_bytes,
            max_total_bytes=settings.snapshot_max_total_bytes,
            max_files=settings.snapshot_max_files,
            max_archive_bytes=settings.snapshot_max_archive_bytes,
        )

    def key(self) -> str:
        """Short id of these limits; snapshots taken under other limits are kept apart."""
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:8]


@dataclass(frozen=True)
class RemoteFile:
    """A file a host lists. `size` is None when the host does not say; `load(max_bytes)`
    downloads it and raises ResponseTooLargeError past `max_bytes`."""

    path: str
    size: int | None
    load: Callable[[int], Awaitable[bytes]]
    # Where the file can be downloaded from, and its SHA-256 when the host publishes one.
    url: str | None = None
    sha256: str | None = None


@dataclass(frozen=True)
class Skipped:
    path: str
    size: int | None
    reason: str


@dataclass(frozen=True)
class Selection:
    chosen: tuple[RemoteFile, ...]
    skipped: tuple[Skipped, ...]
    incomplete: tuple[str, ...]


def safe_relative_path(raw: str) -> str | None:
    """`raw` as a relative POSIX path that is safe to create on Windows, macOS and Linux,
    or None. Rejects absolute paths, `..`, drive letters, backslashes and reserved names."""
    if not raw or len(raw) > _MAX_PATH_CHARS or raw.startswith("/"):
        return None
    parts = [part for part in raw.split("/") if part not in {"", "."}]
    if not parts:
        return None
    for part in parts:
        if (
            part == ".."
            or _BAD_CHARS.search(part)
            or _WINDOWS_RESERVED.match(part)
            or part.endswith((" ", "."))
        ):
            return None
    return "/".join(parts)


def excluded_dir(name: str) -> bool:
    return name.lower() in EXCLUDED_DIRS or name.lower().endswith(".egg-info")


def core_file_reason(path: str) -> str | None:
    """None if `path` is a core file, otherwise why it is skipped."""
    pure = PurePosixPath(path)
    if any(excluded_dir(part) for part in pure.parts[:-1]):
        return EXCLUDED_DIR
    if _rank(pure) is None:
        return NOT_CORE
    return None


def _rank(path: PurePosixPath) -> int | None:
    suffix = path.suffix.lower()
    if suffix in SETTINGS_SUFFIXES:
        rank = 0
    elif suffix in OTHER_SUFFIXES or path.name.lower() in CORE_NAMES:
        rank = 1
    else:
        return None
    if any(part.lower() in DEMOTED_DIRS for part in path.parts[:-1]):
        rank += 2
    return rank


def select_files(files: Iterable[RemoteFile], limits: DownloadLimits) -> Selection:
    """The core files to download within `limits`, in priority order, and why every other
    file is left out. Files of unknown size are budgeted when they are downloaded."""
    candidates: list[RemoteFile] = []
    skipped: list[Skipped] = []
    seen: set[str] = set()
    for file in files:
        path = safe_relative_path(file.path)
        reason = UNSAFE_PATH if path is None else core_file_reason(path)
        if reason is None and file.size is not None and file.size > limits.max_file_bytes:
            reason = TOO_LARGE
        if reason is None and path is not None and path.casefold() in seen:
            reason = CASE_DUPLICATE
        if reason is not None or path is None:
            skipped.append(Skipped(file.path, file.size, reason or UNSAFE_PATH))
            continue
        seen.add(path.casefold())
        candidates.append(RemoteFile(path, file.size, file.load, file.url, file.sha256))

    def priority(file: RemoteFile) -> tuple[int, int, str]:
        pure = PurePosixPath(file.path)
        return (_rank(pure) or 0, len(pure.parts), file.path)

    chosen: list[RemoteFile] = []
    incomplete: list[str] = []
    total = 0
    for file in sorted(candidates, key=priority):
        reason = None
        if len(chosen) >= limits.max_files:
            reason = COUNT_LIMIT
        elif file.size is not None and total + file.size > limits.max_total_bytes:
            reason = TOTAL_LIMIT
        if reason is not None:
            skipped.append(Skipped(file.path, file.size, reason))
            if reason not in incomplete:
                incomplete.append(reason)
            continue
        chosen.append(file)
        total += file.size or 0
    return Selection(tuple(chosen), tuple(skipped), tuple(incomplete))
