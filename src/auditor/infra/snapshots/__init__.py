"""Fetching the core files of a paper's code or data (decisions D8 and D11).

`fetch_snapshot(source, http, settings)` downloads code, configuration and documentation
from GitHub, GitLab, Hugging Face, Kaggle datasets, Anonymous GitHub, Zenodo or a local
folder into the scratch folder, within size limits, and never runs any of it.
"""

from auditor.infra.snapshots.hosts import FileUnavailableError, SnapshotUnavailableError
from auditor.infra.snapshots.policy import (
    DownloadLimits,
    RemoteFile,
    Selection,
    Skipped,
    core_file_reason,
    safe_relative_path,
    select_files,
)
from auditor.infra.snapshots.snapshot import (
    SkippedFile,
    Snapshot,
    SnapshotFile,
    SnapshotManifest,
    fetch_snapshot,
)
from auditor.infra.snapshots.sources import SourceError, SourceKind, SourceSpec, parse_source

__all__ = [
    "DownloadLimits",
    "FileUnavailableError",
    "RemoteFile",
    "Selection",
    "Skipped",
    "SkippedFile",
    "Snapshot",
    "SnapshotFile",
    "SnapshotManifest",
    "SnapshotUnavailableError",
    "SourceError",
    "SourceKind",
    "SourceSpec",
    "core_file_reason",
    "fetch_snapshot",
    "parse_source",
    "safe_relative_path",
    "select_files",
]
