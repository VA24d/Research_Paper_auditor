"""Run ids and run manifests (decision D10)."""

import platform
import secrets
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from pydantic import JsonValue

from auditor.config import Settings
from auditor.infra.files import atomic_write
from auditor.schemas import RunManifest, SystemName

TRACKED_PACKAGES = (
    "research-paper-auditor",
    "langgraph",
    "langchain",
    "langchain-core",
    "langchain-google-genai",
    "langchain-openai",
    "pydantic",
    "httpx",
)

_SECRET_MARKERS = ("key", "token", "secret", "password", "credential")


def new_run_id(system: SystemName, now: datetime | None = None) -> str:
    moment = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{system.value}-{moment}-{secrets.token_hex(3)}"


def redact(value: Any) -> Any:
    """Drop values stored under names that look like credentials."""
    if isinstance(value, dict):
        return {
            key: "<redacted>"
            if any(marker in key.lower() for marker in _SECRET_MARKERS)
            else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def _git(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            continue
    return versions


def build_manifest(
    *,
    run_id: str,
    system: SystemName,
    settings: Settings,
    model: str | None,
    seed: int | None = None,
    dataset_version: str | None = None,
    extra_config: dict[str, JsonValue] | None = None,
) -> RunManifest:
    status = _git("status", "--porcelain")
    config = redact(settings.model_dump(mode="json"))
    if extra_config:
        config.update(redact(extra_config))
    return RunManifest(
        run_id=run_id,
        system=system,
        started_at=datetime.now(UTC),
        git_commit=_git("rev-parse", "HEAD"),
        git_dirty=None if status is None else bool(status),
        model=model,
        llm_cache=settings.llm_cache,
        seed=seed,
        dataset_version=dataset_version,
        python_version=platform.python_version(),
        platform=platform.platform(),
        packages=_package_versions(),
        config=config,
    )


def write_manifest(manifest: RunManifest, directory: Path) -> Path:
    path = directory / "manifest.json"
    atomic_write(path, (manifest.model_dump_json(indent=2) + "\n").encode("utf-8"))
    return path
