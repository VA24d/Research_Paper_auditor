import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from auditor.config import Settings
from auditor.infra.log import configure_logging
from auditor.infra.manifest import build_manifest, new_run_id, redact, write_manifest
from auditor.infra.scope import Scope, enter_scope
from auditor.schemas import RunManifest, SystemName


def test_run_ids_are_readable_and_unique() -> None:
    now = datetime(2026, 10, 10, 12, 30, tzinfo=UTC)
    first = new_run_id(SystemName.B1, now)
    second = new_run_id(SystemName.B1, now)
    assert re.fullmatch(r"b1-20261010T123000Z-[0-9a-f]{6}", first)
    assert first != second


def test_redact_hides_credentials_at_any_depth() -> None:
    config = {
        "llm_options": {"api_key": "sk-123", "temperature": 0},
        "nested": [{"access_token": "t"}],
        "model": "x",
    }
    assert redact(config) == {
        "llm_options": {"api_key": "<redacted>", "temperature": 0},
        "nested": [{"access_token": "<redacted>"}],
        "model": "x",
    }


def test_manifest_never_contains_secrets(tmp_path: Path) -> None:
    settings = Settings(llm_options={"api_key": "sk-secret-value"})
    manifest = build_manifest(
        run_id="r1", system=SystemName.FULL, settings=settings, model="m", seed=7
    )
    path = write_manifest(manifest, tmp_path / "r1")
    text = path.read_text(encoding="utf-8")
    assert "sk-secret-value" not in text
    loaded = RunManifest.model_validate(json.loads(text))
    assert loaded.seed == 7
    assert loaded.packages["langgraph"]
    assert loaded.started_at.tzinfo is not None


def test_logging_includes_scope_ids(tmp_path: Path) -> None:
    log_file = tmp_path / "run.log"
    logger = configure_logging("INFO", log_file)
    with enter_scope(Scope(run_id="r9", task_id="citation-abc")):
        logging.getLogger("auditor.test").info("checking")
    logging.getLogger("auditor.test").info("outside")
    for handler in logger.handlers:
        handler.flush()
    lines = log_file.read_text(encoding="utf-8").splitlines()
    assert "[r9 citation-abc] checking" in lines[0]
    assert "[- -] outside" in lines[1]


def test_configure_logging_is_idempotent(tmp_path: Path) -> None:
    configure_logging("INFO")
    logger = configure_logging("DEBUG", tmp_path / "x.log")
    assert len(logger.handlers) == 2
    assert logger.level == logging.DEBUG


@pytest.fixture(autouse=True)
def reset_auditor_logger() -> object:
    yield
    logger = logging.getLogger("auditor")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
