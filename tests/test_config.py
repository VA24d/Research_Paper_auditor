from pathlib import Path

import pytest
from pydantic import ValidationError

from auditor.config import Settings

ENV_VARS = ("AUDITOR_DATA_DIR", "AUDITOR_CACHE_DIR", "AUDITOR_GROBID_URL", "AUDITOR_LOG_LEVEL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def test_defaults() -> None:
    settings = Settings()
    assert settings.data_dir == Path("data")
    assert settings.cache_dir == Path("data") / "cache"
    assert settings.grobid_url == "http://localhost:8070"
    assert settings.log_level == "INFO"


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDITOR_GROBID_URL", "http://grobid:8070")
    monkeypatch.setenv("AUDITOR_LOG_LEVEL", "DEBUG")
    settings = Settings()
    assert settings.grobid_url == "http://grobid:8070"
    assert settings.log_level == "DEBUG"


def test_reads_dotenv_from_working_directory(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("AUDITOR_DATA_DIR=elsewhere\n", encoding="utf-8")
    assert Settings().data_dir == Path("elsewhere")


def test_environment_wins_over_dotenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("AUDITOR_LOG_LEVEL=ERROR\n", encoding="utf-8")
    monkeypatch.setenv("AUDITOR_LOG_LEVEL", "WARNING")
    assert Settings().log_level == "WARNING"


def test_rejects_unknown_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDITOR_LOG_LEVEL", "VERBOSE")
    with pytest.raises(ValidationError):
        Settings()
