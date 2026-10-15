from pathlib import Path

import pytest
from pydantic import ValidationError

from auditor.config import DEFAULT_ALLOWED_HOSTS, Settings


def test_defaults() -> None:
    settings = Settings()
    assert settings.data_dir == Path("data")
    assert settings.cache_dir == Path("data") / "cache"
    assert settings.grobid_url == "http://localhost:8070"
    assert settings.log_level == "INFO"
    assert settings.allowed_hosts == DEFAULT_ALLOWED_HOSTS
    assert settings.llm_provider is None


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


def test_unrelated_dotenv_lines_are_ignored(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OTHER_TOOL_KEY=abc\n", encoding="utf-8")
    assert Settings().log_level == "INFO"


def test_rejects_unknown_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDITOR_LOG_LEVEL", "VERBOSE")
    with pytest.raises(ValidationError):
        Settings()


def test_complex_values_are_read_as_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDITOR_ALLOWED_HOSTS", '["example.org"]')
    monkeypatch.setenv("AUDITOR_LLM_OPTIONS", '{"thinking_level": "low"}')
    settings = Settings()
    assert settings.allowed_hosts == ("example.org",)
    assert settings.llm_options == {"thinking_level": "low"}


def test_empty_host_tokens_count_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUDITOR_GITHUB_TOKEN", "")
    monkeypatch.setenv("AUDITOR_HF_TOKEN", "hf_x")
    settings = Settings(_env_file=None)
    assert settings.github_token is None
    assert settings.hf_token is not None
    assert settings.hf_token.get_secret_value() == "hf_x"
