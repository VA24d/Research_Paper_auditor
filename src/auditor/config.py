from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_ALLOWED_HOSTS = (
    "localhost",
    "127.0.0.1",
    "arxiv.org",
    "export.arxiv.org",
    "api.openalex.org",
    "api.crossref.org",
    "api.semanticscholar.org",
    "github.com",
    "api.github.com",
    "codeload.github.com",
    "raw.githubusercontent.com",
)


class Settings(BaseSettings):
    """Runtime settings, read from AUDITOR_* environment variables and `.env`.

    Relative paths resolve against the working directory, so commands are
    expected to run from the repository root.
    """

    model_config = SettingsConfigDict(
        env_prefix="AUDITOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("data")
    cache_dir: Path = Path("data/cache")
    runs_dir: Path = Path("runs")
    grobid_url: str = "http://localhost:8070"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    contact_email: str | None = None
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS
    http_timeout_s: float = Field(default=30.0, gt=0)
    http_max_attempts: int = Field(default=4, ge=1)
    http_cache_ttl_days: float = Field(default=30.0, ge=0)

    llm_provider: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_timeout_s: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    llm_structured_method: str | None = None
    llm_options: dict[str, JsonValue] = Field(default_factory=dict)
    llm_cache: bool = True
