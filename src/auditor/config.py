import tempfile
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, SecretStr, field_validator
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
    # Code and data hosts read by auditor.infra.snapshots.
    "gitlab.com",
    "huggingface.co",
    "datasets-server.huggingface.co",
    # Hugging Face serves data files from regional CDN hosts such as us.aws.cdn.hf.co and
    # cas-bridge.xethub.hf.co; an entry starting with "." allows its subdomains.
    ".hf.co",
    "www.kaggle.com",
    # Kaggle answers a download with a signed redirect to Google Cloud Storage.
    "storage.googleapis.com",
    "anonymous.4open.science",
    "zenodo.org",
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
    # Downloaded repository and data snapshots. Disposable: anything here is fetched again.
    scratch_dir: Path = Path(tempfile.gettempdir()) / "auditor-scratch"
    grobid_url: str = "http://localhost:8070"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    contact_email: str | None = None
    allowed_hosts: tuple[str, ...] = DEFAULT_ALLOWED_HOSTS
    http_timeout_s: float = Field(default=30.0, gt=0)
    http_max_attempts: int = Field(default=4, ge=1)
    http_cache_ttl_days: float = Field(default=30.0, ge=0)

    # Limits for one repository or data snapshot (auditor.infra.snapshots). Only core files
    # (code, configs, docs) are downloaded, and only up to these sizes.
    snapshot_max_file_bytes: int = Field(default=1_000_000, gt=0)
    snapshot_max_total_bytes: int = Field(default=10_000_000, gt=0)
    snapshot_max_files: int = Field(default=500, gt=0)
    snapshot_max_archive_bytes: int = Field(default=50_000_000, gt=0)
    # Data analysis mode downloads a dataset's data files to count rows, read labels and
    # find test rows that also occur in training data (decision D4). Off by default; part
    # of the run fingerprint like every other setting.
    data_analysis_enabled: bool = False
    data_analysis_max_file_bytes: int = Field(default=200_000_000, gt=0)
    data_analysis_max_total_bytes: int = Field(default=1_000_000_000, gt=0)
    data_analysis_max_hashed_rows: int = Field(default=2_000_000, gt=0)
    # Optional tokens: GitHub allows 60 API requests an hour without one; Hugging Face
    # needs one for gated repositories.
    github_token: SecretStr | None = None
    hf_token: SecretStr | None = None

    @field_validator("github_token", "hf_token", mode="before")
    @classmethod
    def _empty_token_is_unset(cls, value: object) -> object:
        # `.env.example` leaves these empty; an empty token would fail every request.
        return None if value == "" else value

    llm_provider: str | None = None
    llm_model: str | None = None
    llm_base_url: str | None = None
    llm_timeout_s: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    llm_structured_method: str | None = None
    llm_options: dict[str, JsonValue] = Field(default_factory=dict)
    llm_cache: bool = True
    # The model's context window in tokens. Prompts that do not fit fail loudly instead of
    # being cut by the server (Ollama's default context is only a few thousand tokens).
    llm_context_tokens: int | None = Field(default=None, gt=0)
    # Model calls in flight at once across all papers and tasks of a run (a local server
    # serves only a few requests in parallel).
    llm_max_concurrency: int | None = Field(default=None, gt=0)
    # How the code agent compares a paper's method with its code: "pairs" asks one small
    # question per (paper sentence, code function) pair, which suits small local models;
    # "whole" sends the method text and the related files in one call.
    method_check_mode: Literal["pairs", "whole"] = "pairs"
    method_check_pairs: int = Field(default=12, gt=0)
