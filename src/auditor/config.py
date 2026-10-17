import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, JsonValue, SecretStr, field_validator
from pydantic.fields import FieldInfo
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

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
    # OpenReview serves public submissions, reviews and decisions (scripts/real_cases.py).
    "api2.openreview.net",
    "openreview.net",
)


class _Unless(PydanticBaseSettingsSource):
    """A settings source minus the fields a higher-priority source already sets.

    pydantic-settings merges dict values across sources, so `AUDITOR_LLM_OPTIONS` given for
    one run would otherwise keep every option from `.env` it does not mention (such as a
    temperature or a thinking budget). A value set at a higher level replaces it whole."""

    def __init__(
        self, source: PydanticBaseSettingsSource, *higher: PydanticBaseSettingsSource
    ) -> None:
        super().__init__(source.settings_cls)
        self.source = source
        self.higher = higher

    def get_field_value(self, field: FieldInfo, field_name: str) -> tuple[Any, str, bool]:
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        taken = {key for higher in self.higher for key in higher()}
        return {key: value for key, value in self.source().items() if key not in taken}


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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            _Unless(env_settings, init_settings),
            _Unless(dotenv_settings, init_settings, env_settings),
            file_secret_settings,
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
    # A long paper with high thinking takes Gemini 3.8 Flash up to about 7 minutes.
    llm_timeout_s: float = Field(default=600.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)
    llm_structured_method: str | None = None
    llm_options: dict[str, JsonValue] = Field(default_factory=dict)
    llm_cache: bool = True
    # "run" reuses a cached answer only within the run that made it, so repeated runs stay
    # independent while the copies of a paper and a resumed run share answers (D10).
    llm_cache_scope: Literal["shared", "run"] = "shared"
    # A stronger model for the review steps that judge other checks' findings (the merge of
    # the theory findings), from the same provider; its options default to llm_options.
    llm_review_model: str | None = None
    llm_review_options: dict[str, JsonValue] | None = None
    # The model's context window in tokens. Prompts that do not fit fail loudly instead of
    # being cut by the server (Ollama's default context is only a few thousand tokens).
    llm_context_tokens: int | None = Field(default=None, gt=0)
    # Model calls in flight at once across all papers and tasks of a run (a local server
    # serves only a few requests in parallel).
    llm_max_concurrency: int | None = Field(default=None, gt=0)
    # Model calls one run may make in total; a hard stop for metered or shared credits.
    # Cache hits do not count.
    llm_max_calls: int | None = Field(default=None, ge=0)
    # Dollars one run may spend on model calls, shared by the main and the review model.
    llm_max_cost_usd: float | None = Field(default=None, gt=0)
    # US dollars per million (input, output) tokens; output includes thinking. Gemini API
    # list prices as of 2026-10-17 (3.8 Flash is $1.50 / $7.50 from 2027-01-01).
    llm_prices: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {
            "gemini-2.5-flash": (0.30, 2.50),
            "gemini-3.5-flash-lite": (0.30, 2.50),
            "gemini-3.8-flash": (0.75, 3.75),
        }
    )
    # How the code agent compares a paper's method with its code: "pairs" asks one small
    # question per (paper sentence, code function) pair, which suits small local models;
    # "whole" sends the method text and the related files in one call.
    method_check_mode: Literal["pairs", "whole"] = "pairs"
    method_check_pairs: int = Field(default=12, gt=0)
    # Ask the model, one code line at a time, which of several repository values a setting
    # mismatch is about (only when an LLM is configured).
    code_second_look: bool = True
    # The planner's model pass over citation candidates (Part 2's checks). Runs that only
    # evaluate other parts can leave it out; it was a quarter of the cost of a full run.
    citation_llm_enabled: bool = True
    # Check the mathematics of the paper's formal claims with the LLM (decision D21). The
    # planner lists the claims in one call and groups them into at most `theory_max_tasks`
    # tasks of `theory_batch_size` claims; each task screens its claims in one cheap call,
    # and at most `theory_deep_checks` of them get a deep check with high thinking.
    theory_check_enabled: bool = False
    theory_batch_size: int = Field(default=8, gt=0)
    theory_max_tasks: int = Field(default=6, gt=0)
    theory_deep_checks: int = Field(default=2, ge=0)
    # Tries of a high-risk claim's deep check; an error any try shows counts, and the
    # challenge step filters what is not real. Answers vary from one try to the next.
    theory_deep_samples: int = Field(default=2, ge=1)
    # Let deep checks run one piece of Python in the Docker math sandbox before answering
    # (auditor.infra.sandbox); without Docker or its image they answer without it.
    theory_sandbox_enabled: bool = True
    # Which model does the deep checks: the main (cheap) one, or the review model. On SPOT,
    # Flash-Lite deep checks mostly answered "no error"; the stronger model costs more.
    theory_deep_model: Literal["main", "review"] = "main"
