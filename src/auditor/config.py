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
    # Full-text hosts the citation agent reaches through the indexes' pdf_url and
    # landing_page_url fields (auditor.retrieval.fulltext). doi.org only redirects — every
    # redirect hop is checked against this list, so the target hosts must be here too.
    "doi.org",
    "aclanthology.org",
    "openaccess.thecvf.com",
    "proceedings.mlr.press",
    "papers.nips.cc",
    "proceedings.neurips.cc",
    "jmlr.org",
    "link.springer.com",
    "onlinelibrary.wiley.com",
    "mdpi.com",
    "ouci.dntb.gov.ua",
    # HAL serves institutional copies from many subdomains (inria.hal.science, ...).
    "hal.science",
    ".hal.science",
    # More full-text hosts observed during the set B runs (session 7, 2026-11-02): the
    # indexes hand out publisher and repository landing pages freely, so the list grows
    # with observed hosts. Paywalled hosts 403 without a subscription; the retrieval
    # then falls back to the abstract cleanly. Entries with a "." cover subdomains, so
    # "mdpi.com" (bare) needs ".mdpi.com" to also cover "www.mdpi.com".
    "proceedings.com",
    ".proceedings.com",
    "discovery.ucl.ac.uk",
    "linkinghub.elsevier.com",
    ".sciencedirect.com",
    "ieeexplore.ieee.org",
    ".science.org",
    "academic.oup.com",
    "cds.cern.ch",
    "citeseerx.ist.psu.edu",
    "cocosci.berkeley.edu",
    "dare.uva.nl",
    ".acm.org",
    "springer.com",
    ".springer.com",
    ".springeropen.com",
    ".iop.org",
    ".aaai.org",
    ".rsna.org",
    ".acpjournals.org",
    ".cell.com",
    ".emerald.com",
    ".mdpi.com",
    "nature.com",
    ".nature.com",
    ".openreview.net",
    ".spiedigitallibrary.org",
    ".tandfonline.com",
    "panafrican-med-journal.com",
    ".panafrican-med-journal.com",
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
    # The review model's provider and structured-output method when they differ from the
    # main model's (for example Gemini for the cheap steps and a Vertex partner model,
    # provider "vertex_maas", for the review steps).
    llm_review_provider: str | None = None
    llm_review_structured_method: str | None = None
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
    # A ledger shared by all runs (one JSON line per priced call) and its dollar limit.
    llm_ledger: Path | None = None
    llm_ledger_limit_usd: float | None = Field(default=None, gt=0)
    # US dollars per million (input, output) tokens; output includes thinking. Gemini API
    # list prices as of 2026-10-17 (3.8 Flash is $1.50 / $7.50 from 2027-01-01).
    # The share of the input price a provider charges for input read from its prompt
    # cache (Gemini bills cached tokens at a fraction of the input price). 1 counts them
    # in full, which keeps AUDITOR_LLM_MAX_COST_USD on the safe side.
    llm_cached_input_share: float = Field(default=1.0, ge=0, le=1)
    # Explicit context caching on Vertex Gemini of the long first message that review-model
    # calls about a paper share (auditor.infra.prefix_cache). Each cache is a short-lived
    # resource on the project, deleted at the end of the run; storage is charged to the
    # dollar cap for the whole lifetime, at `llm_cache_storage_price` US dollars per million
    # tokens per hour.
    llm_explicit_cache: bool = False
    llm_explicit_cache_ttl_s: int = Field(default=1200, gt=0)
    llm_explicit_cache_min_chars: int = Field(default=16_000, ge=0)
    llm_cache_storage_price: float = Field(default=1.0, ge=0)
    llm_prices: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {
            "gemini-2.5-flash": (0.30, 2.50),
            "gemini-3.5-flash-lite": (0.30, 2.50),
            "gemini-3.8-flash": (0.75, 3.75),
            # Vertex partner models: placeholders above their list prices, so the dollar cap
            # errs on the safe side until the real prices are filled in.
            "deepseek-ai/deepseek-v4.1-flash-maas": (0.60, 2.40),
            "deepseek-ai/deepseek-v3.2-maas": (0.60, 1.80),
            "moonshotai/kimi-k2-thinking-maas": (0.80, 3.20),
            "zai-org/glm-5-maas": (1.00, 3.50),
            "zai-org/glm-4.7-maas": (0.60, 2.40),
            "qwen/qwen3-235b-a22b-instruct-2507-maas": (0.40, 1.60),
            "qwen/qwen3-next-80b-a3b-instruct-maas": (0.30, 1.50),
            "qwen/qwen3-coder-480b-a35b-instruct-maas": (1.00, 4.00),
            "minimaxai/minimax-m2-maas": (0.40, 1.60),
            "openai/gpt-oss-120b-maas": (0.20, 0.80),
        }
    )
    # How the code agent compares a paper's method with its code: "pairs" asks one small
    # question per (paper sentence, code function) pair, which suits small local models;
    # "whole" sends the method text and the related files in one call.
    method_check_mode: Literal["pairs", "whole", "rag"] = "rag"
    # An OpenAI-compatible embeddings endpoint for code retrieval in "rag" mode (for
    # example LM Studio at http://localhost:1234); without one, BM25 ranks the code.
    embed_url: str | None = None
    embed_model: str = "text-embedding-nomic-embed-text-v1.5"
    # Method sentences per call in "rag" mode, each group with its own code; 0 sends the
    # whole method text in one call.
    method_rag_sentences: int = Field(default=0, ge=0)
    # Which model reads the method and the code: the main model or the review model. On
    # 10 set C pairs, 3.8 Flash found 4 of 28 SciCoQA discrepancies in "rag" mode, 2 in
    # "whole" mode and none in "pairs" mode.
    method_check_model: Literal["main", "review"] = "review"
    method_check_pairs: int = Field(default=12, gt=0)
    # Ask the model, one code line at a time, which of several repository values a setting
    # mismatch is about (only when an LLM is configured).
    code_second_look: bool = True
    # The code and data claim loop (auditor.agents.code.claim_loop). The planner lists up
    # to `claim_loop_max_claims` claims that rest on the released code or data (dataset
    # and split, preprocessing, number of training examples, architecture, training
    # procedure, evaluation protocol) in one cheap call; each becomes a code task that the
    # review model checks against the repository with tools, in at most
    # `claim_loop_max_tool_calls` tool calls and `claim_loop_max_model_calls` model calls.
    claim_loop_enabled: bool = False
    claim_loop_max_claims: int = Field(default=6, gt=0)
    claim_loop_max_tool_calls: int = Field(default=12, gt=0)
    claim_loop_max_model_calls: int = Field(default=8, gt=0)
    claim_loop_plan_max_output: int = Field(default=4096, gt=0)
    # Low thinking: on 3.8 Flash a medium-effort turn took 46 to 86 s on average on set C,
    # and a claim makes up to 8 turns one after another.
    claim_loop_effort: Literal["low", "medium", "high"] = "low"
    # A claim starts no model call after this many seconds and ends without a verdict.
    claim_loop_timeout_s: float = Field(default=540.0, gt=0)
    # What the claims are about: the experiments (data, splits, evaluation), or the
    # method's implementation details, read from the method sections.
    claim_loop_focus: Literal["experiments", "method"] = "experiments"
    # Which model explores: the review model, or the main (cheap) one.
    claim_loop_model: Literal["main", "review"] = "review"
    # The planner's model pass over citation candidates (Part 2's checks). Runs that only
    # evaluate other parts can leave it out; it was a quarter of the cost of a full run.
    citation_llm_enabled: bool = True
    # The abstract-only ablation (P2.4, P2.9): the citation agent never fetches a cited
    # work's full text, so it judges claims against the abstract alone.
    citation_abstract_only: bool = False
    # How passages are ranked against a claim: "bm25" (pure Python, no model, the team's
    # default) or "minilm" (all-MiniLM-L6-v2 as a ~23 MB quantised ONNX model on the CPU,
    # better for paraphrases; needs the `embed` extra, `uv sync --extra embed`, and
    # gracefully falls back to BM25 when it is not installed or the model is offline).
    citation_embedder: Literal["bm25", "minilm"] = "bm25"
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
    # Deep-check tries across all of a paper's theory tasks. Without it each task had its
    # own: one long SPOT paper made 13 deep checks and cost $1.30.
    theory_deep_per_paper: int = Field(default=4, ge=0)
    # Thinking of a deep check's first round, and what of the paper it sees: the whole
    # paper (the same in every deep check, so a provider's prompt cache can reuse it), or
    # only the sections holding the claim and what it uses, with their neighbours. On five
    # SPOT dev papers, medium thinking with the whole paper found the same error as high
    # for $0.45 instead of $1.62; the sections found none.
    theory_deep_effort: Literal["low", "medium", "high"] = "medium"
    # How claims are picked for deep checks: "screen" lists them with the main model and
    # screens them cheaply; "triage" has the review model read the whole paper and name
    # the claims most likely to be wrong, which go straight to deep checks. On the SPOT dev
    # half the screen called 140 of 162 claims fine, including the wrong ones; triage
    # found 6 of 15 errors in 5 of 13 papers, against 3 and then 1 of 15 for the screen.
    theory_selector: Literal["screen", "triage"] = "triage"
    # Output cap of the triage call, thinking included.
    theory_triage_max_output: int = Field(default=16_384, gt=0)
    # Independent looks at a paper, each with its own triage and deep checks; findings are
    # pooled and repeats merged (D20). Runs find mostly different errors: on the SPOT test
    # half two looks caught 5 of 13 papers by both judges, one look 4, for $2.48 against
    # $1.68.
    theory_repeats: int = Field(default=2, ge=1)
    theory_deep_context: Literal["paper", "sections"] = "paper"
    # Let deep checks run one piece of Python in the Docker math sandbox before answering
    # (auditor.infra.sandbox); without Docker or its image they answer without it.
    theory_sandbox_enabled: bool = True
    # A second look, on the review model, at each text-table inconsistency the arithmetic
    # finds: the numbers it used may be the wrong column or the wrong comparison.
    results_challenge_enabled: bool = True
    # Read each captioned table from the PDF's text inside its page box (parser.table_crops)
    # before the layout text and GROBID's tables.
    results_crop_tables: bool = True
    # A second extraction pass for numbers the text quotes from a table (D22): prose
    # sentences with a decimal number that name a table or a row's label, at most
    # `results_number_max_sentences` per paper, are read again with the tables they use,
    # and each number is paired with the cell it is read from. On set D the per-section
    # extraction found none of 7 planted wrong numbers.
    results_number_pass_enabled: bool = False
    # The cell rule (D22), no model: a number a sentence gives for a row of a table it
    # names that no cell of the row holds. Its findings go to the challenge step when a
    # model is configured. Off until measured on more papers.
    results_cell_rule_enabled: bool = True
    results_number_max_sentences: int = Field(default=12, ge=0)
    results_number_effort: Literal["low", "medium", "high"] = "medium"
    # The model that reads the cells: the main model, or the review model.
    results_number_model: Literal["main", "review"] = "main"
    # Which model does the deep checks: the main (cheap) one, or the review model. On SPOT,
    # Flash-Lite deep checks mostly answered "no error"; the stronger model costs more.
    theory_deep_model: Literal["main", "review"] = "main"
