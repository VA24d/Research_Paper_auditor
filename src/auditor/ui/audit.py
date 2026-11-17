"""What the UI needs besides Streamlit: building an `auditor audit` run, reading its
progress and spend, and finding finished reports. Kept free of Streamlit so it is tested
like the rest of the package.

An audit from the UI is the CLI's `auditor audit` in a child process, so it writes the same
run folder (manifest, usage logs, `report.md` and `report.json`) and obeys the same limits.
Every UI audit charges a ledger of its own (`data/ui/ledger.jsonl`) with a total limit, on
top of the per-audit dollar limit, so a demo cannot run away with the shared credits.
"""

import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from auditor.config import Settings
from auditor.experiments import read_dataset
from auditor.infra.llm import ledger_total
from auditor.reporting import REPORT_JSON
from auditor.schemas import DatasetEntry, LLMCallRecord

# Gemini models on Vertex; not every project serves every model (a project created in
# November 2026 has no gemini-2.5-flash).
MODELS = ("gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash")
DEFAULT_PROJECT = "lmaproject6767"
RUN_PREFIX = "ui-"
# Where the model comes from: the team's Vertex project, a Gemini API key the user brings,
# or any OpenAI-compatible server (OpenAI, OpenRouter, Ollama, vLLM, LM Studio, ...).
VERTEX, GEMINI_KEY, OPENAI_COMPATIBLE = "vertex", "gemini_key", "openai_compatible"


@dataclass(frozen=True)
class AuditRequest:
    paper: Path
    repository: str | None = None
    commit: str | None = None
    venue: Path | None = None
    use_llm: bool = True
    model: str = MODELS[0]
    project: str = DEFAULT_PROJECT
    skip_citation: bool = True
    max_cost_usd: float = 0.5
    ledger: Path = Path("data/ui/ledger.jsonl")
    ledger_limit_usd: float = 5.0
    embed_url: str | None = None
    source: str = VERTEX
    # The user's own key: passed to the child process in the provider's own variable,
    # never in AUDITOR_* settings, so it is not written to the manifest or the cache key.
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    # Runs folder for this audit; None uses AUDITOR_RUNS_DIR.
    runs_dir: Path | None = None


def model_choices(settings: Settings) -> tuple[str, ...]:
    """The models the UI offers: the Gemini model configured in `.env` first."""
    configured = settings.llm_model if settings.llm_provider == "google_genai" else None
    return tuple(dict.fromkeys(m for m in (configured, *MODELS) if m))


def default_project(settings: Settings) -> str:
    """The Vertex project configured in AUDITOR_LLM_OPTIONS, else the team's first one."""
    project = settings.llm_options.get("project")
    return project if isinstance(project, str) and project else DEFAULT_PROJECT


def new_run_id(now: datetime | None = None) -> str:
    return RUN_PREFIX + (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")


def command(request: AuditRequest, run_id: str) -> list[str]:
    """The `auditor audit` command line for `request`."""
    args = [sys.executable, "-m", "auditor.cli", "audit", str(request.paper), "--run-id", run_id]
    if request.repository:
        args += ["--repo", request.repository]
        if request.commit:
            args += ["--repo-commit", request.commit]
    if request.venue is not None:
        args += ["--venue", str(request.venue)]
    if not request.use_llm:
        args.append("--no-llm")
    return args


def _default_prices() -> dict[str, tuple[float, float]]:
    return dict(Settings.model_fields["llm_prices"].get_default(call_default_factory=True))


def environment(request: AuditRequest, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The child process's environment: `base` (the current one by default) with the
    request's model, limits and parts. Settings not named here come from `.env` as usual."""
    env = dict(os.environ if base is None else base)
    env["PYTHONUNBUFFERED"] = "1"
    env["AUDITOR_SKIP_PARTS"] = json.dumps(["citation"] if request.skip_citation else [])
    if request.runs_dir is not None:
        env["AUDITOR_RUNS_DIR"] = str(request.runs_dir)
    if not request.use_llm:
        return env
    for name in ("AUDITOR_LLM_BASE_URL", "AUDITOR_LLM_MAX_COST_USD", "AUDITOR_LLM_LEDGER"):
        env.pop(name, None)
    env.update(
        {
            "AUDITOR_LLM_MODEL": request.model,
            "AUDITOR_LLM_REVIEW_MODEL": request.model,
            "AUDITOR_LLM_MAX_CONCURRENCY": "4",
            "AUDITOR_LLM_MAX_RETRIES": "5",
            # With the citation agent left out, its planner pass only costs money.
            "AUDITOR_CITATION_LLM_ENABLED": "false" if request.skip_citation else "true",
        }
    )
    if request.source == VERTEX:
        options: dict[str, object] = {
            "vertexai": True,
            "project": request.project,
            "location": "global",
            "max_output_tokens": 32768,
        }
        env["AUDITOR_LLM_PROVIDER"] = "google_genai"
        # The shared credits: a limit per audit and one for all UI audits together.
        env["AUDITOR_LLM_MAX_COST_USD"] = f"{request.max_cost_usd:g}"
        env["AUDITOR_LLM_LEDGER"] = str(request.ledger)
        env["AUDITOR_LLM_LEDGER_LIMIT_USD"] = f"{request.ledger_limit_usd:g}"
    elif request.source == GEMINI_KEY:
        options = {"max_output_tokens": 32768}
        env["AUDITOR_LLM_PROVIDER"] = "google_genai"
        env.pop("GOOGLE_GENAI_USE_VERTEXAI", None)
        env["GOOGLE_API_KEY"] = request.api_key or ""
    elif request.source == OPENAI_COMPATIBLE:
        options = {}
        env["AUDITOR_LLM_PROVIDER"] = "openai"
        if request.base_url:
            env["AUDITOR_LLM_BASE_URL"] = request.base_url
        # The OpenAI client needs some key; local servers ignore it.
        env["OPENAI_API_KEY"] = request.api_key or "not-needed"
    else:
        raise ValueError(f"unknown model source {request.source!r}")
    if request.source != VERTEX and request.model in _default_prices():
        # The user's own money: only the per-audit limit, and only for a priced model
        # (an unpriced one cannot be limited, see StructuredLLM.from_settings).
        env["AUDITOR_LLM_MAX_COST_USD"] = f"{request.max_cost_usd:g}"
    if request.source != VERTEX:
        env.pop("AUDITOR_LLM_LEDGER_LIMIT_USD", None)
    env["AUDITOR_LLM_OPTIONS"] = json.dumps(options)
    if request.embed_url:
        env["AUDITOR_EMBED_URL"] = request.embed_url
    return env


def spent(ledger: Path) -> float:
    """Dollars the UI's audits have charged so far."""
    return ledger_total(ledger)


@dataclass(frozen=True)
class Progress:
    calls: int
    cost_usd: float
    input_tokens: int
    output_tokens: int


def progress(run_dir: Path, settings: Settings | None = None) -> Progress:
    """Model calls a run has made so far and what they cost, from its usage log."""
    prices = (settings or Settings()).llm_prices
    calls = cost = tokens_in = tokens_out = 0.0
    log = run_dir / "usage" / "llm_calls.jsonl"
    if log.is_file():
        for line in log.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                call = LLMCallRecord.model_validate_json(line)
            except ValueError:
                continue  # a line still being written
            if call.cache_hit:
                continue
            calls += 1
            tokens_in += call.input_tokens
            tokens_out += call.output_tokens
            price = prices.get(call.model.split(":", 1)[-1])
            if price is not None:
                cost += (call.input_tokens * price[0] + call.output_tokens * price[1]) / 1e6
    return Progress(int(calls), cost, int(tokens_in), int(tokens_out))


def reports(runs_dir: Path) -> list[Path]:
    """Every `report.json` under the runs folder, newest run first."""
    found = list(runs_dir.glob(f"*/*/{REPORT_JSON}"))
    return sorted(found, key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)


def report_of(run_dir: Path) -> Path | None:
    """The report of a one-paper run, once it is written."""
    found = sorted(run_dir.glob(f"*/{REPORT_JSON}"))
    return found[0] if found else None


def demo_papers(dataset: Path) -> list[DatasetEntry]:
    """The papers of an experiment dataset whose files exist here, for the demo list."""
    if not dataset.is_file():
        return []
    return [entry for entry in read_dataset(dataset) if Path(entry.paper).is_file()]


def save_upload(name: str, data: bytes, folder: Path) -> Path:
    """An uploaded file, stored once by content under its original suffix."""
    suffix = Path(name).suffix.lower()
    path = folder / f"{hashlib.sha256(data).hexdigest()[:16]}{suffix}"
    if not path.is_file():
        folder.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return path


def tail(lines: Sequence[str], count: int = 15) -> str:
    """The last `count` lines, without the logging prefix, for the progress panel."""
    out = []
    for line in lines[-count:]:
        # "2026-11-08 03:32:21,123 WARNING auditor.x [run task] message" -> "WARNING message"
        parts = line.split(" ", 4)
        if len(parts) == 5 and parts[2] in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            message = parts[4].split("] ", 1)[-1]
            line = f"{parts[2]} {message}"
        out.append(line.rstrip())
    return "\n".join(out)
