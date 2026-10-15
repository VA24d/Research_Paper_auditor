# Evidence-Gated Verification Agent for Research Paper Auditing

Group project for **CL3.410: Language Models and Agents (Monsoon 2026)**.

## Team: Hood Research Department

- Vijay Aravynthan (2023111007)
- Anurag Peddi (2023101090)
- Itikela Bhaskar (2023111032)
- Sudheera Y S (2023111002)

## Overview

An automated verification system that audits research conference submissions for factual correctness (specifically citations and data) before human review. The system decomposes submissions into checkable items, inspects cited full texts and code repositories, and enforces evidence-gated verification where flagged discrepancies must be backed by verbatim quotes.

## Documents

- [Project Proposal (Markdown)](./LMA_Group_Project_Proposal.md)
- [Project Proposal (PDF)](./LMA%20group%20project%20proposal.pdf)
- [Team Plan](./docs/team_plan.md): who does what, task list, timeline
- [Design Decisions](./docs/decisions.md): contracts and rules every part follows

## System Pipeline

1. **Submission & Venue Profile**: Paper text, bibliography, repository, and venue rules.
2. **Planner & State Tracker**: Extracts claims, citations, numbers, and code settings; tracks audit items.
3. **Specialists**:
   - **Citation Agent**: Reference lookup, full-text retrieval, and claim validity check.
   - **Code Agent**: Static repository search for settings, splits, and metrics.
   - **Template Checker**: Deterministic format rules check (no LLM call).
4. **Verifier Agent**: Tests suspected errors against source evidence; rejects flags lacking verbatim quotes.
5. **Output States**: Verified flag (quoted evidence), Needs human review, or Dropped.

## Development Setup

Works on Windows, macOS and Linux. Run every command from the repository root.

### Prerequisites

- Git
- [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.9 or newer. uv installs
  the right Python version (3.13) by itself.
  - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh` (or `brew install uv` on macOS)
- Docker Desktop (Docker Engine on Linux), only if you work on PDF parsing. It runs GROBID.

### First-time setup

```bash
git clone https://github.com/itikelabhaskar/Research_Paper_auditor.git
cd Research_Paper_auditor
uv sync
uv run pre-commit install
uv run pytest
```

Then copy `.env.example` to `.env` (`cp .env.example .env`, or
`Copy-Item .env.example .env` in PowerShell) and fill in any keys you need. `.env` is
git-ignored.

### Everyday commands

| Task | Command |
|---|---|
| Run tests | `uv run pytest` |
| Lint | `uv run ruff check` |
| Format | `uv run ruff format` |
| Type check | `uv run mypy` |
| Add a dependency | `uv add <package>` (commit the updated `uv.lock`) |

CI runs the same checks on Linux, Windows and macOS for every pull request.

### GROBID (PDF parsing)

```bash
docker compose up -d
```

GROBID is ready when <http://localhost:8070/api/isalive> returns `true`, which takes about
20 seconds. Tests that need it are skipped by default; run them with `uv run pytest -m grobid`.
Stop it with `docker compose down`. Give Docker at least 4 GB of memory. The image has an
arm64 build, so it runs natively on Apple Silicon.

### Test markers

Tests that need external services are skipped unless selected explicitly:

| Marker | Needs | Run with |
|---|---|---|
| `grobid` | GROBID running locally | `uv run pytest -m grobid` |
| `network` | Internet access to external APIs | `uv run pytest -m network` |
| `llm` | An LLM API key; costs tokens | `uv run pytest -m llm` |

## Repository Layout

```text
src/auditor/
  config.py        settings from environment and .env
  schemas/         shared contracts between all parts
  infra/           LLM factory, HTTP client and policy gate, caching, logging
  parser/          PDF to ParsedPaper
  planner/         ParsedPaper to AuditTasks
  orchestration/   LangGraph workflow
  baselines/       B0 and B1
  agents/          citation, code, template and verifier agents
  retrieval/       cited-paper retrieval and passage search
  reporting/       audit report generation
  evaluation/      metrics and experiment runner
tests/             mirrors src/auditor
configs/           venue profiles and run configurations
datasets/          dataset manifests and gold labels (no PDFs or repositories)
scripts/           helper scripts (Python only, so they run on every OS)
docs/              plans and design decisions
```

## Using the Auditor

Every command accepts a PDF (parsed with GROBID, so start it first) or a `ParsedPaper`
JSON file such as `tests/fixtures/schemas/parsed_paper.json`.

| Task | Command |
|---|---|
| Parse a PDF | `uv run auditor parse paper.pdf -o paper.json` |
| Full audit | `uv run auditor audit paper.pdf [--repo URL] [--repo-commit SHA] [--venue venue.json] [--no-llm]` |
| Resume an interrupted audit | `uv run auditor audit paper.pdf --run-id <id> --resume` |
| Baseline B0 or B1 | `uv run auditor baseline b0 paper.pdf` (needs an LLM in `.env`) |
| Experiment over a dataset | `uv run auditor batch dataset.jsonl --experiment main [--systems full b0 b1] [--runs 3]` |
| Set C: candidates, injection, validation, scoring | `uv run python scripts/set_c.py --help` |
| Data analysis mode | add `--analyse-data` to `audit`, `baseline` or `batch` (Parquet needs `uv sync --extra data`) |
| Check the LLM provider | `uv run python scripts/probe_llm.py` |
| Parser statistics | `uv run python scripts/parser_report.py data/papers` |
| Citation-link accuracy | `uv run python scripts/citation_sample.py score data/papers datasets/parser_fixtures/citation_gold.jsonl` |

Each run writes to `runs/<run_id>/`: `manifest.json` (git commit, settings with secrets
removed, package versions), `usage/*.jsonl` (every tool call, HTTP request and LLM call),
and per paper `tasks.jsonl`, `results.jsonl`, `verified.jsonl` and `predictions.jsonl`.

### Experiments

A dataset file lists one paper per line as JSON:

```json
{"paper": "papers/2106.09685.pdf", "repository": "https://github.com/microsoft/LoRA", "repository_commit": "<commit sha>"}
{"paper": "injected/2106.09685-citation.json", "venue": "venues/neurips.json"}
```

`paper` is a PDF or a `ParsedPaper` JSON file (such as a copy with injected errors);
`repository`, `repository_commit` and `venue` are optional. Give the commit the gold labels
were checked against, so every system inspects the same snapshot. Relative paths are
relative to the dataset file.

`auditor batch` runs each system `--runs` times. Run ids are `<experiment>-<system>-r<n>`,
so running the same command again skips papers that already have predictions, resumes
interrupted ones and retries failed ones. Every failure, including B1 stopping without an
answer, is kept in the run's `failures.jsonl` and the batch continues. A run is only
continued with the same inputs: if the model, the code, a package version, any paper,
repository or venue, or any setting other than paths and logging changed, the batch refuses
and asks for a new `--experiment` name. `auditor audit --resume` applies the same check. LLM answers are not cached in a batch unless `--llm-cache` is given, and
`--strict` refuses to start while any part is still a mock.

### Docker

For the final runs, the whole pipeline also runs in a container, so everyone gets the same
Python and package versions:

```bash
docker compose run --rm auditor batch dataset.jsonl --experiment main
```

This builds the image on first use, starts GROBID and waits until it is ready, and mounts
the repository at the container's working directory, so paths and `.env` work as they do
on the host. Rebuild after pulling code changes with `docker compose build auditor`.
An API key in `.env` reaches the container as is. Gemini on Vertex AI is different: the
container cannot see your `gcloud` login, so point `GOOGLE_APPLICATION_CREDENTIALS` in
`.env` at a service-account key file kept under `data/` (git-ignored), using the path
relative to the repository root.

## Plugging In Your Part

The pipeline finds each part by convention and uses a mock until it exists (the run
manifest records which parts were mocked):

| Part | Expose | Signature |
|---|---|---|
| Citation agent | `check` in `auditor/agents/citation/__init__.py` | `async def check(task: AuditTask, ctx: RunContext) -> TaskResult` |
| Code agent | `check` in `auditor/agents/code/__init__.py` | same |
| Template checker | `check` in `auditor/agents/template/__init__.py` | same |
| Verifier | `verify` in `auditor/agents/verifier/__init__.py` | `async def verify(results: Sequence[TaskResult], ctx: RunContext) -> list[VerifiedFinding]` |
| Tools for B1 | `make_tools` in `auditor/agents/<name>/tools.py` | `def make_tools(ctx: RunContext) -> list[BaseTool]` |

The shape of a check, in `auditor/agents/citation/__init__.py`. A real citation check also
retrieves the cited work and quotes it as evidence; this one only shows the plumbing:

```python
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from auditor.infra.context import RunContext
from auditor.schemas import AuditTask, CandidateFinding, ErrorClass, TaskOutcome, TaskResult


class Support(BaseModel):
    supported: bool
    reason: str


async def check(task: AuditTask, ctx: RunContext) -> TaskResult:
    if not task.claim:
        return TaskResult.for_task(task, TaskOutcome.UNVERIFIABLE, reason="no claim text")
    result = await ctx.require_llm().generate(Support, [HumanMessage(task.claim)])
    answer = result.value  # a Support instance; result.usage has the token counts
    findings = []
    if not answer.supported:
        findings.append(
            CandidateFinding.for_task(
                task, 0, error_class=ErrorClass.CLAIM_NOT_SUPPORTED, summary=answer.reason
            )
        )
    return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=findings)
```

Inside `check`:

- `ctx.paper` is the `ParsedPaper`; `ctx.repository` is the code repository and commit to
  inspect, when known; `ctx.http` is the only way to reach the network (allowlist, rate
  limits and cache included).
- `await ctx.require_llm().generate(Schema, messages)` returns a `StructuredResult`:
  `result.value` is the validated `Schema` instance, `result.usage` the token counts.
  Invalid replies are retried with feedback before it gives up.
- `TaskResult.for_task` and `CandidateFinding.for_task` copy the task's ids; a finding's
  anchor defaults to the task's, so pass a narrower one when you can point at the exact
  span.
- Decorate shared tool functions with `@tracked_tool()` (`auditor.infra.tools`), so calls
  are counted for the cost metrics and the task budget.
- Return `outcome=unverifiable` with a reason when the check cannot be done. Do not catch
  unexpected errors; the orchestrator retries temporary ones and records the rest as
  `failed`.

For tests, `auditor.testing` provides `offline_context(tmp_path)` (example paper, no
network), `ScriptedChatModel` with `tool_reply(...)` (a fake LLM), and `checked_result(task)`.
Example inputs for every contract are in `tests/fixtures/schemas/` and
`auditor.schemas.examples`.
