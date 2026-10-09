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
