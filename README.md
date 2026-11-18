# Research Paper Auditor: Evidence-Gated Verification

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![GitHub Pages](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-blue.svg)](https://va24d.github.io/Research_Paper_auditor/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B.svg)](https://researchpaperauditor.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Model: Gemini 3.8 Flash](https://img.shields.io/badge/Model-Gemini%203.8%20Flash-purple.svg)](https://cloud.google.com/vertex-ai)

> **Course Project:** CL3.410 Language Models and Agents (Monsoon 2026)  
> **Team:** Hood Research Department  
> **Affiliations:** Software Engineering Research Center (SERC) & Center for Visual Information Technology (CVIT), IIIT Hyderabad  
> **Authors:** Vijay Aravynthan (2023111007), Itikela Bhaskar (2023111032), Anurag Peddi (2023101090), Sudheera Y S (2023111002)  
> **Repository:** [https://github.com/VA24d/Research_Paper_auditor](https://github.com/VA24d/Research_Paper_auditor)

---

## Deliverables & Links

- 🌐 **Project Website (GitHub Pages)**: [https://va24d.github.io/Research_Paper_auditor/](https://va24d.github.io/Research_Paper_auditor/) *(source in [`website/`](./website/))*
- 🖥️ **Live Web UI (Streamlit)**: [https://researchpaperauditor.streamlit.app](https://researchpaperauditor.streamlit.app) *(or run locally at `http://localhost:8501` via `uv run auditor ui`)*
- 📄 **Final Research Report (PDF)**: [`Final_Report/report.pdf`](./Final_Report/report.pdf)
- 🖼️ **Research Poster (ISO A1 Landscape)**: [`Final_Report/poster/A1 size Landscape Research poster - new.pptx`](./Final_Report/poster/A1%20size%20Landscape%20Research%20poster%20-%20new.pptx) ([PDF Export](./website/static/pdf/poster.pdf))
- 📋 **Design Decisions & Contracts**: [`docs/decisions.md`](./docs/decisions.md)

---

## Overview

Over 62,000 papers are submitted to major AI conferences annually. Reviewers have only 15–20 minutes per paper and rarely clone released code repositories or read cited papers in depth. Consequently, factual discrepancies slip into publication undetected.

The **Research Paper Auditor** is an evidence-gated verification agent that audits research conference submissions (PDF or LaTeX) against their **cited full texts and GitHub code repositories**. 

### Core Guarantee: Evidence Gating
**A discrepancy is flagged ONLY when backed by dual verbatim quotes** (from the manuscript and from the raw repository source). If a candidate claim lacks a verifiable quote, it is dropped to prevent false accusations.

---

## Key Results (Set C Benchmark: 80 Paper Versions)

| System / Model | Method F1 | Metric F1 | Setting F1 | **Macro-F1** | Clean Control False Alarms | Dual-Quote Grounding | Cost / Paper |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full System (Ours)** | **0.34** | **0.80** | **0.77** | **0.636 ± 0.004** | **0 / 72 (100% P)** | **100%** | **$0.079 (₹7.4)** |
| No Verifier (Ablation) | 0.34 | 0.80 | 0.86 | 0.667 ± 0.003 | 0 / 72 | 100% | $0.079 (₹7.4) |
| B0: Zero-Shot LLM | 0.00 | 0.00 | 0.14 | 0.048 ± 0.015 | 0 / 72 | 0% | $0.047 (₹4.4) |
| B1: ReAct Agent | 0.00 | 0.00 | 0.14 | 0.046 ± 0.007 | 0 / 72 | 0% | $0.542 (₹50.9) |

- **13× Performance Gain**: Baselines fail completely on method and metric checks ($0.00$ F1).
- **Zero False Alarms**: Across 72 clean control papers, 0 false alarms were raised.
- **7× Cost Reduction**: Specialist routing costs $0.079/paper vs $0.542 for ReAct agents.

---

## Examples of Code Claims Checked

### 1. Real Accepted Paper Contradiction (Mirzadeh et al., NeurIPS 2020)
- **Paper Claim (§B.2 Continual Learning Setup)**:
  > *"… we have not employed any data augmentation or shuffling."*
- **Code Reality (`stable_sgd/data_utils.py:28`)**:
  ```python
  train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
  ```
- **Auditor Outcome**: `VERIFIED_FLAG` (`SETTING_MISMATCH`). Both lines quoted verbatim without prompt engineering.

### 2. Hyperparameter & Architecture Settings
- **Learning Rates**: Paper states `lr = 0.0001`; code configuration parses `parser.add_argument("--lr", default=0.001)`.
- **Batch Sizes**: Paper claims `batch_size = 64`; training script executes `python train.py --batch_size 32`.
- **Optimizers**: Paper specifies `AdamW`; constructor initializes `torch.optim.SGD(lr=0.01)`.

### 3. Metric Mismatch Claims
- **Metric Scope**: Paper reports achieving `F1 = 89.2%` without specifying type; code strictly computes `sklearn.metrics.f1_score(..., average="micro")` rather than standard macro-averaged F1.
- **Loss Formulations**: Paper claims optimizing cross-entropy with label smoothing; code optimizes standard NLL loss without smoothing.

### 4. Method & Algorithmic Consistency
- **Data Augmentation**: Paper text specifies random horizontal flips and color jittering; the PyTorch `transforms.Compose` pipeline only applies `ToTensor()` and `Normalize()`.
- **Freezing / Fine-Tuning**: Paper claims the image encoder backbone was frozen; code leaves `requires_grad=True` across all transformer layers.

### 5. Directional Table-to-Text Errors
- **arXiv:2312.11580**: Paper text claims *"PlaNet-S (0.73) is higher than U-Net (0.78)"*. Caught automatically as a reversed arithmetic comparison against Table 2.

---

## Development Setup

Works on macOS, Linux, and Windows. Run commands from the repository root.

### Prerequisites

- Git
- [uv](https://docs.astral.sh/uv/) 0.9 or newer (installs Python 3.13 automatically).
- Docker Desktop (for GROBID PDF parsing).

### First-Time Setup

```bash
git clone https://github.com/VA24d/Research_Paper_auditor.git
cd Research_Paper_auditor
uv sync
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

### Math sandbox

```bash
docker build -t auditor-math:1 docker/math
```

The theory checker's deep checks can run one piece of model-written Python in this image,
with no network and tight limits (D11). Without the image they run without it. Its tests
run with `uv run pytest -m docker`.

### Test markers

Tests that need external services are skipped unless selected explicitly:

| Marker | Needs | Run with |
|---|---|---|
| `grobid` | GROBID running locally | `uv run pytest -m grobid` |
| `docker` | Docker and the `auditor-math:1` image | `uv run pytest -m docker` |
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

Every command accepts a PDF, a `ParsedPaper` JSON file such as
`tests/fixtures/schemas/parsed_paper.json`, or a paper's main LaTeX file (`.tex`). A PDF
is parsed with GROBID, so start it first; a PDF parsed before is read from the cache in
`data/cache/grobid` without GROBID (`auditor parse --refresh` parses it again).

| Task | Command |
|---|---|
| Parse a PDF | `uv run auditor parse paper.pdf -o paper.json` |
| Full audit | `uv run auditor audit paper.pdf [--repo URL] [--repo-commit SHA] [--venue venue.json] [--no-llm]` |
| Resume an interrupted audit | `uv run auditor audit paper.pdf --run-id <id> --resume` |
| Baseline B0 or B1 | `uv run auditor baseline b0 paper.pdf` (needs an LLM in `.env`) |
| Experiment over a dataset | `uv run auditor batch dataset.jsonl --experiment main [--systems full b0 b1] [--runs 3]` |
| Score runs against gold labels | `uv run auditor evaluate labels.jsonl runs/main-full-r1 runs/main-b1-r1 [--dataset dataset.jsonl] [--area code] [--markdown out.md] [--json out.json]` |
| Rebuild one paper's audit report | `uv run auditor report runs/<run_id>/<paper>-<variant> --paper paper.json` |
| Web UI: run audits and read reports | `uv run auditor ui` (after `uv sync --extra ui`) |
| Rerun the verifier on an old run (no model calls) | `uv run python scripts/verifier_replay.py runs/<run_id> --out <folder>` |
| Set C: candidates, injection, validation, scoring | `uv run python scripts/set_c.py --help` |
| Data analysis mode | add `--analyse-data` to `audit`, `baseline` or `batch` (Parquet needs `uv sync --extra data`) |
| Check the LLM provider | `uv run python scripts/probe_llm.py` |
| Parser statistics | `uv run python scripts/parser_report.py data/papers` |
| Citation-link accuracy | `uv run python scripts/citation_sample.py score data/papers datasets/parser_fixtures/citation_gold.jsonl` |

Each run writes to `runs/<run_id>/`: `manifest.json` (git commit, settings with secrets
removed, package versions), `usage/*.jsonl` (every tool call, HTTP request and LLM call),
and per paper `tasks.jsonl`, `results.jsonl`, `verified.jsonl`, `predictions.jsonl` and the
audit report for a human reader, `report.md` (with `report.json`). A
paper whose checks ran out of model calls (`AUDITOR_LLM_MAX_CALLS`, `AUDITOR_LLM_MAX_COST_USD`)
gets `predictions.incomplete.jsonl` instead, so it is not scored and a batch does not skip
it as done. The dollar limit also counts what the calls still waiting for a reply could
cost, at their prompt and answer cap, so calls sent together cannot run far past it. For B0
and B1, `dropped.jsonl` lists why any answer could not become a prediction (an anchor
outside the paper, an unknown entry).

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

### Web UI

```bash
uv sync --extra ui
uv run auditor ui
```

This opens the auditor in the browser at <http://localhost:8501>, reachable from this computer
only. Pick a set C demo paper (injected copies carry planted errors) or upload a ParsedPaper
JSON or a PDF (PDFs need GROBID: `docker compose up -d`), optionally with a repository and
venue rules, and run the audit. It runs `auditor audit` with live progress and spend, then
shows the report: verified flags with their evidence, items for human review, checks not
done. Past audits, including the experiment runs, open in the second tab at no cost.

Switching "Use the model" off makes an audit free (checks that need a model end as not
checked). With the model on, an audit costs about $0.08 on Gemini 3.8 Flash; every audit
stops at the per-audit limit in the sidebar, and all UI audits share a ledger
(`data/ui/ledger.jsonl`) with a total limit (default $5). Sign in for Vertex first
(`gcloud auth application-default login`).

### Evaluation

`auditor evaluate` scores any set of run folders (or folders of run folders) against a gold
label file with the matching rules of D3: Macro-F1, precision, recall, false-positive rate on
clean controls, flags per original paper (D14), unlabelled flags, review load, coverage,
grounding of the evidence (with `--dataset`) and cost. Runs of the same system are
summarised as mean ± standard deviation, and each full-system run is also scored without its
verifier (`full_no_verifier`, the ablation in P4.7). Results so far are in
[docs/p4-results.md](docs/p4-results.md).

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
