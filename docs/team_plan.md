# Team Plan

**Project:** Evidence-Gated Verification Agent for Research Paper Auditing
**Stack:** Python 3.13, LangGraph, LangChain, Pydantic, uv
**Team:** Bhaskar, Anurag, Vijay, Sudheera

This is the working plan. It replaces the first draft of the work split and follows the
decisions in [decisions.md](decisions.md). Edit it as tasks move. Tick a task when its PR is
merged, and put your name next to a task when you start it.

---

## 1. How the parts fit together

```text
 PDF or ParsedPaper JSON + optional GitHub URL + venue name
                    |
                    v
   [Part 1] Parser ---------------------------> ParsedPaper
                    |
                    v
   [Part 1] Planner --------------------------> AuditTask[]
                    |
   [Part 1] Orchestrator fans tasks out, one branch per task
          +---------+--------------------+
          v                              v
   [Part 2] Citation agent       [Part 3] Code agent / Template checker
          +---------+--------------------+
                    v
           CandidateFinding[]
                    |
                    v
   [Part 4] Verifier -------------------------> VerifiedFinding[]
                    |
          +---------+--------------------+
          v                              v
   [Part 4] Audit report         Prediction[] --> [Part 4] Evaluation (vs GoldLabel[])

 Shared infrastructure (Part 1): config, schemas, LLM factory, HTTP client and policy
 gate, caching, logging, run context and budgets.
 Baselines (Part 1): B0 and B1 read the same ParsedPaper and emit the same Prediction[].
```

## 2. Ownership

| Part | Owner | Code | Data |
|---|---|---|---|
| 1. Parsing, planning, orchestration, infrastructure, baselines | Bhaskar | `auditor.schemas`, `auditor.infra`, `auditor.parser`, `auditor.planner`, `auditor.orchestration`, `auditor.baselines` | Parser and planner fixtures |
| 2. Citation agent and retrieval | Anurag | `auditor.agents.citation`, `auditor.retrieval` | Set A (HALLMARK), set B (claim–source pairs) |
| 3. Code agent and template checker | Vijay | `auditor.agents.code`, `auditor.agents.template`, `configs/venues/` | Set C (paper–code instances), template fixtures |
| 4. Verifier, report, evaluation | Sudheera | `auditor.agents.verifier`, `auditor.reporting`, `auditor.evaluation` | Verifier decision fixtures |

The owner is accountable for a part: they review every PR that touches it and keep its
tests green. Ownership does not mean only the owner writes the code.

## 3. Working on someone else's part

Anyone can pick up any open task.

1. Put your name next to the task in this file (in a small PR or a direct commit to the plan),
   or tell the owner, so two people don't do the same thing.
2. Work on a branch named after the task, for example `p2-5-number-check`.
3. Follow the interfaces in `auditor.schemas` and the decisions in `decisions.md`. If the
   task needs a schema change, raise it first (decision D5).
4. The part owner reviews the PR.

## 4. Tasks

Each task is done when its code, tests and short docs are merged and CI passes on all three
operating systems.

### Part 1 — Bhaskar: parsing, planning, orchestration, infrastructure, baselines

Foundation (others depend on these, so they come first):

- [ ] **P1.1** Repository setup: uv project, src layout, ruff, mypy, pytest, pre-commit,
  CI on Linux/Windows/macOS, GROBID compose file, setup docs.
- [ ] **P1.2** Schemas v0 for every contract in D5, with contract tests and example JSON
  fixtures that other parts can build against.
- [ ] **P1.3** `RunContext`, budgets, logging and `RunManifest`.
- [ ] **P1.4** Shared HTTP client and policy gate (D8): allowlist, per-host rate limits,
  on-disk cache, retries, call log.
- [ ] **P1.5** LLM factory (D9): provider from config, structured-output helper with
  validation and retries, token usage tracking, LLM cache switch.
- [ ] **P1.6** Orchestration graph with mock specialists: fan-out per task, finding
  collection, retries, timeouts, failure isolation, checkpointing and resume.

Parser:

- [ ] **P1.7** GROBID client and TEI to `ParsedPaper` conversion: sections, sentences with
  page coordinates, bibliography, citation mentions linked to entries.
- [ ] **P1.8** Layout extraction with pdfplumber: page count, page sizes, fonts, margins,
  link annotations (GitHub URLs).
- [ ] **P1.9** Parse cache keyed by PDF hash and parser version; `auditor parse` command.
- [ ] **P1.10** Parser fixtures from about 10 papers, hand-checked, plus parser accuracy
  numbers (citation linking, bibliography fields, sentence anchoring).

Planner:

- [ ] **P1.11** Deterministic extraction: citation contexts, numbers with units,
  experimental settings (`ExtractedSetting`), repository links.
- [ ] **P1.12** LLM refinement: decide which claims are worth checking and split sentences
  that cite several works. Output must quote the paper exactly; anchors come from parser
  sentence ids (D3).
- [ ] **P1.13** Task generation: citation, code and template `AuditTask`s with deterministic
  task ids.

Integration and baselines:

- [ ] **P1.14** `auditor audit` command: runs the full pipeline on one paper and writes
  findings, predictions and the run manifest.
- [ ] **P1.15** B0 runner: one model call over the whole paper, no tools.
- [ ] **P1.16** B1 runner: one tool-calling agent, one pass, no verifier, using the tools
  from P2.6 and P3.7.
- [ ] **P1.17** Failure analysis for parsing, planning and orchestration for the write-up.

### Part 2 — Anurag: citation agent and retrieval

- [ ] **P2.1** Clients for arXiv, OpenAlex, Crossref and Semantic Scholar, all through the
  shared HTTP client (P1.4).
- [ ] **P2.2** Reference resolution from DOI, arXiv ID, or title and authors; a clear
  "not found" result with the reason.
- [ ] **P2.3** Full-text retrieval for open-access sources. Reuse the Part 1 parser for PDFs,
  or arXiv HTML where available. Fall back to the abstract when the full text is unavailable,
  and record which one was used.
- [ ] **P2.4** Chunking and passage retrieval with exact character offsets into the source,
  and an abstract-only mode for the ablation. Any embedding model must run on all three
  operating systems.
- [ ] **P2.5** Citation agent `check()`: existence, metadata, support (supported,
  contradicted, insufficient) and number checks, returning `CandidateFinding`s with
  `Evidence` quotes.
- [ ] **P2.6** Citation tools for B1 (D7).
- [ ] **P2.7** HALLMARK test split: loader and existence-check evaluation.
- [ ] **P2.8** Set B: 60 recent arXiv papers, about 600 claim–source pairs, injection scripts
  that edit `ParsedPaper` copies, `GoldLabel`s including clean controls, and the gold
  evidence passage for each pair (needed for Recall@5).
- [ ] **P2.9** Citation module evaluation: Recall@5, precision, recall and F1, full text
  against abstract only.

### Part 3 — Vijay: code agent and template checker

- [ ] **P3.1** Repository fetching through the shared infrastructure: pinned commit, size
  limits, no code execution.
- [ ] **P3.2** Static extraction of settings from `.py` (including argparse and config
  defaults), `.yaml`, `.json` and `.toml` files, as `ExtractedSetting`s with file and line.
- [ ] **P3.3** Normalisation and comparison: units, number formats (`3e-4` and `0.0003`),
  and name aliases (`lr` and `learning_rate`).
- [ ] **P3.4** Code agent `check()` returning `CandidateFinding`s with file/line `Evidence`.
- [ ] **P3.5** Venue profiles in `configs/venues/` (start with the venues of the papers in
  sets B and C).
- [ ] **P3.6** Template checker `check()`: deterministic rules over `ParsedPaper.layout`
  and sections. No LLM calls. Rules that need visual inspection are marked unresolved.
- [ ] **P3.7** Code and template tools for B1 (D7).
- [ ] **P3.8** Set C: 40 repositories, 160 instances, injected mismatches (paper side by
  editing `ParsedPaper` copies, or code side by patching a pinned copy), `GoldLabel`s
  including clean controls.
- [ ] **P3.9** Template fixtures and evaluation; code module evaluation (precision, recall,
  F1, false positives on matching settings).

### Part 4 — Sudheera: verifier, report, evaluation

- [ ] **P4.1** Quote validation: a quote counts only if it appears verbatim in the source
  text after the agreed normalisation (whitespace, hyphenation, ligatures). File and line
  evidence is checked against the fetched repository; rule findings must name their rule.
- [ ] **P4.2** Gating logic with the three states and a reason for every decision. Failed
  tasks go to `NEEDS_HUMAN_REVIEW`. Bounded rechecks reuse the specialist tools. Missing
  evidence alone never produces `VERIFIED_FLAG`.
- [ ] **P4.3** Verifier fixtures (true errors, false accusations, uncertain cases) and
  decision accuracy.
- [ ] **P4.4** Report generator: JSON and Markdown, advisory wording that never asserts
  intent, verified items separated from items for human review.
- [ ] **P4.5** Evaluation runner: load `GoldLabel`s and `Prediction`s, apply the matching
  rule (D3), compute Macro-F1, precision, recall, false-positive rate on clean items,
  grounding rate, and cost from the call log and token usage.
- [ ] **P4.6** Experiment configs for B0, B1 and the full system, three runs each, with run
  manifests.
- [ ] **P4.7** Ablations: no verifier, abstract-only retrieval, single agent.
- [ ] **P4.8** Result tables and failure analysis.

### Shared

- [ ] **S1** Review schemas v0 (P1.2) within a day of the PR opening.
- [ ] **S2** LLM provider decision (D9) by the end of week 1.
- [ ] **S3** Write-up: each person writes the method and results for their part; everyone
  edits the introduction, the discussion and the limitations.
- [ ] **S4** Demo.

## 5. Dependencies and what to use until they land

| You need | From | Until it lands |
|---|---|---|
| Schemas (P1.2) | Part 1, day 2 | Nothing; review the draft PR |
| `ParsedPaper` fixtures | Part 1 (P1.2, P1.10) | The example JSON fixtures in P1.2 |
| Shared HTTP client (P1.4) | Part 1 | Write the client code against the interface in the P1.2 PR; plug in when merged |
| LLM factory (P1.5) | Part 1 | `langchain_core` fake chat models in tests |
| `AuditTask`s from the planner | Part 1 (P1.13) | Hand-written task JSON in your own fixtures |
| `CandidateFinding`s | Parts 2 and 3 | Hand-written findings (P4.3) |
| Tools for B1 | Parts 2 and 3 (P2.6, P3.7) | B1 is built and tested with mock tools first |
| `GoldLabel`s | Parts 2 and 3 (P2.8, P3.8) | A small hand-made gold set for the evaluation runner |
| Venue profiles | Part 3 (P3.5) | One example profile in the P1.2 fixtures |

## 6. Timeline

| Week | Goal | Checkpoint |
|---|---|---|
| 1 | Foundation and prototypes. P1.1–P1.3 by day 2–3, P1.6 with mocks by day 5. Everyone prototypes against the schema fixtures. Provider decision (S2). | The mock pipeline runs end to end; every part demos its module on fixtures. |
| 2 | Core functionality. Parser and planner (Part 1), citation agent (Part 2), code agent and template checker (Part 3), verifier and evaluation runner (Part 4). Datasets B and C started. | Every module passes its own tests and the contract tests. |
| 3 | Integration in pipeline order: planner to citation agent, planner to code and template, specialists to verifier, verifier to report. Datasets B and C finished. B0 and B1 working. | One real paper runs through the whole system and produces an evidence-backed report. |
| 4 | Experiments, ablations, failure analysis, write-up, demo. | Final results, tables, report and demo. |

Integration is continuous rather than a single event in week 3: the orchestrator runs on
mocks from week 1, and each real module replaces its mock as soon as it merges.

## 7. Evaluation summary

| Set | Owner | Size | Ground truth | Used for |
|---|---|---|---|---|
| A. Reference existence (HALLMARK) | Anurag | Released test split | Released labels | Sanity check of citation lookup |
| B. Claim–source support | Anurag | ~600 pairs from 60 papers | `GoldLabel`s from injection log | Citation agent and RQ1 |
| C. Paper–code consistency | Vijay | 160 instances from 40 repositories | `GoldLabel`s from injection log | Code agent and RQ1 |
| Template fixtures | Vijay | One case per supported rule | Expected result per rule | Template checker |
| Verifier fixtures | Sudheera | True, false and uncertain cases | Expected state | Verifier |
| Parser and planner fixtures | Bhaskar | ~10 papers | Hand-checked extraction | Parser and planner |

Systems: B0, B1 and the full system, three runs each, on the same injected `ParsedPaper`
files. Metrics: Macro-F1 over the classes in D4, precision, recall, false-positive rate on
clean items, grounding rate of quotes, Recall@5, tool calls, tokens, steps and cost per
paper. Results are reported as measured, including if the target is missed.

## 8. Definition of project completion

- A PDF or `ParsedPaper` goes in; an evidence-backed audit report comes out.
- Citations are checked against the cited sources' text where available.
- Paper–code consistency is checked statically; no repository code is executed.
- Supported venue rules are checked deterministically.
- The verifier confirms, drops or escalates every candidate finding, with a reason.
- B0, B1 and full-system results, ablations, grounding and cost are reported.
- Every part is merged, tested on all three operating systems, documented and demonstrated.

## 9. Setup

See the README for setup on Windows, macOS and Linux.
