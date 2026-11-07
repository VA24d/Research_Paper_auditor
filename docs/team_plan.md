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
   TaskResult[] (outcome + CandidateFinding[])
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

The owner is the person to ask about a part and keeps its tests green. Ownership does not
mean only the owner writes the code, and nobody needs the owner's approval to change it.

## 3. Working on someone else's part

Anyone can pick up any open task.

1. Put your name next to the task in this file (in a small PR or a direct commit to the plan),
   or tell the owner, so two people don't do the same thing.
2. Work on a branch named after the task, for example `p2-5-number-check`.
3. Follow the interfaces in `auditor.schemas` and the decisions in `decisions.md`. If the
   task needs a schema change, raise it first (decision D5).
4. Merge it yourself once CI passes. Ask the owner to look first only if you want a second
   opinion.

## 4. Tasks

Each task is done when its code, tests and short docs are merged and CI passes on all three
operating systems.

### Part 1 — Bhaskar: parsing, planning, orchestration, infrastructure, baselines

Status: P1.1 is merged (PR #1). P1.2–P1.11 and P1.13–P1.16 are implemented and tested
(PR #2 and the follow-up PR). P1.10 includes a gold sample of 100 hand-checked citation
links (see [parser_evaluation.md](parser_evaluation.md)). P1.12 has been run live on
Gemini 2.5 Flash (Vertex AI). P1.17 is open. Also available for Part 4: hyperparameter
tables read into settings, `auditor batch` for repeated runs over a dataset (P4.6), and a
Docker image for the final runs.

Foundation (others depend on these, so they come first):

- [x] **P1.1** Repository setup: uv project, src layout, ruff, mypy, pytest, pre-commit,
  CI on Linux/Windows/macOS, GROBID compose file, setup docs.
- [x] **P1.2** Schemas v0 for every contract in D5, with contract tests and example JSON
  fixtures that other parts can build against.
- [x] **P1.3** `RunContext`, budgets, logging, `RunManifest`, and tool-call
  instrumentation on the shared tool functions (counted separately from HTTP requests, D8).
- [x] **P1.4** Shared HTTP client and policy gate (D8): allowlist, per-host rate limits,
  on-disk cache, retries, request log.
- [x] **P1.5** LLM factory (D9): provider from config, structured-output helper with
  validation and retries, token usage tracking, LLM cache switch.
- [x] **P1.6** Orchestration graph with mock specialists: fan-out per task, result
  collection, retries, timeouts, failure isolation, checkpointing and resume, with one
  checkpoint thread per run, system, paper and variant (D10).

Parser:

- [x] **P1.7** GROBID client and TEI to `ParsedPaper` conversion: sections, sentences with
  page coordinates, bibliography, citation mentions linked to entries.
- [x] **P1.8** Layout extraction with pdfplumber: page count, page sizes, fonts, margins,
  link annotations (GitHub URLs).
- [x] **P1.9** Parse cache keyed by PDF hash and parser version; `auditor parse` command.
- [x] **P1.10** Parser fixtures from about 10 papers, hand-checked, plus parser accuracy
  numbers (citation linking, bibliography fields, sentence anchoring).

Planner:

- [x] **P1.11** Deterministic extraction: citation contexts, numbers with units,
  experimental settings (`ExtractedSetting`), repository links.
- [x] **P1.12** LLM refinement: decide which claims are worth checking and split sentences
  that cite several works. Output must quote the paper exactly; anchors come from parser
  ids (D3). Live run on Gemini 2.5 Flash, 2 November: 154 citation candidates in two
  papers (2310.06825, 2005.11401), 9 calls, $0.04. Gemini left the citation markers out of
  most quotes ("REALM and ORQA" for "REALM [20] and ORQA [31]"), so 48 of 126 claims in
  2005.11401 fell back to the whole sentence. A quote now also counts when it matches the
  sentence without its markers, and the claim keeps the sentence's own text. Replayed on
  the same answers, 89 of 126 are narrowed, 23 are the whole sentence by the model's
  choice, 9 are empty on candidates judged not check-worthy, and 5 are refused because
  they skip words with "...". All 28 in 2310.06825 are narrowed.
- [x] **P1.13** Task generation with deterministic task ids and the right anchor kind (D3):
  reference tasks (one per bibliography entry, for existence and metadata), citation tasks
  (one per claim), code tasks and template tasks. Every citation mention gets a task,
  including mentions the parser could not link (D6).

Integration and baselines:

- [x] **P1.14** `auditor audit` command: runs the full pipeline on one paper and writes
  task results, findings, predictions and the run manifest.
- [x] **P1.15** B0 runner: one model call over the whole paper, no tools.
- [x] **P1.16** B1 runner: one tool-calling agent, one pass, no verifier, using the tools
  from P2.6 and P3.7.
- [ ] **P1.17** Failure analysis for parsing, planning and orchestration for the write-up.

### Part 2 — Anurag: citation agent and retrieval

Status: P2.1-P2.7 are implemented and tested on the `p2-citation-checker` branch (not yet
merged). P2.8's dataset is BUILT and validated (201 labels; awaiting the D18 20% second
review and the LLM-backed module runs); P2.9's offline machinery is done and set A was
measured (HALLMARK test_public: accuracy 0.726, flagging P 0.732 / R 0.891 / F1 0.804,
FPR on valid 0.557).

- [x] **P2.1** Clients for arXiv, OpenAlex, Crossref and Semantic Scholar, all through the
  shared HTTP client (P1.4).
- [x] **P2.2** Reference resolution from DOI, arXiv ID, or title and authors; a clear
  "not found" result with the reason.
- [x] **P2.3** Full-text retrieval for open-access sources. Reuse the Part 1 parser for PDFs,
  or arXiv HTML where available. Fall back to the abstract when the full text is unavailable,
  and record which one was used.
- [x] **P2.4** Chunking and passage retrieval with exact character offsets into the source,
  and an abstract-only mode for the ablation. Any embedding model must run on all three
  operating systems.
- [x] **P2.5** Citation agent `check()` for reference tasks (existence, metadata) and citation
  tasks (support: supported, contradicted, insufficient; number checks), returning a
  `TaskResult` with `CandidateFinding`s and `Evidence` quotes. Unlinked citation mentions are
  resolved from the marker text where possible, otherwise returned as `unverifiable` (D6).
- [x] **P2.6** Citation tools for B1 (D7).
- [x] **P2.7** HALLMARK test split: loader and existence-check evaluation.
- [ ] **P2.8** Set B: 60 recent arXiv papers, about 600 claim–source pairs, injection scripts
  that edit `ParsedPaper` copies (the unchanged originals are kept too), `GoldLabel`s
  including clean controls, and the gold evidence passage for each pair (needed for
  Recall@5). Every label is validated against its source and a sample is reviewed by a
  second person before freezing (D18). *(DONE except the second review: 60 papers built,
  201 labels (154 injected: 96 reference_not_found, 38 reference_metadata_mismatch,
  15 claim_number_mismatch, 5 claim_not_supported; 47 clean) validated by
  `scripts/set_b.py validate` — the only open D18 item is the 20% second review
  (`reviewed_by`), and the claim classes are thin and can be grown by hand.)*
- [ ] **P2.9** Citation module evaluation: Recall@5, precision, recall and F1, full text
  against abstract only, and tool-call accuracy of reference lookups (D8). *(Machinery
  done and set A measured (HALLMARK test_public, 831 entries: accuracy 0.726, P 0.732 /
  R 0.891 / F1 0.804, FPR 0.557). The set B module runs (full-text vs abstract-only)
  need an LLM configured (AUDITOR_LLM_*) and are the last step.)*

### Part 3 — Vijay: code agent and template checker

Code tasks arrive with an `ExtractedSetting` whose `key` says what to look for. Hyperparameter
and architecture keys are listed in `auditor.planner.tables.ALIASES` (`learning_rate`,
`batch_size`, `hidden_size`, ...). The evaluation setup uses `dataset`, `data_split`
(`"80/10/10"`), `test_fraction`, `validation_fraction` and `eval_metric` (`macro_f1`,
`exact_match`, ...; see `auditor.planner.evaluation_setup`). Gold labels for set C should
use the same keys. A repository's `commit` may be a branch or tag name when the dataset gives
one; record the commit hash actually checked out.

- [x] **P3.1** Repository fetching through the shared infrastructure: pinned commit, size
  limits, no code execution. `auditor.infra.snapshots` fetches core files from GitHub,
  GitLab, Hugging Face, Kaggle datasets, Anonymous GitHub, Zenodo and local folders into a
  scratch folder; data files only in data analysis mode (`--analyse-data`).
- [x] **P3.2** Static extraction of settings from `.py` (including argparse and config
  defaults), `.yaml`, `.json` and `.toml` files, as `ExtractedSetting`s with file and line.
  `auditor.agents.code.extract`, plus shell command lines, split lists and dataset cards.
- [x] **P3.3** Normalisation and comparison: units, number formats (`3e-4` and `0.0003`),
  and name aliases (`lr` and `learning_rate`).
- [x] **P3.4** Code agent `check()` returning a `TaskResult` with `CandidateFinding`s and
  file/line `Evidence`; settings that cannot be located are `unverifiable` with a reason.
  `auditor.agents.code.check` (settings in `settings_check`, evaluation data in
  `data_checks`, data analysis in `auditor.agents.code.data`).
- [x] **P3.5** Venue profiles in `configs/venues/` (start with the venues of the papers in
  sets B and C). NeurIPS 2025, ICLR 2026, ICML 2025, ACL 2025 and EMNLP 2025, long and
  short, for the submission version. Every limit cites its official source; what was left
  out and why is in `configs/venues/README.md`.
- [x] **P3.6** Template checker `check()`: deterministic rules over `ParsedPaper.layout`
  and sections. No LLM calls. Findings use page or document anchors where there is no
  sentence (D3). Rules that need visual inspection return `unverifiable`. Checks
  `max_pages`, `min_body_font_size`, `paper_size`, `margins`, `required_sections`,
  `forbidden_sections`, `references_present` and `anonymity` in
  `auditor.agents.template.rules`.
- [x] **P3.7** Code and template tools for B1 (D7): `auditor.agents.code.tools` (list,
  search, read, find setting, data report) and `auditor.agents.template.tools`
  (`check_template_rule`, `layout_summary`).
- [ ] **P3.8** Set C: 40 repositories, 160 instances, injected mismatches (paper side by
  editing `ParsedPaper` copies, or code side by patching a pinned copy; the unchanged
  originals are kept too), `GoldLabel`s including clean controls. Every label is validated
  against the paper and the code the experiment actually uses, and a sample is reviewed by
  a second person before freezing (D18). Built from SciCoQA: 40 pairs, 63 injected setting
  errors, 36 clean controls and 92 method discrepancies (`datasets/set_c/README.md`).
  Waiting for the 20% second review (`datasets/set_c/review.jsonl`) before freezing.
- [ ] **P3.9** Template fixtures and evaluation; code module evaluation (precision, recall,
  F1, false positives on matching settings, tool-call accuracy of repository searches).
  Done: template fixtures (`auditor.agents.template.fixtures`, `scripts/template_eval.py`)
  and code module scoring (`scripts/set_c.py score`). Open: tool-call accuracy of
  repository searches, which needs set C.

- [ ] **P3.10** Theory checker (D21): `auditor.agents.theory` checks the mathematics of a
  paper's formal claims for `math_error` (plan, screen, deep check, challenge), opt-in with `AUDITOR_THEORY_CHECK_ENABLED`.
  Outside benchmark: SPOT's math and CS proof subset (`datasets/spot`, 26 papers, 29
  errors), scored by an LLM judge whose grades a person checks. Open: the first run, which
  needs the team's Vertex credentials.

### Part 4 — Sudheera: verifier, report, evaluation

- [x] **P4.1** (Vijay, PR #72) Quote validation: a quote counts only if it appears verbatim in the source
  text after the agreed normalisation (whitespace, hyphenation, ligatures). File and line
  evidence is checked against the fetched repository; rule findings must name their rule.
- [x] **P4.2** (Vijay, PR #72) Gating logic with the three states and a reason for every decision.
  Suspicions attached to `unverifiable` or `failed` task results go to
  `NEEDS_HUMAN_REVIEW`. Bounded rechecks reuse the specialist tools. Missing evidence alone
  never produces `VERIFIED_FLAG`.
- [x] **P4.3** (Vijay, PR #72; decisions on real runs in `auditor evaluate`) Verifier fixtures (true errors, false accusations, uncertain cases) and
  decision accuracy.
- [x] **P4.4** (Sudheera) Report generator: JSON and Markdown, advisory wording that never asserts
  intent, with three sections: verified flags, items for human review, and items not
  checked with their reasons (D6).
- [x] **P4.5** (Sudheera) Evaluation runner: load `GoldLabel`s and `Prediction`s, apply the matching
  rule for every anchor kind (D3), compute Macro-F1, precision, recall, false-positive
  rate on clean controls, flags per original paper and share of original papers flagged
  (D14), unlabelled flags, coverage (share of tasks checked), grounding rate, and cost
  from token usage, tool calls and HTTP requests.
- [ ] **P4.6** Experiment configs for B0, B1 and the full system, three runs each on both
  the injected and the original papers, with run manifests. `auditor batch` (README,
  "Experiments") runs a dataset file this way and resumes after interruptions.
- [ ] **P4.7** (Sudheera; no-verifier done in `auditor evaluate`) Ablations: no verifier, abstract-only retrieval, single agent.
- [ ] **P4.8** Result tables and failure analysis.

### Shared

- [ ] **S1** Review schemas v0 (P1.2) within a day of the PR opening.
- [ ] **S2** LLM provider decision (D9) by the end of week 1.
- [ ] **S3** Write-up: each person writes the method and results for their part; everyone
  edits the introduction, the discussion and the limitations.
- [ ] **S4** Demo.
- [ ] **S5** Adjudicate a random sample of flags raised on original papers (D14): split
  by area, each flag checked by one person against the source, unclear ones by two.

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
| B. Claim–source support | Anurag | ~600 pairs from 60 papers | Validated `GoldLabel`s (D18) | Citation agent and RQ1 |
| C. Paper–code consistency | Vijay | 160 instances from 40 repositories | Validated `GoldLabel`s (D18) | Code agent and RQ1 |
| Template fixtures | Vijay | One case per supported rule | Expected result per rule | Template checker |
| Verifier fixtures | Sudheera | True, false and uncertain cases | Expected state | Verifier |
| Parser and planner fixtures | Bhaskar | ~10 papers | Hand-checked extraction | Parser and planner |

Systems: B0, B1 and the full system, three runs each, on the same injected `ParsedPaper`
files and on the unchanged originals. Metrics: Macro-F1 over the classes in D4,
precision, recall, false-positive rate on clean controls, flags per clean paper with an
adjudicated false-alarm estimate (D14), coverage, grounding rate of quotes, Recall@5,
tool-call accuracy, tool calls, HTTP requests, tokens, steps and cost per paper. Results
are reported as measured, including if the target is missed.

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
