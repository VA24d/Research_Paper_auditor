# Full System Technical and Scientific Audit Report: Research Paper Auditor

**Target System**: Research Paper Auditor (`src/auditor/`)  
**Git Branch**: `p4-evaluation-report` (Clean working tree, commit `5e1aacb`)  
**Execution Environment**: Anaconda Python 3.12.13 (`/Users/va/optiq/bin/python`)  
**Audit Date**: November 10, 2026  
**Auditor**: Independent Multi-Agent Technical & Scientific Audit Team  
**Audit Scope**: Baseline Proposal Alignment, Scientific & Evaluation Rigor, Code Quality & Robustness, Forensic Defect Verification  
**Audit Deliverable Path**: `/Users/va/Research_Paper_auditor/FULL_SYSTEM_AUDIT_REPORT.md`  

---

## Table of Contents

1. [Executive Summary & Proposal Scorecard](#1-executive-summary--proposal-scorecard)
   - [1.1 Executive Overview of System Health & Engineering Scale](#11-executive-overview-of-system-health--engineering-scale)
   - [1.2 Component-by-Component Implementation Matrix](#12-component-by-component-implementation-matrix)
2. [Empirical & Evaluation Audit](#2-empirical--evaluation-audit)
   - [2.1 Validation of Results in `results/p4/` and `docs/p4-results.md`](#21-validation-of-results-in-resultsp4-and-docsp4-resultsmd)
   - [2.2 In-Depth Audit of Metric Computation Logic (`src/auditor/evaluation/metrics.py`)](#22-in-depth-audit-of-metric-computation-logic-srcauditorevaluationmetricspy)
   - [2.3 Audit of Experimental Validity on Set C](#23-audit-of-experimental-validity-on-set-c)
   - [2.4 Audit of Baselines B0 and B1](#24-audit-of-baselines-b0-and-b1)
   - [2.5 Scientific Validation & Refutation of the Verifier Ablation Claim](#25-scientific-validation--refutation-of-the-verifier-ablation-claim)
3. [Subsystem Health Reviews](#3-subsystem-health-reviews)
   - [3.1 Part 1: PDF Parser & Task Planner](#31-part-1-pdf-parser--task-planner)
   - [3.2 Part 2: Citation Agent & Scholarly Retrieval](#32-part-2-citation-agent--scholarly-retrieval)
   - [3.3 Part 3: Code Agent & Template Checker](#33-part-3-code-agent--template-checker)
   - [3.4 Specialized Extensions: Theory / Math Agent & Integrity Agent](#34-specialized-extensions-theory--math-agent--integrity-agent)
   - [3.5 Part 4: Evidence-Gated Verifier](#35-part-4-evidence-gated-verifier)
   - [3.6 Reporting Engine & Streamlit Web UI](#36-reporting-engine--streamlit-web-ui)
   - [3.7 Core Infrastructure, LLM Factory & LangGraph Orchestrator](#37-core-infrastructure-llm-factory--langgraph-orchestrator)
4. [Verified Findings & Suggested Improvements Ranked by Severity](#4-verified-findings--suggested-improvements-ranked-by-severity)
   - [4.1 Critical Findings](#41-critical-findings)
     - [Finding 1: Set B Citation Evaluation Gap & Citation Mocking in Final Experiments](#finding-1-set-b-citation-evaluation-gap--citation-mocking-in-final-experiments)
     - [Finding 2: Artificial 1.000 Precision & Unlabelled Flag Exclusion in `metrics.py`](#finding-2-artificial-1000-precision--unlabelled-flag-exclusion-in-metricspy)
   - [4.2 High Severity Findings](#42-high-severity-findings)
     - [Finding 3: Verifier Ablation Mechanism Refutation (0 Dropped Findings, Demoted Real Errors)](#finding-3-verifier-ablation-mechanism-refutation-0-dropped-findings-demoted-real-errors)
     - [Finding 4: Loose Document-Level Method Matching & Spurious True Positives](#finding-4-loose-document-level-method-matching--spurious-true-positives)
     - [Finding 5: Baseline Prompt & Schema Asymmetry (B0/B1 Structurally Disadvantaged)](#finding-5-baseline-prompt--schema-asymmetry-b0b1-structurally-disadvantaged)
     - [Finding 6: Parser Splitting Artifacts Causing 64% of Unlabelled Flags](#finding-6-parser-splitting-artifacts-causing-64-of-unlabelled-flags)
     - [Finding 7: Pipeline Crash Risk in Unwrapped LangGraph Nodes & Unhandled LLM Parsing](#finding-7-pipeline-crash-risk-in-unwrapped-langgraph-nodes--unhandled-llm-parsing)
   - [4.3 Medium Severity Findings](#43-medium-severity-findings)
     - [Finding 8: Benchmark Circularity & Missing Second Review on Set C](#finding-8-benchmark-circularity--missing-second-review-on-set-c)
     - [Finding 9: Zero Test Coverage in Web UI & Missing Subprocess Timeout](#finding-9-zero-test-coverage-in-web-ui--missing-subprocess-timeout)
     - [Finding 10: Resource Leaks & Zombie Subprocesses in Evaluation and Sandbox](#finding-10-resource-leaks--zombie-subprocesses-in-evaluation-and-sandbox)
   - [4.4 Low Severity Findings](#44-low-severity-findings)
     - [Finding 11: Baseline B1 Spending Runaway & Prompt Bloat](#finding-11-baseline-b1-spending-runaway--prompt-bloat)
     - [Finding 12: Packaging Standards Omission (Missing `py.typed` Marker)](#finding-12-packaging-standards-omission-missing-pytyped-marker)
5. [Architectural & Scientific Roadmap](#5-architectural--scientific-roadmap)
   - [Phase 1: Critical Evaluation & Metric Integrity (Days 1–5)](#phase-1-critical-evaluation--metric-integrity-days-15)
   - [Phase 2: Pipeline Hardening & Crash Isolation (Days 6–10)](#phase-2-pipeline-hardening--crash-isolation-days-610)
   - [Phase 3: Method Matching & Specialist Calibration (Days 11–16)](#phase-3-method-matching--specialist-calibration-days-1116)
   - [Phase 4: Baseline Alignment & Benchmark Expansion (Days 17–22)](#phase-4-baseline-alignment--benchmark-expansion-days-1722)
   - [Phase 5: Packaging & Production Deployment Readiness (Days 23–28)](#phase-5-packaging--production-deployment-readiness-days-2328)

---

## 1. Executive Summary & Proposal Scorecard

### 1.1 Executive Overview of System Health & Engineering Scale

The Research Paper Auditor is a multi-agent system designed to detect discrepancies between claims in machine learning scientific papers and their concrete implementations, citations, and venue formatting rules. This comprehensive audit evaluates the system state on git branch `p4-evaluation-report` (commit `5e1aacb`), assessing engineering scale, architectural alignment with the original proposal (`LMA_Group_Project_Proposal.md`) and team plan (`docs/team_plan.md`), test suite health, and scientific validity.

```
                      +-------------------------------------------------+
                      |            Research Paper Auditor Core          |
                      |    134 Files | 27,896 Python Lines in src/      |
                      +-------------------------------------------------+
                                               |
             +---------------------------------+--------------------------------+
             |                                 |                                |
             v                                 v                                v
+--------------------------+     +--------------------------+     +--------------------------+
|   Engineering Health     |     |   Subsystem Execution    |     |   Scientific Rigor       |
| 1,302 Unit Tests Passing |     | 4 Core Subsystems Built  |     | Macro-F1: 0.703 vs 0.046 |
| 91.5% Statement Coverage |     | 2 Specialized Extensions |     | Spend Ledger: $125.04    |
| Ruff: 100% Passing       |     | Orchestrator: LangGraph  |     | 12 Forensic Findings     |
| Strict Type Safety       |     | Web UI + CLI Interfaces  |     | Identified & Confirmed   |
+--------------------------+     +--------------------------+     +--------------------------+
```

#### Codebase Volume and Distribution
The production implementation in `src/auditor/` comprises **134 Python files** totaling **27,896 lines of code** (and 237 total project files, 3,041 indexed symbols, and 10,659 context graph edges in `.trail/cache/`):
- `src/auditor/schemas/` (11 files, 1,718 lines): Complete Pydantic v2 data contracts.
- `src/auditor/infra/` (18 files, 3,844 lines): HTTP caching policy gates, LLM factories, sandboxing, and token ledgers.
- `src/auditor/parser/` (10 files, 2,939 lines): GROBID TEI parsing, layout analysis, table cropping, and direct LaTeX parsing.
- `src/auditor/planner/` (10 files, 2,048 lines): Deterministic extraction and LLM candidate task refinement.
- `src/auditor/agents/citation/` (8 files, 2,202 lines): Citation resolution, full-text parsing, and NLI verification.
- `src/auditor/retrieval/` (6 files, 1,560 lines): Crossref/arXiv/OpenAlex/Semantic Scholar clients and passage ranking.
- `src/auditor/agents/code/` (19 files, 4,960 lines): Static code AST extraction, data leakage analysis, Ruff defect detection, and RAG method comparison.
- `src/auditor/agents/template/` (5 files, 963 lines): Deterministic venue compliance verification.
- `src/auditor/agents/theory/` (5 files, 864 lines): Mathematical proof verification with Docker sandboxed SymPy execution.
- `src/auditor/agents/integrity/` (4 files, 2,540 lines): Structural placeholder analysis and result-vs-table consistency checks.
- `src/auditor/agents/verifier/` (5 files, 560 lines): Grounding verification and tri-state evidence gating.
- `src/auditor/orchestration/` (6 files, 668 lines): LangGraph StateGraph, task runner, and state synthesis.
- `src/auditor/baselines/` (4 files, 418 lines): Single-turn B0 and tool-augmented B1 agent baselines.
- `src/auditor/reporting/` (2 files, 506 lines): Markdown and JSON report generation.
- `src/auditor/evaluation/` (5 files, 1,090 lines): Evaluation harnesses and metrics computation.
- `src/auditor/ui/` (3 files, 532 lines): Streamlit interactive user interface.

#### Test Suite and Static Analysis Summary
- **Unit and Integration Test Suite**: The automated test suite (`python -m pytest tests -q`) executes **1,302 passing unit tests in 10.08 seconds** with 0 failures, 0 errors, and 8 intentionally deselected integration tests requiring running external daemons (Docker, live GROBID, external network).
- **Test Coverage**: Across 14,699 executable Python statements in `src/auditor/`, the test suite achieves **91.46% overall statement coverage** (13,443 statements covered, 1,256 missed).
- **Static Linting**: `ruff check .` passes with **zero warnings or errors** across all 272 files under strict modern rules (`E, W, F, I, B, UP, SIM, C4, PT, PTH, RUF, N, ASYNC, DTZ, TID, PLW1514`). `ruff format --check .` confirms 100% adherence.
- **Type Checking**: `mypy src/auditor tests scripts` passes strictly with **0 errors across 236 files** when `MYPYPATH=src` and `types-PyYAML` are configured.

#### Overall Audit Verdict
The project represents an engineering achievement of rare scope and sophistication for a course project. The team implemented every core architectural subsystem proposed, exceeded their original design with four production-grade extensions (LaTeX e-print parser, 300 dpi table vision extraction, Docker math verification sandbox, and Streamlit Web UI), and created a clean, modular, and exceptionally well-tested codebase.

However, the scientific evaluation contains several methodological vulnerabilities:
1. **Set B Evaluation Omission**: Citation claim verification was mocked out in the final experiments, and Set B evaluation was never executed with the verifier in Part 4.
2. **Metric Distortions**: Unlabelled flags on original papers were excluded from false positives in `metrics.py`, mathematically locking reported precision to 1.000 for all systems and concealing hundreds of false alarms.
3. **Verifier Ablation Refutation**: The verifier dropped zero findings on Set C; its sole effect was demoting 26 true errors to human review due to specialist uncertainty flags, reducing Macro-F1 by 3.3% and recall by 5.3%.
4. **Permissive Matching**: Document-level method matching credited the system with true positives for unrelated code differences.
5. **Baseline Asymmetry**: Baselines B0 and B1 were starved of schema targeting instructions and suffered unbounded context bloat.

---

### 1.2 Component-by-Component Implementation Matrix

The table below cross-references every deliverable, milestone, and architectural feature promised in `LMA_Group_Project_Proposal.md` and `docs/team_plan.md` against the concrete source code in `src/auditor/`:

| Proposal Milestone / Component | Architectural Commitment in Proposal & Team Plan | Concrete Implementation Source Files | Status |
|:---|:---|:---|:---:|
| **P1.1 Repository & CI Setup** | uv project, src layout, ruff, mypy, pytest, pre-commit, GROBID compose, multi-OS CI | `pyproject.toml`, `.github/workflows/ci.yml`, `compose.yaml` | **Fully Implemented** |
| **P1.2 Contracts & Schemas** | Pydantic v2 schemas: ParsedPaper, ExtractedSetting, Anchor, AuditTask, Evidence, TaskResult, Finding, Prediction, GoldLabel | `src/auditor/schemas/` (`paper.py`, `anchors.py`, `tasks.py`, `findings.py`, `evaluation.py`, `venue.py`) | **Fully Implemented** |
| **P1.3 RunContext & Instrumentation** | RunContext, budgets, logging, RunManifest, tool instrumentation | `src/auditor/infra/context.py`, `manifest.py`, `usage.py`, `tools.py` | **Fully Implemented** |
| **P1.4 Shared HTTP Policy Gate** | Host allowlist, per-host rate limiting, on-disk HTTP cache, retries, request logging | `src/auditor/infra/http.py:L295-L628` (`HttpClient`) | **Fully Implemented** |
| **P1.5 LLM Factory** | Single LLM factory, structured output with validation/retries, token tracking, spend ledger | `src/auditor/infra/llm.py:L420-L825` (`StructuredLLM`) | **Fully Implemented** |
| **P1.6 Orchestration Graph** | LangGraph StateGraph, fan-out per task, collection, retries, timeouts, thread checkpointing & resume | `src/auditor/orchestration/graph.py`, `runner.py`, `registry.py` | **Fully Implemented** |
| **P1.7 GROBID Parser & TEI** | GROBID client and TEI to `ParsedPaper` conversion: sections, sentences with coords, citations | `src/auditor/parser/grobid.py`, `tei.py`, `linking.py` | **Fully Implemented** |
| **P1.8 Layout Extraction** | pdfplumber layout extraction: page count, sizes, fonts, margins, GitHub URL link annotations | `src/auditor/parser/layout.py` | **Fully Implemented** |
| **P1.9 Parse Caching & CLI** | Parse cache keyed by SHA-256 and parser version; `auditor parse` CLI | `src/auditor/parser/pipeline.py:L54-L113`, `cli.py` | **Fully Implemented** |
| **P1.10 Parser Fixtures & Eval** | 10 hand-checked papers, citation linking, bibliography accuracy, sentence anchoring | `tests/parser/`, `src/auditor/parser/accuracy.py`, `docs/parser_evaluation.md` | **Fully Implemented** |
| **P1.11 Deterministic Planner** | Deterministic extraction: citation contexts, numbers with units, settings, repository links | `src/auditor/planner/settings.py`, `claims.py`, `repository.py` | **Fully Implemented** |
| **P1.12 LLM Candidate Refinement** | Refine claims worth checking and split sentences citing multiple works; exact quotes | `src/auditor/planner/claims.py:L54-L146` (`refine_candidates`) | **Fully Implemented** |
| **P1.13 Task Generation** | Deterministic task ids, 4 anchor kinds (sentence, bib, page, doc), unlinked citations kept | `src/auditor/planner/tasks.py:L202-L264` (`plan_paper`) | **Fully Implemented** |
| **P1.14 Pipeline CLI** | `auditor audit` single paper and `auditor batch` multi-paper execution | `src/auditor/cli.py`, `runner.py` | **Fully Implemented** |
| **P1.15 Baseline B0 Runner** | Same LLM, single pass, no tools, structured prediction output | `src/auditor/baselines/b0.py`, `common.py` | **Fully Implemented** |
| **P1.16 Baseline B1 Runner** | Same LLM, same tools, single agent, one pass, no verifier, middleware limits | `src/auditor/baselines/b1.py`, `common.py` | **Fully Implemented** |
| **P2.1 Scholarly API Clients** | arXiv, OpenAlex, Crossref, Semantic Scholar via shared `HttpClient` | `src/auditor/retrieval/clients.py` | **Fully Implemented** |
| **P2.2 Reference Resolution** | Cascade resolution by DOI, arXiv ID, or title/authors search with clear "not found" | `src/auditor/retrieval/resolve.py` | **Fully Implemented** |
| **P2.3 Full-Text Retrieval** | Open-access PDF/HTML retrieval, GROBID parsing, abstract fallback | `src/auditor/retrieval/fulltext.py` | **Fully Implemented** |
| **P2.4 Passage Retrieval** | BM25 + ONNX embedding ranking, character offsets, abstract-only mode | `src/auditor/retrieval/passages.py`, `embeddings.py` | **Fully Implemented** |
| **P2.5 Citation Agent Check** | `check()`: Reference existence/metadata checks, citation NLI support & number checks | `src/auditor/agents/citation/reference.py`, `support.py` | **Fully Implemented** |
| **P2.6 Citation Tools for B1** | `lookup_reference`, `retrieve_cited_text`, `find_supporting_passages` BaseTools | `src/auditor/agents/citation/tools.py` | **Fully Implemented** |
| **P2.7 Set A (HALLMARK) Eval** | Test-public split loader, existence scoring (831 entries, Acc 0.851, F1 0.879, FPR 0.160) | `src/auditor/agents/citation/hallmark.py`, `docs/p2-mid-results.md` | **Fully Implemented** |
| **P2.8 Set B Dataset (~600 pairs)** | 60 arXiv papers, ~600 claim-source pairs, gold passages for Recall@5 | `src/auditor/agents/citation/set_b.py`, `datasets/set_b/` | **Divergent / Incomplete** |
| **P2.9 Citation Module Eval** | Recall@5, P/R/F1, full-text vs abstract-only ablation | `src/auditor/agents/citation/eval.py`, `scripts/citation_eval.py` | **Partially Implemented** |
| **P3.1 Repository Snapshots** | Pinned commit, size limits, no repo code execution, multi-host support | `src/auditor/infra/snapshots/` (`hosts.py`, `policy.py`, `snapshot.py`) | **Fully Implemented** |
| **P3.2 Static Settings Extraction** | Extract from `.py` (AST/argparse), `.yaml`, `.json`, `.toml`, notebooks, shell, split lists | `src/auditor/agents/code/extract/` | **Fully Implemented** |
| **P3.3 Setting Normalisation** | Units, number formats (`3e-4` vs `0.0003`), name aliases (`lr` vs `learning_rate`) | `src/auditor/agents/code/extract/normalise.py`, `names.py` | **Fully Implemented** |
| **P3.4 Code Agent Check** | `check()`: `settings_check`, `data_checks`, `data/` file analysis, `defects`, `method_check` | `src/auditor/agents/code/` (`settings_check.py`, `data_checks.py`, etc.) | **Fully Implemented** |
| **P3.5 Venue Profiles** | NeurIPS 2025, ICLR 2026, ICML 2025, ACL 2025, EMNLP 2025 profiles with citations | `configs/venues/*.json`, `configs/venues/README.md` | **Fully Implemented** |
| **P3.6 Template Checker** | Deterministic rule checking (max pages, fonts, margins, sections, anonymity), no LLM calls | `src/auditor/agents/template/rules.py`, `shared.py` | **Fully Implemented** |
| **P3.7 Code & Template Tools for B1** | List, search, read files, find setting, data report, check rule, layout summary | `src/auditor/agents/code/tools.py`, `src/auditor/agents/template/tools.py` | **Fully Implemented** |
| **P3.8 Set C Dataset (160 instances)** | 40 repos, 160 instances (SciCoQA), injected errors and clean controls | `datasets/set_c/labels.jsonl` (163 errors, 72 clean controls on 80 versions) | **Fully Implemented** |
| **P3.9 Code & Template Eval** | Template fixtures eval, Set C evaluation (P/R/F1 on settings, methods, metrics) | `tests/agents/template/test_fixtures.py`, `docs/p4-results.md` | **Fully Implemented** |
| **P3.10 Theory / Math Agent** | Cascade: plan, screen, deep check (Docker math sandbox), review challenge; SPOT proof set | `src/auditor/agents/theory/` (`__init__.py`, `review.py`, `spot.py`) | **Fully Implemented (Scope Expansion)** |
| **P4.1 Quote Grounding Validation** | Verbatim quote check in source (normalising ligatures/hyphens/spaces), code line verification | `src/auditor/agents/verifier/grounding.py` | **Fully Implemented** |
| **P4.2 Evidence Gating Logic** | Tri-state: `VERIFIED_FLAG`, `NEEDS_HUMAN_REVIEW`, `DROPPED`; two-sided proof requirement | `src/auditor/agents/verifier/decide.py`, `policy.py` | **Fully Implemented** |
| **P4.3 Verifier Fixtures & Decisions** | Fixtures, decision tracking split by gold label category | `tests/agents/verifier/test_decide.py`, `src/auditor/evaluation/metrics.py` | **Fully Implemented** |
| **P4.4 Report Generator** | JSON and Markdown reports with advisory wording, 3 sections, deduplication | `src/auditor/reporting/report.py` | **Fully Implemented** |
| **P4.5 Evaluation Runner** | Macro-F1, precision, recall, FPR, unlabelled flags, grounding rate, cost tracking | `src/auditor/evaluation/metrics.py`, `runner.py`, `runs.py` | **Fully Implemented (Metric Limitation)** |
| **P4.6 Set C Experiments** | 3 runs Full, 3 runs B0, 2 runs B1, Gemini 2.5 Flash, run manifests | `results/p4/set_c-rq1.md`, `.json`, `docs/p4-results.md` | **Fully Implemented** |
| **P4.7 Ablations** | Verifier ablation, single-agent ablation (B1), abstract-only retrieval ablation | `results/p4/set_c-verifier-ablation.md`, `docs/p4-results.md` | **Partially Implemented** |
| **P4.8 Result Tables & Failure Analysis** | Failure stage analysis, stability across runs, unlabelled flag analysis | `results/p4/set_c-failures.md`, `scripts/p4_analysis.py` | **Fully Implemented** |
| **D22 Integrity Agent** | Check structural placeholder markers (`??`, `[?]`, `TODO`, `XX%`) and result consistency | `src/auditor/agents/integrity/` (`__init__.py`, `results.py`, `set_d.py`) | **Fully Implemented (Scope Expansion)** |
| **Web UI** | Streamlit UI (`auditor ui`), child process execution, live progress & report viewing | `src/auditor/ui/app.py`, `audit.py` | **Fully Implemented (Scope Expansion)** |

---

## 2. Empirical & Evaluation Audit

### 2.1 Validation of Results in `results/p4/` and `docs/p4-results.md`

All primary headline metrics published in `docs/p4-results.md` were independently reproduced by inspecting the raw run JSON outputs in `results/p4/runs/` and evaluating `results/p4/set_c-rq1.json`.

#### Headline Performance on Set C (Benchmark RQ1)
The evaluation was executed across 80 paper versions (40 clean originals, 40 versions containing injected discrepancies) comprising 163 gold planted errors and 72 clean control settings:

| System Configuration | Macro-F1 | Precision | Recall | Clean False Alarm Rate | Flags / Original Paper | Cost / Paper ($) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Full System** (3 runs) | **0.703 ± 0.006** | 1.000 ± 0.000 | **0.468 ± 0.009** | 0.000 ± 0.000 | 3.12 ± 0.04 | $0.079 ± 0.000 |
| **Full w/o Verifier** (3 runs) | **0.736 ± 0.006** | 1.000 ± 0.000 | **0.521 ± 0.011** | 0.000 ± 0.000 | 4.71 ± 0.04 | $0.079 ± 0.000 |
| **Baseline B0** (Single-turn LLM) | **0.048 ± 0.015** | 1.000 ± 0.000 | **0.029 ± 0.009** | 0.000 ± 0.000 | 1.63 ± 0.25 | $0.047 ± 0.002 |
| **Baseline B1** (ReAct Agent) | **0.046 ± 0.007** | 1.000 ± 0.000 | **0.028 ± 0.004** | 0.000 ± 0.000 | 0.39 ± 0.23 | $0.542 ± 0.014 |

Exact verification in `results/p4/set_c-rq1.json`:
- Full System Macro-F1: $\mu = 0.702588, \sigma = 0.005539$.
- Full System w/o Verifier Macro-F1: $\mu = 0.735905, \sigma = 0.006108$.
- Baseline B0 Macro-F1: $\mu = 0.047878, \sigma = 0.014728$.
- Baseline B1 Macro-F1: $\mu = 0.046474, \sigma = 0.006799$.

#### Spend Ledger Audit
The evaluation spend recorded in `results/p4/ledger-p4.jsonl` was forensically verified:
- **Total Ledger Calls**: 6,997 structured model invocations.
- **Total Spend Across All P4 Runs**: Exactly **$125.04** (matching the reported figure down to the exact cent).
- Distribution: Pilot run: $1.70; Full system (3 runs): $18.84; Baseline B0 (3 runs): $11.35; Baseline B1 (2 complete runs): $86.78; Baseline B1 Run 3 (aborted at 10 papers): $6.36.

---

### 2.2 In-Depth Audit of Metric Computation Logic (`src/auditor/evaluation/metrics.py`)

A forensic audit of the scoring engine in `src/auditor/evaluation/metrics.py` revealed a critical flaw in how precision and false positives are computed:

```python
# src/auditor/evaluation/metrics.py:145-167
for prediction in predictions:
    if not prediction.is_positive:
        continue
    if classes is not None and prediction.error_class not in classes:
        continue
    matched = next((label for label in unmatched if matches(prediction, label)), None)
    if matched is None:
        scores.unlabelled += 1
        unlabelled.add(_unlabelled_key(prediction))
        continue
    unmatched.remove(matched)
    predicted = scores.per_class.setdefault(prediction.error_class, ClassScore())
    if not matched.is_error:
        scores.clean_flagged += 1
        predicted.false_positives += 1
        scores.outcomes[matched.label_id] = "flagged"
    elif matched.error_class == prediction.error_class:
        predicted.true_positives += 1
        scores.outcomes[matched.label_id] = "found"
    else:
        predicted.false_positives += 1
        scores.per_class[ErrorClass(matched.error_class)].false_negatives += 1
        scores.outcomes[matched.label_id] = "wrong_class"
```

#### Mechanism of the Flaw
1. Line 151 intercepts any positive prediction that does not match a gold label in `unmatched`.
2. It increments `scores.unlabelled += 1` and immediately calls `continue`, completely bypassing the `false_positives` accumulator.
3. `false_positives` is incremented **only** when a prediction matches a gold label that is either an explicit clean control (`not matched.is_error`) or a gold error of a different class (`matched.error_class != prediction.error_class`).
4. Set C contains only 72 clean controls tied to specific setting keys. If the system flags any other setting, method, metric, or formatting discrepancy on an original paper where no clean control label was placed, it is categorized as `unlabelled` rather than a false positive.
5. In all evaluation runs, no prediction ever collided with the 72 clean control keys, and no system assigned the wrong class to a matched label. Consequently, **False Positives were strictly zero ($FP = 0$) across all systems, runs, and classes**.

#### Mathematical Distortion
Because $FP \equiv 0$:
$$\text{Precision} = \frac{TP}{TP + FP} = \frac{TP}{TP + 0} \equiv 1.000$$
$$\text{Macro-F1} = \frac{2 \cdot P \cdot R}{P + R} = \frac{2 \cdot 1.0 \cdot R}{1.0 + R} = \frac{2R}{1 + R}$$

This calculation creates an artificial precision of 100% across all systems (Full, Full without Verifier, B0, and B1), regardless of whether the system emitted 28 unlabelled flags (B1) or 374 unlabelled flags (Full without verifier). Macro-F1 is mathematically reduced to a monotonic transformation of recall, completely concealing false alarms.

---

### 2.3 Audit of Experimental Validity on Set C

#### 1. Dataset Construction Circularity
`datasets/set_c/README.md` (lines 38–47 and 99–101) documents the pipeline used to build Set C:
> *"1. `collect` downloaded each arXiv PDF... and ran the code agent on every setting the planner found, against the repository..."*  
> *"2. `select` kept pairs where the code confirmed at least two settings... changed the paper's value for up to two of them and kept up to two others as clean controls."*  
> *"Read the setting scores with care. Injections were only made where the code agent had already found the paper's value in the repository, so recall on them measures whether the agent notices a changed value, not whether it can find settings in unfamiliar code."*

By filtering the benchmark to settings that the code agent was already capable of locating during dataset creation, the evaluation introduces selection bias: the 0.771 setting recall reflects performance on pre-filtered easy targets rather than an unbiased sample of ML codebases.

#### 2. Missing Independent Review (Violation of Decision D18)
Decision D18 mandated an independent 20% second review on all gold labels before freezing evaluation benchmarks. Inspection of all 235 entries in `datasets/set_c/labels.jsonl` shows that every entry contains `"reviewed_by": []`. The required second review was never conducted.

#### 3. Loose Document-Level Method Matching
In `src/auditor/schemas/anchors.py` (lines 71–72):
```python
case DocumentAnchor(), DocumentAnchor():
    return True
```
All 100 `method_mismatch` gold labels in `datasets/set_c/labels.jsonl` specify `target = SettingTarget(key="method")` and `anchor = DocumentAnchor()`. The code agent's method check (`src/auditor/agents/code/method_check.py:175-181`) hardcodes the exact same target and anchor for every method discrepancy it generates.

Consequently, `matches()` declares a match between any method prediction and any method gold label on the same paper version, without requiring any semantic overlap, quote match, or concept alignment. Forensic audit of Run 1 (`p4-setc-full-r1`) revealed:
- **Paper 2106.01342v1**: Gold label describes intersample attention vs token self-attention; prediction describes an MLP hidden layer with ReLU on the [CLS] token. **Scored as True Positive**.
- **Paper 2107.12710v2**: Gold label describes additive vs multiplicative graph attention weights; prediction describes graph layer output dimensionality (32 vs 16). **Scored as True Positive**.
- **Paper 2006.13807v2**: Gold label describes SoftMax vs multi-label CheXNet output activations; prediction describes convolutional kernel sizes (5 vs 3x3). **Scored as True Positive**.
- **Paper 2102.06514v3**: Contains two distinct method gold labels (`m-640fdd3e` for decay schedule $\tau$, `m-39619375` for stop-gradient parameter update). The code agent produced two duplicate variations of the $\tau$ schedule (`-f0` and `-f1`). The evaluator greedily paired `-f0` to `m-640fdd3e` and `-f1` to `m-39619375`, **awarding 2 True Positives for a duplicate finding** while the stop-gradient error was missed.

---

### 2.4 Audit of Baselines B0 and B1

#### 1. Schema and Formatting Asymmetry
In `src/auditor/baselines/common.py` (lines 63–72), the prompt template instructs baselines:
```python
target_kind and target: what the error is about. Use "bibliography" with an entry id,
"marker" with the citation marker text when it has no entry, "setting" with a snake_case
setting name such as "learning_rate", or "rule" with a venue rule id.
```
The baselines were **never instructed** that method mismatches must be targeted as `target="method"` with `anchor_kind="document"`, or that evaluation metric errors must be targeted as `target="eval_metric"`.

When Baseline B0 correctly identified the method discrepancy on paper `2102.06514v3`, it outputted:
```json
{"target_kind": "setting", "target": "target_network_initialization", "anchor_kind": "sentence", "anchor": "s352"}
```
Because the evaluator strictly looked for `target="method"` and `anchor="document"`, this valid detection was rejected by `matches()` and routed to `unlabelled`. Across all runs, B0 and B1 scored **0 True Positives out of 100 on method_mismatch** and **0 True Positives out of 3 on metric_mismatch**. Because metric mismatch (3 labels) and method mismatch (100 labels) are weighted equally as individual classes in Macro-F1, this schema omission artificially depressed baseline Macro-F1 scores by ~27 percentage points.

#### 2. Baseline B1 Token Runaway & Context Explosion
In `src/auditor/baselines/b1.py` (lines 96–109), Baseline B1 executes a ReAct tool loop (up to 20 turns). On each turn, B1 re-sent the full rendered paper text alongside the expanding history of tool call outputs.
- Average input tokens per paper version: **~680,000 tokens**.
- Cost per paper version: **$0.542** (6.9x more expensive than the full multi-agent system at $0.079).
- Total spend for 2 runs: **$86.78** (69.4% of the entire project budget).
- Outcome: Baseline B1 Run 3 had to be aborted after 10 papers when the $125 project budget ledger limit was hit, leaving B1 with only 2 completed runs instead of the planned 3.

---

### 2.5 Scientific Validation & Refutation of the Verifier Ablation Claim

The central research hypothesis of the project proposal was:
> *"The evidence-gated verifier suppresses ungrounded candidate findings without destroying recall."*

A forensic analysis of all 1,377 verifier decisions across the three Set C runs refutes this hypothesis:

```
                            All 1,377 Candidate Findings
                                         |
               +-------------------------+-------------------------+
               |                                                   |
               v                                                   v
      Verified Flag: 944                                  Human Review: 433
   (229 Real Errors, 691 Unlabelled,                   (26 Real Errors, 377 Unlabelled,
      0 Clean Controls, 0 Dropped)                        0 Clean Controls, 0 Dropped)
                                                                   |
                                                +------------------+------------------+
                                                |                                     |
                                                v                                     v
                                       Specialist Marked                     Missing Repo
                                        Uncertain: 421                       Snapshot: 12
                                    (Includes 26 Real Errors)
```

1. **Zero Dropped Findings**: Across all 1,377 findings submitted to the verifier, **exactly zero findings were dropped** (`DROPPED = 0`). In 100% of candidate findings, the quotes existed verbatim in the paper and code lines existed in the repository snapshot. The verifier never encountered or suppressed an ungrounded hallucination on Set C.
2. **Mechanism of Action**: The verifier functioned exclusively as an uncertainty threshold. In `src/auditor/agents/verifier/decide.py` (line 125):
   ```python
   if finding.uncertain:
       return Decision(state=VerificationState.NEEDS_HUMAN_REVIEW, ...)
   ```
3. **Recall Degradation**: The specialist code agent marked 26 true planted errors (22 setting mismatches, 4 method mismatches) with `uncertain=True`. The verifier demoted these 26 findings to `NEEDS_HUMAN_REVIEW`. Under the evaluation protocol (`src/auditor/schemas/evaluation.py:64`), only `VERIFIED_FLAG` counts as positive, causing these 26 true errors to be counted as False Negatives.
4. **Net Scientific Impact**:
   - Macro-F1 dropped by **3.3 percentage points** (0.736 to 0.703).
   - Recall dropped by **5.3 percentage points** (52.1% to 46.8%).
5. **Real-World Calibration (Correcting the Metric)**:
   If unlabelled flags on original papers (238/run) are accounted for as potential false alarms, the verifier's filtering of uncertain findings actually improves real precision from 0.182 to 0.240 and real F1 from 0.268 to 0.316. The reported conclusion that the verifier "imposes a cost of 3.3 points of Macro-F1" is an artifact of excluding false alarms from precision in `metrics.py`.

---

## 3. Subsystem Health Reviews

### 3.1 Part 1: PDF Parser & Task Planner

```
[Raw PDF / LaTeX]
       |
       +---> GROBID (Docker/HTTP) ------> TEI XML ------> Section/Sentence Anchors
       +---> pdfplumber Layout ---------> Layout Metrics, Fonts, Margins, GitHub Links
       +---> pdftotext -layout ---------> Plaintext Layout Tables (72% Table Recall)
       +---> pdfplumber (300 dpi) ------> Table Crop Images (96-98% Table Recall)
       +---> Direct LaTeX Parser -------> TeX Macros, Math Formulas, Citations
                                                 |
                                                 v
                                    ParsedPaper Unified Contract
                                                 |
       +-----------------------------------------+-----------------------------------------+
       |                                                                                   |
       v                                                                                   v
[Deterministic Planner]                                                       [LLM Candidate Refinement]
Regex & AST parameter extraction                                              Splits multi-citation sentences
Extracts numbers with units                                                   Filters non-check-worthy claims
Discovers GitHub/GitLab repositories                                          Strictly preserves exact quote anchors
       |                                                                                   |
       +-----------------------------------------+-----------------------------------------+
                                                 |
                                                 v
                                 AuditTask Graph (Sentence, Bib, Page, Doc)
```

- **GROBID & TEI Pipeline (`grobid.py`, `tei.py`, `linking.py`)**: Robust HTTP client interfacing with GROBID service with 300s timeouts. Accurately extracts TEI XML and converts sections, paragraphs, sentences, and citation callouts into `ParsedPaper` instances.
- **Visual Table Extraction Scope Expansion (`layout_tables.py`, `table_crops.py`)**: While the original proposal envisioned basic text parsing, the team engineered a dual-engine table extractor: layout text parsing via `pdftotext -layout` and high-resolution (300 dpi) visual table cropping via `pdfplumber`. On the 39-paper LaTeX benchmark (`docs/decisions.md:1075-1095`), visual table cropping boosted table number recall from 72% to 96–98%.
- **LaTeX Direct Parser (`latex.py`)**: Added support for arXiv source bundles (`.tar.gz`), parsing raw LaTeX macros, math formulas, and BibTeX trees directly, bypassing PDF OCR ambiguities.
- **Task Planner (`tasks.py`, `claims.py`, `settings.py`)**: Combines deterministic extraction (numbers, units, snake_case hyperparameters, repository URLs) with LLM candidate refinement (`refine_candidates`). The LLM splits sentences citing multiple references into single atomic claims while guaranteeing that the resulting quotes remain verbatim substrings of the original paper sentences.

---

### 3.2 Part 2: Citation Agent & Scholarly Retrieval

- **Scholarly API Integration (`src/auditor/retrieval/clients.py`, `resolve.py`)**: Implements asynchronous clients for arXiv, OpenAlex, Crossref, and Semantic Scholar. Cascading resolution queries DOI, arXiv ID, or title/author fuzzy search through the shared `HttpClient` rate-limiter and disk cache.
- **Passage Retrieval & Ranking (`passages.py`, `embeddings.py`)**: Dual-engine retriever combining BM25 keyword matching with local ONNX-runtime `all-MiniLM-L6-v2` dense embeddings, ranking source passages with character offset tracking.
- **Citation Verification (`agents/citation/reference.py`, `support.py`)**: Verifies reference existence, metadata fields (allowing publication year $\pm 1$ tolerance and author spelling discrepancies), citation claim support via LLM NLI, and quantitative number consistency.
- **Evaluation Status & The Part 4 Mocking Gap**:
  - The Citation Agent was independently validated on Set A (HALLMARK benchmark, 831 entries), achieving **0.851 Accuracy, 0.879 F1, and a low False Positive Rate of 0.160**.
  - However, in Part 4 full-system experiments (`results/p4/`), **the citation agent was completely mocked out** (`mocked: citation` in all run manifests). The promised evaluation on Set B (~600 claim-source pairs) was never executed in Part 4 due to missing local source PDFs and credit conservation rules. Furthermore, `datasets/set_b/labels.jsonl` currently contains only 7 claim-level error labels.

---

### 3.3 Part 3: Code Agent & Template Checker

- **Repository Snapshot Isolation (`infra/snapshots/`)**: Strictly enforces read-only snapshot retrieval from GitHub/GitLab, enforcing repository size limits and guaranteeing that no untrusted repository code is executed during audits.
- **Static Parameter Extraction (`agents/code/extract/`)**: Comprehensive static analysis extracting parameters from Python ASTs (`ast.parse`), `argparse` definitions, YAML, JSON, TOML, Jupyter notebook cells, and shell script environment variables.
- **Parameter Normalization (`extract/normalise.py`, `names.py`)**: Automatically resolves unit conversions (e.g., seconds vs milliseconds), scientific notation (`3e-4` vs `0.0003`), and naming aliases (`lr` vs `learning_rate`, `bs` vs `batch_size`).
- **Defect and Method Checks (`defects.py`, `method_check.py`, `code_retrieval.py`)**:
  - AST defect detection runs `compile()` to catch syntax errors and executes Ruff in a subprocess with 120s timeouts to catch undefined variables (`F821`, `F822`, `F823`).
  - Method consistency checks use semantic code search (fastembed/Nomic) to retrieve relevant functions and compare implementation logic against paper claims.
- **Template Checker (`agents/template/rules.py`, `shared.py`)**: 100% deterministic rule engine verifying page budgets, font size minimums, margins, required sections, and double-blind anonymity against official profiles for NeurIPS, ICLR, ICML, ACL, and EMNLP with zero LLM API costs.

---

### 3.4 Specialized Extensions: Theory / Math Agent & Integrity Agent

- **Theory / Math Agent (`agents/theory/`)**:
  - Implements a four-stage verification cascade: claim planning, initial screening, deep symbolic verification, and review challenge.
  - Deep verification executes in an isolated Docker container (`auditor-math:1`) running SymPy under non-root permissions, dropped Linux capabilities, blocked network access, and a 20-second timeout.
  - Evaluated on the SPOT proof set (`results/spot/`), successfully identifying 8 out of 14 planted mathematical defects.
- **Integrity Agent (`agents/integrity/`)**:
  - Inspects structural placeholder markers (`??`, `[?]`, `TODO`, `TBD`, `XX%`) and audits result consistency across text claims and LaTeX/crop tables.
  - Evaluated on Set D (`datasets/set_d/`), detecting 168 out of 192 planted integrity defects.
  - **Identified Failure Mode**: The agent is highly susceptible to parser sentence-splitting artifacts, erroneously flagging hyphenated line breaks and formula splits as "unfinished text" (accounting for 64.1% of all unlabelled flags on Set C).

---

### 3.5 Part 4: Evidence-Gated Verifier

- **Two-Sided Proof Policy (`agents/verifier/policy.py`, `decide.py`)**: Requires that any discrepancy finding provide verifiable evidence from both the paper side (exact quote present in `paper.sentences`) and the external side (exact line in repository snapshot, or registered rule in venue profile).
- **Grounding Checks (`grounding.py`)**: Re-verifies verbatim quotes against source texts, normalizing ligatures (e.g., `ﬁ` $\rightarrow$ `fi`), discretionary hyphens, and whitespace breaks.
- **Tri-State Decision Gating**:
  - `VERIFIED_FLAG`: Two-sided evidence confirmed, quote grounded, specialist confident.
  - `NEEDS_HUMAN_REVIEW`: Evidence grounded, but specialist flagged `uncertain=True` or finding rests on absence.
  - `DROPPED`: Evidence missing or quote not grounded in source text.
- **Review Passes (`recheck.py`, `theory/review.py`)**: Optional second-pass model reviews designed to challenge high-severity flags before final synthesis.

---

### 3.6 Reporting Engine & Streamlit Web UI

- **Reporting Engine (`src/auditor/reporting/report.py`)**: Generates structured JSON reports and publication-grade Markdown audit summaries. Follows strict advisory phrasing conventions (`"Consider verifying..."` rather than accusatory language), groups findings into three clear severity sections, and deduplicates redundant findings across specialists.
- **Streamlit Web UI (`src/auditor/ui/app.py`, `audit.py`)**: Provides an interactive browser dashboard enabling users to upload PDF papers, configure repository snapshots, select venue profiles, track live audit progress via streaming logs, and inspect findings with interactive quote anchors.
- **UI Testing & Robustness Vulnerabilities**:
  - `src/auditor/ui/app.py` has **0.00% automated test coverage** (185/185 statements unexecuted).
  - The UI launches the audit runner via `subprocess.Popen` without an execution timeout or cancellation guard, risking infinite worker thread hangs if the CLI deadlocks.

---

### 3.7 Core Infrastructure, LLM Factory & LangGraph Orchestrator

- **LangGraph StateGraph (`orchestration/graph.py`, `runner.py`)**: Implements dynamic task fan-out using LangGraph `Send` branches, thread checkpointing via SQLite, retry loops with exponential backoff and jitter, and synthesis reducers.
- **HTTP Policy Gate (`infra/http.py`)**: Robust network client with strict domain allowlists, per-host token bucket rate limiting (arXiv: 0.33 req/s; Semantic Scholar: 1.0 req/s), SHA-256 on-disk response caching, and automatic retry on HTTP 429/5xx.
- **LLM Factory & Spend Ledger (`infra/llm.py`, `usage.py`)**: Encapsulates model interactions behind `StructuredLLM`. Implements an advanced self-correction loop with prose fallback rewriting, detailed schema retry prompts, and an atomic spend ledger recording every dollar incurred.

---

## 4. Verified Findings & Suggested Improvements Ranked by Severity

All 12 findings below have been independently verified through source code inspection, static analysis, raw data file parsing, and empirical reproduction under the Anaconda environment.

```
+----------------------------------------------------------------------------------------------------+
|                                    12 VERIFIED AUDIT FINDINGS                                      |
+----------------------------------------------------------------------------------------------------+
|  CRITICAL SEVERITY                                                                                 |
|  [F1] Set B Citation Evaluation Gap & Citation Mocking in Final Experiments                        |
|  [F2] Artificial 1.000 Precision & Unlabelled Flag Exclusion in metrics.py                         |
+----------------------------------------------------------------------------------------------------+
|  HIGH SEVERITY                                                                                     |
|  [F3] Verifier Ablation Mechanism Refutation (0 Dropped Findings, Demoted Real Errors)             |
|  [F4] Loose Document-Level Method Matching & Spurious True Positives                              |
|  [F5] Baseline Prompt & Schema Asymmetry (B0/B1 Structurally Disadvantaged)                         |
|  [F6] Parser Splitting Artifacts Causing 64% of Unlabelled Flags                                   |
|  [F7] Pipeline Crash Risk in Unwrapped LangGraph Nodes & Unhandled LLM Parsing                     |
+----------------------------------------------------------------------------------------------------+
|  MEDIUM SEVERITY                                                                                   |
|  [F8] Benchmark Circularity & Missing Second Review on Set C                                       |
|  [F9] Zero Test Coverage in Web UI & Missing Subprocess Timeout                                    |
|  [F10] Resource Leaks & Zombie Subprocesses in Evaluation and Sandbox                              |
+----------------------------------------------------------------------------------------------------+
|  LOW SEVERITY                                                                                      |
|  [F11] Baseline B1 Spending Runaway & Prompt Bloat                                                 |
|  [F12] Packaging Standards Omission (Missing py.typed Marker)                                      |
+----------------------------------------------------------------------------------------------------+
```

---

### 4.1 Critical Findings

#### Finding 1: Set B Citation Evaluation Gap & Citation Mocking in Final Experiments
- **Severity**: **CRITICAL**
- **Exact Source & Data Citations**:
  - `results/p4/set_c-rq1.md`: lines 36–46 (`Notes` column: `mocked: citation` for all runs).
  - `results/p4/runs/p4-setc-full-r1/manifest.json`: `"mocked_parts": ["citation"]`, `"skip_parts": ["citation"]`.
  - `docs/p4-results.md`: lines 31–34 and 200–205.
  - `datasets/set_b/labels.jsonl`: lines 1–158 (158 total entries; only 7 claim error labels).
- **Verified Evidence & Technical Root Cause**:
  While the Citation Agent codebase is fully written and tested on unit mocks, it was completely excluded from the Part 4 evaluation experiments reported in `docs/p4-results.md`. Every Set C run explicitly mocked out the citation agent. The promised evaluation on Set B (~600 claim-source pairs) was never executed in Part 4. The raw dataset `data/set_b` exists only on an external uncommitted machine, and `datasets/set_b/labels.jsonl` contains only 7 claim-level error labels (4 number mismatches, 3 unsupported claims).
- **Impact Analysis**:
  The system's claim of end-to-end citation verification remains scientifically unverified in the final integrated pipeline. The interaction between citation retrieval, LLM NLI verification, and the evidence-gated verifier was never tested at scale.
- **Step-by-Step Remediation Guidance**:
  1. Commit or reconstruct the `data/set_b` paper pairs and cached source PDFs into the project repository or a persistent cloud bucket.
  2. Expand `datasets/set_b/labels.jsonl` to at least 100 well-annotated claim-source pairs covering reference existence, number mismatches, and unsupported claims.
  3. Execute the full system and Baseline B1 on Set B with `citation_llm_enabled = True` and report end-to-end precision, recall, and Macro-F1.

---

#### Finding 2: Artificial 1.000 Precision & Unlabelled Flag Exclusion in `metrics.py`
- **Severity**: **CRITICAL**
- **Exact Source & Data Citations**:
  - `src/auditor/evaluation/metrics.py`: lines 151–154 (`if matched is None: scores.unlabelled += 1; continue`).
  - `results/p4/set_c-rq1.json`: `summary.*.Precision.mean = 1.0`, `clean_flagged = 0`, `fp = 0`.
  - `results/p4/set_c-rq1.md`: lines 7–10 and 25–30.
- **Verified Evidence & Technical Root Cause**:
  In `src/auditor/evaluation/metrics.py`, line 151 skips any positive prediction that does not match an existing gold label, routing it to `unlabelled` and skipping `predicted.false_positives += 1`. Because Set C only has 72 clean controls on specific setting keys, the hundreds of unlabelled flags raised on original papers (averaging 238 per run for Full, 374 for Full without verifier) never increment false positives. Consequently, False Positives are strictly zero for all systems, mathematically locking reported precision to 1.000 and forcing Macro-F1 to be a pure function of Recall.
- **Impact Analysis**:
  The reported 1.000 precision is scientifically invalid. It hides over 238 potential false alarms per run on clean papers and invalidates the claim that the full system achieved 0.703 Macro-F1 without generating false alarms.
- **Step-by-Step Remediation Guidance**:
  1. Update `src/auditor/evaluation/metrics.py` to support an evaluation mode where unlabelled positive predictions on original papers are penalized as false positives:
     ```python
     if matched is None:
         scores.unlabelled += 1
         unlabelled.add(_unlabelled_key(prediction))
         if prediction.variant == "original":
             predicted = scores.per_class.setdefault(prediction.error_class, ClassScore())
             predicted.false_positives += 1
         continue
     ```
  2. Execute Task S5: Hand-adjudicate a random sample of 50 unlabelled flags on original papers to determine true false alarm rates vs undisclosed bugs.
  3. Recompute and publish calibrated precision and Macro-F1 metrics.

---

### 4.2 High Severity Findings

#### Finding 3: Verifier Ablation Mechanism Refutation (0 Dropped Findings, Demoted Real Errors)
- **Severity**: **HIGH**
- **Exact Source & Data Citations**:
  - `src/auditor/agents/verifier/decide.py`: lines 85–133.
  - `results/p4/set_c-rq1.md`: lines 25–30 (Table: Real error: 229 Verified, 26 Review, 0 Dropped; Unlabelled: 691 Verified, 377 Review, 0 Dropped).
  - `results/p4/set_c-failures.md`: line 17.
  - `src/auditor/schemas/evaluation.py`: line 64 (`is_positive` requires `VERIFIED_FLAG`).
- **Verified Evidence & Technical Root Cause**:
  Across all 1,377 findings submitted across three Set C runs, the verifier dropped **zero findings** (100% of candidate findings had evidence found in source). The verifier acted solely as an uncertainty filter: it demoted 26 real planted errors to `NEEDS_HUMAN_REVIEW` because the specialist set `uncertain=True`. Because evaluation counts only `VERIFIED_FLAG` as positive, these 26 true errors became False Negatives, **reducing Macro-F1 from 0.736 to 0.703 (-3.3%) and recall from 52.1% to 46.8% (-5.3%)**.
- **Impact Analysis**:
  The proposal's core scientific claim—that the verifier suppresses ungrounded candidate findings without destroying recall—is refuted by the project's own empirical data on Set C.
- **Step-by-Step Remediation Guidance**:
  1. Calibrate specialist uncertainty thresholds so confident setting and method discrepancies are not marked uncertain.
  2. Revise `decide.py` so that findings with strong two-sided textual grounding are not downgraded to human review solely based on a specialist uncertainty flag.
  3. Test the verifier on tasks where ungrounded hallucinations genuinely occur (e.g., citation NLI or mathematical proofs).

---

#### Finding 4: Loose Document-Level Method Matching & Spurious True Positives
- **Severity**: **HIGH**
- **Exact Source & Data Citations**:
  - `src/auditor/schemas/evaluation.py`: lines 86–93 (`matches()` logic).
  - `src/auditor/schemas/anchors.py`: lines 71–72 (`case DocumentAnchor(), DocumentAnchor(): return True`).
  - `src/auditor/agents/code/method_check.py`: lines 175–181, 249–255.
  - `datasets/set_c/labels.jsonl`: all 100 `method_mismatch` gold labels.
- **Verified Evidence & Technical Root Cause**:
  `matches()` matches predictions to gold labels if `paper_id`, `variant`, `target`, and `anchor` match. Because all 100 method gold labels and all generated method findings hardcode `target = SettingTarget(key="method")` and `anchor = DocumentAnchor()`, any method prediction on an injected paper matches any method gold label. In Run 1, predictions describing MLP activations, kernel sizes, and dimensionality were scored as True Positives for attention and loss function defects. Furthermore, duplicate variations of the same finding on paper `2102.06514v3` claimed two distinct gold labels, double-counting recall.
- **Impact Analysis**:
  Method mismatch recall (reported at 0.536) and true positive counts are significantly inflated by coincidental and duplicate matches.
- **Step-by-Step Remediation Guidance**:
  1. Add semantic or text overlap requirements to `matches()` for `method_mismatch`:
     ```python
     if prediction.error_class == ErrorClass.METHOD_MISMATCH:
         if not semantic_overlap(prediction.summary, label.rationale, threshold=0.5):
             return False
     ```
  2. Implement prediction deduplication so multiple predictions from the same paper cannot be greedily assigned to multiple distinct gold labels without distinct evidence anchors.

---

#### Finding 5: Baseline Prompt & Schema Asymmetry (B0/B1 Structurally Disadvantaged)
- **Severity**: **HIGH**
- **Exact Source & Data Citations**:
  - `src/auditor/baselines/common.py`: lines 63–72 (`ANSWER_GUIDE`).
  - `results/p4/set_c-rq1.json`: B0 and B1 scored 0 True Positives on `method_mismatch` (0/100) and 0 True Positives on `metric_mismatch` (0/3).
  - `results/p4/runs/p4-setc-b0-r1`: paper `2102.06514v3` (valid method finding rejected due to fine-grained sentence anchor).
- **Verified Evidence & Technical Root Cause**:
  The baseline prompt template `ANSWER_GUIDE` instructs baselines to use snake_case setting names and sentence anchors, but never discloses that method mismatches must be targeted as `target="method"` with `anchor="document"` or that metric errors must be targeted as `target="eval_metric"`. When B0 detected method issues, it produced sentence anchors and fine-grained target keys, causing the evaluator to reject them. This schema starvation guaranteed 0% recall on method and metric classes, artificially depressing baseline Macro-F1 scores by ~27 points.
- **Impact Analysis**:
  The comparison between the full system (Macro-F1 0.703) and baselines (Macro-F1 ~0.047) is methodologically unfair. A substantial portion of the performance gap stems from schema formatting asymmetry rather than auditing capability.
- **Step-by-Step Remediation Guidance**:
  1. Update `ANSWER_GUIDE` in `src/auditor/baselines/common.py` to describe all valid target kinds and expected anchor granularities for methods and metrics.
  2. Re-run B0 and B1 with the aligned schema instructions.

---

#### Finding 6: Parser Splitting Artifacts Causing 64% of Unlabelled Flags
- **Severity**: **HIGH**
- **Exact Source & Data Citations**:
  - `results/p4/runs/p4-setc-full-r1/*/predictions.jsonl`: 152 out of 237 unlabelled predictions (64.14%) are `unfinished_or_placeholder_content`.
  - `results/p4/runs/p4-setc-full-r1/*/verified.jsonl`: 152 out of 152 verified as `verified_flag`.
  - `src/auditor/agents/integrity/__init__.py`: sentence splitting heuristic checks.
  - `datasets/set_c/labels.jsonl`: 0 gold labels for `unfinished_or_placeholder_content`.
- **Verified Evidence & Technical Root Cause**:
  The Integrity Agent flagged sentences that ended abruptly before continuing on another line or after a math formula as "unfinished placeholder content". In reality, these were standard PDF sentence-splitting artifacts generated by GROBID and pdftotext. Because the truncated sentence text existed verbatim in `paper.sentences`, the verifier verified 100% of these spurious findings. Because Set C had no labels for this class, all 152 escaped into `unlabelled` without penalizing precision.
- **Impact Analysis**:
  This represents the single largest source of false alarms in the system, generating 1.9 spurious flags per paper and undermining user trust.
- **Step-by-Step Remediation Guidance**:
  1. Add boundary heuristics in `src/auditor/agents/integrity/` to suppress flags when the subsequent sentence begins with a lowercase letter, formula, or continuation punctuation.
  2. Require true placeholder markers (`??`, `TODO`, `XX%`) for `unfinished_or_placeholder_content` rather than grammatical truncation.

---

#### Finding 7: Pipeline Crash Risk in Unwrapped LangGraph Nodes & Unhandled LLM Parsing
- **Severity**: **HIGH**
- **Exact Source & Data Citations**:
  - `src/auditor/orchestration/graph.py`: lines 110–114 (`plan` node) and lines 130–136 (`verify` node).
  - `src/auditor/agents/verifier/recheck.py`: line 68 (`await ctx.require_review_llm().generate(Rechecks, ...)`).
  - `src/auditor/agents/theory/review.py`: line 73 (`await ctx.require_review_llm().generate(MathVerdicts, ...)`).
- **Verified Evidence & Technical Root Cause**:
  While specialist task execution is safely isolated by `run_task()` in `runner.py`, the `plan` and `verify` nodes in `graph.py` contain no `try ... except` error handling. Furthermore, `recheck.py` and `theory/review.py` call `llm.generate()` directly without catching `StructuredOutputError`. If the LLM produces malformed JSON or times out after retries, an unhandled exception propagates through `verify` to LangGraph, crashing the entire pipeline and abandoning the audit run without saving results.
- **Impact Analysis**:
  A single malformed LLM response during verification or review crashes the entire multi-paper audit batch.
- **Step-by-Step Remediation Guidance**:
  1. Wrap `plan` and `verify` in `src/auditor/orchestration/graph.py` with try/except blocks that log errors and return fallback states.
  2. Surround `generate()` calls in `recheck.py` and `theory/review.py` with `try ... except StructuredOutputError:` and fall back to the unreviewed findings.

---

### 4.3 Medium Severity Findings

#### Finding 8: Benchmark Circularity & Missing Second Review on Set C
- **Severity**: **MEDIUM**
- **Exact Source & Data Citations**:
  - `datasets/set_c/README.md`: lines 38–47, 99–101.
  - `datasets/set_c/labels.jsonl`: lines 1–235 (all entries have `"reviewed_by": []`).
  - `docs/decisions.md`: Decision D18 (mandatory 20% second review).
- **Verified Evidence & Technical Root Cause**:
  Set C paper-code pairs were selected by first running the code agent and keeping only settings that the agent successfully extracted from the repository. Injected errors were then placed exclusively on those confirmed settings. Furthermore, none of the 235 labels underwent the mandatory 20% second review required by Decision D18.
- **Impact Analysis**:
  Benchmark circularity limits the evaluation's ecological validity, testing only whether the agent recognizes a modified value on code structures it is already tailored to parse.
- **Step-by-Step Remediation Guidance**:
  1. Perform an independent second review on at least 20% of Set C labels, recording reviewer IDs in `reviewed_by`.
  2. Augment Set C with independently curated paper-code repositories where settings were not pre-screened by the agent.

---

#### Finding 9: Zero Test Coverage in Web UI & Missing Subprocess Timeout
- **Severity**: **MEDIUM**
- **Exact Source & Data Citations**:
  - `src/auditor/ui/app.py`: lines 8–352 (0.00% coverage, 185 statements unexecuted).
  - `src/auditor/ui/app.py`: lines 191–225 (`subprocess.Popen` execution loop).
- **Verified Evidence & Technical Root Cause**:
  Automated pytest runs report 0.00% coverage for `src/auditor/ui/app.py`. The Streamlit application launches the CLI audit subprocess using `subprocess.Popen` inside a `while True:` polling loop with no overall timeout. If the audit subprocess hangs or enters an infinite loop, the UI worker thread spins indefinitely, and closing the browser does not terminate the orphaned process.
- **Impact Analysis**:
  Production UI hangs and resource exhaustion can occur on deadlocked audits, with zero automated test regression protection.
- **Step-by-Step Remediation Guidance**:
  1. Implement an execution timeout (e.g., `max_timeout_s = 600`) and enforce process termination (`process.kill()`) in `app.py`.
  2. Write automated unit/integration tests for `ui/app.py` using Streamlit's `AppTest` framework.

---

#### Finding 10: Resource Leaks & Zombie Subprocesses in Evaluation and Sandbox
- **Severity**: **MEDIUM**
- **Exact Source & Data Citations**:
  - `src/auditor/evaluation/runs.py`: lines 183, 186, 235 (direct `.open()` calls without context managers).
  - `src/auditor/infra/sandbox.py`: lines 85–94 (`_docker()` timeout does not kill child process).
  - `src/auditor/config.py`: line 112 (`<scratch_dir>` defaults to `/tmp/auditor-scratch` with no eviction).
- **Verified Evidence & Technical Root Cause**:
  In `runs.py`, files are opened via `.open()` without a `with` context manager, relying on garbage collection to close descriptors. In `sandbox.py`, if `asyncio.wait_for` times out during image build, inspect, or kill, the child Docker process is never reaped. Additionally, `/tmp/auditor-scratch` accumulates repository checkouts, parquet tables, and PDFs indefinitely with no TTL or LRU cache eviction.
- **Impact Analysis**:
  Large batch runs risk file descriptor exhaustion, zombie process accumulation, and unbounded disk space consumption.
- **Step-by-Step Remediation Guidance**:
  1. Replace direct `.open()` calls in `runs.py` with `with path.open(...) as file:`.
  2. Add `process.kill(); await process.wait()` in `sandbox.py:_docker` when `TimeoutError` occurs.
  3. Implement an LRU cache eviction policy with maximum disk quota in `config.py` for `<scratch_dir>`.

---

### 4.4 Low Severity Findings

#### Finding 11: Baseline B1 Spending Runaway & Prompt Bloat
- **Severity**: **LOW**
- **Exact Source & Data Citations**:
  - `src/auditor/baselines/b1.py`: lines 78–109.
  - `docs/p4-results.md`: lines 174–182.
  - `results/p4/set_c-rq1.md`: lines 45–46 (55.7M input tokens in run 2, $44.19).
- **Verified Evidence & Technical Root Cause**:
  Baseline B1 re-sends the complete paper text and cumulative tool interaction history on every turn across 20 turns, averaging ~680,000 input tokens per paper version. B1 cost $0.542/paper, exhausting the budget and forcing the third run to abort after 10 papers.
- **Impact Analysis**:
  Wasted $86.78 in cloud credits, prevented B1 from completing its third evaluation run, and caused token context degradation in the baseline LLM.
- **Step-by-Step Remediation Guidance**:
  1. Implement a sliding message window or summarize previous tool outputs in B1 middleware (`_Accounted`).
  2. Send only relevant paper sections rather than the entire rendered paper on intermediate tool calls.

---

#### Finding 12: Packaging Standards Omission (Missing `py.typed` Marker)
- **Severity**: **LOW**
- **Exact Source & Data Citations**:
  - Root directory check: `src/auditor/py.typed` is absent.
  - Command: `python -m mypy tests scripts` (yields 572 errors across 99 files without `MYPYPATH=src`).
  - `pyproject.toml`: `types-PyYAML` listed under dev dependency groups but omitted from default runtime environment.
- **Verified Evidence & Technical Root Cause**:
  `src/auditor/` lacks the PEP 561 `py.typed` marker file. When downstream users or tests import `auditor` as an installed package, Mypy skips analyzing inline type hints, emitting 572 errors.
- **Impact Analysis**:
  Violates Python packaging standards and hinders external libraries from type-checking against `auditor` schemas.
- **Step-by-Step Remediation Guidance**:
  1. Create an empty marker file at `src/auditor/py.typed`.
  2. Ensure `types-PyYAML` is installed in standard developer and CI environments.

---

## 5. Architectural & Scientific Roadmap

A prioritized, phased roadmap to resolve all verified audit findings and prepare the Research Paper Auditor for production deployment and peer-reviewed publication:

```
+----------------------------------------------------------------------------------------------------+
|                                    5-PHASE REMEDIATION ROADMAP                                     |
+----------------------------------------------------------------------------------------------------+
|  Phase 1: Critical Evaluation & Metric Integrity (Days 1-5)                                        |
|  - Penalize original paper unlabelled flags as false positives in metrics.py                       |
|  - Execute Task S5: Hand-adjudicate 50 unlabelled flags to calibrate true false alarm rate        |
|  - Reconstruct data/set_b and execute full system on Set B with citation enabled                   |
+----------------------------------------------------------------------------------------------------+
|  Phase 2: Pipeline Hardening & Crash Isolation (Days 6-10)                                         |
|  - Wrap plan and verify nodes in graph.py with error boundaries and fallback reporting            |
|  - Catch StructuredOutputError in recheck.py and theory/review.py                                 |
|  - Enforce execution timeouts and process cleanup on subprocess.Popen in Streamlit Web UI          |
+----------------------------------------------------------------------------------------------------+
|  Phase 3: Method Matching & Specialist Calibration (Days 11-16)                                    |
|  - Implement semantic/quote overlap matching in matches() for method_mismatch                      |
|  - Deduplicate predictions matching the same gold label                                            |
|  - Add hyphen and math boundary heuristics in Integrity Agent to eliminate 152 false alarms/run    |
+----------------------------------------------------------------------------------------------------+
|  Phase 4: Baseline Alignment & Benchmark Expansion (Days 17-22)                                    |
|  - Align ANSWER_GUIDE in baselines/common.py with target="method" and anchor="document"            |
|  - Implement tool history pruning in Baseline B1 to eliminate token runaway                        |
|  - Complete Decision D18 mandatory 20% second review across all Set C gold labels                  |
+----------------------------------------------------------------------------------------------------+
|  Phase 5: Packaging & Production Deployment Readiness (Days 23-28)                                 |
|  - Add src/auditor/py.typed marker for PEP 561 compliance                                          |
|  - Implement AppTest automated test suite for src/auditor/ui/app.py                                |
|  - Add LRU cache eviction and disk quota management for <scratch_dir>                              |
+----------------------------------------------------------------------------------------------------+
```

### Phase 1: Critical Evaluation & Metric Integrity (Days 1–5)
1. **Remediate Metric Computation (`metrics.py`)**:
   - Modify `src/auditor/evaluation/metrics.py` so that unlabelled predictions on original (clean) papers increment `false_positives`.
   - Publish dual metrics: strict evaluation (unlabelled = FP) and permissive evaluation (unlabelled ignored).
2. **Execute Task S5 (Unlabelled Flag Adjudication)**:
   - Sample 50 random unlabelled flags generated on original papers.
   - Adjudicate whether each flag represents a true undisclosed discrepancy in the published paper/code or a pipeline false alarm.
   - Establish the true empirical False Alarm Rate ($FAR$).
3. **Resolve the Set B Citation Evaluation Gap**:
   - Reconstruct or retrieve `data/set_b` containing parsed papers and retrieved source PDFs.
   - Expand `datasets/set_b/labels.jsonl` to $\ge 100$ balanced claim-source pairs.
   - Run the full system and Baseline B1 on Set B and publish verified citation precision, recall, and Macro-F1.

### Phase 2: Pipeline Hardening & Crash Isolation (Days 6–10)
1. **LangGraph StateGraph Isolation**:
   - Surround `planner()` in `src/auditor/orchestration/graph.py` with try/except, returning an empty task list and logging failure rather than aborting.
   - Surround `verifier()` and `pipeline.reviews` with try/except, synthesizing unverified candidate findings as fallback rather than crashing the pipeline.
2. **LLM Structured Output Exception Handling**:
   - Add explicit `except StructuredOutputError:` handling in `src/auditor/agents/verifier/recheck.py:68` and `src/auditor/agents/theory/review.py:73`.
3. **Web UI Subprocess Management**:
   - Add a 600-second execution deadline to `src/auditor/ui/app.py`.
   - Implement an explicit process termination handler (`process.kill()`) triggered on timeout or UI session disconnect.

### Phase 3: Method Matching & Specialist Calibration (Days 11–16)
1. **Semantic Verification in Method Matcher**:
   - Refactor `src/auditor/schemas/evaluation.py:matches()` to require semantic text embedding similarity ($\ge 0.50$) or quote overlap between prediction summary and gold label rationale.
   - Restrict greedy matching so that duplicate findings cannot match distinct gold labels.
2. **Integrity Agent False Alarm Suppression**:
   - Add sentence continuation heuristics in `src/auditor/agents/integrity/` to detect sentences split across column breaks, discretionary hyphens, or display math.
   - Eliminate the 152 spurious `unfinished_or_placeholder_content` flags generated per run.
3. **Verifier Policy Recalibration**:
   - Adjust `src/auditor/agents/verifier/decide.py` so that high-confidence findings with two-sided textual grounding are not demoted to `NEEDS_HUMAN_REVIEW` solely because of a specialist uncertainty flag.

### Phase 4: Baseline Alignment & Benchmark Expansion (Days 17–22)
1. **Baseline Prompt Realignment**:
   - Update `src/auditor/baselines/common.py` to specify schema conventions for method and metric errors.
   - Re-evaluate Baselines B0 and B1 under fair schema conditions.
2. **Baseline B1 Context Pruning**:
   - Implement tool history sliding windows in `src/auditor/baselines/b1.py` to bound prompt tokens per call below 15,000 tokens, eliminating the $0.54/paper token runaway.
3. **Benchmark Quality Assurance**:
   - Conduct the Decision D18 mandatory 20% second review on Set C labels.
   - Curate 20 additional paper-code pairs mined independently of the code agent.

### Phase 5: Packaging & Production Deployment Readiness (Days 23–28)
1. **Packaging Compliance**:
   - Add `src/auditor/py.typed` to enable PEP 561 compliant downstream type checking.
   - Ensure `types-PyYAML` is added to base dependencies in `pyproject.toml`.
2. **UI Test Suite**:
   - Write comprehensive unit tests for `src/auditor/ui/app.py` using Streamlit's `AppTest` harness, raising UI coverage from 0.00% to $\ge 80\%$.
3. **Resource & Disk Management**:
   - Convert all file open operations in `src/auditor/evaluation/runs.py` to `with` context managers.
   - Add orphan process cleanup to `src/auditor/infra/sandbox.py:_docker` upon timeout.
   - Implement a 7-day TTL and 10 GB LRU eviction policy for `<scratch_dir>`.

---

## 6. Audit Sign-Off & Verification Commands

All empirical findings, test results, and metric values in this audit report can be independently reproduced by executing the following commands in the active Anaconda Python environment (`/Users/va/optiq/bin/python`):

```bash
# 1. Verify Test Suite Execution (1,302 passed in ~10s)
python -m pytest tests -q

# 2. Verify Static Analysis (Ruff lint and format check)
python -m ruff check .
python -m ruff format --check .

# 3. Verify Strict Type Checking across entire repo (236 files)
python -m pip install types-PyYAML
MYPYPATH=src python -m mypy src/auditor tests scripts

# 4. Verify Spend Ledger ($125.04 across 6,997 calls)
python -c '
import json
lines = open("results/p4/ledger-p4.jsonl").readlines()
total = sum(json.loads(l).get("usd", 0.0) for l in lines)
print(f"Ledger entries: {len(lines)}, Total spend: ${total:.2f}")
assert round(total, 2) == 125.04
'

# 5. Verify 0 Dropped Findings across all Set C runs
python -c '
import json
from pathlib import Path
from collections import Counter
states = Counter()
for r in [1, 2, 3]:
    for vf in Path(f"results/p4/runs/p4-setc-full-r{r}").glob("*/verified.jsonl"):
        for line in open(vf):
            states[json.loads(line).get("state")] += 1
print("Verifier states across runs:", dict(states))
assert states["dropped"] == 0
'

# 6. Verify Artificial 1.000 Precision Across All Systems
python -c '
import json
data = json.load(open("results/p4/set_c-rq1.json"))
for r in data["runs"]:
    assert r["precision"] == 1.0
    assert r["clean_flagged"] == 0
print("Confirmed: Precision is 1.000 for all runs due to unlabelled exclusion.")
'

# 7. Verify UI Zero Automated Test Coverage
python -m pytest tests/ --cov=src/auditor/ui/app.py -q
```

---
*Report compiled and certified by Teamwork Master Audit Subagent on branch `p4-evaluation-report`.*
