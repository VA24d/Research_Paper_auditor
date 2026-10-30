# P2 mid-term results (2026-10-30, end of the fix session)

All measured numbers for the P2 citation checker as of this date, on branch
`p2-citation-checker` at commit `2b659de` (see git log for the full fix-session series
`598b04d`..`2b659de`). Companion to [p2_audit_report.md](p2_audit_report.md) (the audit
and its fixes) and [p2_context.md](p2_context.md) (the working context). Everything here
is reproducible with the commands in §6. LLM = `google_genai` / `gemini-2.5-flash` via
Vertex AI with ADC (the team's shared free-trial project).

## 1. Test suites

| Suite | Command | Result |
|---|---|---|
| Offline (full, project-wide) | `python -m pytest tests -q` | **1229 passed, 1 skipped, 8 deselected** |
| Network | `pytest tests -m network` | 1 passed, 1 skipped (`embed` extra not installed), rest deselected |
| GROBID | `pytest tests -m grobid` | 3 passed (GROBID 0.9.1, docker) |
| LLM | `pytest tests -m llm` | no tests carry the marker yet |
| ruff | `ruff check` (project-wide) | **clean** |
| ruff format | `ruff format --check` | **clean** (228 files) |
| mypy | `mypy` | 2 errors, both `src/auditor/infra/llm.py:317` (`fcntl` — Part 1's file, pre-existing on Windows) |

The one offline skip is Part 3's `test_data_analysis.py` (needs `pyarrow`); the network
skip is the MiniLM embedding test (needs the `embed` extra, which is not synced).

## 2. Set A — HALLMARK (reference existence/metadata, 831 test-public entries)

Final run `runs/hallmark/results_fixed.jsonl` (all fixes active, HTTP-cached lookups):

| Metric | Audit session (stale file) | Fresh pre-fix baseline | **Now** |
|---|---|---|---|
| Accuracy | 0.726 | 0.750 | **0.851** |
| Flagging precision | 0.732 | 0.811 | **0.902** |
| Flagging recall | 0.891 | 0.786 | **0.857** |
| Flagging F1 | 0.804 | 0.798 | **0.879** |
| FPR on valid entries | 0.557 (170/305) | 0.313 (96/307) | **0.160 (49/307)** |
| Unverifiable (not scored) | 4 | 2 | 2 |

Per type (now: P / R / F1, with the audit-session recall for comparison):

| Hallucination type | P | R | F1 | R before |
|---|---|---|---|---|
| arxiv_version_mismatch | 1.000 | 0.800 | 0.889 | 0.933 |
| chimeric_title | 1.000 | 1.000 | 1.000 | 1.000 |
| fabricated_doi | 1.000 | 1.000 | 1.000 | 0.966 |
| future_date | 1.000 | 1.000 | 1.000 | 1.000 |
| hybrid_fabrication | 1.000 | 1.000 | 1.000 | 0.938 |
| merged_citation | 1.000 | 1.000 | 1.000 | 0.897 |
| near_miss_title | 1.000 | 0.700 | 0.824 | 0.725 |
| nonexistent_venue | 1.000 | 0.378 | 0.549 | 0.919 |
| partial_author_list | 1.000 | 0.935 | 0.967 | 0.581 |
| placeholder_authors | 1.000 | 1.000 | 1.000 | 0.788 |
| plausible_fabrication | 1.000 | 1.000 | 1.000 | 1.000 |
| preprint_as_published | 1.000 | 0.500 | 0.667 | 0.900 |
| swapped_authors | 1.000 | 0.968 | 0.984 | 0.905 |
| wrong_venue | 1.000 | 0.657 | 0.793 | 0.800 |

What the fixes did: new author checks (missing names, silent truncation, wrong first
author) took `partial_author_list` from R 0.581 to **0.935** and `placeholder_authors` /
`swapped_authors` to ~1.0; arXiv-DOI routing, folded queries and the prefix title rule
fixed the resolution failures; the year ±1 tolerance (Anurag's decision A) removed the
single biggest false-positive class (46 of 96 FPs); journal-abbreviation and
repository-copy venue matching removed the rest of the venue FPs.

Honest trade-offs, both deliberate: `nonexistent_venue` (R 0.919 → 0.378) and
`preprint_as_published` (R 0.900 → 0.500) lost recall because a repository/preprint
placeholder venue on the index side can no longer contradict the entry — those FNs are
indistinguishable from the FPs the tolerance removes at the venue level. Precision was
chosen because every flag costs human review. `arxiv_version_mismatch` (R 0.800): its
old "flags" were resolution failures counted as TPs; detecting it properly needs an arXiv
version comparison (future work).

## 3. Set B — claim–source support

### 3.1 Dataset state after the corrections

`datasets/set_b/`: **158 gold labels** (112 injection records + 46 clean controls) over
the 60 arXiv papers; `scripts/set_b.py validate datasets/set_b --sources
data/set_b/sources` exits 1 **only** for the D18 second review (Anurag's task).

| Class | Labels |
|---|---|
| reference_not_found | 96 |
| reference_metadata_mismatch | 9 |
| claim_number_mismatch | 4 |
| claim_not_supported | 3 |
| none (clean controls) | 46 |

Three corrections were applied this session (2026-10-30), documented in the set B README:

1. **Year injections +1 → +3** (38 records, `runs/_regen_years.py`): one year of drift is
   normal in real bibliographies, so the checker tolerates ±1 and the injected year must
   be beyond it. `propose()` now injects +2/+3 and the builder verifies the injected year
   is not within ±1 of the resolved work's year.
2. **Collision labels dropped** (32, `runs/_drop_collisions.py`): labels stacked on an
   entry that also carries a `reference_not_found` injection are unreachable — a
   fabricated title can never yield a metadata-mismatch report, and a claim citing that
   entry can never be resolved for support.
3. **Table-fragment labels dropped** (11 of 15 claim-number, `runs/_drop_table_junk.py`):
   the builder had extracted numbers from whole sentence texts, and GROBID merges table
   rows into sentences (e.g. 00692 s68: "10.0 20.2 PG19 2.50 6.75 … Fine-tuning methods
   …"), so the perturbed value was table data no claim states. `propose()` now skips
   sentences crowded with bare numbers (>3).

Net effect: 201 → 158 labels; the remaining claim classes are thin (4 + 3) and growing
them by hand is the documented follow-up.

### 3.2 Full-system mini-run (the citation agent as the product runs it)

`auditor batch --experiment full-mini --systems full --runs 1 --llm-cache` over the
injected copy and the kept original of 2501.00692 (2 done, 0 failed). Scored with
`--before-verifier` — **required until Part 4's verifier exists**, because the placeholder
verifier sends every finding to human review and `needs_human_review` predictions do not
count as positive (D3).

| Class (00692's 2 labels: both reference_not_found) | P | R | F1 |
|---|---|---|---|
| reference_not_found | 1.000 | 1.000 | 1.000 |
| **macro-F1** | | | **1.000** |
| FPR on the paper's clean controls | 0.000 (0/14) | | |

LLM cost of the run: **123 model calls, 98,134 input / 41,883 output tokens** (~60 calls
and ~50k tokens per paper pair). HTTP full-text retrieval used arXiv HTML where available
and fell back to the abstract elsewhere; the GROBID PDF path is wired in and is used when
the HTML is unavailable.

Prediction mix on the run (59 predictions): 32 on the injected copy, 27 on the original —
11 metadata mismatches each (real bibliography variance the checker now reports), 10/8
not-found (2 true positives on the injected entries; the rest are resolution failures on
real references, counted as unlabelled flags), 5/3 claim-not-supported and 2/2
claim-number findings (unlabelled — 00692's claim labels were removed as unreachable).

### 3.3 Recall@5 (retrieval)

`scripts/citation_eval.py recall` on the current injections: **Recall@5 = 1.000 (7 of
7)** — every label that carries a gold passage has its passage in the top-5 retrieved
chunks. 105 of 112 records carry no gold passage (bibliography injections and the early
claim injections never had one), so the number is real but its base is small.

### 3.4 B1 baseline (characterisation, audit session)

B1 (the one-pass tool-calling baseline) was characterised on 3 papers
(`runs/b1-smoke-b1-r2`): 25 model calls / 570,347 in / 14,032 out tokens, 30 tool calls
(`lookup_reference` 11, `find_supporting_passages` 5, code/template tools), no loops; it
caught real injected errors but emitted few findings (22 predictions on 8 of 26 papers in
the subset runs). The audit-session subset scores (full-text and abstract-only both
macro-F1 0.026) are **superseded**: they were measured against the pre-cleanup label set
and with the pre-fix agent; re-running B1 on the cleaned labels (~250 LLM calls) is a
documented follow-up if current B1 numbers are wanted. The abstract-only ablation showed
no difference at B1's granularity; the module-level ablation needs the full system.

## 4. Known issues and observations

- **Scoring needs `--before-verifier`** for the full system until Part 4's verifier
  exists (placeholder verifier → everything `needs_human_review`).
- **The scholarly full-text allowlist** was extended this session (doi.org redirector,
  ACL Anthology, PMLR, NeurIPS/CVF proceedings, JMLR, Springer/Wiley/MDPI, ouci mirror,
  HAL + subdomains; polite rate limits for doi.org and ACL). Institutional repositories
  (e.g. `mural.maynoothuniversity.ie`) stay blocked — unbounded set; those works fall
  back to the abstract. Flagged to Bhaskar (shared `config.py`/`http.py`).
- **Semantic Scholar's unauthenticated ~1 rps** makes failing lookups slow (bounded
  retries honor Retry-After); a hallmark run takes ~30-40 min on a cold cache, ~10 min
  warm.
- **P3's code tools** (reached through the full system and B1): an unauthenticated
  GitHub 403 (`api.github.com`, 60 req/h limit) kills a whole paper run; a malformed
  JSON from a repo tool kills it too; B1 hit LangGraph's recursion limit (90) on one
  paper — the bounded failure mode is a failed paper, not a hang. Reported to
  Bhaskar/Vijay.
- **B1 drops** model-reported errors whose `sentence_id`/`bib_id` is empty (logged
  warning). Part 1's `baselines/common.py`.
- The `embed` extra (`AUDITOR_CITATION_EMBEDDER=minilm`) is still not synced; the default
  stays BM25. A BM25-vs-MiniLM Recall@5 comparison is offline and cheap once synced.
- The D18 second review (20% of labels, `reviewed_by`) is Anurag's task and is the only
  `validate` failure.

## 5. The runs that produced these numbers

| Run | Directory | What it is |
|---|---|---|
| HALLMARK final | `runs/hallmark/results_fixed.jsonl` | 831 entries, all fixes, cached lookups |
| Full-system mini | `runs/full-mini-full-r1` | 00692 injected + original, full system, `--llm-cache` |
| B1 smoke | `runs/b1-smoke-b1-r2` | 3 papers through B1, `--llm-cache` (audit session) |

## 6. How to reproduce

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q                 # 1229 passed
.\.venv\Scripts\python.exe -m pytest tests -m network         # needs APIs
.\.venv\Scripts\python.exe -m pytest tests -m grobid          # needs docker compose up -d
.\.venv\Scripts\ruff.exe check ; .\.venv\Scripts\ruff.exe format --check
.\.venv\Scripts\python.exe -m mypy                            # 2 pre-existing Part-1 errors

# Set A
.\.venv\Scripts\python.exe scripts\hallmark.py run --file data\hallmark\test_public.jsonl --out runs\hallmark\results_fixed.jsonl
.\.venv\Scripts\python.exe scripts\hallmark.py score --results runs\hallmark\results_fixed.jsonl

# Set B mini-run (full system; scoring MUST use --before-verifier until Part 4)
.\.venv\Scripts\auditor.exe batch --experiment full-mini --systems full --runs 1 --llm-cache runs\full_mini_dataset.jsonl
.\.venv\Scripts\python.exe scripts\citation_eval.py score --labels runs\full_mini_labels.jsonl --run runs\full-mini-full-r1 --before-verifier

# Set B dataset integrity
.\.venv\Scripts\python.exe scripts\set_b.py validate datasets\set_b --sources data\set_b\sources   # 1 problem: D18 review

# Retrieval
.\.venv\Scripts\python.exe scripts\citation_eval.py recall --injections datasets\set_b\injections.jsonl --sources data\set_b\sources
```

## 7. What is next for P2

- [ ] D18 second review (Anurag) → freeze the dataset.
- [ ] Full-system subset run over several papers (~60 calls + ~50k tokens per paper pair)
      for the module's real set B numbers; then the full-dataset runs.
- [ ] Optional: B1 re-run on the cleaned labels for current baseline numbers.
- [ ] Grow the claim classes by hand (4 claim-number, 3 claim-not-supported is thin).
- [ ] arXiv version check; BM25-vs-MiniLM retrieval comparison (needs the `embed` extra);
      at least one `llm`-marked regression test.
