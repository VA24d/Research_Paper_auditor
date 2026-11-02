# P2 citation checker — audit report (2026-10-29, updated 2026-10-30 after the fix session)

All measured numbers are consolidated in [p2-mid-results.md](p2-mid-results.md) (the
results deliverable); this file records the audit, the findings and the fixes.

Audit of the P2 implementation (citation agent, retrieval package, set A/B machinery,
evaluation scripts) on branch `p2-citation-checker` at `d1019e0` + audit commits. Scope:
static audit of every P2 module, the full test suites (offline/network/llm/grobid), a live
end-to-end smoke test (GROBID + real LLM), and the P2.9 evaluation runs on a documented
subset of set B. This file is the audit deliverable; `docs/p2_context.md` holds the working
context.

## 1. Test results

| Suite | Command | Result |
|---|---|---|
| Offline (full) | `python -m pytest tests -q` | **1194 passed, 1 skipped, 8 deselected** (~2m37s) |
| Network | `pytest tests -m network` | **1 passed**, 1 skipped (onnxruntime not installed), 1199 deselected |
| LLM | `pytest tests -m llm` | **no tests carry the marker** (1201 deselected) |
| GROBID | `pytest tests -m grobid` | **3 passed** (GROBID 0.9.1 via docker compose) |
| ruff check | `ruff check src scripts` | **clean** |
| ruff format | `ruff format --check src scripts` | **clean** (139 files) |
| mypy | `mypy` | **2 errors, both in `src/auditor/infra/llm.py:317`** (Part 1's file, pre-existing; see §5) |

Skips explained: the skipped offline test is Part 3's `test_data_analysis.py` (needs
`pyarrow`, not installed). The skipped network test is `tests/retrieval/test_embeddings.py`
(needs the `embed` extra; see §5). The `llm` marker is defined in `pyproject.toml` but no
test in the tree uses it — there is no LLM-level regression test; the LLM paths are
exercised by scripts and the live runs instead.

## 2. Static audit findings

Severity: **high** = breaks a live run or a score; **medium** = wrong/costly behaviour in
real conditions; **low** = dead code or doc/code drift; **info** = observation.

| # | Where | Severity | Finding | Action |
|---|---|---|---|---|
| 1 | `src/auditor/agents/citation/support.py` (`_mismatch_finding`, `_number_check`) | **high** | Every `claim_number_mismatch` finding got `number=0`, so two number mismatches in one task, or a contradiction plus a number mismatch, produced duplicate `finding_id`s. `TaskResult` validation rejects duplicates, so `run_task` turned the whole task into `failed` — observed live in the smoke audit (`citation-7ed141ecf06c90b4`). This hits exactly the injected classes set B scores. | **FIXED** in this audit: the running findings count is threaded into `_number_check`/`_mismatch_finding` (commit `17ec48c`), with two regression tests in `tests/agents/citation/test_support.py`. |
| 2 | `src/auditor/agents/citation/__init__.py:66`, `tools.py:69,90`, `scripts/citation_eval.py:175` | **medium** | `retrieve_source` is always called with `grobid=None` — `RunContext` carries no GrobidClient — so `SourceLevel.FULL_PDF` is unreachable from the agent, B1's tools and the fetch script. A cited work whose only open text is a PDF falls back to the abstract in every P2 run even with GROBID running. | Needs a small design change (how the client reaches `check`/tools); ask before implementing. |
| 3 | `src/auditor/agents/citation/lookup.py:93-101`, `retrieval/resolve.py` (title-search step) | **medium** | The four title searches (OpenAlex, Crossref, arXiv, S2) are built as a tuple of awaited calls, so all four always run even when the first has hits; and any one of them raising aborts the whole lookup even if an earlier one had hits. ~4x the HTTP calls per title lookup, plus fragility. | Small, clearly-correct fix (sequential with early exit + per-search exception capture); ask before implementing since it changes behaviour observable in `queries`. |
| 4 | `src/auditor/agents/citation/support.py:279-280` vs code | **low** | Docstring says "a quote must be verbatim in the source", but the model's quote is never checked against `source.text`; a non-verbatim quote is attached as evidence with no offsets. Scoring is unaffected. | Either check verbatim (grounding improvement) or fix the docstring. |
| 5 | `src/auditor/agents/citation/eval.py:48-51` | **low** | Dead code: `shorter in longer` duplicates the containment test on the line above. | Trivial deletion. |
| 6 | `src/auditor/agents/citation/reference.py:79-88` | **low** | `check_reference` re-fetches the bibliography entry that `check` already resolved and would already have returned unverifiable for a missing one — the KeyError path is unreachable. | Re-checked 2026-11-02: NOT unreachable — `check_reference` is also called directly on bare tasks (7 tests in `test_reference.py` rely on the re-fetch). Kept as is; the re-fetch makes the check usable without a pre-resolved entry. |
| 7 | `src/auditor/agents/citation/set_b.py:215-218` | **low** | `_inject_sentence` slices `sentence.text[anchor.start:anchor.end]` before the span-None check (harmless — the slice with None is the whole text and the error is raised before use, but the order reads wrong). | Trivial reorder. |
| 8 | `src/auditor/agents/citation/support.py:334-336` | **info** | An invalid verdict string is mapped to `insufficient` but the quote computed before the fix is still used. Harmless in practice. | None. |

Decisions D1-D22: no violations found in the P2 code (D1 raw/field agreement holds;
D3/D4 classes as owned; D6/D7 entry points present (`check`, `make_tools`); D8 counted
tool calls via `tracked_tool`; D9 all model calls through `StructuredLLM`; D10 caching
optional per run; D15 HALLMARK machinery; D18 validate implements the checks and honestly
reports the missing second review). `citation_abstract_only` **is** wired into `check()`
(`__init__.py:70`) — the previously flagged open item is resolved.

## 3. Live smoke test (GROBID + real LLM)

Setup: docker GROBID 0.9.1 up, provider `google_genai` / `gemini-2.5-flash` via Vertex AI
with ADC (shared free-trial project), probes OK (structured output in all four methods +
tool calling).

### 3a. Full pipeline, one paper

`auditor audit data/set_b/papers/2501.00684.json --run-id audit-smoke-p2`:

- 56 tasks: **54 checked, 1 unverifiable, 1 failed**; verifier: **0 verified flags, 21
  needs_human_review, 0 dropped**.
- The 1 failed task was finding #1 above (duplicate finding ids) — after the fix the same
  task yields two distinct findings (covered by the regression tests).
- Citation-class findings carry real evidence: a LOOKUP record (queries + source level)
  and a QUOTE with exact char offsets (e.g. `char_start 10770`), e.g.:
  - `reference_not_found` — "The reference could not be resolved: no work with this title
    was found in the scholarly indexes" (tweets and not-yet-indexed items).
  - `claim_not_supported` — quote from the cited work's full HTML text with offsets.
- Output: `runs/audit-smoke-p2/2501.00684-original/{tasks,results,predictions,verified}.jsonl`.

### 3b. B1 tool-calling agent, 3 papers

`auditor batch --experiment b1-smoke --systems b1 --llm-cache` over 3 entries (2 injected +
1 original), default 3 runs. Clean run (r2):

| Metric | Value |
|---|---|
| Model calls | 25 (≈8 per paper) |
| Tokens | 570,347 in / 14,032 out |
| Tool calls | 30 — `lookup_reference` 11, `find_supporting_passages` 5, `find_repository_setting` 10, `layout_summary` 3, `list_repository_files` 1 |
| Loops/wrong tools | none observed |

B1 caught real injected errors on `2501.00684-injected`: `reference_metadata_mismatch` 2,
`reference_not_found` 1, `claim_not_supported` 1 (plus template/integrity noise).

Observations from the batch (owned by other parts, flagged here):
- r1 lost paper `2501.00691-injected` to `HttpStatusError: HTTP 403 api.github.com`
  (unauthenticated GitHub rate limit) from a code tool — one failed tool call kills the
  whole paper run. Owner: whoever owns the code tools + the HTTP gate (Bhaskar/Vijay).
- `auditor.baselines.common` **drops** a model-reported error when its `sentence_id`/`bib_id`
  is empty (logged warning, finding silently lost). Owner: Bhaskar (baselines).
- `auditor batch` runs **3 repetitions by default**; with `--llm-cache` the repeats are
  nearly free, uncached they triple the cost.

## 4. P2.9 evaluation runs (the owed numbers)

Scale decision (with Anurag's confirmation): measured cost was ~8 model calls / ~23k input
tokens per paper, so the full 119-entry dataset would have been ~2000 calls / ~46M input
tokens for both runs — too much for the shared free-trial credits. A **documented
stratified subset of 30 entries (15 papers: injected copy + original of each)** was run
once per condition, uncached (D10). Subset selection (`runs/_inspect_smoke.py`, throwaway):
papers ranked by claim-class coverage first, so the subset contains

| Labels inside the subset | Count |
|---|---|
| claim_number_mismatch | **15 (all of set B's)** |
| claim_not_supported | **5 (all of set B's)** |
| reference_not_found | 26 |
| reference_metadata_mismatch | 18 |
| clean controls (`none`) | 14 |

Commands (run directory `runs/setb-subset-b1-r1`, abstract-only `runs/setb-subset-abs-b1-r1`):

```text
.venv\Scripts\auditor.exe batch --experiment setb-subset --systems b1 --runs 1 runs\setb_subset_dataset.jsonl
.venv\Scripts\auditor.exe batch --experiment setb-subset-abs --systems b1 --runs 1 runs\setb_subset_dataset.jsonl   # with AUDITOR_CITATION_ABSTRACT_ONLY=1
.venv\Scripts\python.exe scripts\citation_eval.py score --labels datasets\set_b\labels.jsonl --run runs\setb-subset-b1-r1
.venv\Scripts\python.exe scripts\citation_eval.py score --labels datasets\set_b\labels.jsonl --run runs\setb-subset-abs-b1-r1 --abstract-only
.venv\Scripts\python.exe scripts\citation_eval.py recall --injections datasets\set_b\injections.jsonl --sources data\set_b\sources
```

### Results

**Run integrity.** The full-text run finished with **26 of 30 entries completed** after
resumes; the 4 failures are deterministic and none of them are P2's: `2501.00873-injected`
and `2501.00884-original` (unauthenticated GitHub API 403 from a code tool), both variants
of `2501.01473` (`JSONDecodeError` inside a code tool on the `rank_bm25` repo), and
`2501.00692-injected` (`GraphRecursionError`: B1's agent looped to LangGraph's limit of 90).
One OAuth SSL hiccup and a few scholarly-API read errors were retried successfully by
re-running the (resumable) batch. **All conclusions below are on the 26 completed entries.**

**Full-text run (`runs/setb-subset-b1-r1`, B1, uncached, 26 entries, 78 subset labels):**

| Class | P | R | F1 | tp / fp / fn |
|---|---|---|---|---|
| claim_not_supported | 0.000 | 0.000 | 0.000 | 0/0/5 |
| claim_number_mismatch | 0.000 | 0.000 | 0.000 | 0/0/15 |
| reference_metadata_mismatch | 1.000 | 0.056 | 0.105 | 1/0/17 |
| reference_not_found | 0.000 | 0.000 | 0.000 | 0/0/26 |
| **macro-F1** | | | **0.026** | |
| FPR on clean controls | 0.000 (0/14) | | | |

Scoring note: `scripts/citation_eval.py score` scores a run against **all** labels passed
in; for a subset run the labels must be restricted to the subset's papers
(`runs/setb_subset_labels.jsonl`, built by `runs/_inspect_smoke.py`). Scoring with the full
201-label file gives the same shape with wrong denominators (macro-F1 0.013).

**What the run actually shows.** B1 is a *whole-paper, one-pass* agent: on the 26 completed
entries it emitted only **22 predictions on 8 papers** (10 of them citation-class: 6
reference_not_found, 3 reference_metadata_mismatch, 1 claim_number_mismatch). Verified
against the labels, the low numbers are **real B1 behaviour, not a scoring bug**: the one
`reference_metadata_mismatch` hit is correct (2501.00799 b0), while e.g. its
`reference_not_found` on 2501.00799 targets `b28` where the labels sit on `b0`/`b1` — it
flags *real* open-access-poor references instead of the injected ones, and those count as
unlabelled flags by design (D3). The citation **module** (the per-task agent the full
system runs) demonstrably works — the smoke audit in §3a produced correct, evidence-backed
citation findings on the same papers — but no full-system set B run has happened yet; that
is roughly an order of magnitude more expensive (one model call per citation task ≈ 40
calls/paper) and needs its own scale decision.

**Abstract-only ablation (`runs/setb-subset-abs-b1-r1`, `AUDITOR_CITATION_ABSTRACT_ONLY=1`,
uncached, 25 entries):**

| Class | P | R | F1 |
|---|---|---|---|
| claim_not_supported | 0.000 | 0.000 | 0.000 |
| claim_number_mismatch | 0.000 | 0.000 | 0.000 |
| reference_metadata_mismatch | 1.000 | 0.056 | 0.105 |
| reference_not_found | 0.000 | 0.000 | 0.000 |
| **macro-F1** | | | **0.026** |
| FPR on clean controls | 0.000 (0/14) | | |

Same 4 deterministic failures (GitHub 403 ×2, repo JSON ×1, GraphRecursion ×1) plus one
extra transient (`2501.00891-injected`, scholarly-API ReadError — retryable).

**Full text vs abstract only (B1, this subset): no measurable difference** — both
conditions land on macro-F1 0.026 with the single metadata hit. The prediction *mix* did
change (abstract-only: 13 citation-class predictions, 8 reference_not_found vs 10/6
full-text — "not found" is easier to claim against an abstract), but neither condition
recovers injected items. At B1's granularity the ablation cannot show a retrieval effect;
that comparison needs the full system, where every claim gets its own support judgement
over the ranked passages.

Recall@5 (offline, all 154 injection records): **1.000 — but on the 20 records that carry
a gold passage** (134 have none). Set A re-score: reproduced exactly (§5).

## 5. Observations, shortcomings, honest limits

- **Set A (HALLMARK) reproduces exactly** on re-score: accuracy 0.726, flagging P 0.732 /
  R 0.891 / F1 0.804, FPR on valid 0.557 (170/305), 4 unverifiable. Weak types are the
  known ones: `partial_author_list` (R 0.581 — the resolver only sees the entry as written,
  and the index record often carries the full author list) and `near_miss_title` (R 0.725).
- **Recall@5 is measured on only 20 records** (1.000): 134 of 154 injection records carry
  no gold passage (bibliography injections and early claim injections have none by
  construction). The number is real but its base is tiny; growing gold passages is the
  claim-class growth task in the set B README.
- **PDF-level full text is unreachable** from the agent (finding #2): every "full text"
  in the runs above is arXiv HTML (`full_html`); a cited work that only exists as a PDF is
  judged on its abstract.
- **The support judge quotes are not verbatim-checked** (finding #4): a grounding rate of
  quotes is not yet measured anywhere and is still owed in the evaluation summary.
- **The embedding extra was never synced** (`uv sync --extra embed`,
  `AUDITOR_CITATION_EMBEDDER=minilm`): the ONNX MiniLM ranker exists and is only exercised
  by its skipped test. Default stays BM25 (works everywhere). The BM25 vs MiniLM Recall@5
  comparison is cheap to run offline once synced (no LLM) and is worth one session.
- **No llm-marked tests exist** although the marker is defined; the obvious first one is a
  single-task `check_citation` against the live model.
- **mypy fails on `src/auditor/infra/llm.py:317`** (`fcntl.flock`, `LOCK_EX`): pre-existing
  Part-1 code reading `fcntl` unguarded — Windows-locale issue. **Not fixed by me** (Part 1
  owns the file); same family as the `layout_tables` Windows-locale fix I already owe
  Bhaskar (commit `4457c46`, mine but belongs to Part 1 — do not let it merge with P2).
- **D18 second review**: `reviewed_by` is still empty on all 201 labels; `scripts/set_b.py
  validate` exits 1 on exactly that check. That is Anurag's own task and was **not** faked
  in this audit.
- The claim classes remain thin in set B as a whole (20 of 154); the subset therefore
  over-represents them (by design, for the module comparison).

## 6. How to run everything manually

```powershell
# environment (from the repo root; uv is not on PATH)
.\.venv\Scripts\python.exe -m pytest tests -q            # offline suite
.\.venv\Scripts\python.exe -m pytest tests -m network    # real APIs
.\.venv\Scripts\python.exe -m pytest tests -m grobid     # needs docker compose up -d
.\.venv\Scripts\python.exe -m ruff check src scripts
.\.venv\Scripts\python.exe -m ruff format src scripts
.\.venv\Scripts\python.exe -m mypy

# GROBID
docker compose up -d ; docker compose ps

# LLM probe (1 call)
.\.venv\Scripts\python.exe scripts\probe_llm.py

# full pipeline on one paper
.\.venv\Scripts\auditor.exe audit data\set_b\papers\2501.00684.json --run-id <id>

# B1 batch (paths in the dataset file are relative to the file itself!)
.\.venv\Scripts\auditor.exe batch --experiment <name> --systems b1 [--llm-cache] [--runs 1] <dataset.jsonl>

# set A
.\.venv\Scripts\python.exe scripts\hallmark.py download --split test_public --out data\hallmark
.\.venv\Scripts\python.exe scripts\hallmark.py run --file data\hallmark\test_public.jsonl --out runs\hallmark\results.jsonl
.\.venv\Scripts\python.exe scripts\hallmark.py score --results runs\hallmark\results.jsonl

# set B scoring / recall (offline)
.\.venv\Scripts\python.exe scripts\citation_eval.py score --labels datasets\set_b\labels.jsonl --run runs\<run_id> [--abstract-only]
.\.venv\Scripts\python.exe scripts\citation_eval.py recall --injections datasets\set_b\injections.jsonl --sources data\set_b\sources
```

## 7. What is left in P2 (checklist)

- [ ] D18 second review (20% of the 201 labels, `reviewed_by`) — Anurag's own task; then
      `scripts/set_b.py validate` exits 0 and the set freezes.
- [ ] Full-dataset set B runs (the subset numbers above are a documented stand-in).
- [ ] Grow the claim classes (20 of 154) by hand injections
      (`scripts/set_b.py inject --kind claim ...`).
- [ ] Wire GROBID into the agent/B1 full-text path (finding #2) — needs a design choice.
- [ ] Sequential title-search fallback in `lookup.py`/`resolve.py` (finding #3).
- [ ] Verbatim-quote check or docstring fix (finding #4) + a measured grounding rate.
- [x] Sync the `embed` extra, run the MiniLM vs BM25 Recall@5 comparison offline — DONE
      2026-11-02 (extra synced, comparison run: BM25 1.000 vs MiniLM 0.667 Recall@5 on
      the 6 gold-passage records; BM25 ships — see p2-mid-results §3.3).
- [x] At least one `llm`-marked regression test — DONE 2026-11-02
      (`tests/agents/citation/test_support_live.py`, one live model call, passes).
- [ ] `scripts/citation_eval.py tool-accuracy` on labelled lookups (machinery exists, no
      labelled fixture yet).
- [x] Trivial cleanups: eval.py dead code (already gone) and set_b.py slice order
      (already reordered in the fix session); reference.py re-fetch KEPT — finding 6's
      "unreachable" was wrong, direct callers rely on it.
- [x] Builder verification hole found by the D18 second review (a number proposal passed
      although the recorded source states neither value) — FIXED 2026-11-02 with tests.

## 8. Needs from others / asks

- **Bhaskar**: `parser/layout_tables` Windows-locale fix (my commit `4457c46`) belongs to
  Part 1 — take it from this branch, do not let it merge with P2; `infra/llm.py:317`
  `fcntl` mypy errors on Windows; baselines drop findings with empty ids (§3b); batch
  default of 3 runs is an easy way to triple LLM cost.
- **Vijay / HTTP gate owner**: unauthenticated `api.github.com` 403 kills a whole paper
  run (§3b).
- **Anurag**: the D18 second review; the scale decision for the full-dataset set B runs;
  whether to implement findings #2/#3 now or after the datasets freeze.

## 9. Fix session (2026-10-30) — what was fixed and what it changed

Follow-up session implementing the audit's findings and repairing the results. All
changes are on `p2-citation-checker` (commits `598b04d`..`180609f`), full suite **1216
passed / 1 skipped / 8 deselected**, ruff clean, mypy unchanged (2 pre-existing Part-1
errors in `infra/llm.py`).

### 9.1 HALLMARK (set A): 0.726 → 0.851 accuracy

| Metric | Before (stale file) | Fresh pre-fix baseline | **After the fixes** |
|---|---|---|---|
| Accuracy | 0.726 | 0.750 | **0.851** |
| Flagging P / R / F1 | 0.732 / 0.891 / 0.804 | 0.811 / 0.786 / 0.798 | **0.902 / 0.857 / 0.879** |
| FPR on valid entries | 0.557 (170/305) | 0.313 (96/307) | **0.160 (49/307)** |

(The "before" file predated merged PRs; the fresh baseline re-measured the old resolver.)

Fixes, each with tests: sequential title searches with early exit and per-index exception
capture; arXiv-DOI routing (`10.48550/arXiv.<id>` lives in DataCite — no longer reported
as a fabricated DOI); ASCII-folded search queries with a shortened retry (mojibake titles
like "DemoirΘing"); title-prefix matching (index keeps only "NeRF"); **author checks**
(missing names, silently truncated lists, wrong first author — `partial_author_list`
R 0.258 → **0.935**, `placeholder_authors` → **1.000**, `swapped_authors` → **0.968**);
near-miss title rule; journal-abbreviation and repository-copy venue matching; ±1 year
tolerance (Anurag's decision A); GROBID wired into the agent path; hallmark results now
carry the finding summary as the reason.

Honest trade-offs: `nonexistent_venue` R 0.919 → 0.378 and `preprint_as_published`
R 0.667 → 0.500 — the repository/placeholder venue tolerance removes FPs whose venue
signal is genuinely indistinguishable from those FNs. `arxiv_version_mismatch` R 0.800:
its old "flags" were resolution failures counted as TPs; a real version check is future
work.

### 9.2 Set B: the horrible numbers were three stacked problems

1. **The evaluation used B1** (the one-pass baseline), which emits few citation findings —
   documented in §4. The full system (the per-task citation agent) is the module's real
   test.
2. **The dataset had unreachable labels** (all fixed, `validate` exits 1 only for the
   pending D18 review): 38 year injections at +1 (undetectable under the correct ±1
   tolerance — regenerated at +3); 32 labels stacked on entries that also carry a
   `reference_not_found` injection (a metadata mismatch can never be reported and a claim
   citing a fabricated entry can never be judged — dropped); 11 of 15 claim-number
   injections perturbed numbers inside GROBID-merged table fragments, not in claims
   (dropped; the builder now skips sentences crowded with bare numbers). Final: **158
   labels** (96 RNF, 9 metadata, 4 claim-number, 3 claim-not-supported, 46 controls).
3. **The agent over-flagged real bibliographies** (author parse junk like `*`/`xlstm`,
   100-author collaborations, number words vs suffixes, a trigger-happy "contradicted"
   verdict) — fixed: junk-surname filtering, a ≤10-author cap on the truncation rule,
   number-word/suffix handling in `numbers_agree`, and a support prompt that requires an
   incompatible statement for "contradicted".

### 9.3 Full-system mini-run (one paper pair, `--llm-cache`)

`auditor batch --experiment full-mini --systems full --runs 1` on 2501.00692 (injected +
original), scored with `--before-verifier` (the placeholder verifier sends everything to
human review, so nothing counts positive without it — **any full-system scoring before
Part 4's verifier lands must use `--before-verifier`**):

| Class (00692 labels) | P | R | F1 |
|---|---|---|---|
| reference_not_found | 1.000 | 1.000 | 1.000 |
| **macro-F1** | | | **1.000** |
| FPR on clean controls | 0.000 (0/14) | | |

The earlier zero scores were the stale subset-labels file (colliding labels matched
first, since D3 matching does not compare classes) plus the broken labels — both fixed.
Remaining known limits: the mini-run covers one paper pair; the claim classes have too
few usable labels to measure (4 claim-number, 3 claim-not-supported dataset-wide); the
remaining reference-check false positives on the original paper (resolution failures on
~8 of ~40 real references) are unlabelled flags, not clean-control hits — they feed the
D14 false-alarm estimate.

### 9.4 What is still owed in P2

- [ ] D18 second review (Anurag) — `validate` exits 1 on exactly that.
- [ ] Full-dataset set B runs, and a full-system subset run over several papers
      (~60-100 calls/paper) for the module's real numbers.
- [ ] B1 subset re-run on the cleaned labels if current B1 numbers are wanted (~250 calls).
- [ ] arXiv version check for `arxiv_version_mismatch`.
- [ ] GROBID-in-path is fixed; the sequential-search and quote-anchoring findings are
      fixed; remaining: none of the audit's medium findings — all resolved this session.
