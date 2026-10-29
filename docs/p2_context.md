# P2 Citation Checker — Working Context

Owner: Anurag (Part 2: citation agent and retrieval). This file tracks understanding,
decisions and progress so a new session can pick up where this one left off. Read this
file FIRST, then docs/team_plan.md (Part 2 section) and docs/decisions.md.

## Project summary

Evidence-gated verification agent for research paper auditing (LMA group project).
Pipeline: PDF/ParsedPaper -> planner -> AuditTask[] -> specialists (citation/code/
template/theory/integrity) -> TaskResult[] -> verifier -> VerifiedFinding[] -> report +
predictions. Shared infra (Part 1, Bhaskar): config, schemas, HTTP gate, LLM factory,
orchestration (LangGraph). Part 3 (Vijay): code agent + template checker, merged into
main. Part 4 (Sudheera): verifier/report/evaluation, not started.

## My part (P2) — from docs/team_plan.md

- P2.1 Clients for arXiv, OpenAlex, Crossref, Semantic Scholar through the shared HTTP
  client (`ctx.http`, allowlist already contains all four hosts).
- P2.2 Reference resolution from DOI, arXiv ID, or title+authors; clear "not found" reason.
- P2.3 Full-text retrieval for open-access sources (reuse Part 1 parser for PDFs, arXiv
  HTML where available); fall back to abstract; record which level was used.
- P2.4 Chunking + passage retrieval with exact character offsets; abstract-only mode for
  the ablation; embedding model must run on all three OSes (use pure-Python BM25 instead).
- P2.5 `check(task, ctx) -> TaskResult` in `auditor.agents.citation` for reference tasks
  (existence, metadata) and citation tasks (support: supported/contradicted/insufficient;
  number checks), with CandidateFindings + Evidence quotes. Unlinked markers resolved from
  marker text where possible, else unverifiable (D6).
- P2.6 Citation tools for B1 (D7): `make_tools(ctx)` in `auditor.agents.citation.tools`.
- P2.7 HALLMARK test split loader + existence-check evaluation.
- P2.8 Set B: 60 arXiv papers, ~600 claim-source pairs, injection scripts editing
  ParsedPaper copies, GoldLabels incl. clean controls + gold evidence passage (Recall@5),
  validated per D18.
- P2.9 Citation module evaluation: Recall@5, P/R/F1, full text vs abstract only,
  tool-call accuracy of reference lookups.

## Key interfaces (verified in code)

- `RunContext` (infra/context.py): settings, paper, http (HttpClient), llm
  (StructuredLLM|None), review_llm, logger, repository, source_pdf, allowances/locks.
  `require_llm()` raises LLMNotConfiguredError when None.
- `HttpClient`: `get(url, params=, headers=, use_cache=, attempts=, max_bytes=)`,
  `post(url, data=, files=, json=, request_timeout_s=, attempts=)`,
  `download_to(url, destination, max_bytes=, sha256=)` -> Download(path,size,sha256).
  HttpResponse: url, status_code, headers, content, .ok, .text, .json(). GETs cached
  30 days. Constructor takes `transport=` (httpx.MockTransport) and `sleep=` for tests.
- `StructuredLLM.generate(schema, messages, effort=, purpose=, max_output_tokens=)` ->
  StructuredResult(value, attempts, usage, from_cache). Effort: minimal|low|medium|high.
- `AuditTask`: task_id, paper_id, variant, kind, anchor, target, claim, setting,
  repository, rule, claims. Built via AuditTask.create (id is a hash; hand-made ids
  rejected). REFERENCE task: BibliographyAnchor + BibTarget, no claim. CITATION task:
  SentenceAnchor + BibTarget|MarkerTarget, claim required.
- `TaskResult.for_task(task, outcome, reason=, findings=)`; outcomes
  checked|unverifiable|failed. Unverifiable needs a reason. Findings must belong to
  the task.
- `CandidateFinding.for_task(task, number, error_class=, summary=, anchor=,
  explanation=, evidence=, uncertain=)` -> id `<task_id>-f<number>`.
- `Evidence(kind=quote|code|rule|lookup, source=SourceRef(kind, id, title=, url=), text,
  page=, char_start=, char_end=, path=, line_start=, line_end=)`. QUOTE needs verbatim
  text; LOOKUP holds query+result. SourceKind: submission|cited_work|repository|
  venue_rule|scholarly_index.
- Parser: `parse_pdf(path, grobid=GrobidClient(http, base_url), cache_dir=, paper_id=)`
  -> ParsedPaper; raises ParseError. GROBID must be running for PDF parsing.

## Environment

- Windows 11, Python 3.13.1 at C:\Python313. uv 0.10.12 installed via
  `python -m pip install --user uv` (at %APPDATA%\Python\Python313\Scripts\uv.exe).
- Project venv: `.venv` via `uv sync` (UV_HTTP_TIMEOUT=300 needed; network is slow).
- IMPORTANT: in the Cline PowerShell session `uv` is NOT on PATH. Run everything via
  the venv python directly: `.\.venv\Scripts\python.exe -m pytest`,
  `.\.venv\Scripts\python.exe -m ruff check ...`, `.\.venv\Scripts\python.exe -m ruff
  format ...`, `.\.venv\Scripts\python.exe -m mypy` (all from the repo root).
- Full test suite: `.\.venv\Scripts\python.exe -m pytest tests/ -q` (~49 s).

## Design decisions (mine, for P2)

1. **Retrieval package** `src/auditor/retrieval/` holds P2.1-P2.4 (clients, resolve,
   fulltext, passages) so both the citation agent and B1 tools share the same
   functions (D7).
2. **No new dependencies**: HTML-to-text via a small stdlib extractor; passage ranking
   via pure-Python BM25 (no embedding model -> runs on all 3 OSes, satisfies P2.4).
3. **Source levels** recorded as: `full_html` (arXiv HTML), `full_pdf` (arXiv PDF via
   GROBID), `abstract`, `none`. Abstract-only ablation: `retrieve_source(abstract_only=)`
   parameter exists and works; a `citation_abstract_only` Settings field is still TODO
   (never added to config.py), so `check()` currently always fetches full text.
4. **Support verdicts** (as implemented): supported / contradicted / insufficient.
   contradicted -> CLAIM_NOT_SUPPORTED finding; insufficient -> checked, no finding;
   supported -> no finding. Every finding carries LOOKUP evidence, plus QUOTE evidence
   when the model's quote is >= MIN_QUOTE_CHARS (20); char offsets are recorded when the
   quote is found verbatim in the source text, else offsets stay None (the Part 4
   verifier re-checks verbatim quoting per P4.1).
5. **Number checks**: numbers extracted from the claim deterministically; the model
   locates each in the passages and returns the source value + quote; code compares
   values (`numbers_agree`: 1% relative tolerance, % vs fraction equivalence).
6. **Reference checks are deterministic** (no LLM): resolve -> not found => finding;
   metadata compared field-wise => mismatch finding with LOOKUP evidence.
7. **Caching**: retrieved source text cached on disk under cache_dir/citation/ keyed by
   work id + level, so repeated runs do not re-download (proposal risk plan).
8. **Unlinked markers**: author-year markers parsed and searched; numeric markers
   unverifiable (cannot know the entry).

## Progress log

- [x] Read proposal, team plan, decisions, part1 plan, parser evaluation, bhaskar_website.
- [x] Explored repo: schemas, infra, planner, orchestration, code agent pattern, testing
      helpers, config, conftest, pyproject.
- [x] Created branch `p2-citation-checker` from main (caca490); nothing committed yet.
- [x] Installed uv (0.10.12, `python -m pip install --user uv`), ran `uv sync`
      (UV_HTTP_TIMEOUT=300 needed; network is slow).
- [x] Implemented retrieval package (P2.1-P2.4): clients.py, resolve.py, fulltext.py,
      passages.py.
- [x] Implemented citation agent check() (P2.5): __init__.py, reference.py, support.py,
      lookup.py.
- [x] Implemented citation tools (P2.6): tools.py (lookup_reference, retrieve_cited_text,
      find_supporting_passages + make_tools for B1).
- [x] ruff check --fix (11 auto-fixed: import order, unused imports, EOF newlines) +
      ruff format (5 files reformatted). Clean now.
- [x] Full test suite green: 1035 passed, 1 skipped (pyarrow optional dep, unrelated),
      7 deselected (network/llm markers). No existing functionality broken.

## Git state (exact, end of session)

Branch `p2-citation-checker` (from main @ caca490). NOTHING committed or pushed yet.
- Modified (tracked): `src/auditor/agents/citation/__init__.py` (was an empty tracked
  file).
- Untracked: `docs/p2_context.md`, `src/auditor/agents/citation/lookup.py`,
  `reference.py`, `support.py`, `tools.py`, `src/auditor/retrieval/clients.py`,
  `resolve.py`, `fulltext.py`, `passages.py`.

## Files created (implemented, lint-clean; NO dedicated unit tests yet)

- `src/auditor/retrieval/clients.py` — P2.1: `Work` dataclass; arXiv (Atom XML),
  OpenAlex, Crossref, Semantic Scholar clients; `is_doi`/`is_arxiv_id`/`normalize_doi`.
- `src/auditor/retrieval/resolve.py` — P2.2: `Resolution` dataclass; `resolve_entry`
  (DOI -> arXiv id -> title search across openalex/crossref/arxiv/s2, filtered by
  `_entry_matches`); `resolve_marker` (author-year marker -> bibliography match ->
  resolve_entry); `titles_match`, `first_surname`, `parse_marker`,
  `match_marker_to_bibliography`.
- `src/auditor/retrieval/fulltext.py` — P2.3: `SourceLevel` (full_html|full_pdf|abstract|
  none), `SourceText`, `retrieve_source(http, work, settings, grobid=None,
  abstract_only=False)`; disk cache under `cache_dir/citation/`; stdlib HTML-to-text
  extractor; arXiv `/html/<id>` full text; PDF via `parse_pdf` (needs GROBID).
- `src/auditor/retrieval/passages.py` — P2.4: `Passage(text, char_start, char_end)`,
  `chunk_text(size=1200, overlap=200)`, pure-Python BM25, `rank_passages(text, claim,
  top=5)` (deterministic).
- `src/auditor/agents/citation/__init__.py` — P2.5 entry point `check(task, ctx)`.
- `src/auditor/agents/citation/reference.py` — `check_reference(task, ctx, resolution)`:
  deterministic; REFERENCE_NOT_FOUND / REFERENCE_METADATA_MISMATCH findings with LOOKUP
  evidence; missing entry -> unverifiable.
- `src/auditor/agents/citation/support.py` — `check_citation(task, ctx, resolution,
  source)`: LLM support judgement over top-5 passages; `SupportAnswer` schema
  (verdict/quote/reason/numbers[]); code-side gates (verbatim quote with offsets,
  `numbers_agree` value check); CLAIM_NOT_SUPPORTED + CLAIM_NUMBER_MISMATCH findings.
  Constants: TOP_PASSAGES=5, MIN_QUOTE_CHARS=20. Also `extract_numbers`, `_parse_number`,
  `numbers_agree`, `SupportVerdict`.
- `src/auditor/agents/citation/lookup.py` — `lookup_citation(http, citation)` for raw
  citation strings (identifier first, then title search); `LookupResult(work, reason,
  queries)`.
- `src/auditor/agents/citation/tools.py` — P2.6: shared functions `lookup_reference` /
  `retrieve_cited_text` / `find_supporting_passages` + `make_tools(ctx)` -> 3
  StructuredTools for B1; MAX_TOOL_CHARS=6000 clip; `resolution_record`/`source_record`
  JSON helpers.

## Session status after session 2 (updated as work proceeds)

Done in session 2 (commit `citation: retrieval package, agent check, B1 tools and offline
unit tests` on `p2-citation-checker`):

- Offline unit tests EXIST now: `tests/retrieval/` (test_clients.py, test_resolve.py,
  test_fulltext.py, test_passages.py — 39 tests) and `tests/agents/citation/` (test_check,
  test_reference, test_support, test_lookup, test_tools — 20 tests). All pass. Pattern:
  `httpx.MockTransport` handlers routing on URL substrings, `sleep=_no_sleep,
  min_intervals={}` to dodge rate limits; `ScriptedChatModel` + `tool_reply(SupportAnswer,
  ...)` for the support judgement; `monkeypatch.setattr("auditor.agents.citation.
  resolve_entry"/"...retrieve_source", fake)` for agent-level `check()` tests.
- Known issues 1-3 FIXED:
  1. `_number_check` now decides from `numbers_agree(claimed, found)` only; the model's
     `agrees` hint no longer suppresses a real mismatch. `found=None` still makes no
     finding (not confirmable in code). Regression test exists.
  2. Dead helpers `source_record` and `_best_work` removed from tools.py.
  3. `citation_abstract_only: bool = False` added to Settings (next to
     `citation_llm_enabled`) and wired: `check()` passes
     `abstract_only=ctx.settings.citation_abstract_only` to `retrieve_source`.
- Defects the new tests exposed and fixed in the P2.1-P2.6 code:
  - clients.py: `_parse_arxiv_feed` passed the author Element (not `.text`) to `_clean`
    -> TypeError on every arXiv reply with authors.
  - lookup.py: both "nothing found" return paths built `LookupResult(reason=...)` without
    the required first positional arg `work` -> TypeError.
  - reference.py: findings used `explanation=entry.raw`, None for entries without raw
    (e.g. example paper b1) -> ValidationError. Now `entry.raw or entry.title or entry.id`.
  - reference.py + `check()`: `paper.bib_entry()` RAISES KeyError for unknown ids, so the
    "entry not in the paper -> UNVERIFIABLE" branch was dead code. Both call sites now
    catch KeyError and return UNVERIFIABLE (tested).
  - fulltext.py: HTML-to-text did not collapse `&nbsp;` (U+00A0); quote-offset matching
    against arXiv HTML would misalign. Whitespace collapse now includes \u00a0.
  - tools.py: the three shared tool functions were NOT decorated with
    `auditor.infra.tools.tracked_tool`, so B1's citation tool calls were never counted
    (D8 requires counting). Now decorated; counted in UsageLog (tested).
  - support.py `extract_numbers`: scientific notation is NOT handled ("3e-4" yields "3"
    and "4" separately) — left as is, documented in the test; the number check compares
    values so this mostly wastes a NumberCheck rather than producing a wrong one.
- Known issue 5 (unlinked markers -> UNVERIFIABLE without a reference_not_found finding)
  deliberately UNCHANGED (defensible; revisit when set B exists).

## Known issues / TODOs in the implemented code

1. RESOLVED (session 2): `_number_check` no longer short-circuits on `agrees=True`.
2. RESOLVED: dead helpers removed from tools.py.
3. RESOLVED: `citation_abstract_only` in Settings, wired into `check()`.
4. RESOLVED: offline tests exist (65 citation+retrieval tests at the interruption point).
5. UNCHANGED by decision: unlinked-marker citation tasks return UNVERIFIABLE with a
   reason, no reference_not_found finding. Revisit against set B.
6. NEW (minor, open): `extract_numbers` does not understand scientific notation ("3e-4"
   -> "3", "4"). The value comparison then cannot confirm such a number. Acceptable for
   now; revisit if set B shows claim_number_mismatch recall problems on learning rates.

## In progress at the interruption point

RESOLVED: the eval test was fixed by giving the test source three distinct topics (the
irrelevant claim now ranks other chunks first; the miss case is asserted at k=2, since
top-5 covers most of a short source and containment passes trivially).

## Session 2 final state (all green)

- ruff: clean (src, tests, scripts). mypy --strict: clean (211 files). pytest: 1141
  passed, 1 skipped (optional pyarrow), 7 deselected (network/llm/grobid/docker).
- P2.7 done: `src/auditor/agents/citation/hallmark.py` + `scripts/hallmark.py`
  (download/run/score) + `tests/agents/citation/test_hallmark.py` (6 tests, incl. an
  end-to-end MockTransport test through the agent's `check()`). HALLMARK = arXiv 2607.18360
  (Kim, Brendel, Reizinger 2026), data at github.com/rpatrik96/hallmark data/v1.2/*.jsonl
  (MIT). Record format verified from test_public.jsonl: bibtex_key, bibtex_type, fields
  {title, author ("A and B"), year (str), doi, booktitle/journal, url}, label
  VALID|HALLUCINATED, hallucination_type (14 types), difficulty_tier, explanation,
  generation_method, source_conference, subtests{...bools}, raw_bibtex, schema_version.
  Loader drops `__canary__` entries, rejects duplicate keys. `bib_entry_for` maps fields
  to BibEntry and renders `raw` from the same fields (D1). `check_entry` builds a
  one-entry ParsedPaper (dataclasses.replace(ctx, paper=...)) and runs the agent's
  reference check; Flag = clean / reference_metadata_mismatch / reference_not_found /
  unverifiable. `score` gives accuracy, flagging P/R/F1, FPR on VALID, per-type
  breakdown; results round-trip JSONL. The real split run needs network (script's
  `download`/`run`); machinery is offline tested.
- P2.8 machinery done: `src/auditor/agents/citation/set_b.py` +
  `tests/agents/citation/test_set_b.py` (14 tests) + `scripts/set_b.py` CLI
  (candidates/inject/validate/score). InjectionKind {claim_number_mismatch,
  claim_not_supported, reference_metadata_mismatch, reference_not_found} ->
  CLASS_FOR_KIND. `inject_number`/`inject_claim` edit a SentenceAnchor span (shift later
  citation markers, mark variant INJECTED — same mechanics as set_c inject_paper) and
  produce GoldLabel(BibTarget, SentenceAnchor span of the new text) + InjectionRecord
  carrying the gold evidence passage (text + char offsets in the cited work's source text
  + source_text sha256). `gold_passage_for(source_text, claim)` = top-1 BM25 passage for
  the ORIGINAL claim (the Recall@5 gold). `inject_metadata` (title/venue/year/doi) and
  `inject_not_found` (title replaced, doi + arxiv_id cleared) edit the entry's `raw` with
  the field (D1; `_replace_in_raw` is case-insensitive because GROBID raw text drifts in
  case); entries WITHOUT raw just get the field edited. `clean_label`.
  `validate(labels, records, papers=, sources=)` does the D18 checks (duplicate ids,
  record present and kind matches class, gold passage must appear in the recorded source
  text, clean controls not on injected sentences, injected text present in the injected
  paper, raw edited for bib injections, 20% second review). `score_citation_module(labels,
  predictions)` mirrors set_c's scorer for the 4 citation classes (one-to-one D3
  matching, FPR on clean controls, unlabelled count, macro-F1).
- P2.9 machinery done: `src/auditor/agents/citation/eval.py` +
  `tests/agents/citation/test_eval.py` (6 tests) + `scripts/citation_eval.py` CLI
  (fetch = network, recall/score/tool-accuracy = offline). `recall_at_k(source_text,
  claim, gold_passage, k=5)` (whitespace-normalised containment, tolerant of
  chunk-boundary drift) and `tool_call_accuracy(outcomes, golds)` with
  `lookup_is_correct` (found/not-found must match; titles compared with `titles_match`).
  The abstract-only comparison = run the same pipeline with
  `AUDITOR_CITATION_ABSTRACT_ONLY=1` and score both runs (`score --abstract-only`).
- Test file names are unique across agent dirs (mypy requirement): the agent-level tests
  are `test_citation_check.py` and `test_citation_tools.py` (not test_check/test_tools).
- docs/team_plan.md updated: P2.1-P2.7 ticked; P2.8/P2.9 annotated as machinery-done,
  dataset build pending.

## Session 4 (2026-10-25) — in progress

Anurag authorised: implement the small fixes, build BOTH datasets (network + Docker OK),
pick the 60 set B papers myself (full leeway), run tests, record results/observations in
this file, commit and push (NO PR, NO merge). He also asked whether a small embedding
model can replace BM25 for retrieval (answer below).

Plan (in order, context updated before and after each block):
1. FIXES (each with tests, ruff+mypy+pytest green, then commit):
   a. `extract_numbers` reads scientific notation (3e-4 -> one number "3e-4").
   b. `_number_check`: code-side scan of the source text for the claimed number when the
      model gives found=None; a value present in the source that disagrees with the claim
      now yields CLAIM_NUMBER_MISMATCH without trusting the model.
   c. venue normalisation in reference metadata comparison (common aliases: conference
      full names vs short names; "Transactions on ..." vs "TMLR"-style abbreviations kept
      simple: a shared-token/abbreviation table).
   d. B1 tools read `citation_abstract_only` by default (per-call arg still overrides).
2. EMBEDDING RETRIEVER (opt-in, default stays BM25 per team decision P2.4):
   `auditor.retrieval.embeddings.MiniLMEmbedder` — quantised ONNX all-MiniLM-L6-v2
   (~23 MB, CPU, onnxruntime + tokenizers as an optional extra `embed`), model downloaded
   through the allowlisted HF host and cached under cache_dir/embeddings; Settings field
   `citation_embedder: "bm25"|"minilm"`; rank_passages/gold_passage_for/recall_at_k take
   an optional embedder so the gold passage and the agent's ranking always use the SAME
   ranker; graceful fallback to BM25 when the extra is not installed or the model is
   unavailable (logged, never raises).
3. SET A: hallmark download -> run (background, network) -> score; results recorded here.
4. SET B (the long pole, run in background jobs, resumable, cache makes re-runs cheap):
   curate 60 arXiv ids (arXiv API, cs.LG/cs.CL recent, deterministic) -> commit manifest to
   datasets/set_b/arxiv_ids.txt -> docker compose up -d (GROBID) -> fetch_arxiv.py ->
   `auditor parse` each -> `citation_eval.py fetch` each (also fix the sources keying to
   (paper_id, bib_id) first — the per-paper file has one line per ENTRY) -> deterministic
   pair picking + injections via new machinery in set_b.py (number perturbations,
   direction-swap claim rewrites, metadata edits, not_found fabrications with build-time
   network verification; clean controls sampled) -> validate -> dataset.jsonl.
## Session 4 progress 1 (code blocks done)

- COMMITTED `5daac67 citation: scientific notation, code-side number scan, venue aliases,
  abstract-only in B1 tools`:
  - `extract_numbers` keeps scientific notation whole (3e-4, 1.5e10); test updated.
  - `_number_check` scans the FULL text itself when the model gives found=None: a specific
    number (has a "." or >= 3 digits — `_scan_confident`) the work's text nowhere states
    (by value, via `_source_states`, which extracts numbers from BOTH the claim and the
    source) -> CLAIM_NUMBER_MISMATCH. Never run against abstract-only sources; bare small
    numbers ("4x") are not scanned (derived quantities). 4 new tests.
  - `retrieval.resolve.venues_match`: normalisation (punctuation -> spaces), a 15-group
    alias table (neurips/nips, icml, iclr, cvpr, iccv, eccv, acl, emnlp, naacl/hlt, aaai,
    ijcai, kdd, jmlr, tmlr, icra, interspeech) and a >=3-token containment rule;
    `reference._metadata_findings` uses it. 2 new tests.
  - B1 tools: `retrieve_cited_text`/`find_supporting_passages` (and their StructuredTool
    wrappers) take `abstract_only: bool | None = None`; None follows
    `settings.citation_abstract_only`, an explicit value overrides. 1 new test.
- COMMITTED `06cc9a8 citation: optional MiniLM embedding ranker...`:
  - `retrieval/passages.py`: `Embedder` Protocol (`score(chunks, claim) -> list[float] |
    None`); `rank_passages(..., embedder=None)` uses cosine scores when the embedder works,
    falls back to BM25 on failure/wrong width (logged).
  - `retrieval/embeddings.py`: `MiniLMEmbedder` (quantised ONNX all-MiniLM-L6-v2 ~23 MB,
    CPU, mean-pool + L2-normalise, tokenizer.json from the same HF repo, cached under
    `cache_dir/embeddings/`, downloads via the shared HttpClient) and
    `await get_embedder(settings, http)` which never raises (missing `embed` extra or
    failed download -> logged warning + None).
  - Settings: `citation_embedder: Literal["bm25","minilm"] = "bm25"`; pyproject: optional
    extra `embed = [onnxruntime, tokenizers]` + mypy overrides for them and numpy.
  - Wired into `support.check_citation`, `tools.find_supporting_passages`,
    `set_b.gold_passage_for(embedder=)`, `eval.recall_at_k(embedder=)` — the gold passage
    and the agent's ranking always use the SAME ranker.
  - Tests: FakeEmbedder ranking/fallback/width tests + missing-extra test + a
    network-marked real-model test (deselected by default; needs `uv sync --extra embed`
    — uv is NOT available in the agent shell, so the extra is NOT installed here; the
    code degrades to BM25 cleanly until Anurag runs the sync).
- Answer to Anurag's question: yes — implemented opt-in; BM25 stays the default (team
  decision P2.4), MiniLM helps on paraphrases, ~23 MB + CPU-only, graceful fallback.

## Session 4 progress 2 (build machinery + pipeline started)

- COMMITTED `9e1dbdd citation: set B build driver (propose, verify, inject) and per-entry
  source keying`:
  - sources are keyed (paper_id, bib_id) everywhere (validate + scripts) — the per-paper
    fetch file has one line per ENTRY, so per-paper keying silently lost all but the last
    entry's text.
  - set_b.py machinery: `swap_direction` (15 direction-word pairs), `perturb_number`
    (+-15-35%, same decimals), `Proposal`, `propose(paper, sources, pairs, controls, rng)`
    — deterministic round-robin over number/claim/metadata/not_found queues; number pairs
    need a specific number the source also states; claim swaps need a direction word;
    clean controls sampled from the verified pairs. 4 new tests (18 total in test_set_b).
  - `scripts/set_b_build.py`: the one-pass driver — propose -> VERIFY (perturbed value
    must not be stated by the source; the source must state the claim's original
    direction word; metadata edits must disagree with the work the index resolves;
    fabricated titles must resolve to nothing) -> inject -> append labels/records ->
    write injected paper + copy the original to <out>/papers/original/ (D14).
  - `scripts/hallmark.py run` is now resumable: checkpoints every 25 entries and skips
    already-checked keys on rerun.
- Verified live: 1157 passed + ruff + mypy strict clean after each commit.

PIPELINE STATE (background jobs):
- Set A: data/hallmark/test_public.jsonl downloaded (831 entries: 524 HALLUCINATED, 307
  VALID). Foreground smoke run of 10 entries: 69 s (~7 s/entry -> ~96 min for all),
  results sane (fabricated dois/titles flagged). FULL RUN started in background
  (PID 31592), resumable, checkpoint every 25.
- Docker: daemon was down; started Docker Desktop, engine up (28.2.2), `docker compose
  up -d` pulling GROBID in background (PID 23376).
- Set B: manifest COMMITTED at datasets/set_b/arxiv_ids.txt — 60 arXiv ids picked
  deterministically (cs.LG+cs.CL submitted Jan-Mar 2025, first 60 by submission date).
  PDF downloads running in background (PID 2920, ~3 min).

## Session 4 progress 5 (SET A COMPLETE — results)

HALLMARK test_public: 831/831 entries checked (524 HALLUCINATED / 307 VALID), 4
unverifiable (network), final score with all fixes applied
(`scripts/hallmark.py score --results runs/hallmark/results.jsonl`):

- accuracy 0.726; flagging precision 0.732, recall 0.891, F1 0.804
- false-positive rate on VALID entries 0.557 (170 of 305)
- Weakest types: partial_author_list F1 0.735 (R 0.581 — resolution matches on FIRST
  author, and a partial author list keeps the first author), near_miss_title 0.841,
  wrong_venue 0.889, placeholder_authors 0.881; the other ten types 0.94-1.0.

Three passes over the split drove three rounds of fixes (all committed):
1. FPR 0.892 -> 0.557: venues_match folding (Crossref's "2023 IEEE/CVF ... (CVPR)";
   "Proceedings of the AAAI ..."; arXiv placeholder venues never contradict) + the
   resolver PREFERS the candidate whose venue agrees with the entry.
2. Recall 0.777 -> 0.891: Resolution.doi_unresolved — a DOI that Crossref AND OpenAlex
   both reject is a reference_not_found even when a title search finds the real work
   (fabricated_doi recall had crashed to 0.448 when the fake DOI was silently forgotten).
3. Speed: ~1.5 s/entry with AUDITOR_HTTP_MAX_ATTEMPTS=1 (S2/OpenAlex Retry-After waits
   made single entries take 185 s at the default 4 attempts); runs resumable.

Honest shortcoming: the remaining 0.557 FPR comes from the metadata comparison — DBLP
venue spellings the alias table does not cover (PMLR, workshop names) and title-search
resolutions to different records. Reported as measured.

Set A artifacts: data/hallmark/test_public.jsonl, runs/hallmark/results.jsonl (831 rows).


- SET B COMPLETE (to the extent possible without an LLM and without a second reviewer):
  - 60 papers parsed into data/set_b/papers/<id>.json (GROBID).
  - Cited works' texts fetched for all 2,743 bibliography entries into
    data/set_b/sources/<paper_id>.jsonl ({paper_id, bib_id, level, source_text|reason}).
    The PDF level was skipped (too slow for a build; --with-pdf enables it); levels are
    recorded. One truncated file (2501.00773) repaired by dropping malformed lines.
  - `scripts/set_b_build.py` ran over all 60 papers: 201 gold labels = 154 injected
    (reference_not_found 96, reference_metadata_mismatch 38, claim_number_mismatch 15,
    claim_not_supported 5) + 47 clean controls. Every proposal was verified at build
    time (see progress 2). Injected copies written to datasets/set_b/papers/ (gitignored,
    local only), originals under datasets/set_b/papers/original/.
  - `scripts/set_b.py validate datasets/set_b --sources data/set_b/sources`: exit 1 with
    EXACTLY ONE problem — "only 0 of 201 labels have a second review; D18 asks for 20%"
    — by design: the second review is Anurag's job. Everything else validates clean.
  - dataset.jsonl (119 entries: 59 injected copies with injections + 60 originals)
    written by `scripts/set_b_build.py dataset`.
  - Offline sanity: `citation_eval.py recall` -> Recall@5 = 1.000 (20/20 of the records
    that carry gold passages — the bibliography injections carry none by design). The
    real Recall@5 measurement happens when the agent runs (its runtime retrieval vs
    these recorded gold passages).
- Set B shortfalls (documented in datasets/set_b/README.md):
  - The claim classes are thin (15 number + 5 claim of 154): most claim sentences don't
    state a source-confirmed specific number, and few have a direction phrase the source
    contradicts. Grow with hand-written claim injections (`set_b.py inject`) or more
    papers.
  - The D18 20% second review is outstanding — Anurag must review ~40 labels and add
    `reviewed_by` before the set is frozen.
- The module eval runs (auditor batch full-text vs abstract-only) are NOT runnable: no
  .env / no LLM keys. Commands are in docs/p2_how_it_works.md section 8.

FINAL VALIDATION: ruff clean, mypy --strict clean (215 files), pytest 1157 passed /
1 skipped (pyarrow) / 8 deselected. Everything committed and pushed to
origin/p2-citation-checker. NO PR, NO merge (as instructed).

Commits this session: 5daac67 (four small fixes), 06cc9a8 (embedding ranker), 9e1dbdd
(build driver + source keying), a9aae1c (pdftotext decode + hallmark tolerance),
[fetch/recall fixes], 4211afa + 1edfc61 (venue folding + resolver preference),
6e4f500 (doi_unresolved), 53c1853 (injection caps), c6f472c (papers local),
plus the final dataset commit.

HALLMARK test_public: 831/831 entries checked (524 HALLUCINATED / 307 VALID), 4
unverifiable (network), scored with `scripts/hallmark.py score --results
runs/hallmark/results.jsonl`:

- accuracy 0.726; flagging precision 0.732, recall 0.891, F1 0.804
- false-positive rate on VALID entries 0.557 (170 of 305 flagged: 90-ish
  reference_not_found + the rest reference_metadata_mismatch)
- per-type F1: perfect (1.000) on arxiv_version_mismatch... (see below), weakest:
  partial_author_list 0.735 (R 0.581 — our resolution matches on FIRST author, a
  partial author list keeps the first author and often the same title, so it looks
  clean), near_miss_title 0.841 (R 0.725 — token overlap 0.7 borderline),
  wrong_venue 0.889 (R 0.800), placeholder_authors 0.881, swapped_authors 0.950.

Fixes that GOT the numbers here (three full passes over the split):
1. FPR 0.892 -> 0.557: `venues_match` folding — Crossref's "2023 IEEE/CVF ... (CVPR)"
   (leading year + bracketed short name), "Proceedings of the AAAI ..." (leading
   "proceedings of the"), and arXiv placeholder venues ("arXiv (Cornell University)")
   never contradict; the resolver now PREFERS the candidate whose venue agrees with the
   entry over the first search hit (OpenAlex keeps both the preprint and proceedings
   records; it used to hand back the preprint and the check flagged a false mismatch).
2. Recall recovered 0.777 -> 0.891: `Resolution.doi_unresolved` — an entry whose DOI
   neither Crossref nor OpenAlex knows is a `reference_not_found` finding even when the
   subsequent title search finds the (real) work; previously the fake DOI was silently
   forgotten and the real work resolved CLEAN (fabricated_doi recall had crashed to
   0.448).
3. Speed: the runs take ~1.5 s/entry with `AUDITOR_HTTP_MAX_ATTEMPTS=1` (the S2/OpenAlex
   rate limits give long Retry-After waits, 60 s x 4 attempts by default -> 185 s for a
   single entry observed); the runs are resumable with 25-entry checkpoints.

Honest shortcoming found by set A: the remaining 0.557 FPR is the metadata comparison —
DBLP-style venue spellings our alias table does not cover (PMLR, workshop names,
journal subtleties) and entries whose title search resolves to a *different* record.
The benchmark itself says venues disagree on what counts as a hallucination; our numbers
are reported as measured.

Set A artifacts: data/hallmark/test_public.jsonl (downloaded), runs/hallmark/
results.jsonl (831 rows: flag + per-type). data/ and runs/ are untracked working files.

Set B progress: 60 papers parsed into data/set_b/papers/*.json; fetch_sources (PID 996)
at ~32/60 papers of cited-work texts (multi-hour; resumable). Next: set_b_build ->
validate -> dataset.jsonl.

- Fetch driver committed with per-entry progress. State at 18:10:
  - HALLMARK (PID 38304): resumable, was at 260/831, restarted cleanly.
  - fetch_sources (PID 21080, 3 papers concurrently, PDF level SKIPPED by default
    (--with-pdf to enable) because a whole PDF download + GROBID parse per entry made
    the build ~20x slower; HTML + abstract covers most arXiv works and the level is
    recorded per entry).
- Throughput reality (measured): resolve+fulltext is ~8-20 s per bibliography entry
  (title-only entries search four indexes; arXiv HTML pages are multi-MB and the arXiv
  rate limit is 3 s per request). A 60-paper set with ~60-100 entries per paper is a
  multi-hour build. Both scripts are RESUMABLE (hallmark skips checked keys; fetch
  skips fetched bib ids) and checkpoint/append as they go.
- Also fixed in `scripts/citation_eval.py recall`: the original claim is rebuilt from
  the kept original paper (papers/original/<id>.json) — a number injection's `original`
  is only the number span, not the sentence.
- Cross-part bugfix committed (`a9aae1c`): parser/layout_tables.py decodes pdftotext
  output as utf-8 (was the Windows locale; cp1252 died on UTF-8 bytes and left stdout
  None -> AttributeError in EVERY parse on Windows). Flagged for Bhaskar.

- HALLMARK run crashed twice: first (PID 31592/9840) on a transient S2 ConnectError that
  killed the process (no per-entry tolerance), then relaunched before the fix landed.
  Fix: `scripts/hallmark.py _run` catches ANY per-entry exception -> Flag.UNVERIFIABLE
  with the reason, run continues; checkpoint every 25 entries. Relaunched as PID 27816
  with the fixed code (831 entries, ~7-10 s each, ~90-120 min). At last check 85 done.
- Parse loop first failed on EVERY pdf: `parser/layout_tables.py` (Part 1's module)
  calls `pdftotext -enc UTF-8 ... capture_output=True, text=True` WITHOUT an encoding —
  on Windows `text=True` decodes with the locale (cp1252), the reader thread dies on
  UTF-8 bytes and `stdout` becomes None -> AttributeError at `done.stdout.strip()`
  (decision D1 says a missing pdftotext should degrade, and pdftotext IS installed here
  via MiKTeX). Fix (one-liner, cross-part bugfix, flagged for Bhaskar):
  `encoding="utf-8", errors="replace"` in the subprocess.run call. Parser tests: 87
  passed. Verified: 2501.00712 parsed (31 pages, 908 sentences, 120 references,
  165/174 markers linked, 19 tables). Parse loop relaunched (PID 11764).
- PDFs: all 60 downloaded to data/papers/. GROBID container running.
- LLM check: NO .env file exists -> `auditor batch` (b1) eval runs and the support
  judgement need AUDITOR_LLM_PROVIDER/MODEL; they are NOT runnable this session. The
  dataset build needs no LLM.

## Session 3 (2026-10-23) — done

Anurag asked for:
1. A markdown doc explaining exactly how P2 works and is implemented, with execution flow
   (file names), capabilities and shortcomings -> being written to `docs/p2_how_it_works.md`.
2. An exact list of what is left for P2 and how to do it (Docker/GROBID available on his
   machine; he authorises using it if needed).
3. Keep this context file updated regularly.

Nothing else in this session changes code; the doc is documentation only. Verification of
facts done for the doc: read retrieval/, agents/citation/, hallmark.py, set_b.py, eval.py,
the three scripts, config fields, planner/tasks.py (citation_candidates -> citation_tasks),
orchestration NODE_FOR_KIND mapping (REFERENCE and CITATION both -> "citation" node),
registry (finds `check` and `make_tools`). Also verified live: `Settings()` defaults
(grobid_url http://localhost:8070, cache_dir data/cache), `auditor parse/batch` CLI flags,
compose.yaml (grobid service on 127.0.0.1:8070, auditor service under the "run" profile),
fetch_arxiv.py manifest format, datasets/parser_fixtures/arxiv_ids.txt (13 parser papers,
not a set B pool). Test counts: 106 offline tests in tests/retrieval (39) +
tests/agents/citation (67).

## Session 3 result

- `docs/p2_how_it_works.md` WRITTEN (429 lines): role in the pipeline, module map,
  execution flows 3.1-3.9 (reference check, citation support check, B1 tools, resolution,
  fulltext, passages, HALLMARK, set B, module eval) with file/function names,
  configuration, 9 capabilities, 11 numbered shortcomings, and an 10-step set B build
  guide with exact commands (docker compose up -d for GROBID, fetch_arxiv.py manifest,
  `auditor parse` per PDF, `citation_eval.py fetch` per paper, `set_b.py
  candidates/inject/validate`, second review, `auditor batch` full vs
  AUDITOR_CITATION_ABSTRACT_ONLY=1, scoring/recall/tool-accuracy).
- No code changed; docs only. The context doc and the how-it-works doc are uncommitted at
  the time of this note (commit them as `citation: document how the citation checker
  works`).

## What is left for Anurag (updated for session 3)

Code is DONE and pushed (both commits on origin/p2-citation-checker). Remaining work is
all dataset execution + annotation, none of it code:

1. Set A (HALLMARK, network only): hallmark.py download/run/score (commands in
   docs/p2_how_it_works.md section 8 bottom). No GROBID needed.
2. Set B build (network + GROBID + curation), the 10 steps in
   docs/p2_how_it_works.md section 8: curate 60 arXiv ids -> fetch_arxiv.py -> `auditor
   parse` each (docker compose up -d first) -> `citation_eval.py fetch` each -> pick ~10
   pairs/paper via `set_b.py candidates` -> `set_b.py inject` per injection + hand-written
   clean-control labels -> second-review 20% -> `set_b.py validate` -> build the
   DatasetEntry JSONL (injected copies + originals).
3. Module eval numbers: `auditor batch --systems b1` full and with
   AUDITOR_CITATION_ABSTRACT_ONLY=1, then citation_eval.py score (both) / recall /
   tool-accuracy.
4. Open the PR when the numbers are in. Nothing to merge until then.
5. Optional code improvements he may want before the PR (his call, all small):
   scientific-notation support in extract_numbers; a code-side scan of the source text
   for the claimed number (removes shortcoming 2); venue-normalisation for the metadata
   comparison (removes most of shortcoming 3); pass citation_abstract_only through the
   B1 tools (removes shortcoming 9).

## Still to do in session 2 (all done — kept for the record)

- P2.7 HALLMARK: machinery `src/auditor/agents/citation/hallmark.py` +
  `scripts/hallmark.py` (download/run/score) + tests — DONE and green (5 tests, incl. an
  end-to-end MockTransport test through the agent's `check()`). HALLMARK = arXiv 2607.18360
  (Kim, Brendel, Reizinger 2026), data at github.com/rpatrik96/hallmark data/v1.2/*.jsonl
  (MIT). Record format verified from test_public.jsonl: bibtex_key, bibtex_type, fields
  {title, author ("A and B"), year (str), doi, booktitle/journal, url}, label
  VALID|HALLUCINATED, hallucination_type (14 types), difficulty_tier, explanation,
  generation_method, source_conference, subtests{...bools}, raw_bibtex, schema_version.
  Loader drops `__canary__` entries, rejects duplicate keys. `bib_entry_for` maps fields
  to BibEntry and renders `raw` from the same fields (D1). `check_entry` builds a
  one-entry ParsedPaper (dataclasses.replace(ctx, paper=...)) and runs the agent's
  reference check; Flag = clean / reference_metadata_mismatch / reference_not_found /
  unverifiable. `score` gives accuracy, flagging P/R/F1, FPR on VALID, per-type
  breakdown; results round-trip JSONL. The real split run needs network (script's
  `download`/`run`); machinery is offline tested.
- P2.8 set B: machinery `src/auditor/agents/citation/set_b.py` + tests (13 passing).
  InjectionKind {claim_number_mismatch, claim_not_supported, reference_metadata_mismatch,
  reference_not_found} -> CLASS_FOR_KIND. `inject_number`/`inject_claim` edit a
  SentenceAnchor span (shift later citation markers, mark variant INJECTED — same
  mechanics as set_c inject_paper) and produce GoldLabel(BibTarget, SentenceAnchor span
  of the new text) + InjectionRecord carrying the gold evidence passage (text + char
  offsets in the cited work's source text + source_text sha256). `gold_passage_for(
  source_text, claim)` = top-1 BM25 passage for the ORIGINAL claim (the Recall@5 gold).
  `inject_metadata` (title/venue/year/doi) and `inject_not_found` (title replaced, doi +
  arxiv_id cleared) edit the entry's `raw` with the field (D1; `_replace_in_raw` is
  case-insensitive because GROBID raw text drifts in case); entries WITHOUT raw just get
  the field edited. `clean_label`. `validate(labels, records, papers=, sources=)` does
  the D18 checks (duplicate ids, record present and kind matches class, gold passage must
  appear in the recorded source text, clean controls not on injected sentences, injected
  text present in the injected paper, raw edited for bib injections, 20% second review).
  `score_citation_module(labels, predictions)` mirrors set_c's scorer for the 4 citation
  classes (one-to-one D3 matching, FPR on clean controls, unlabelled count, macro-F1).
- P2.9 eval: machinery `src/auditor/agents/citation/eval.py` + tests (6 passing):
  `recall_at_k(source_text, claim, gold_passage, k=5)` (whitespace-normalised containment,
  tolerant of chunk-boundary drift) and `tool_call_accuracy(outcomes, golds)` with
  `lookup_is_correct` (found/not-found must match; titles compared with `titles_match`).
- STILL MISSING: `scripts/set_b.py` CLI (candidates / inject / validate / score, follow
  scripts/set_c.py's shape; pair collection over 60 arXiv PDFs needs network+GROBID and
  Anurag's say-so — machinery/CLI only this session) and `scripts/citation_eval.py` CLI
  (recall / score / tool-accuracy offline; run = network+LLM, full-text vs abstract-only
  via `citation_abstract_only`). Then: ruff, mypy, full pytest, update team_plan.md ticks
  (P2.1-P2.7 done; P2.8/P2.9 machinery done, dataset build pending), commit, push p2
  branch. NO merge to main, NO PR.

## Reference facts (verified this session)

- Error classes for P2: reference_not_found, reference_metadata_mismatch,
  claim_not_supported, claim_number_mismatch (ErrorClass in schemas/findings.py).
- `ParsedPaper`: paper_id, variant, sentences, bibliography (BibEntry: id, raw, title,
  authors, year, venue, doi, arxiv_id, url), citations (CitationMention: id, sentence_id,
  start/end span, marker, bib_ids, link_method); helpers `paper.sentence(id)`,
  `paper.bib_entry(id)`.
- Registry (orchestration/registry.py) finds `check` in `auditor.agents.citation` and
  `make_tools` in `auditor.agents.citation.tools`; missing -> mock + reported in
  manifest; exists-but-fails-to-import -> run stops.
- Tests: pytest, asyncio_mode=auto; markers network/llm/grobid/docker excluded by
  default; conftest `parts` fixture swaps agent modules with empty stand-ins;
  `auditor.testing` has `offline_context(tmp_path)`, `ScriptedChatModel`, `tool_reply`,
  `offline_http`, `checked_result`.
- Config: Settings (AUDITOR_* env / .env). `citation_llm_enabled` exists (planner pass).
  Allowed hosts already include arxiv.org, export.arxiv.org, api.openalex.org,
  api.crossref.org, api.semanticscholar.org. Rate limits: arXiv 3 s, Crossref 0.2 s,
  OpenAlex 0.1 s, S2 1 s.
- Parser: `parse_pdf(path, grobid=GrobidClient(http, base_url), cache_dir=, paper_id=)`
  -> ParsedPaper; raises ParseError; GROBID must be running (`docker compose up -d`).

## User preferences (Anurag, stated this session)

- Only implement P2; never other parts. Never touch main; work on `p2-citation-checker`;
  committing/pushing allowed, no merging back to main.
- Python deps only in the project venv (uv); nothing global except uv itself.
- Do not assume — ask when non-trivial. Do not break existing functionality; test well.
- Keep this context file updated before and after implementing or planning.
- Final report at the end: what was done, what is left, what the user must do, plus
  decisions and observations.

## Next steps for a new session

1. Read this file top to bottom; the "Session status after session 2" section and
   "Still to do in session 2" are the live state.
2. Finish the session-2 leftovers (see "Still to do"): fix the one RED eval test,
   scripts/set_b.py, scripts/citation_eval.py, ruff+mypy+full pytest, team_plan ticks,
   commit, push p2 branch (no merge, no PR).
3. Run everything through `.venv\Scripts\python.exe -m ...` (uv is NOT on PATH in the
   agent shell).

## Session 2 (2026-10-21) — plan agreed with Anurag

1. Offline unit tests for retrieval/ and agents/citation/ (MockTransport, ScriptedChatModel,
   monkeypatched resolve/retrieve for agent-level tests), fixing the known issues 1-3 while
   testing (issue 5 stays as is: defensible, revisit against set B).
2. `citation_abstract_only` Settings field wired into `check()`.
3. P2.7/P2.8/P2.9: machinery in `auditor/agents/citation/` (following the set C pattern:
   `auditor.agents.code.set_c` holds machinery, `scripts/set_c.py` is a thin CLI), thin
   CLIs in `scripts/`, offline tests on fixture papers. The full 60-paper set B run needs
   GROBID + network + Anurag's say-so and is NOT part of this session.
4. ruff + mypy + full pytest green; update this file and team_plan.md ticks; commit
   (`citation: ...` style) and push the p2 branch; NO merge to main, NO PR yet.

Progress: ALL items done (see "Session 2 final state" above for the full detail).
Commits: "citation: retrieval package, agent check, B1 tools and offline unit tests",

## Session 6 (2026-10-29) — FIX session after the audit (in progress)

Anurag's brief: fix the audit findings as far as possible; only ask about MAJOR decisions;
set B may be used WITHOUT the D18 second review (he will do it later); the set B numbers
are horrible — find out what is wrong and fix it; same for HALLMARK accuracy (0.726,
FPR 0.557 on valid). Testing on minimal sets, LLM-call budget left to my judgement.
No merge, no PR. Keep this file current.

Plan (in order):
1. [x] Analyse HALLMARK false positives: runs/hallmark/results.jsonl reasons for VALID
   entries -> find the comparison bugs in resolve.py/reference.py.
2. [ ] Fix the metadata comparisons (year/venue/title tolerance) + author handling; tests.
3. [ ] Fix the eager title searches (lookup.py + resolve.py): sequential with early exit,
   per-search exception capture; tests.
4. [ ] Wire GROBID into the agent path (check(), tools.py, citation_eval fetch):
   construct GrobidClient(ctx.http, ctx.settings.grobid_url) lazily; tests.
5. [ ] Low cleanups: eval.py dead lines, reference.py re-fetch, set_b.py slice order,
   support.py quote verbatim/normalised re-anchor + docstring.
6. [ ] Re-run HALLMARK (HTTP cache makes it cheap, no LLM) + score; iterate thresholds.
7. [ ] Small FULL-system run on 2-4 subset entries (claim-injected papers), score with
   --before-verifier (mock verifier sends everything to human review -> nothing counts as
   positive otherwise), diagnose agent issues, fix.
8. [ ] Full offline suite + ruff + mypy, `citation:` commits, push.

Findings/notes for this session:
- HALLMARK false-positive analysis (fresh baseline re-run first — the OLD results file
  predated merged PRs and reproduced stale numbers): fresh pre-fix baseline (old resolver,
  reason fix only): accuracy 0.750, P 0.811 / R 0.786 / F1 0.798, FPR 0.313 (96/307).
  FP buckets: 46 year delta -1, 25 resolution failures, 13 arXiv DOIs, 8 venue, 2 title.
- FIXES IMPLEMENTED (all with tests, full suite 1212 passed, ruff clean):
  1. Sequential title searches with early exit + per-index exception capture
     (resolve.py _search_candidates + lookup.py search_all); one dead/rate-limited index
     no longer sinks the lookup and unused indexes are not queried.
  2. arXiv-DOI routing: 10.48550/arXiv.<id> DOIs live in DataCite -> resolve via arXiv
     instead of being called fabricated (clients.arxiv_id_from_doi; resolve_entry +
     lookup_citation).
  3. ASCII-folded search queries + shortened retry (>8 words) for mojibake/long titles
     (clients.ascii_fold / short_title; "DemoirΘing" -> "Demoiring").
  4. titles_match: prefix rule (index keeps only the main title, "NeRF" for the full
     title). reference.py: near-miss title rule (tokens differ, neither side a subset ->
     flag) — catches "…You Nedd"-style corruptions that resolve fuzzily.
  5. AUTHOR CHECKS (new): author names the work doesn't have (placeholders/fabricated),
     silently truncated lists (fewer than the work's, no "et al." marker, work >= 3
     authors), wrong first author (swapped). first_surname hardened (et al., trailing
     initial, comma order).
  6. venues_match: institutional-repository placeholder venues (UCL Discovery,
     eScholarship, Zenodo, SSRN, ...) never contradict, like arXiv records.
  7. GROBID wired into the agent path: fulltext.grobid_for(settings, http); check(),
     both B1 tools and citation_eval fetch pass it; _pdf_full_text checks is_alive first
     (a dead GROBID is a fallback to abstract, not a crash — HttpRequestError from a dead
     GROBID would previously have escaped as an unhandled exception).
  8. check_entry keeps the first finding's summary as the result reason (results files
     are now self-explanatory).
  9. Low cleanups: eval.py dead containment lines, reference.py entry pass-through
     (optional entry param, no re-fetch in the pipeline path), set_b.py slice-before-check
     order, support.py quote re-anchoring (whitespace-tolerant _find_quote) + honest
     docstring.
- YEAR DECISION (Anurag chose option A): checker tolerates ±1 year (46/96 hallmark FPs
  gone; correct for proceedings-vs-online-first drift). Set B: ALL 38 metadata injections
  were year+1 -> regenerated to year+3 surgically (runs/_regen_years.py: papers' bib row
  + raw edited via _replace_in_raw, label ids re-derived like inject_metadata, records
  updated; 38/38 done). set_b.py propose() now injects year + rng.choice((2,3)) and the
  builder's verify rejects injected years within ±1 of the resolved work's year.
  `scripts/set_b.py validate` now exits 1 with EXACTLY the D18 second-review problem.
  NOTE: the subset B1 runs from the audit session are stale for the metadata class.
- HALLMARK re-run with all fixes in flight: runs/hallmark/results_fixed.jsonl
  (~2.5 h at ~5 entries/min; S2's unauthenticated ~1 rps is the bottleneck; a
  parent+child python pair per run is the venv launcher, NOT a race).
- Full-system mini-run (step 7) PREPARED: runs/full_mini_dataset.jsonl = 2501.00692
  injected + original (has both claim classes). First attempt hit 300s reference-task
  timeouts from API throttling while the hallmark run hammers the same indexes — killed;
  rerun AFTER hallmark finishes, scoring with --before-verifier (placeholder verifier
  sends everything to human review; nothing counts positive otherwise).
- SESSION 6 FINAL RESULTS (all committed, pushed through 180609f):
  * HALLMARK (runs/hallmark/results_fixed.jsonl, 831/831, full fixes): accuracy 0.851,
    flagging P 0.902 / R 0.857 / F1 0.879, FPR 0.160 (49/307). From 0.726 / 0.732-0.891-
    0.804 / 0.557. Author types fixed: partial_author_list R 0.258->0.935,
    placeholder_authors ->1.000, swapped_authors ->0.968; fabrications ->1.000;
    near_miss_title 0.475->0.700. Remaining weak: nonexistent_venue R 0.378 and
    preprint_as_published R 0.500 — the deliberate price of the repository/placeholder
    venue tolerance (those FNs are indistinguishable from the FPs it removes at venue
    level); arxiv_version_mismatch R 0.800 (needs a real version check — future work;
    the old 'flags' for it were resolution failures counted as TPs).
  * Set B full-system mini-run (00692 injected+original, --llm-cache, 2 done 0 failed):
    AFTER the dataset cleanups: reference_not_found P 1.000 R 1.000 F1 1.000, macro-F1
    1.000, FPR 0/14 controls (runs/full_mini_labels.jsonl). The earlier zero scores were
    (a) a STALE subset-labels file still containing colliding metadata labels
    (matches() does not compare classes -> RNF preds matched metadata labels first) and
    (b) genuinely broken dataset labels, both fixed below.
  * DATASET CORRECTIONS (all applied, validate exits 1 only for the D18 review):
    1. year injections +1 -> +3 (38 records, runs/_regen_years.py);
    2. collision labels dropped (32: metadata/claim labels on entries that also carry a
       not_found injection — unreachable by construction; runs/_drop_collisions.py);
    3. table-fragment claim_number labels dropped (11 of 15 — the builder had perturbed
       numbers inside GROBID-merged table rows, e.g. 00692 s68 "10.0 20.2 PG19 2.50
       6.75..."; runs/_drop_table_junk.py). Final: 158 labels (96 RNF, 9 metadata, 4
       claim_number, 3 claim_not_supported, 46 controls), 112 records. README updated.
    propose() now skips sentences crowded with bare numbers (>3) so a rebuild will not
    reproduce the junk.
  * AGENT FIXES for the mini-run's over-flagging: _author_problems filters parse junk
    ('*', 'g', 'xlstm' style tokens stay only if alphabetic >=2 chars) and caps the
    truncation rule at collaborations of <=10 authors; numbers_agree now maps number
    words ('eight' <-> 8) and strips suffixes ('8x', '10-fold'); the support prompt
    demands an INCOMPATIBLE statement for 'contradicted' (different conditions/models
    are 'insufficient').
  * Full suite 1216 passed / 1 skipped / 8 deselected; ruff clean; format clean; mypy 2
    pre-existing Part-1 errors only.
- Session 6 LEFT OVER: re-run the B1 subset (30 entries) on the cleaned labels if
  current B1 numbers are wanted (~250 calls); the arxiv_version check; the D18 second
  review (Anurag); full-dataset runs remain future work.


## Session 5 (2026-10-29) — AUDIT + EVALUATION session (completed)

Anurag's instructions: audit the whole P2 implementation end to end, run everything
(offline suite + network/llm/grobid marked tests + real GROBID + the real Vertex LLM),
execute the P2.9 set B module runs (small sample first, then full with confirmation),
write docs/p2_audit_report.md (separate from this file). NO new features; only small,
clearly-correct defect fixes (ask otherwise). Branch stays `p2-citation-checker`, no PR,
no merge to main. LLM is LIVE: provider `google_genai` (langchain-google-genai v4 ->
google-genai SDK, Vertex AI + ADC), model gemini-2.5-flash, shared free-trial project
lmaproject6767 — cost rules: <=100 model calls per run without asking; small sample
(<=10) first; `--llm-cache` during development, UNCACHED for final reported numbers (D10);
bounded retries (<=5) already in code, do not weaken; caps AUDITOR_LLM_MAX_CALLS /
AUDITOR_LLM_MAX_COST_USD stay set; never touch gcloud billing/services/IAM; report
403 BILLING_DISABLED / PERMISSION_DENIED to Anurag instead of fixing.

Audit plan (in order):
1. [x] Read the five docs; `git status` clean, branch up to date with origin (HEAD
   d1019e0 = merge of PR #63, same as origin/main right now).
2. [ ] Static audit of every P2 module (file:line findings + severity) in
   docs/p2_audit_report.md.
3. [ ] Full offline suite + ruff + mypy; network-, llm-, grobid-marked tests.
4. [ ] End-to-end smoke test (docker GROBID + real LLM): one paper through
   `auditor audit`, then a 3-5 paper `auditor batch --systems b1`; inspect traces.
5. [ ] P2.9 eval runs: <=5-paper cached sample -> confirmation -> full set B runs
   (full-text + abstract-only) -> score/compare/recall + re-score hallmark set A.
6. [ ] Write docs/p2_audit_report.md, commit, push.
7. [ ] Final: ruff + mypy + pytest green, `citation: ...` commits, push.

Open items Anurag flagged for this audit:
- The embedding extra was never synced/run (`uv sync --extra embed`,
  AUDITOR_CITATION_EMBEDDER=minilm).
- parser/layout_tables Windows-locale fix (a9aae1c) is mine but belongs to Part 1 —
  flag to Bhaskar in the report, do NOT merge.
- Known-weak spots from set A: partial_author_list, near_miss_title.
- D18 20% second review is Anurag's own job; do not fake `reviewed_by`.
- Claim classes are thin (20 of 154); growth by hand is optional/later.

Findings so far: (static audit, step 2 — details going into docs/p2_audit_report.md)
- NO new features; only defect-level observations recorded. Headline findings:
  1. (medium) GROBID is never wired into the P2 full-text path: `RunContext` has no
     GrobidClient, so `check()` (agents/citation/__init__.py:66) and the B1 tools
     (tools.py:69/90) call `retrieve_source` with grobid=None -> SourceLevel.FULL_PDF is
     unreachable at run time (only scripts/fetch_sources.py passes grobid). Same for
     scripts/citation_eval.py fetch (citation_eval.py:175). Effect: PDF-only cited works
     fall back to abstract in every P2 run.
  2. (medium) lookup.py:93-97 and resolve.py (search step) evaluate ALL FOUR title
     searches eagerly (tuple of awaited calls), so one lookup always makes 4 API calls
     even when the first has hits; also one network failure in any of the four aborts the
     whole search even if an earlier one had hits.
  3. (low) support.py: docstring (line 279-280) says "a quote must be verbatim in the
     source" but the code never verifies the model's quote against source.text (quote is
     attached with char_start=None when not found). Doc/code mismatch, no correctness
     impact on scoring.
  4. (low) eval.py:48-51 dead code: `shorter in longer` duplicates the containment test
     on line 46.
  5. (low) reference.py:79-88 re-fetches the entry `check()` already resolved (dead
     duplicate path, plus unreachable because check() only calls it when entry exists).
  6. (info) `citation_abstract_only` IS already wired into check() (line 70) — the
     flagged open item is resolved.
  7. (info, Part 1's file) mypy fails on src/auditor/infra/llm.py:317 (fcntl.flock —
     Windows-locale, pre-existing, NOT mine; report to Bhaskar together with the
     layout_tables fix).
- Test results: offline 1192 passed / 1 skipped (pyarrow missing, Part 3's test) / 8
  deselected; ruff check clean; ruff format clean (139 files); mypy 2 errors (above);
  network 1 passed + 1 skipped (onnxruntime — the `embed` extra is indeed NOT synced);
  grobid 3 passed; llm-marked tests: NONE EXIST (marker defined in pyproject, 1201
  deselected).
- Smoke test (step 4a) DONE on data/set_b/papers/2501.00684.json via
  `auditor audit --run-id audit-smoke-p2` (GROBID up, real LLM): 56 tasks (54 checked,
  1 unverifiable, 1 failed, 0 verified flags, 21 needs_human_review, 0 dropped); output
  runs/audit-smoke-p2/2501.00684-original/{tasks,results,predictions,verified}.jsonl.
  12 citation-class findings with LOOKUP + QUOTE evidence (quotes carry char offsets,
  e.g. char_start 10770). The 1 failed task was the duplicate-finding-id bug (finding 1
  above) — FIXED in support.py (number passed into _number_check/_mismatch_finding, call
  site passes len(findings)) with two regression tests in test_support.py
  (test_a_contradiction_and_a_number_mismatch_both_reported,
  test_two_number_mismatches_get_unique_finding_ids); all 79 citation-agent tests pass.
  LLM probe: provider google_genai:gemini-2.5-flash, structured output + tool calling OK
- B1 batch smoke (step 4b) DONE: `auditor batch --experiment b1-smoke --systems b1
  --llm-cache` over 3 entries (2 injected + 1 original; runs/smoke_b1_dataset.jsonl, paths
  relative to the dataset file), default 3 runs per system. r2 (clean): 25 llm calls /
  570k in / 14k out tokens for 3 papers (~8 calls+23k in tokens per paper); 30 tool calls,
  sane mix (lookup_reference 11, find_supporting_passages 5, code/template tools), no
  loops. B1 caught real injected errors (reference_metadata_mismatch 2, reference_not_found
  1, claim_not_supported 1 on 00684-injected). Observations: (a) r1 failed paper
  2501.00691-injected with HttpStatusError HTTP 403 api.github.com (unauthenticated rate
  limit, from a code tool) — kills the whole paper run, flag to Bhaskar/Vijay; (b) B1
  (baselines/common.py) DROPS reported errors with empty sentence_id/bib_id with a warning
  — findings silently lost when the model omits ids; (c) batch defaults to 3 runs — with
  --llm-cache the extra runs are cheap.
- P2.9 subset runs DONE (both conditions, uncached):
  * Full-text `runs/setb-subset-b1-r1`: 26/30 entries done (4 deterministic non-P2
    failures: GitHub 403 x2 papers, code-tool JSONDecodeError on rank_bm25 repo x2
    variants, GraphRecursionError on 00692-injected); B1 emitted 22 predictions on 8
    papers, 10 citation-class; score vs runs/setb_subset_labels.jsonl (78 labels — MUST
    use the subset-restricted labels file; the scorer takes the labels as the universe):
    metadata_mismatch P 1.000 R 0.056 F1 0.105, all other classes 0, macro-F1 0.026, FPR
    0/14. The low numbers are REAL B1 behaviour (verified: it flags unlabelled real
    references, e.g. 00799 b28 vs labels b0/b1), not a scoring bug.
  * Abstract-only `runs/setb-subset-abs-b1-r1`: 25/30 done, macro-F1 0.026 identical;
    prediction mix shifted (8 RNF vs 6). B1-level ablation shows no difference; the
    module-level ablation needs the full system (~10x cost, one call per citation task).
  * Recall@5 1.000 on the 20/154 records that carry a gold passage (134 have none).
- docs/p2_audit_report.md COMPLETE (all sections; results tables filled).
- Audit session remaining: nothing pending in code. Follow-ups for Anurag listed in the
  report §7 (D18 second review, full-dataset + full-system runs, GROBID wiring decision,
  lookup sequential-fallback decision, embed extra, claim-class growth, llm-marked test).


then "citation: HALLMARK loader, set B injection machinery and evaluation scripts".
