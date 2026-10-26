# Part 2 — the citation checker: how it works

Owner: Anurag. This document explains what the citation agent does, how it is implemented,
and the exact execution flow with file names. Read [p2_context.md](p2_context.md) for
session history and [decisions.md](decisions.md) for the binding decisions (D1, D3, D4, D6,
D8, D18, D15 are the ones this part implements).

## 1. Role in the pipeline

The pipeline runs `PDF/ParsedPaper -> planner -> AuditTask[] -> specialists -> TaskResult[]
-> verifier -> VerifiedFinding[] -> report + predictions`. The citation agent is one
specialist. It consumes two of the task kinds the planner emits and answers two questions:

- **REFERENCE tasks** (`BibliographyAnchor` + `BibTarget`, no claim): does this bibliography
  entry exist in the scholarly indexes, and does its metadata (title, year, venue) match?
  Answered **without any model** — pure lookups and field comparison.
- **CITATION tasks** (`SentenceAnchor` + `BibTarget`/`MarkerTarget` + claim): does the cited
  work actually support what the paper claims about it? Answered `supported`,
  `contradicted` or `insufficient`, with the numbers the claim attributes to the work
  compared by value in code.

Its outputs are `TaskResult`s whose `CandidateFinding`s carry the error classes it owns
(D4): `reference_not_found`, `reference_metadata_mismatch`, `claim_not_supported`,
`claim_number_mismatch`. Every finding quotes evidence: a LOOKUP record of what was
queried, and a QUOTE copied exactly from the cited work's text with character offsets.

Where the tasks come from (Part 1, not this part): `src/auditor/planner/claims.py`
(`citation_candidates`, optionally refined by a model pass when
`settings.citation_llm_enabled`) and `src/auditor/planner/tasks.py`
(`reference_tasks`, `citation_tasks`). The orchestrator
(`src/auditor/orchestration/graph.py`, `NODE_FOR_KIND`) sends both REFERENCE and CITATION
tasks to the same `"citation"` node; `src/auditor/orchestration/registry.py` discovers the
node's implementation by finding `check` in `auditor.agents.citation` and `make_tools` in
`auditor.agents.citation.tools` (decisions D6/D7). A missing `check` makes the pipeline run
with a mock and say so in the manifest; a broken import stops the run.

## 2. Module map

```text
src/auditor/
  retrieval/                      # the retrieval package (P2.1-P2.4), no LLM anywhere
    clients.py                    #   arXiv / OpenAlex / Crossref / S2 API clients -> Work
    resolve.py                    #   entry/marker -> Work (identifier first, then title)
    fulltext.py                   #   Work -> SourceText (HTML > PDF > abstract > none)
    passages.py                   #   chunking + BM25 ranking with exact char offsets
  agents/citation/                # the agent (P2.5-P2.6) and the dataset machinery
    __init__.py                   #   check(task, ctx) -> TaskResult  (the entry point)
    reference.py                  #   reference checks (no model)
    support.py                    #   support judgement (one model call + code checks)
    lookup.py                     #   lookup_citation for raw citation strings
    tools.py                      #   the three B1 tools (P2.6, D7/D8)
    hallmark.py                   #   set A machinery (P2.7)
    set_b.py                      #   set B machinery (P2.8): injections, labels, scoring
    eval.py                       #   Recall@k + tool-call accuracy (P2.9)
scripts/
  hallmark.py                     # set A CLI (download / run / score)
  set_b.py                        # set B CLI (candidates / inject / validate / score)
  citation_eval.py                # eval CLI (fetch / recall / score / tool-accuracy)
  fetch_arxiv.py                  # downloads arXiv PDFs from a manifest (Part 1's script)
tests/
  retrieval/                      # 39 offline unit tests (MockTransport)
  agents/citation/                # 67 offline unit tests (MockTransport, ScriptedChatModel)
```

Everything goes through the shared infrastructure: `src/auditor/infra/http.py`
(`HttpClient`: host allowlist, per-host rate limits, 30-day GET cache, retries, request
log) and `src/auditor/infra/context.py` (`RunContext`: settings, paper, http, llm,
logger). Nothing in this part opens its own connection.

## 3. Execution flows

### 3.1 A reference task (no model)

```text
planner.tasks.reference_tasks(paper)          # one task per bibliography entry
  -> orchestration.graph (node "citation")
  -> auditor.agents.citation.check(task, ctx)          # agents/citation/__init__.py
       1. narrow task.target to BibTarget; ctx.paper.bib_entry(task.target.bib_id)
          (KeyError -> TaskResult UNVERIFIABLE "entry ... is not in the paper")
       2. retrieval.resolve.resolve_entry(ctx.http, entry)   # see 3.4
       3. reference.check_reference(task, ctx, resolution)   # agents/citation/reference.py
            resolution not found -> CandidateFinding REFERENCE_NOT_FOUND
                                    + LOOKUP evidence (every query and its outcome)
            resolution found     -> _metadata_findings: title (titles_match, token
                                    overlap >= 0.7), year (exact), venue
                                    (case-insensitive)
                                    -> CandidateFinding REFERENCE_METADATA_MISMATCH
                                    listing every disagreeing field
            all match            -> CHECKED with no findings
```

### 3.2 A citation task (one model call, checked in code)

```text
planner.claims.citation_candidates -> planner.tasks.citation_tasks  # claim = sentence text
  -> auditor.agents.citation.check(task, ctx)
       1. resolve the target:
            BibTarget    -> resolve_entry(ctx.http, entry)
            MarkerTarget -> retrieval.resolve.resolve_marker(ctx.http, marker,
                            paper.bibliography)
                            author-year markers matched to the bibliography by surname
                            and year; numeric markers get a reason -> UNVERIFIABLE (D6)
       2. not resolved -> UNVERIFIABLE with the resolver's reason
       3. retrieval.fulltext.retrieve_source(ctx.http, work, ctx.settings,
                          abstract_only=ctx.settings.citation_abstract_only)  # see 3.5
       4. support.check_citation(task, ctx, resolution, source)  # agents/citation/support.py
            ctx.require_llm() raises  -> UNVERIFIABLE "no LLM is configured..."
            source.level NONE/empty   -> UNVERIFIABLE with the retrieval reason
            passages = passages.rank_passages(source.text, claim, top=TOP_PASSAGES=5)
            one StructuredLLM call (effort "low"), SupportAnswer schema:
                verdict: supported | contradicted | insufficient
                quote:   exact quote from ONE passage (empty if insufficient)
                reason:  one or two sentences
                numbers: [NumberCheck(claimed, found, quote, agrees)]
            code checks of the answer:
                verdict not one of the three -> treated as insufficient
                verdict contradicted -> CandidateFinding CLAIM_NOT_SUPPORTED with LOOKUP
                    + QUOTE evidence; the quote is located in the source (str.index) so
                    the Evidence carries char_start/char_end; quotes shorter than
                    MIN_QUOTE_CHARS=20 get LOOKUP evidence only
                numbers -> _number_check per NumberCheck:
                    found is None                -> no finding (cannot confirm in code)
                    numbers_agree(claimed,found) -> no finding
                    otherwise                    -> CLAIM_NUMBER_MISMATCH
                    (the model's `agrees` hint is ignored: a model that says "agrees"
                    about values that disagree is exactly the failure this catches)
            no findings -> CHECKED, else CHECKED with findings
```

`numbers_agree` (support.py) normalises both sides — percent ("12.5%" vs "0.125"),
thousands separators, scientific notation — and accepts anything within a 1% relative
tolerance ("92.1" vs "92.15" agrees; "92.1" vs "88.4" does not).


### 3.3 The B1 tools (shared with the tool-using baseline, D7/D8)

`agents/citation/tools.py`. Each function is decorated with
`auditor.infra.tools.tracked_tool`, so every call is charged to the task's budget and
recorded in `tool_calls.jsonl` (HTTP requests are counted separately; one tool call may
make several requests). `make_tools(ctx)` returns three `StructuredTool`s for the B1
agent:

| Tool | Does | Answer |
|---|---|---|
| `lookup_reference` | `resolve_entry` | JSON `resolution_record`: found, id, source, title, authors, year, venue, doi, arxiv_id, queries, pdf/html urls — or found:false + reason |
| `retrieve_cited_text` | `resolve_entry` + `retrieve_source` | "source level: ..." + the text, clipped at `MAX_TOOL_CHARS=6000` |
| `find_supporting_passages` | resolve + retrieve + `rank_passages` | "source level: ..." + the passages numbered `[1]..[5]` |

`agents/citation/lookup.py` (`lookup_citation`) is the raw-string variant of the same
lookup: a DOI or arXiv URL in the text is looked up by identifier first, otherwise the
title part is searched across OpenAlex, Crossref, arXiv and S2 in that order. It records
every query it made for the evidence trail.

### 3.4 Resolution internals (`src/auditor/retrieval/resolve.py`)

`resolve_entry` tries identifiers before search, and every attempt is logged into
`Resolution.queries` (this is what the LOOKUP evidence shows the verifier):

1. `entry.doi` (if it looks like a DOI) -> Crossref by DOI, then OpenAlex by DOI.
2. `entry.arxiv_id` -> the arXiv API.
3. title (+ first author) -> title search in order OpenAlex, Crossref, arXiv, S2; each hit
   is filtered by `_entry_matches` (title token overlap, year within one, first surname
   present in the work's authors) before it is accepted.

Nothing found anywhere -> `Resolution(reason="no work with this title was found in the
scholarly indexes")`. An entry with no identifiers at all -> a reason saying so. The
clients themselves (`clients.py`) parse each API's dialect into one `Work` dataclass:
arXiv's Atom XML, OpenAlex's inverted abstract index (rebuilt into plain text), Crossref's
JATS-tagged abstracts (tags stripped), S2's externalIds/openAccessPdf.

### 3.5 Full-text retrieval internals (`src/auditor/retrieval/fulltext.py`)

`retrieve_source` walks the levels best first and stops at the first hit; the level used
is recorded on the `SourceText`, because the ablation and the evaluation need to know what
the agent actually read:

1. **full_html**: for arXiv works, `https://arxiv.org/html/<id>` (the ar5iv-style
   rendering); HTML is reduced to text with the standard library only (`html.parser`;
   block tags become line breaks, `script`/`style` dropped, whitespace collapsed,
   `&nbsp;` included). Pages whose text is under 2000 characters are treated as "no HTML
   rendering" stubs.
2. **full_pdf**: `work.pdf_url` -> `http.download_to` (50 MB cap) into
   `scratch_dir/citation/<sha>.pdf` -> Part 1's `parser.pipeline.parse_pdf` (GROBID must
   be running, `docker compose up -d`) -> the sentences joined with newlines.
3. **abstract**: the abstract the lookup returned; `reason="no full text was available"`.
4. **none**: nothing retrievable; `reason` says so and the check returns UNVERIFIABLE.

`abstract_only=True` (the ablation, `AUDITOR_CITATION_ABSTRACT_ONLY=1`) skips levels 1-2
entirely. Retrieved text is cached on disk under `cache_dir/citation/<sha24>.txt`, so
repeated runs of the same paper do not re-download anything.

### 3.6 Passage ranking (`src/auditor/retrieval/passages.py`)

`chunk_text` splits a source into ~1200-character chunks with 200 characters of overlap,
breaking at whitespace so sentences stay whole; every `Passage` carries the exact
`char_start`/`char_end` in the source. `rank_passages` scores the chunks with a
pure-Python Okapi BM25 (k1=1.5, b=0.75) against the claim's tokens — no embedding model,
so it runs on Windows, macOS and Linux unchanged (decision P2.4) — and returns the top
passages, deterministic (ties keep text order).


### 3.7 Set A — HALLMARK (`agents/citation/hallmark.py`, `scripts/hallmark.py`)

HALLMARK (arXiv 2607.18360, data at github.com/rpatrik96/hallmark, MIT) labels BibTeX
entries VALID or HALLUCINATED across 14 hallucination types. This is set A, the sanity
check of the citation lookup (D15). Flow:

1. `scripts/hallmark.py download --split test_public --out data/hallmark` (network) —
   fetches the split JSONL from the benchmark's repository.
2. `load_split` drops the benchmark's `__canary__` contamination-detector entries and
   rejects duplicate `bibtex_key`s.
3. `bib_entry_for` maps the record's BibTeX fields to a `BibEntry` and renders `raw` from
   the same fields (D1: raw and fields never disagree); `paper_for` builds a one-entry
   `ParsedPaper` per record.
4. `scripts/hallmark.py run` (network): `check_entry` runs the agent's real reference
   check on each one-entry paper and records `Flag` = clean / reference_metadata_mismatch
   / reference_not_found / unverifiable. Results go to a JSONL file.
5. `scripts/hallmark.py score` (offline): accuracy, flagging precision/recall/F1, the
   false-positive rate on the VALID entries (the number that limits deployment, since
   every flag needs human review), and a per-hallucination-type breakdown.

### 3.8 Set B — claim-source pairs (`agents/citation/set_b.py`, `scripts/set_b.py`)

Set B is the main test of the support judgement: ~600 pairs from 60 recent arXiv papers.
Errors are injected into copies of the `ParsedPaper` (D1), the originals are kept, and
every injected error and every sampled clean pair gets a `GoldLabel` (D3/D18). Four
injection kinds map to the four citation error classes:

| CLI `--kind` | Function | Edits | Gold class |
|---|---|---|---|
| `number` | `inject_number` | a number span in the claim sentence | claim_number_mismatch |
| `claim` | `inject_claim` | the claim text itself (human-authored rewrite) | claim_not_supported |
| `metadata` | `inject_metadata` | one bib field (title/venue/year/doi) | reference_metadata_mismatch |
| `not_found` | `inject_not_found` | title replaced, doi+arxiv_id cleared | reference_not_found |

Every injection returns `(injected ParsedPaper, GoldLabel, InjectionRecord)`:

- The sentence edits shift later citation markers with the text and mark the copy
  `variant=injected` (same mechanics as set C's `inject_paper`), so the copy still
  validates.
- Bibliography edits also rewrite the entry's `raw` text (`_replace_in_raw`,
  case-insensitive because GROBID's raw text drifts in case), so raw and fields never
  disagree — a mismatch would give the injection away (D1). Entries without a `raw` just
  get the field edited.
- The **gold evidence passage**: `gold_passage_for(source_text, claim)` is the top-1 BM25
  passage of the cited work's text for the *original* (uninjected) claim. It is stored in
  the `InjectionRecord` with its exact char offsets and the source text's sha256 — this
  is the Recall@5 gold.
- Clean controls: `clean_label` records a `none` error on the original variant.

The D18 validation (`validate(labels, records, papers=, sources=)`) refuses to freeze
while any of these hold: duplicate label ids; an error label without an injection record
or with a record whose kind disagrees with the label's class; a gold passage that does
not appear in the recorded source text; a clean control sitting on a sentence an
injection edits; injected text missing from the injected paper or from the edited `raw`;
fewer than 20% of labels second-reviewed (`reviewed_by`).

Scoring (`score_citation_module`, same shape as set C's scorer): one-to-one matching of
positive predictions to labels (D3), per-class P/R/F1 over the four citation classes,
macro-F1 over the classes the labels contain, the false-positive rate on the clean
controls, and the count of unlabelled flags.

### 3.9 Module evaluation (`agents/citation/eval.py`, `scripts/citation_eval.py`)

- `recall_at_k(source_text, claim, gold_passage, k=5)`: whether the gold passage is
  inside one of the top-k passages the retriever ranks for the claim
  (whitespace-normalised containment, so chunk-boundary drift between build time and
  eval time does not matter). This is the retrieval number the support judgement rests
  on.
- `tool_call_accuracy(outcomes, golds)`: the share of labelled reference lookups that
  came out right (D8's tool-call accuracy) — found/not-found must match the label, and
  titles compared with the resolver's own `titles_match`.
- `scripts/citation_eval.py fetch` (network): resolves each of a paper's bibliography
  entries and writes the text retrieved at the best level available to a JSONL file, so
  the offline commands never need the network again.
- `score --abstract-only`: scores the run made with `AUDITOR_CITATION_ABSTRACT_ONLY=1`
  next to the full-text run — the ablation comparison.

## 4. Configuration

`AUDITOR_*` env / `.env` (src/auditor/config.py). This part reads:

- `citation_llm_enabled` (default True) — the planner's model pass over citation
  candidates; turning it off makes the planner use the rule-based candidates only.
- `citation_abstract_only` (default False) — the ablation: `check()` never fetches a
  cited work's full text.
- The shared knobs: `grobid_url` (default `http://localhost:8070`; the compose service
  answers there), allowed hosts (the four scholarly APIs are pre-allowlisted), per-host
  rate limits (arXiv 3 s, Crossref 0.2 s, OpenAlex 0.1 s, S2 1 s), HTTP cache TTL.


## 5. Capabilities

- Resolves references through four scholarly indexes with identifier-first fallbacks, and
  says exactly what was queried and why nothing was found.
- Recovers unlinked author-year markers from the bibliography without any network call.
- Reads cited works at the best level available (arXiv HTML, PDF via GROBID, abstract),
  records the level, and caches the text on disk.
- Ranks passages with exact character offsets, so quotes are traceable to their place in
  the source.
- Judges claim support with one cheap model call, then checks the answer in code: the
  quote must be verbatim (its offsets are recorded), and every number is compared by
  value with a 1% tolerance — the model's own "agrees" cannot hide a real mismatch.
- Runs reference checks with no model at all (deterministic, cheap, reproducible).
- Serves B1 with the same three capabilities, each call counted for the cost/step
  accounting (D8).
- Ships tested machinery for all three evaluation sets it owns: a HALLMARK loader +
  existence-check scorer, set B injections with gold labels and gold evidence passages +
  D18 validation + module scorer, and Recall@5 / tool-call accuracy.
- Works offline in tests (httpx.MockTransport; the whole retrieval stack is testable
  without a network).

## 6. Shortcomings and limitations

1. **`extract_numbers` does not read scientific notation**: "3e-4" becomes the digits
   "3" and "4". The value comparison then cannot confirm such a number (it finds nothing
   to agree with). Learning-rate claims in set B may therefore see weak number checks.
2. **The number check needs the model to find the value**: a `NumberCheck` with
   `found=None` makes no finding, even though the number may be wrong — the code never
   scans the source text for the claimed value itself. A recall gap by design (a
   code-side confirmation was preferred over trusting the model).
3. **The metadata comparison is shallow**: title token-overlap >= 0.7, exact year,
   case-insensitive venue. A venue written differently ("NeurIPS" vs "Advances in Neural
   Information Processing Systems") is reported as a mismatch, and authors/DOI are not
   compared in the mismatch report (authors are only used for resolution matching).
4. **Full text is effectively arXiv-only**: the HTML level is the arXiv rendering; other
   open-access venues go through the PDF path, which needs GROBID; closed-access works
   fall back to the abstract or nothing, where the support judgement is usually
   `insufficient`.
5. **Retrieval is BM25 over one work's text** — no embeddings, no cross-work search. If
   the gold passage shares few tokens with the claim (paraphrases), Recall@5 suffers.
6. **Unlinked numeric markers cannot be resolved** (no author names to search) -> those
   citation tasks return UNVERIFIABLE with a reason rather than a finding (D6; kept
   deliberately, revisit against set B).
7. **HALLMARK evaluation covers the existence check only** — the benchmark has no
   claim-support labels, so it never exercises the support judgement.
8. **Set B's `not_found` injection is validated against the paper, not against the live
   indexes**: a fabricated title that happens to collide with a real work would still
   resolve; the D18 `validate` does not re-query the network.
9. **`citation_abstract_only` is wired into `check()` only** — the B1 tools take
   `abstract_only` per call and ignore the setting, so the ablation covers the agent but
   not a B1 run.
10. **arXiv's 3-second rate limit** makes full-text retrieval of many arXiv works slow
    (sequential, uncached misses); a 600-pair build fetches hundreds of sources.
11. The support judgement needs a configured LLM; without one every citation task
    returns UNVERIFIABLE (reference tasks still work — they need no model).


## 7. What is left to do (owner: Anurag)

The code and its offline tests are done and green (ruff, mypy --strict, 1141 tests pass,
both commits pushed to `origin/p2-citation-checker`). What remains is running the real
datasets and publishing the numbers:

1. **Set A (HALLMARK, network only, ~10 minutes of compute):** commands in section 8 step
   10 and below.
2. **Set B (network + GROBID + curation):** the full build, section 8.
3. **Module evaluation** once set B exists (full-text vs abstract-only, Recall@5,
   tool-call accuracy): `scripts/citation_eval.py`.
4. **Open the PR** for `p2-citation-checker` when the numbers are in.

## 8. Building set B, step by step

Run everything from the repository root with the project venv
(`.venv\Scripts\python.exe` or `uv run`); `uv` works in your own shell.

1. **Start GROBID** (needs Docker): `docker compose up -d` — the compose file starts only
   the `grobid` service (grobid/grobid:0.9.1-crf on 127.0.0.1:8070, healthchecked).
   `docker compose ps` shows when it is healthy; the first pull is ~1 GB.
2. **Curate the 60 papers**: write a manifest of versioned arXiv ids, one per line, same
   format as `datasets/parser_fixtures/arxiv_ids.txt` (`<id>v<n>`), e.g.
   `data/set_b/arxiv_ids.txt`. That file holds the parser's 13 papers and shows the
   format; pick 60 recent cs.LG/cs.CL papers with dense citation claims. The choice of
   papers is yours — the machinery does not pick them.
3. **Download the PDFs** (arXiv's 3 s/request limit applies; 60 PDFs ~ 3 minutes):
   `uv run python scripts/fetch_arxiv.py data/set_b/arxiv_ids.txt`
   (writes `data/papers/<id>.pdf`).
4. **Parse them** (GROBID running), one command per PDF, loopable:
   `uv run auditor parse data/papers/<id>.pdf -o data/set_b/papers/<id>.json`
   (the parse cache makes a re-run cheap; `--refresh` forces a re-parse).
5. **Fetch the cited works' texts** (network; the HTTP cache makes a re-run cheap):
   `uv run python scripts/citation_eval.py fetch data/set_b/papers/<id>.json --out data/set_b/sources/<id>.jsonl`
   One JSON line per bibliography entry: `{paper_id, bib_id, level, source_text}`.
6. **List the claim-source pairs and pick what to inject**:
   `uv run python scripts/set_b.py candidates data/set_b/papers/<id>.json --source-text data/set_b/sources/<id>.jsonl`
   prints one line per claim with the gold passage. Choose ~10 pairs per paper (~600 over
   60 papers): which to inject, with which kind and value, and which to keep as clean
   controls.
7. **Inject** (one command per injection; appends to the dataset files):
   `uv run python scripts/set_b.py inject data/set_b/papers/<id>.json --sentence s12 --start 30 --end 34 --kind number --value 88.4 --bib b3 --rationale "the source reports 88.4" --evidence evidence.json --source-text data/set_b/sources/<id>.jsonl --out-dir datasets/set_b`
   - `--kind claim` takes the rewritten claim in `--value`;
   - `--kind metadata` takes `--field title|venue|year|doi` plus `--value`;
   - `--kind not_found` takes the fabricated title in `--value`;
   - `--evidence` is a JSON file of `Evidence` records (the source quote the label rests
     on; the same shape set C uses).
   Clean controls are labels with `error_class: "none"` on the ORIGINAL paper — nothing
   is edited for them, so append them to `datasets/set_b/labels.jsonl` directly (one
   `GoldLabel` line each, anchor = the claim's `SentenceAnchor`, target = the
   `BibTarget`), e.g. with a two-line Python snippet calling
   `auditor.agents.citation.set_b.clean_label`.
8. **Second-review 20% of the labels** (D18): read a sample, add
   `"reviewed_by": ["anurag"]` to the reviewed label lines (both injected and clean).
9. **Validate and freeze**:
   `uv run python scripts/set_b.py validate datasets/set_b --sources data/set_b/sources.jsonl`
   (exit 0 means it can be frozen; fix every reported problem first).
10. **Run and score the module**:
    - full text: `uv run auditor batch --experiment setb --systems b1 datasets/set_b/dataset.jsonl`
    - abstract-only: the same with `AUDITOR_CITATION_ABSTRACT_ONLY=1` (or set it in `.env`);
    - `uv run python scripts/citation_eval.py score --labels datasets/set_b/labels.jsonl --run runs/<run_id>`
      for both runs (add `--abstract-only` to the second);
    - `uv run python scripts/citation_eval.py recall --injections datasets/set_b/injections.jsonl --sources data/set_b/sources.jsonl`
      for Recall@5;
    - `uv run python scripts/citation_eval.py tool-accuracy --file lookups.jsonl --gold gold.jsonl`
      for tool-call accuracy (the lookups come from the run's `tool_calls.jsonl` or a
      dedicated HALLMARK-style labelling).

The dataset file for `auditor batch` is a JSONL of `DatasetEntry` records — one line per
injected copy and per kept original (D14 runs the unchanged originals too):
`{"paper": "papers/<id>-injected.json", "venue": null}` per line, paths relative to the
file.

And the HALLMARK run (set A), for completeness:

```text
uv run python scripts/hallmark.py download --split test_public --out data/hallmark
uv run python scripts/hallmark.py run --file data/hallmark/test_public.jsonl --out runs/hallmark/results.jsonl
uv run python scripts/hallmark.py score --results runs/hallmark/results.jsonl
```

