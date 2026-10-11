# Part 1 Plan: Parsing, Planning, Orchestration, Infrastructure, Baselines

Owner: Bhaskar. Task ids refer to [team_plan.md](team_plan.md); decision ids refer to
[decisions.md](decisions.md).

Part 1 is on everyone's critical path, so the order below puts what the others need first:
schemas, then shared infrastructure, then an orchestrator that runs on mocks. Parser and
planner quality work comes after that.

---

## Working rules

- Every milestone ends with passing `ruff check`, `ruff format --check`, `mypy` and
  `pytest` locally, and a green CI run on Linux, Windows and macOS.
- Tests that need GROBID, the network or a paid LLM are marked `grobid`, `network` or
  `llm`. They are skipped by default and run explicitly (`uv run pytest -m grobid`).
- Library APIs are checked against current docs (LangChain docs MCP, DeepWiki) before use,
  not written from memory. Versions are pinned through `uv.lock`.
- Small commits, one logical change each, with messages in the form `area: what changed`
  (for example `schemas: add AuditTask and CandidateFinding`).
- No placeholder code. A module lands when it works and is tested.

---

## M0. Repository setup (P1.1)

Done in the setup PR:

- uv project with `src/auditor` layout, `uv.lock`, Python 3.13 pinned in `.python-version`.
- ruff (lint and format, explicit UTF-8 encoding enforced), mypy strict with the Pydantic
  plugin, pytest with markers, pre-commit hooks.
- GitHub Actions: lint and types on Linux; tests on Linux, Windows and macOS.
- `.gitattributes` (LF everywhere), `.editorconfig`, `.env.example`, `compose.yaml` for
  GROBID.
- `auditor.config.Settings` (pydantic-settings, `AUDITOR_*` variables and `.env`).
- README setup instructions for all three operating systems.

Verified: the lock file resolves with both uv 0.9 and uv 0.12; the test suite passes on
Windows natively and on Linux in Docker; the GROBID container starts and parses a paper.

## M1. Schemas v0 (P1.2) — target: day 2

Models in `auditor.schemas` (frozen, `extra="forbid"`, `schema_version` on top-level records):

- `ParsedPaper`: `paper_id`, `source` (PDF hash, parser name and version), `metadata`,
  `sections[]`, `sentences[]` (id, section id, text, character offsets, page and boxes),
  `bibliography[]` (id, raw string, title, authors, year, venue, DOI, arXiv ID),
  `citation_mentions[]` (sentence id, character span, marker text, linked bibliography ids,
  link confidence), `links[]` (URLs with page), `layout` (page count, page sizes, font
  statistics, margins).
- `SourceLocation`: page, boxes, section id, sentence id, character span.
- `Anchor`: a discriminated union of `sentence` (sentence id, optional character span),
  `bibliography` (entry id), `page` (page numbers) and `document` (D3).
- `ExtractedSetting`: key, raw value, normalised value, unit, location (paper or repository
  file and line).
- `VenueProfile`, `VenueRule`.
- `AuditTask`: deterministic `task_id`, `paper_id`, `kind` (reference, citation, code,
  template), claim text, `anchor`, target (bibliography id or unlinked marker text, setting
  keys, or rule id), context.
- `Evidence`: kind (quote, code, rule), source identity, exact text, location.
- `TaskResult`: task id, outcome (`checked`, `unverifiable`, `failed`), reason, candidate
  findings (D6).
- `CandidateFinding`, `VerifiedFinding` with `VerificationState` (`VERIFIED_FLAG`,
  `NEEDS_HUMAN_REVIEW`, `DROPPED`).
- `ErrorClass` enum (D4), `Prediction` and `GoldLabel` (both with paper variant and
  `Anchor`), `RunManifest`.

Also:

- Example JSON fixtures for each model under `tests/fixtures/schemas/`, so other parts can
  build against them on day 2.
- A script that exports JSON Schema files, so the contracts can be read without Python.

Tests:

- Every fixture validates and round-trips through JSON unchanged, including one of each
  anchor kind and each task outcome.
- Unknown fields are rejected.
- Task ids are stable across runs and processes.
- Enums match the class table in D4.

## M2. Run context, logging, HTTP gate, LLM factory (P1.3–P1.5)

`RunContext` holds settings, a logger bound to `run_id`/`paper_id`/`task_id`, the HTTP
client, the LLM factory and a per-task budget (tool calls and LLM calls). Exceeding a budget
raises a specific error that the orchestrator records as a failed task.

Tool calls are counted by a decorator that the shared tool functions (D7) use, so local
tools such as reading a cloned repository are counted too. Each record holds the tool name,
task id, an argument hash, duration and outcome. This is separate from the HTTP request log,
and it is what the B1 tool budget and the tool-call metrics use.

HTTP gate (`auditor.infra.http`), built on `httpx.AsyncClient`:

- Host allowlist from config; any other host raises before a request is sent.
- Per-host async rate limiter (token bucket), defaults from D8.
- On-disk response cache for GET requests, keyed by URL and relevant headers. hishel is
  evaluated first; if it does not handle the scholarly APIs' cache headers well, a small
  SQLite cache with an explicit TTL is used instead.
- Retries with exponential backoff and jitter on 429, 5xx and connection errors only;
  `Retry-After` is respected.
- A request log record per request (host, path, status, retries, cache hit, duration,
  task id).
- A contact address from config (`AUDITOR_CONTACT_EMAIL`) in the User-Agent, and as
  `mailto` for Crossref.

LLM factory (`auditor.infra.llm`):

- Builds chat models with `langchain.chat_models.init_chat_model` from config (provider,
  model ID, temperature, timeout, retries). The provider is pending (D9), so the first
  version supports Gemini and an OpenAI-compatible endpoint, and is tested with fake models.
- A `structured()` helper: picks the structured-output method that the provider supports,
  validates against the Pydantic model, retries a bounded number of times on parse
  failures, and returns the parsed object together with token usage.
- Token usage collected with `UsageMetadataCallbackHandler` and recorded per task.
  Cache hits are counted separately, because LangChain reports the original token counts on
  a hit.
- LLM cache switch (D10).
- A small probe script that checks tool calling and each structured-output mode against a
  real provider, to support the provider decision.

Tests:

- HTTP: `httpx.MockTransport` for allowlist, retries, `Retry-After`, caching and request
  log; the rate limiter with a fake clock.
- Tool-call counting for both a remote and a local tool, and budget enforcement.
- LLM: a fake chat model that supports `bind_tools`, used for the structured-output helper,
  retries and usage accounting.

## M3. Orchestration graph with mocks (P1.6) — target: end of week 1

LangGraph `StateGraph` (v1.2):

- State is a `TypedDict` whose values are Pydantic models; the list of `TaskResult`s uses an
  `operator.add` reducer.
- The planner node is followed by a fan-out through `Send`, one branch per `AuditTask`,
  routed by task kind to the citation, code and template nodes.
- Each specialist node wraps a `check(task, ctx)` function from a registry (D6). Mocks are
  registered by default until the real modules land.
- Failure isolation lives inside the node wrapper, not in LangGraph's node policies. In the
  pinned LangGraph 1.2.14, an exception that leaves one `Send` branch still aborts the whole
  run when other branches run in parallel, even though the node's `error_handler` has
  already recorded the failure (reproduced locally with sync and async nodes; a single
  branch is handled correctly). So the wrapper:
  - retries transient errors itself (bounded attempts, exponential backoff with jitter);
  - applies a per-attempt timeout with `asyncio.timeout`;
  - converts any error left after that, including budget errors, into a `failed`
    `TaskResult`, so no exception ever leaves a specialist node.
  A regression test pins the upstream behaviour, so the workaround is revisited only when a
  LangGraph upgrade makes that test fail.
- Specialist nodes connect to the verifier with separate edges, so a paper with no code
  tasks still reaches the verifier.
- Checkpointing with `AsyncSqliteSaver` (package `langgraph-checkpoint-sqlite`), because the
  graph is async and the sync `SqliteSaver` raises `NotImplementedError` from its async
  methods. The connection stays open for the whole run.
- Thread id `{run_id}/{system}/{paper_id}/{variant}` (D10): re-invoking an existing thread
  continues its state and the reducer would add the old results to the new ones, so a
  thread id is reused only to resume the same interrupted execution.
- `max_concurrency` is set at the top level of the run config.

Tests:

- Mixed task kinds all reach the verifier.
- No tasks of one kind.
- One task raising, timing out or exceeding its budget, while other tasks run in parallel,
  yields a `failed` result and the rest complete.
- A transient error is retried; a validation error is not.
- A run interrupted mid-way resumes on the same thread without re-running finished tasks.
- Two executions with different thread ids do not share results.
- Regression test for the upstream fan-out behaviour described above.
- The concurrency cap is respected.
- A paper with zero tasks finishes cleanly.

## M4. Parser (P1.7–P1.10)

GROBID client (`auditor.parser.grobid`):

- `processFulltextDocument` with `segmentSentences=1`, `teiCoordinates` for `s`, `ref`,
  `biblStruct`, `head` and `figure`, and `includeRawCitations=1`.
- `consolidateCitations=0`: reference resolution belongs to Part 2, which avoids duplicate
  Crossref lookups.
- Retries on 503. Timeouts are handled. 204 and 500 responses become explicit parse errors.

TEI conversion (`auditor.parser.tei`), written with lxml:

- Sections, sentences with character offsets and boxes, bibliography entries, and citation
  mentions resolved through `target="#bN"`.
- Unresolved markers are kept with `linked=[]` instead of being dropped.

Layout (`auditor.parser.layout`), with pdfplumber:

- Page count and sizes.
- Font size distribution of body text.
- Text-bounding margins per page.
- Link annotations: GitHub URLs with their page.

Fallbacks:

- If GROBID resolves too few markers, a numeric and author-year linker over the bibliography
  fills the gaps, with lower link confidence.
- If GROBID fails entirely, the paper is marked unparseable with the reason. A partial
  result is never presented as complete.

Cache: parsed output is stored under `data/cache/parsed/<sha256>-<parser version>.json`.

CLI: `auditor parse paper.pdf -o paper.json`.

Fixtures:

- About 10 recent arXiv ML papers, with at least two using author-year citations.
- Each is fetched by a script from a manifest (D13).
- Hand-check the bibliography and citation links on a sample of each.
- Commit trimmed `ParsedPaper` JSON fixtures, preferring CC-BY papers.

Tests:

- TEI conversion on small handwritten TEI snippets (ranges, author-year, unresolved
  markers, multi-column coordinates) and on recorded real GROBID output.
- Layout extraction on the proposal PDF in this repository.
- Live GROBID tests are marked `grobid`.

Metrics for the write-up:

- Citation-link precision and recall against the hand-checked sample.
- Bibliography field accuracy.
- Share of sentences with valid anchors.

## M5. Planner (P1.11–P1.13)

Deterministic pass:

- Every sentence with a citation mention gives one candidate claim per linked bibliography
  entry. A mention the parser could not link still gives a candidate, carrying the marker
  text as its target, so it ends up checked or `unverifiable` instead of disappearing (D6).
- Numbers with units and experimental settings are found with patterns: learning rate,
  batch size, epochs, optimiser, weight decay, dropout, warmup, seeds, splits, dataset
  names, metrics. Hyperparameter tables are included once table extraction is added.
- Repository links come from layout links and text.

LLM pass (structured output):

- Filter claims to those worth checking (background statements that make a factual
  assertion about the cited work).
- Split sentences that cite several works into per-citation claims.
- Normalise settings.
- The returned claim text must be a substring of the anchored sentence; anything else is
  rejected and logged.

Task generation:

- Reference tasks, one per bibliography entry, for existence and metadata
  (`bibliography` anchor).
- Citation tasks per claim (`sentence` anchor).
- Code tasks per setting group when a repository URL exists (`sentence` anchor of the
  sentence that states the setting).
- Template tasks per venue rule (`page` or `document` anchor).
- `task_id` is a hash of paper id, kind, anchor and target, so ids are stable across runs.

Tests:

- Deterministic extraction on fixture papers against expected outputs.
- Unlinked citation mentions still produce tasks.
- LLM pass with a fake model, including a model returning text that is not in the paper.
- Task-id stability.

## M6. End-to-end command, B0, B1 (P1.14–P1.16)

- `auditor audit <pdf | parsed.json> [--repo URL] [--venue NAME]` runs the full pipeline and
  writes task results, candidate and verified findings, predictions and the run manifest
  under `runs/`.
- B0: one model call with the full paper text, where sentences, bibliography entries and
  pages are shown with their ids, and instructions to list errors in the `Prediction`
  format with an anchor of the right kind (D3). No tools.
- B1: `langchain.agents.create_agent` with the registered tools from Parts 2 and 3,
  a structured response in the `Prediction` format, and call-limit middleware to enforce a
  single pass with a fixed tool budget. No verifier.
- Both baselines go through the same LLM factory, tool-call counting and request log, so
  cost is comparable.

Tests:

- B0 and B1 with fake models and mock tools produce valid predictions.
- B1 stops at its tool budget.
- `auditor audit` runs end to end on a fixture paper with mock specialists.

## M7. Analysis (P1.17)

- Parser failures by category: unresolved markers, missing bibliography fields, broken
  sentences.
- Planner misses against gold labels: errors injected into sentences that produced no
  task.
- Orchestration statistics: failed tasks, retries, cost per paper.

---

## Risks and how they are handled

| Risk | Handling |
|---|---|
| GROBID misses author-year links or garbles references in some templates | Fallback linker, link confidence on every mention, parser metrics reported honestly |
| LLM provider decision is late | Provider-agnostic factory, fake models in tests, probe script ready |
| A provider's structured output is unreliable | Validation and bounded retries in one helper, raw output logged on failure |
| An exception escaping one parallel branch aborts the whole fan-out (LangGraph 1.2.14, reproduced) | Specialist wrapper does its own retries and timeouts and never lets an exception escape; regression test tracks the upstream behaviour |
| Large fan-out slows LangGraph (hundreds of tasks per paper) | One graph thread per paper and variant; batch tasks per node if a paper exceeds a few hundred tasks |
| SQLite checkpoint contention under concurrency (`AsyncSqliteSaver`) | One writer per run; tune durability; resume tested |
| Schema churn breaks other parts | Versioned schemas, small dedicated PRs, contract tests in CI |
| Windows or macOS-only bugs | CI on all three operating systems; pathlib and UTF-8 enforced by lint |
