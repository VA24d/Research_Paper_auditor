# Design Decisions

Decisions the team has agreed on before implementation. Each one fixes a gap in the
original proposal or work split. If a decision needs to change, edit it here in its own
PR so everyone sees the change, and update the affected schemas and tests in the same PR.

Status legend: **Agreed**, **Pending** (needs a choice before the date shown).

---

## D1. The pipeline input is a parsed paper, not a PDF — Agreed

PDF parsing is stage 0 and produces a `ParsedPaper` (JSON-serialisable). Everything after
it (planner, specialists, verifier, B0, B1) reads a `ParsedPaper`.

Why: evaluation sets B and C are built by injecting errors. Injecting into a PDF means
re-rendering it, which is slow and lossy. Injecting into `ParsedPaper` JSON is exact, and it
means B0, B1 and the full system all read the same injected text. The parser is evaluated
separately on clean PDFs (D2).

Consequences:
- Dataset builders inject errors by editing a copy of a `ParsedPaper`: sentence text,
  citation targets, bibliography fields, or extracted settings.
- The CLI accepts either a PDF (parsed on the fly, cached) or a `ParsedPaper` JSON file.

## D2. Two evaluation levels — Agreed

| Level | Input | Used for |
|---|---|---|
| Paper level (end to end) | Injected `ParsedPaper` files | RQ1 numbers for B0, B1 and the full system |
| Component level | Hand-built `AuditTask` / `CandidateFinding` fixtures, and hand-annotated parser fixtures | Module metrics for each part (Recall@5, parser extraction accuracy, verifier decision accuracy, template rule accuracy) |

## D3. Gold labels and how predictions are matched to them — Agreed

Every injected error and every sampled clean item gets a `GoldLabel`:

- `paper_id`, `label_id`
- `error_class` (D4), or `none` for clean control items
- `anchor`: the sentence id in the `ParsedPaper` (or a character span within it)
- `target`: what the error is about — a bibliography entry id, a setting key, or a rule id

Every system (B0, B1, full) emits `Prediction` records with the same `anchor` and `target`
fields. A prediction matches a gold label when the paper, the target and the anchor agree
(same sentence id, or overlapping character spans). Matching is one-to-one.

For headline metrics only `VERIFIED_FLAG` predictions count as positive. Items the full
system sends to `NEEDS_HUMAN_REVIEW` are reported separately as review load, not hidden.
B0 and B1 have no verifier, so all of their flags count as positive.

Because of this rule, the planner must assign task anchors from parser sentence ids, never
from free text produced by an LLM.

## D4. Error classes — Agreed

Macro-F1 is computed over these classes (the ones present in a given evaluation set):

| Area | Class | Meaning |
|---|---|---|
| Citation | `reference_not_found` | Cited work cannot be found (likely fabricated) |
| Citation | `reference_metadata_mismatch` | Work exists but title, authors, year or venue are wrong |
| Citation | `claim_not_supported` | Cited source does not support, or contradicts, the claim |
| Citation | `claim_number_mismatch` | A number attributed to the source differs from the source |
| Code | `setting_mismatch` | Hyperparameter or configuration in the paper differs from the code |
| Code | `data_split_mismatch` | Dataset or split described differs from the code |
| Code | `metric_mismatch` | Metric or evaluation procedure described differs from the code |
| Template | `template_violation` | A venue formatting rule is broken |

`none` marks clean control items and is used for the false-positive rate. Adding a class
means updating this table, the enum in `auditor.schemas`, and the dataset README.

## D5. Shared schemas — Agreed

- All contracts live in `src/auditor/schemas/` as Pydantic v2 models, owned by Bhaskar.
- Models are immutable (`frozen=True`), reject unknown fields, and carry a `schema_version`.
- Changes go through a small dedicated PR reviewed by the schema owner and by the owner of
  every module that consumes the changed model. Contract tests must pass before merge.

Contracts:

| Model | Produced by | Consumed by |
|---|---|---|
| `ParsedPaper` (sections, sentences, bibliography, citation mentions, layout) | Parser (Part 1) | Planner, template checker, verifier, B0/B1 |
| `ExtractedSetting` (name, value, unit, normalised value, source location) | Parser/planner (paper side), code agent (repo side) | Code agent, verifier |
| `VenueProfile` / `VenueRule` | Template checker owner writes the rule files | Planner, template checker |
| `AuditTask` | Planner | Citation agent, code agent, template checker |
| `Evidence` (quote or file/line or rule, plus source identity) | Specialists | Verifier, report |
| `CandidateFinding` | Specialists | Verifier |
| `VerifiedFinding` | Verifier | Report, evaluation |
| `Prediction` | Full system, B0, B1 | Evaluation |
| `GoldLabel` | Dataset builders | Evaluation |
| `RunManifest` | Orchestration | Evaluation, report |

## D6. Specialist interface hides LangGraph — Agreed

Each specialist is a plain async function:

```python
async def check(task: AuditTask, ctx: RunContext) -> list[CandidateFinding]: ...
```

`RunContext` gives access to settings, the shared LLM factory, the shared HTTP client,
a logger and the task's budget. Specialists never import LangGraph; the orchestration layer
wraps them in graph nodes. This lets each part be built and tested on its own, and lets mock
specialists stand in until the real ones are ready.

Failure rules:
- Missing or unreachable evidence is not an error. Return a finding marked uncertain, or no
  finding. Never turn "could not find evidence" into an accusation.
- Unexpected exceptions are caught by the orchestrator, recorded as a failed task, and the
  verifier routes that task to `NEEDS_HUMAN_REVIEW`.

## D7. Tools for baseline B1 come from the specialist owners — Agreed

B1 must use the same tools as the full system. Each specialist owner exposes their
capabilities as plain async functions plus thin LangChain tool wrappers in
`auditor.agents.<name>.tools` (for example `find_reference`, `retrieve_passages`,
`search_repository`, `read_repository_file`, `check_template_rule`). Their own specialist
uses the same functions. Part 1 builds the B1 agent around whatever tools are registered.

## D8. All network access goes through one client (the policy gate) — Agreed

No module calls `httpx`, `requests` or `urllib` directly. Everything goes through
`auditor.infra.http`, which provides:

- a host allowlist (arXiv, OpenAlex, Crossref, Semantic Scholar, GitHub and the LLM
  provider; others must be added explicitly);
- per-host rate limits (arXiv: 1 request every 3 seconds; Crossref: within polite-pool limits);
- an on-disk HTTP cache, always on;
- retries with backoff on 429 and 5xx only;
- a call log, which gives the "tool calls per paper" metric for free;
- per-task budgets (maximum tool calls and LLM calls), enforced through `RunContext`.

Cloning GitHub repositories also goes through an `infra` helper, which pins the commit and
never executes repository code.

## D9. One LLM factory, provider chosen by config — Pending (decide by end of week 1)

- All model access goes through `auditor.infra.llm`. No module constructs a chat model itself.
- Every component uses the same model ID in reported runs, as RQ1 requires.
- Structured output goes through one helper that validates against the Pydantic model and
  retries on parse failures, because providers differ in what structured-output modes they
  support.
- Token usage is recorded for every call and rolled up per task, per paper and per run.

Provider options under consideration: Google Gemini (paid from Google Cloud credits) or Z.ai
GLM. The choice is made after a short probe of tool calling and structured output on the
candidate models. Code written before then must not depend on the provider.

Facts that affect the choice (checked October 2026):
- The Google Cloud $300 trial credit cannot pay for Gemini API keys created in AI Studio
  (accounts opened after 2 March 2026). It can pay for Gemini on Vertex AI, which
  authenticates with `gcloud auth application-default login` rather than an API key. The
  credit expires 90 days after sign-up.
- The newest Gemini models ignore `temperature`, `top_p` and `top_k`, and thinking cannot be
  switched off completely. Run-to-run variation is expected either way (see D10).
- GLM-5.3-Flash has no `seed`, cannot disable thinking, and does not support JSON-schema
  structured output.
- Rough cost for the planned runs (about 200M input tokens): roughly $80–110 on
  `gemini-3.1-flash-lite` or `gemini-3.5-flash-lite`, and roughly $225 on `gemini-3.8-flash`.
  These are estimates; the real numbers come from the probe and the first full runs.

## D10. Caching and repeated runs — Agreed

- The HTTP cache is always on.
- The LLM response cache is on during development and **off** for reported runs, otherwise
  repeated runs would be identical and the variance estimate would be meaningless.
- Every run writes a `RunManifest`: git commit, config, model ID, package versions, dataset
  version and seed (where the provider supports one).
- Reported numbers are the mean and range over three runs, as in the proposal.

## D11. Code is analysed statically only — Agreed

No repository code is executed anywhere in the system. The "optional run in sandbox" box in
the proposal diagram is dropped.

## D12. PDF parsing stack — Agreed

- GROBID 0.9.1 (Docker image `grobid/grobid:0.9.1-crf`) for structure, bibliography,
  sentence segmentation and in-text citation linking, with page coordinates.
- pdfplumber for page count, page sizes, fonts, margins and link annotations.
- Parsed output is cached by the PDF's SHA-256 and the parser version.
- `ParsedPaper` JSON fixtures are committed for tests, so only people working on the parser
  need Docker running.

## D13. Paper and repository files are not committed — Agreed

PDFs, LaTeX sources and cloned repositories stay out of git (size, and arXiv licence terms).
Datasets commit a manifest (arXiv ID or repository URL, commit, SHA-256) and a fetch script
downloads into the git-ignored `data/` directory. Small text fixtures committed for tests
should come from CC-BY papers where possible.

## D14. Which baseline answers RQ1 — Agreed

We report the full system against both baselines. B1 (same model, same tools, one pass, no
verifier) is the comparison that isolates the verification loop and the specialised roles,
so RQ1 is answered against B1. B0 shows the combined effect of tools and agents. The
"+10 points Macro-F1, no increase in false alarms" target from the proposal is checked
against both and reported as measured.

## D15. Audit item count — Agreed

Sets B and C give 600 + 160 = 760 audit items. HALLMARK (set A) is a sanity check of the
citation lookup and is reported separately. The "960 items" figure in the proposal is
corrected to 760 in the write-up and the token budget is recomputed.

## D16. Repository layout and tooling — Agreed

- Single package `auditor` in a `src/` layout.
- uv for environments and the lock file, Python 3.13.
- ruff (lint and format), mypy (strict), pytest, pre-commit.
- GitHub Actions runs lint and type checks, and tests on Linux, Windows and macOS for every PR.

Cross-platform rules, since the team uses all three systems:
- Use `pathlib`; never build paths by joining strings with `/` or `\`.
- Always pass `encoding="utf-8"` when reading or writing text (ruff enforces this).
- Write helper scripts in Python under `scripts/`, not shell or batch files.
- Docker is only needed for GROBID.
- Line endings are normalised to LF by `.gitattributes`.

## D17. Branches and reviews — Agreed

- Feature branches per part, merged into `main` through PRs.
- CI must pass, and at least one other member reviews.
- Schema changes go in their own small PRs (D5).
- Anyone may pick up a task from another part. Claim it on the task list in
  `docs/team_plan.md` first, and ask the part owner to review the PR.
