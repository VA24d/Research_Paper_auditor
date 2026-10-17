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
  citation targets, bibliography fields, or extracted settings. An edit to a bibliography
  field (authors, title, year, venue, identifiers) also edits the entry's `raw` text, so the
  two never disagree: a mismatch between them would give the injection away.
- B0 and B1 are shown every bibliography field the specialists can use, with all authors.
- The CLI accepts either a PDF (parsed on the fly, cached) or a `ParsedPaper` JSON file.

## D2. Two evaluation levels — Agreed

| Level | Input | Used for |
|---|---|---|
| Paper level (end to end) | Injected `ParsedPaper` files, plus the unchanged original of every paper | RQ1 numbers for B0, B1 and the full system; false alarms on clean papers (D14) |
| Component level | Hand-built `AuditTask` / `CandidateFinding` fixtures, and hand-annotated parser fixtures | Module metrics for each part (Recall@5, parser extraction accuracy, verifier decision accuracy, template rule accuracy) |

## D3. Gold labels and how predictions are matched to them — Agreed

Every injected error and every sampled clean item gets a `GoldLabel`:

- `paper_id`, `variant` (`injected` or `original`), `label_id`
- `error_class` (D4), or `none` for clean control items
- `anchor`: where the error is (see below)
- `target`: what the error is about — a bibliography entry id, a setting key, or a rule id

Not every finding sits in a sentence (a page-limit violation or a missing section does not),
so an anchor is one of four kinds:

| Anchor kind | Fields | Used for | Matches when |
|---|---|---|---|
| `sentence` | sentence id, optional character span | Claims, numbers and settings stated in the text | Same sentence id; if both sides give spans, the spans overlap |
| `bibliography` | bibliography entry id | `reference_not_found`, `reference_metadata_mismatch` | Same entry id |
| `page` | page numbers | Layout rules such as margins or font size | Page sets overlap |
| `document` | none | Document-level rules such as the page limit or a missing required section | Always (the target decides) |

Every system (B0, B1, full) emits `Prediction` records with the same `anchor` and `target`
fields. A prediction matches a gold label when the paper, the variant, the target and the
anchor kind agree, and the anchors match as in the table. Matching is one-to-one; the error
class is then compared for per-class scores.

For headline metrics only `VERIFIED_FLAG` predictions count as positive. Items the full
system sends to `NEEDS_HUMAN_REVIEW` are reported separately as review load, not hidden.
B0 and B1 have no verifier, so all of their flags count as positive.

A prediction that matches no gold label is an unlabelled flag. It may be a false alarm or a
real error already present in the original paper, so it is not scored as a false positive.
Unlabelled flags are reported per paper, and a random sample is adjudicated by hand (D14).

Because of these rules, the planner must assign anchors from parser ids (sentence,
bibliography entry, page), never from free text produced by an LLM.

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
| Data | `dataset_size_mismatch` | Number of examples stated for a dataset or split differs from the data |
| Data | `label_distribution_mismatch` | Number of classes or class balance stated differs from the data |
| Data | `data_leakage` | Test or validation examples also appear in the training data |
| Code | `method_mismatch` | The code implements a different loss, algorithm step, model component, data processing, training or evaluation procedure than the paper describes |
| Theory | `math_error` | An equation, derivation or proof is wrong: a stated result does not follow, a proof step fails, or a formula is incorrect |
| Template | `template_violation` | A venue formatting rule is broken |

`none` marks clean control items and is used for the false-positive rate. Adding a class
means updating this table, the enum in `auditor.schemas`, and the dataset README.

The three data classes were added in October 2026. Their paper-side keys are
`train_examples`, `validation_examples`, `test_examples` and `total_examples`
(`dataset_size_mismatch`), `num_classes` and `class_balance`
(`label_distribution_mismatch`), and `test_overlap` (`data_leakage`). `test_overlap` is not
stated in the paper: the planner adds it for every paper that describes its data split or
dataset, anchored on that sentence, because presenting a test set means presenting it as
held out. Example and class counts can be checked against a Hugging Face dataset card;
class balance and leakage need the data files, so they are checked only in the opt-in data
analysis mode (`AUDITOR_DATA_ANALYSIS_ENABLED`). Without it, these tasks are unverifiable,
which counts as a miss (D6), so set C is run with data analysis on for every system, and B1
gets the same data tool (D7). Results are reported per class.

`method_mismatch` was added at the same time, so the code agent also covers the discrepancies
real reports are about (SciCoQA, ACL 2026). The planner adds one `method` code task per paper
with a repository, anchored on the first sentence of its method section. The code agent sends
the method text and the most related repository files to the LLM and keeps a discrepancy only
if it quotes the paper verbatim and points at existing code lines. Findings and gold labels
use a `document` anchor with target `method`: a method discrepancy is about the paper as a
whole, so matching is per paper (one to one), as in SciCoQA's own evaluation.

## D5. Shared schemas — Agreed

- All contracts live in `src/auditor/schemas/` as Pydantic v2 models, owned by Bhaskar.
- Models are immutable (`frozen=True`), reject unknown fields, and carry a `schema_version`.
- Every part depends on these models, so tell the team when you change one, and keep the
  contract tests passing.

Contracts:

| Model | Produced by | Consumed by |
|---|---|---|
| `ParsedPaper` (sections, sentences, bibliography, citation mentions, layout) | Parser (Part 1) | Planner, template checker, verifier, B0/B1 |
| `ExtractedSetting` (name, value, unit, normalised value, source location) | Parser/planner (paper side), code agent (repo side) | Code agent, verifier |
| `VenueProfile` / `VenueRule` | Template checker owner writes the rule files | Planner, template checker |
| `Anchor` (sentence, bibliography, page or document; D3) | Parser ids, used by the planner | Every model that points into the paper |
| `AuditTask` | Planner | Citation agent, code agent, template checker |
| `Evidence` (quote or file/line or rule, plus source identity) | Specialists | Verifier, report |
| `TaskResult` (outcome, reason, findings; D6) | Specialists, orchestrator | Verifier, report, evaluation |
| `CandidateFinding` | Specialists | Verifier |
| `VerifiedFinding` | Verifier | Report, evaluation |
| `Prediction` | Full system, B0, B1 | Evaluation |
| `GoldLabel` | Dataset builders | Evaluation |
| `RunManifest` | Orchestration | Evaluation, report |

## D6. Specialist interface hides LangGraph — Agreed

Each specialist is a plain async function:

```python
async def check(task: AuditTask, ctx: RunContext) -> TaskResult: ...
```

`RunContext` gives access to settings, the shared LLM factory, the shared HTTP client,
a logger and the task's budget. Specialists never import LangGraph; the orchestration layer
wraps them in graph nodes. This lets each part be built and tested on its own, and lets mock
specialists stand in until the real ones are ready.

A `TaskResult` holds the task id, exactly one outcome, a reason, and zero or more
`CandidateFinding`s:

| Outcome | Set by | Meaning |
|---|---|---|
| `checked` | Specialist | The check ran to completion. No findings means nothing wrong was found. |
| `unverifiable` | Specialist | The check could not be completed (source not available, repository too large, rule needs visual inspection). The reason is required. Findings may be attached if something suspicious was seen. |
| `failed` | Orchestrator | The specialist raised an exception after retries. The error is recorded. |

Rules:
- Missing or unreachable evidence is never an accusation and never disappears silently. It
  is an `unverifiable` result with a reason.
- The planner emits a task for every citation mention, including mentions the parser could
  not link to a bibliography entry, so unlinked citations end up as checked or unverifiable
  instead of vanishing.
- The report has three sections: verified flags; items needing human review (suspected
  issues the verifier could not settle); and items not checked (`unverifiable` or `failed`
  with no suspicion), listed with their reasons so a human can check them by hand.
- In evaluation, a gold error on a task that was not checked counts as a miss.

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
  provider; others must be added explicitly). An entry starting with "." allows the
  subdomains of a host, for CDNs that change by region (`.hf.co`);
- per-host rate limits (arXiv: 1 request every 3 seconds; Crossref: within polite-pool limits);
- an on-disk HTTP cache, always on;
- retries with backoff on 429 and 5xx only;
- a request log (host, status, retries, cache hits, duration, task id);
- per-task budgets (maximum tool calls and LLM calls), enforced through `RunContext`.

Repositories and data hosts are read through `auditor.infra.snapshots`, which pins the
revision, downloads only core files (code, configs, docs) within size limits, and never
executes repository code. Data files are downloaded only in data analysis mode, through
`HttpClient.download_to`, which streams to disk within a size limit, checks the expected
SHA-256 and never uses the HTTP cache.

HTTP requests are not tool calls: one tool can make several requests, and local tools (for
example reading a cloned repository) make none. Tool calls are counted separately by
instrumenting the shared tool functions (D7), local ones included, and both counts are
reported. Tool-call accuracy, which the proposal lists for the tool-calling capability, is
measured on labelled fixtures: the share of reference lookups that resolve to the gold work
(sets A and B) and of repository searches that reach the gold file (set C).

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
  `auditor batch` turns it off unless given `--llm-cache`.
- Every run writes a `RunManifest`: git commit, config, model ID, package versions, dataset
  version and seed (where the provider supports one).
- Reported numbers are the mean and range over three runs, as in the proposal.
- Every execution gets its own checkpoint identity: the LangGraph thread id is
  `{run_id}/{system}/{paper_id}/{variant}`. Invoking a graph again on an existing thread
  continues that thread's state, so findings from the earlier invocation would be added to
  the new ones. A thread id is reused only to resume the same interrupted execution.

## D11. Code is analysed statically only — Agreed

No repository code is executed anywhere in the system. The "optional run in sandbox" box in
the proposal diagram is dropped.

## D12. PDF parsing stack — Agreed

- GROBID 0.9.1 (Docker image `grobid/grobid:0.9.1-crf`) for structure, bibliography,
  sentence segmentation and in-text citation linking, with page coordinates.
- pdfplumber for page count, page sizes, fonts, margins and link annotations.
- GROBID's TEI output is cached by the PDF's SHA-256. The conversion to `ParsedPaper` and
  the citation linking run on every parse, so improvements to them apply to cached papers.
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

The proposal promises no increase in false alarms "on clean, correct papers", so that part
is measured on the unchanged original of every paper (D2), not only on clean items inside
injected papers. For each system we report:
- false-positive rate on the sampled clean control items (item level, from gold labels);
- flags per original paper, and the share of original papers with at least one flag;
- an adjudicated estimate: a random sample of flags on original papers is checked by hand
  (shared task S5), because an original paper can contain real errors. The estimated number
  of false alarms per paper is reported together with that caveat.

## D15. Audit item count and other corrections — Agreed

Sets B and C give 600 + 160 = 760 audit items. HALLMARK (set A) is a sanity check of the
citation lookup and is reported separately. Running the unchanged originals (D14) adds 100
paper runs per system and run, and the token budget is recomputed with them.

Corrections for the final write-up (the submitted proposal files stay unchanged):
- "about 960 audit items" becomes 760.
- "ICLR 2026 had over 60K submissions" is wrong. The official ICLR 2026 fact sheet reports
  19,525 submissions (11,603 in 2025):
  <https://media.iclr.cc/Conferences/ICLR2026/ICLR2026_Fact_Sheet.pdf>

## D16. Repository layout and tooling — Agreed

- Single package `auditor` in a `src/` layout.
- uv for environments and the lock file, Python 3.13.
- ruff (lint and format), mypy (strict), pytest, pre-commit.
- GitHub Actions runs lint and type checks, and tests on Linux, Windows and macOS for every PR.

Cross-platform rules, since the team uses all three systems:
- Use `pathlib`; never build paths by joining strings with `/` or `\`.
- Always pass `encoding="utf-8"` when reading or writing text (ruff enforces this).
- Pass `newline="\n"` when writing files that are hashed, compared byte for byte or committed
  (fixtures, caches, gold labels). Without it, Python writes `\r\n` on Windows.
- Write helper scripts in Python under `scripts/`, not shell or batch files.
- Docker is needed for GROBID. The `auditor` image in `compose.yaml` is optional; it pins
  the environment for the final runs, and `uv run` gives the same results without it.
- Line endings are normalised to LF by `.gitattributes`.

## D17. Branches and reviews — Agreed

- Everyone decides how they work. Nobody's approval or review is required, and none is
  requested automatically.
- PRs into `main` are the usual route because CI then runs on all three operating systems;
  merge your own once it passes. Ask someone to look at a PR only when you want a second
  opinion.
- Anyone may pick up a task from another part. Put your name on it in `docs/team_plan.md`
  first, so two people don't do the same work.

## D18. Gold labels are validated before they are frozen — Agreed

An injection log alone is not ground truth. A swapped citation can still support the claim,
and a changed configuration value can be irrelevant to the experiment the paper reports.
So every label is checked against its source before the evaluation data is frozen:

- **Clean controls.** The builder records the evidence that makes the item correct: the
  supporting passage in the cited work (this is also the gold passage for Recall@5), or the
  repository file and line that the reported experiment actually uses.
- **Injected errors.** After the mutation, the builder confirms against the source that the
  result really is an error, and records why. For example: the newly cited work does not
  contain the claim, or the paper's value now differs from the value the experiment's
  configuration uses. Mutations that turn out not to be errors are discarded.
- **Second review.** Another team member independently reviews a random sample of at least
  20% of each set's labels (all of them for sets under 100 items). Disagreements are
  resolved, the labelling rule is clarified if needed, and the agreement rate is reported.
- **Freeze.** The labels are then versioned and frozen. Any later change means re-running the
  affected experiments.

## D19. How parts plug into the pipeline — Agreed

Each part exposes fixed entry points, and the pipeline finds them by name
(`auditor.orchestration.registry`):

| Part | Entry point |
|---|---|
| Citation agent, code agent, template checker | `check(task, ctx) -> TaskResult` in `auditor.agents.<name>` |
| Verifier | `verify(results, ctx) -> list[VerifiedFinding]` in `auditor.agents.verifier` |
| Tools for B1 (D7) | `make_tools(ctx) -> list[BaseTool]` in `auditor.agents.<name>.tools` |

Until an entry point exists, a mock runs in its place (specialists check nothing; the
verifier sends every finding to human review). Every run manifest lists which parts were
mocked, so the evaluation runner can refuse runs that used a mock. An entry point that
exists but fails to import stops the run instead of falling back.

## D20. Duplicate findings are merged and findings are ranked after the verifier — Agreed

Specialists check one task at a time, so one problem can come back as several findings. A
paper that names its metric in two sentences gets two metric mismatches with the same
repository evidence. Running more checks also raises recall at the cost of precision, as
Google's Paper Assistant Tool reports (arXiv 2606.28277). PAT ends with a synthesis step
that removes repeats and checks severity. Ours runs after the verifier
(`auditor.orchestration.synthesis`):

- Two findings report the same problem when the paper, variant, error class, target and
  summary are equal and all evidence that does not come from the paper itself is equal.
  The first one stays. The others get `duplicate_of` set to its finding id. A finding whose
  only evidence is from the paper itself is never merged.
- Nothing is deleted, so matching against gold labels (D3) is unchanged. The report and
  the human review load count each group once.
- Every finding gets a `severity`:
  - `high`: the results or the submission are in doubt. This covers `reference_not_found`,
    `claim_not_supported`, `claim_number_mismatch`, `metric_mismatch`, `data_leakage`,
    `method_mismatch`, and the desk-reject template rules `max_pages` and `anonymity`.
  - `medium`: the work is harder to reproduce or trust. This covers `setting_mismatch`,
    `data_split_mismatch`, `dataset_size_mismatch` and `label_distribution_mismatch`.
  - `low`: a detail to correct. This covers `reference_metadata_mismatch` and the other
    template rules.
- Findings are ordered for the reader by verifier state (verified flags, then human
  review, then dropped), then by severity, with certain findings before uncertain ones.

## D21. The theory checker and the `math_error` class — Agreed

Google's Paper Assistant Tool shows that a model can find many of the proof and equation
errors behind real retractions. On the math and CS proof subset of SPOT (arXiv 2505.11855),
it reaches a recall of 89.7%, against 55.2% for a single model call (arXiv 2606.28277). The
auditor gains a fourth specialist, `auditor.agents.theory`, for the class `math_error` (D4).

The check is a cascade, so that deep thinking is spent only on claims that earn it. Each
call sets its own thinking level, and the usage log records the level and the stage.

1. **Plan** (`auditor.planner.theory`). One call to the main model with minimal thinking
   reads the paper and lists its formal claims. Each claim has a kind, a short statement,
   the sentences that state and argue it, the sentences it builds on, a risk level, and
   whether checking it takes real reasoning. Which claims count, and how risky they are,
   is the model's reading of the paper, not a list of keywords, so this works in any field.
   The claims become `theory` tasks of `AUDITOR_THEORY_BATCH_SIZE` (8) claims each, at most
   `AUDITOR_THEORY_MAX_TASKS` (6) of them. Over that budget, low-risk claims that only need
   reading are left out first.
2. **Screen.** One call per task with low thinking. It sees each claim's own sentences and
   the definitions it uses, and calls each claim fine, suspicious or needs reasoning.
3. **Deep check.** Suspicious claims, high-risk claims that need reasoning (a quick read
   cannot clear a delicate argument, so whatever the screen said), and medium-risk ones the
   screen could not settle are checked one at a time with high thinking, at most `AUDITOR_THEORY_DEEP_CHECKS` (2) per task, with the
   whole paper as context. The paper comes first in these prompts, so a provider's prompt
   cache can reuse it. "No error" is the default answer. An error needs a witness: the step
   that fails, a counterexample, or the correct value.
4. **Challenge.** One call per paper to the review model (`AUDITOR_LLM_REVIEW_MODEL`, or the
   main model when none is set), with medium thinking. It judges every candidate with the
   whole paper in view: real errors stay, repeats point at the first one, and the rest are
   dropped with its reason.

- **Evidence gate.** An error is kept only if it quotes the paper verbatim, at least six
  words long, ignoring spacing. The finding is anchored on the quoted sentence and span,
  so D3 sentence matching works for injected math errors. Confidence below high makes it
  uncertain.
- **Opt-in.** Theory checks cost model calls on every formal paper, so they run only with
  `AUDITOR_THEORY_CHECK_ENABLED`, which is part of the run fingerprint. B0 and B1 learn the
  class from the class guide. B1 gets no extra tools, since the paper is already in its
  prompt.
- **Evaluation.** SPOT's labels describe an error in words and give a rough location, with
  no sentence id. SPOT runs are therefore scored as the SPOT and PAT papers do it: an LLM
  judge decides whether a reported issue is the same error as the annotation. A person
  checks every judgement before the numbers are reported (`scripts/spot.py`).
