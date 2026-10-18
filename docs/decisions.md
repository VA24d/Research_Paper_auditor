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
- Amended October 2026: GROBID scrambles many table columns, so the parser also reads
  tables from Poppler's `pdftotext -layout` text (`auditor.parser.layout_tables`). These
  tables carry their page number. A page whose tables cannot be read that way stays in
  the `ParsedPaper` as a table with no rows, and so does every captioned page when
  `pdftotext` is not installed. A run started from a PDF keeps the PDF's path in the run
  context (`RunContext.source_pdf`), and the results check (D22) reads such pages from an
  image of the page. Injected `ParsedPaper` files are unchanged: they hold their tables
  as rows, and with no PDF there is nothing to look at.

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
| Integrity | `unfinished_or_placeholder_content` | The paper says its own results are placeholders, simulated or to be replaced, or it contains unfinished text, author notes or unresolved references |
| Code | `code_defect` | The released code would fail as published: a file the experiment uses does not compile, uses a name defined nowhere, or imports a local module the repository lacks |
| Integrity | `result_inconsistency` | What the paper says about its own results disagrees with its tables or with arithmetic: a number not in the table, a wrong difference or relative gain, a wrong claim of the best result, an impossible value |
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
- The LLM response cache is shared during development. In reported runs an answer is
  reused **only within the run that made it** (`AUDITOR_LLM_CACHE_SCOPE=run`, the default of
  `auditor batch`; `--llm-cache` shares answers across runs for development). Within a run,
  the original and injected copies of a paper ask many identical questions, and a resumed
  run asks the same questions again; answering each once does not change the result. Across
  the repeated runs nothing is shared, so the spread between them still measures the
  model's own variation, which is what this rule protects. (Changed in October 2026; the
  cache used to be off for reported runs.)
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

Reading code with tools is not running it. The code-defect check
(`auditor.agents.code.defects`, class `code_defect`) parses the Python files with
`compile()`, which builds the syntax tree and never executes anything. It also runs Ruff's
undefined-name rules (F821, F822, F823) on the files the experiment uses: the scripts with
an `if __name__ == "__main__":` block, and the local modules they import. It reports three
kinds of defect:

- files that do not compile;
- names that are used but defined nowhere;
- imports of a local package's module that the repository lacks.

Model-written code is different: it is the model's own computation, not the paper's. The
theory checker's deep checks may run one piece of Python in the math sandbox
(`auditor.infra.sandbox`, image built from `docker/math/Dockerfile`). The sandbox runs:

- with no network;
- with a read-only file system, except a 64 MB `/tmp`;
- as an unprivileged user with no capabilities;
- with limits of 1 GB memory, 1 CPU, 64 processes, 20 s and 20 KB of output.

Without Docker, the deep checks answer without it.

First run on the SPOT dev half (October 2026): Flash-Lite plan and screen, 3.8 Flash deep
checks with the sandbox. The $3 cap stopped it after 6 of 13 papers with no findings. The
deep checks asked for a test in 27 of 38 first answers. 19 of the 22 answers that
followed a test were a bare "no error", which the prompt allows. They were first read as
broken answers, but the same pattern came back after the prompt was changed. The second
round now gets the deep-check prompt without the offer of a test, with medium thinking
and a 16k output cap (`TESTED_MAX_OUTPUT`): at high thinking it used about 30k output
tokens, most of a paper's cost, and with the cap it used 0.7k to 16k. Theory and integrity
tasks also have their own limits (`KIND_POLICIES`): the default 10 model calls and 5
minutes per task made several fail and be retried, paying for the same calls again. The
cap is now checked again when a call gets a free slot, since calls already queued went
out after the money ran out.

Undefined names are marked uncertain, because the code that uses them may never run. Code
written for Python 2 is reported as unverifiable. On the 40 set C repositories, the check
found 40 defects in 14 repositories. One is certain: a file left with git merge-conflict
markers (`>>>>>>>`).

**Code and data claim loop** (October 2026). The settings check compares values, and the
method check sends the method text with the most related files to the model in one call.
Neither follows one claim through the code. The claim loop
(`auditor.agents.code.claim_loop`, off by default, `AUDITOR_CLAIM_LOOP_ENABLED`) does that
for claims about the experiments that rest on the released code or data:

- **Plan.** One call to the main model, with low thinking and an answer capped at 4,096
  tokens (`AUDITOR_CLAIM_LOOP_PLAN_MAX_OUTPUT`), lists up to
  `AUDITOR_CLAIM_LOOP_MAX_CLAIMS` (6) claims in six categories: dataset and split,
  preprocessing, number of training examples, architecture, training procedure and
  evaluation protocol. Plain hyperparameter values are left to the settings check. Each
  claim must quote at least 20 characters of its sentence verbatim, and the task is
  anchored on that sentence by its parser id, narrowed to the quote. A dataset, size or
  evaluation claim on a sentence that already has a settings or data task for the same
  key is dropped.
- **Explore.** Each claim is a code task (target `code_claim`). A tool-using agent on the
  review model (`langchain.agents.create_agent`, as in B1) gets the claim, its sentence
  with one sentence on each side, and the repository's file list. It uses the B1 code
  tools: list, search and read files, look up a setting's values, and read the data
  report in data analysis mode. Each claim has at most
  `AUDITOR_CLAIM_LOOP_MAX_TOOL_CALLS` (12) tool calls and
  `AUDITOR_CLAIM_LOOP_MAX_MODEL_CALLS` (8) model calls, at
  `AUDITOR_CLAIM_LOOP_EFFORT` (medium), and never more than the task's budget has left.
  A tool call past the limit is not run, and the model is told to answer. Every model call
  goes through `StructuredLLM.direct_call`, so it counts against the run's call and dollar
  limits and is logged with purpose `claim-loop`. These calls are not cached.
- **Analyse.** The agent answers contradicted, supported or unsettled. A contradiction must
  cite lines of a file in the snapshot, at most 40 of them, that a search or read in the
  same loop showed it. Otherwise it is dropped and the task is unverifiable. Supported is
  checked with no finding. Unsettled, or no answer within the limits, is unverifiable with
  its reason, never a finding.
- **Findings.** The finding quotes the paper sentence and gives the cited lines as CODE
  evidence. Its class follows the category: `data_split_mismatch` (target `data_split`),
  `dataset_size_mismatch` (`train_examples`) and `metric_mismatch` (`eval_metric`) keep
  the sentence anchor. Preprocessing, architecture and training claims are
  `method_mismatch` with a document anchor and target `method`, so they are matched per
  paper like the method check's (D4). A finding is uncertain unless the model calls the
  cited lines direct evidence with high confidence.

Nothing in the repository is run.

First run on set C (`scripts/claim_eval.py`). It covered 10 pairs, each original and its
injected copy, with Flash-Lite exploring (`AUDITOR_CLAIM_LOOP_MODEL=main`; 3.8 Flash cost
$0.51 for one pair) and 3.8 Flash judging:

- 82 claims were planned. Half were checked and half were unverifiable within the limits.
- 4 findings came out, 2 of them on the originals.
- 1 of the 28 SciCoQA discrepancies of those pairs was found.
- The run cost $0.77.

The planned claims are mostly about datasets and evaluation. SciCoQA's discrepancies are in
method details (a loss term, which features are concatenated), which the method check
covers. The loop stays off by default.

## D12. PDF parsing stack — Agreed

- GROBID 0.9.1 (Docker image `grobid/grobid:0.9.1-crf`) for structure, bibliography,
  sentence segmentation and in-text citation linking, with page coordinates.
- pdfplumber for page count, page sizes, fonts, margins and link annotations.
- GROBID's TEI output is cached by the PDF's SHA-256. The conversion to `ParsedPaper` and
  the citation linking run on every parse, so improvements to them apply to cached papers.
- `ParsedPaper` JSON fixtures are committed for tests, so only people working on the parser
  need Docker running.

**LaTeX source as input (plan D7, October 2026).** `auditor audit` and `batch` also take
the main .tex file of a paper's source (`auditor.parser.latex`).
`scripts/fetch_arxiv.py --source` fetches an arXiv e-print and unpacks it, refusing
members that would leave their folder. Formulas stay as LaTeX: on two SPOT papers the
source kept about 800 math marks, where GROBID kept 3 and 4. With the same theory
checker on both inputs:

- 2405.05635: the annotated error was found from either input.
- 2408.16162: found from the PDF, not from the source in this run.

Two papers show no gain for the source; their errors are in the reasoning, which
survives GROBID. The parser stays as the input for papers whose errors sit in the
formulas themselves. The bibliography is not read from the source.

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

1. **Plan** (`auditor.planner.theory`). The paper is cut into chunks of about 15,000
   characters, and one call per chunk to the main model, with low thinking and an answer
   capped at 8,192 tokens, lists its formal claims. On the full SPOT set, a single call
   per paper with minimal thinking made the model repeat itself until it hit the output
   cap on 6 of 26 papers. A formal chunk whose plan comes back empty or broken becomes one
   high-risk claim covering its formal sentences, so no formal text goes unchecked.

   Each claim has a kind, a short statement, the sentences that state and argue it, the
   sentences it builds on, a risk level, and whether checking it takes real reasoning. Which claims count, and how risky they are,
   is the model's reading of the paper, not a list of keywords, so this works in any field.
   The claims become `theory` tasks of `AUDITOR_THEORY_BATCH_SIZE` (8) claims each, at most
   `AUDITOR_THEORY_MAX_TASKS` (6) of them. Over that budget, low-risk claims that only need
   reading are left out first.
2. **Screen.** One call per task with low thinking. It sees each claim's own sentences and
   the definitions it uses, and calls each claim fine, suspicious or needs reasoning.
3. **Deep check.** These claims are checked one at a time with high thinking:
   - suspicious claims;
   - high-risk claims that need reasoning, whatever the screen said, since a quick read
     cannot clear a delicate argument;
   - medium-risk claims the screen could not settle.

   At most `AUDITOR_THEORY_DEEP_CHECKS` (2) claims per task get one, with the whole paper as
   context. The paper comes first in these prompts, so a provider's prompt cache can reuse
   it. "No error" is the default answer. An error needs a witness: the step that fails, a
   counterexample, or the correct value. A high-risk claim gets
   `AUDITOR_THEORY_DEEP_SAMPLES` (2) independent tries, since one try can miss what another
   finds. An error from any try goes on to the challenge.

4. **Challenge.** One call per paper to the review model (`AUDITOR_LLM_REVIEW_MODEL`, or the
   main model when none is set), with medium thinking. It judges every candidate with the
   whole paper in view: real errors stay, repeats point at the first one, and the rest are
   dropped with its reason.

- **Evidence gate.** An error is kept only if it quotes the paper verbatim, at least six
  words long, ignoring spacing. The finding is anchored on the quoted sentence and span,
  so D3 sentence matching works for injected math errors. Confidence below high makes it
  uncertain. An answer that leaves the quote empty but names a sentence that exists is
  kept, anchored on the whole sentence; 3.8 Flash did this on a retracted SPOT paper with
  the right error as its witness.
- **Main result.** When the screen calls every claim fine, the claim that needs reasoning
  and rests on the most others still gets a deep check. On SPOT 2408.16162 (retracted)
  the screen cleared all five claims. With these two changes, one run of that paper
  found the annotated error, which survived the challenge and matched in the judge's
  grade, for $0.16.
- **Five dev papers, $1.50 cap** (2401.09543, 2405.01133, 2405.05635, 2406.01892,
  2408.16162). 2406.01892 alone cost $1.30: five theory tasks, 13 deep checks and 13
  tested rounds, each with the whole paper (48k tokens) in the prompt. The cap then stopped
  2401.09543, and stopped 2408.16162 before its challenge, though its deep check had found
  the annotated error again. The other three gave no findings. A paper's theory tasks now
  share at most `AUDITOR_THEORY_DEEP_PER_PAPER` (4) deep-check tries.
- **Cost settings, measured on the same five papers.**

  | Setup | Cost | Errors found |
  |---|---|---|
  | high thinking, no per-paper limit | $1.62, cap reached | 0 counted (the cap stopped the one found) |
  | medium thinking, whole paper | $0.45 | 1 of 5 (2408.16162, precision 1 of 1) |
  | medium thinking, nearby sections only | $0.28 | 0 of 5 |

  Medium is now the default (`AUDITOR_THEORY_DEEP_EFFORT`). The whole paper stays the
  default context (`AUDITOR_THEORY_DEEP_CONTEXT=sections` is the other choice). To let
  the provider cache the paper, both rounds of a deep check share one system prompt, and
  a paper's first deep check runs alone before the others start. That raised the cached
  share of deep-check input from 27% to 37%.
  `AUDITOR_LLM_CACHED_INPUT_SHARE` sets the price of cached input for the dollar cap; it
  defaults to 1, which counts cached tokens in full.
- **SPOT dev half with the defaults** (Flash-Lite plan and screen, 3.8 Flash medium deep
  checks with the sandbox, 4 tries per paper). The five papers above were not run again;
  the other eight ran once with the same code. Together: 3 of 15 errors found, in 3 of 13
  papers, from 3 reports, all 3 matched; $1.73 in all, $0.58 per error found. The best
  earlier setup on the same half, one 3.8 Flash call per paper, found 2 of 15 with 18
  reports. The judge's grades have not been checked by a person.
- **Where the dev half went wrong.** A trace of the run above, stage by stage:
  - The Flash-Lite plan listed 162 claims, and missed some of the wrong ones entirely
    (2406.06431 got two claims; Theorem 4.2 was not one of them).
  - The Flash-Lite screen called 140 of the 162 fine and 2 suspicious, so it did not
    filter. Deep checks went to the claims the plan rated high risk, which were often
    not the wrong ones (2401.09543's Claim 7 was rated low risk; 2406.11623's Green
    function lemma medium).
  - The 3 errors found all came from deep checks that happened to get the wrong claim:
    picking the claim is the weak step, not checking it.
  - Running the same setup again on the same 13 papers found 1 of 15, not 3: picking
    claims this way is mostly luck.
- **Triage** (`AUDITOR_THEORY_SELECTOR=triage`). One review-model call (medium thinking)
  reads the whole paper and names up to `AUDITOR_THEORY_DEEP_PER_PAPER` claims it
  suspects, each with its reason; these go straight to deep checks, with no plan or
  screen. On the three dev papers the screen had missed (2401.09543, 2405.05635,
  2406.11623) it found 1 annotated error (Proposition 3.9 of 2405.05635), from 4
  reports, for $0.33. With a restated claim in its answer, 3.8 Flash repeated phrases
  until the output cap in two of three papers, so the answer has only a reason, a kind
  and sentence ids.
- **Triage on the whole dev half.** The other ten papers ran once and were added to the
  three above: 6 of 15 errors found, in 5 of 13 papers, from 13 reports, of which 6
  matched; $2.08 in all, $0.35 per error found. The screen setup found 3 of 15, and 1 of
  15 when the same setup ran again. One report of 2406.03201 was matched to both of that
  paper's errors, which is generous. 2408.16162, found by the screen setup, was missed.
  Triage is now the default. Every review-model call about a paper (triage, deep checks,
  challenge) now starts with the same message holding the paper, so the prompt cache can
  reuse it; Vertex still reported only 8% of the 3.8 Flash input as cached.
- **Test half, once, with the frozen defaults** (triage, after the dev-half choice): 8 of
  14 errors found, in 7 of 13 papers (54%), from 12 reports of which 9 matched (75%);
  $1.68, $0.21 per error found. The screen setup's one test-half run had found none, in
  the 11 papers it finished. The test half was not looked at paper by paper. The grades
  are the SPOT judge's, and no person has checked them.
- **A second judge.** `scripts/spot.py rejudge` grades a run again with another model.
  With 3.8 Flash as the second judge:
  - test-half triage run: the first judge (the configured model, 2.5 Flash) matched 8
    errors, the second 5, and both 5, in 4 of 13 papers;
  - dev triage run: 6 by the first, 5 by both, also in 4 of 13 papers.

  The first judge is generous. The conservative test-half figure, where both judges
  agree, is 4 of 13 papers (31%), not 7 of 13.
- **Pooled triage** (two triage calls, 6 deep checks, a broader prompt; branch
  `p3-triage-pool`, not merged). On the test half it found 7 of 14 errors by the first
  judge and 5 by both, in 4 of 13 papers. Its reports were right 7 times in 19, and it
  cost $3.02 against $1.68. It does no better than single triage and costs more, so
  single triage stays the default.
- **Ordinary papers.** On 15 set C papers (machine learning, parsed from the PDF), the
  theory checker alone (`scripts/math_alarms.py`: triage, deep checks, challenge) named
  suspects in 3 papers and kept 1 finding, for $0.21. That finding says a DenseNet
  related-work sentence gives n! skip connections for an n-layer block; the right count
  is quadratic, so it may be a real error. Triage named nothing in the other 12 papers.
- **Explicit context caching** (`AUDITOR_LLM_EXPLICIT_CACHE`, off by default;
  `auditor.infra.prefix_cache`). Vertex's implicit cache reused 8-9% of the 3.8 Flash
  input, so the message holding the paper goes into an explicit Vertex context cache the
  first time a review-model call sends it; later calls about the paper name the cache and
  send only the instructions. Each cache is a resource on the Google Cloud project: it
  expires after `AUDITOR_LLM_EXPLICIT_CACHE_TTL_S` (1200 s), the run deletes the ones it
  made when it ends, and its storage for the whole lifetime is charged to the dollar cap
  (`AUDITOR_LLM_CACHE_STORAGE_PRICE`, $1 per million tokens per hour). Messages shorter
  than `AUDITOR_LLM_EXPLICIT_CACHE_MIN_CHARS` (16,000) are sent as usual, and so is any
  message whose cache cannot be made or has expired. On 2408.16162, 4,798 of the about
  5,500 input tokens of each 3.8 Flash call (triage, two deep checks, challenge) were read
  from the cache. The team lead approved creating these caches on the shared project.
- **Triage output cap.** Six dev papers had used nearly all of the triage call's 16k
  output on thinking. Run again with a 32k cap (`AUDITOR_THEORY_TRIAGE_MAX_OUTPUT`),
  four of them thought for 24k to 32k tokens, but the dev half's score stayed at 6 of 15
  errors in 5 of 13 papers (now from 11 reports, 5 matched), for $1.05 more. The cap
  stays at 16k.
- **Opt-in.** Theory checks cost model calls on every formal paper, so they run only with
  `AUDITOR_THEORY_CHECK_ENABLED`, which is part of the run fingerprint. B0 and B1 learn the
  class from the class guide. B1 gets no extra tools, since the paper is already in its
  prompt.
- **Evaluation.** SPOT's labels describe an error in words and give a rough location, with
  no sentence id. SPOT runs are therefore scored as the SPOT and PAT papers do it: an LLM
  judge decides whether a reported issue is the same error as the annotation. A person
  checks every judgement before the numbers are reported (`scripts/spot.py`).

## D22. The integrity screen and the `unfinished_or_placeholder_content` class — Agreed

Some papers carry their own reason for rejection. One example is an ICLR 2027 submission
whose table captions say "Numbers are simulated drafting results for manuscript
preparation and should be replaced by actual experimental results before submission".
Neither the code agent nor the citation agent would ever look at such a sentence. A fifth
specialist, `auditor.agents.integrity`, gets one task per paper (document anchor, target
rule `unfinished_content`). It checks in two ways:

- **Structural markers, no model.** Unresolved references (`??`, `[?]`, `(?)`, Word's
  "Error! Reference source not found") and leftover markers (TODO, TBD and FIXME in
  capitals, `XX%` or `X.X%`, lorem ipsum, `[citation needed]`). On 195 real papers (set C
  and SPOT), "XXX" and "X%" turned up as variable names and `[]` as an empty list in
  pseudocode, so those are left to the model. The only marker hit left was a real one,
  "Algorithm ??".
- **One cheap model call per paper.** Flash-Lite with low thinking and an answer capped at
  4,096 tokens reads every caption, footnote and paragraph. It reports statements that
  the paper's own results are placeholders, simulated, invented or still to be replaced,
  notes to the authors left in the text, and unfinished text. Which wording counts is the
  model's reading, so this works in any field. A finding must quote its sentence exactly;
  prose is copied, not rendered, so a word match is not enough here.

Findings have severity `high` (D20). B0 and B1 learn the class from the class guide, and B1
gets no extra tools.

The integrity specialist also has a second task per paper (rule `result_consistency`,
class `result_inconsistency`). It checks the paper's statements about its own results
against its tables:

- **Extraction.** One Flash-Lite call with low thinking lists each claim with the numbers
  it rests on, copied from the table as written: a stated value, a difference in points, a
  relative gain in percent, or "best".
- **Computation.** The arithmetic is done in code, taking each number's rounding into
  account.
- **Gate.** A number the model gives counts only if it really is in the table row it names,
  so a misread number never becomes a finding.
- **Without a model,** tables are checked only for negative standard deviations. Values
  over 100% are left alone: on 195 real papers, every one was a relative measure or a
  number the parser had merged into a cell, never an error.
- **Field test.** On 12 set C papers with tables, the two integrity checks raised no
  findings, at $0.10 for 24 calls.

Rules added in October 2026 after running on real cases from ICLR reviews:

- **Placeholder spellings.** "TODOXXX", "Section X" or "Figure X" with a capital X, and
  "[insert ...]" now count as placeholders. In papers that number tables or figures in
  Roman numerals, "Table X" means the tenth table, so it is not flagged there.
- **Dropped: references past the last table, figure or section.** On 229 real papers this
  rule fired 74 times in 34 papers, nearly always because the parser had lost a caption
  or a section number, so it was removed.
- **Same result in two places.** A new claim kind, `same_value`, covers one result
  reported in two places, such as a baseline in the main table and the matching row of an
  ablation table. Its findings are uncertain, since which results are "the same" is the
  model's reading.
- **Stated drops.** A drop written as "-8.6 points" is compared by size, and so is a
  relative change in percent.

Changes from the table-reading work (October 2026):

- **Tables.**
  - The check reads the tables the parser took from the PDF's layout text (D1).
  - Pages whose tables could not be read that way are shown to the model as images, at
    most 6 per paper. A transcription that runs a spread into its number ("60.50.39") is
    not used.
  - A table read from the layout or an image replaces the parser's table of the same
    label.
- **One extraction call per section.** Each call sees a section's sentences with digits
  and the tables the section names or quotes, for at most 10 sections. One more call
  checks the remaining sections against every table. With one call for the whole paper,
  Flash-Lite listed four claims for a 20-page paper and missed the real error.
- **Gates.**
  - A claim with no table id is matched to the table that holds all its numbers.
  - A number the text gives for a row is checked against that row when the row holds a
    single number; the model sometimes copies the text's number instead of the table's.
  - A row named only by a number ("128") is not used for a stated value.
- **Reversed comparisons, no model.** A sentence that calls one stated number higher or
  lower than another when it is the other way round is found by a rule ("higher IoU
  (0.73) than U-Net (0.78)"). On 252 real papers, the rule fired on one real error, which
  appears in three versions of one paper.
- **Challenge step.** Every candidate goes to one call to the review model (medium
  thinking). The call sees the sentence with two before and after, the table and its
  caption, and decides whether the text really disagrees with the table. It drops
  readings of the wrong column or a different quantity
  (`AUDITOR_RESULTS_CHALLENGE_ENABLED`).
- **Measured on real cases.** With Flash-Lite extraction and 3.8 Flash challenges:
  - 2 real errors were found: the reversed comparison in arXiv 2312.11580, and the mean
    rating of 8.93 against Table 6's 6.30 in arXiv 2512.21706.
  - Before the fixes, 8 set C papers gave 2 false alarms. Those 2 papers gave none after
    the fixes.
  - The ICLR review cases mostly cannot be tested: the reviewers' errors are fixed in the
    arXiv versions or are not stated numbers.


Results on the 40 set C originals (October 2026). Flash-Lite did the extraction and 3.8
Flash the challenge, at $0.58 for all 40 (379 calls). The 32 papers not used for tuning
gave 4 findings, in 2 papers:

- One looks like a real error. arXiv 2002.00718v2 says LwF exceeds LwF-MC on new classes
  "by 18.9%", but Table 2 gives 25.6 against 10.5, which is 15.1 points. 18.9 is ILT's
  value in the same column.
- Three were false alarms, and two rules now remove them:
  - **A relative change may be taken against either number.** "A relative degradation of
    58% (0.0788 cf. 0.0335)" is 58% of 0.0788.
  - **Another column of the same two rows.** A stated change that the two named rows give
    in some other column is accepted. The model had read the 100-50 "all" column where
    the text was about 100-10. The rows' numbers are taken in order after the label, not
    by cell, since extraction often merges cells; spreads after "±" are left out.

That is 1 real finding in 4 before the new rules, and 1 in 1 after them. The rules were
written for these same papers, so the second figure is not an independent precision.

Three more rules with no model (October 2026), each uncertain and each reviewed by the
challenge step when a model is configured:

- **Copied cell.** Two rows of a table give the same mean and spread ("85.31 \u00b1 1.27") in
  one column, and the rows agree on no other cell. Rows that share other cells repeat a
  setting by design; a value three or more rows share is left alone.
- **Wrong average or total.** A last column headed "Avg", "Mean", "Total" or "Sum" is not
  the mean or sum of the row's other cells, within rounding, in a table where at least
  two thirds of the rows (and at least two) agree. A column that no row averages plainly
  is another quantity, a weighted mean or a subset.
- **GRIM.** A sentence gives one mean to two decimals and one sample size N under 100, and
  no whole number of points over N rounds to that mean ("M = 3.47, N = 28"). A mean over
  several items per person has a finer grain, so this is a hint.

Over the 195 local papers with a parse, the first versions gave 43 findings, nearly all
false alarms: a "Mean" column that held another metric, and settings repeated by design.
With the conditions above they give 1, a value shared by a model and its ablation without
features, which may well be by design.

**Set D** (`scripts/set_d.py`, `auditor.agents.integrity.set_d`; plant records and labels in
`datasets/set_d`). Errors are planted in copies of the 40 set C papers and repositories,
with the originals as clean controls. Placement follows a hash of the paper id. Planted
sentences use the paper's own table labels, row names and numbers. The placeholder and
GRIM sentences come from short generic templates, since nothing in the paper could supply
them. The run checks each paper with the integrity screen, the results check (Flash-Lite
extraction, 3.8 Flash challenge, layout tables from the PDF) and the code-defect check:

| Plant | Found |
|---|---|
| placeholder or simulated-results sentence | 40 / 40 |
| GRIM-impossible mean | 40 / 40 |
| reversed comparison of two table rows | 28 / 33 |
| wrong size of a gain between two table rows | 21 / 33 |
| number the text quotes from a table, changed | 0 / 7 |
| undefined name in an experiment file | 26 / 26 |
| syntax error in an experiment file | 13 / 13 |

That is 168 of 192 plants, at least one in each of the 40 papers. On the 40 originals the
same checks give 3 placeholder findings, 5 result findings and 43 code defects. The code
defects are the repositories' own, found before; the other 8 have not been checked by
hand. The changed-number plant is the weak spot, and only 7 papers have a prose sentence
that quotes a table number unambiguously. The set D runs led to five changes:

- The comparison rule also reads a level set off by a comma just before the comparative
  word ("reaches 54.81, higher than Y (56.06)"). Without the comma it is often a
  difference ("3.16% higher than"); the 195 local papers give no hits either way.
- A gain whose sentence gives both results is not excused by another column of the
  same rows.
- A gain the extraction model gave without the compared result is completed from its
  sentence, when the sentence holds exactly the two results.
- GRIM findings skip the challenge, which can only guess whether the data are whole
  numbers. It had dropped 12 of the 40 planted cases.
- In the code-defect check, a script that does not compile counts as an entry point when
  its text has a main guard. Before this, breaking a script took it out of the scope that
  was checked.
