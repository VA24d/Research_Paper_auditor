# Part 4: evaluation, audit report and the set C experiments

7 to 8 November 2026, Sudheera Y S.

## Summary

On set C (40 papers, each as an injected copy and as the unchanged original), the full
system reaches Macro-F1 **0.636 ± 0.004** over three runs, against **0.048 ± 0.015** for B0
(three runs) and **0.046 ± 0.007** for B1 (two runs). The proposal's target of +10 points is
met against both baselines, by about 59 points. No system flags any of the 72 clean control
items. On the unchanged originals the full system raises more flags that no label covers
(3.1 per paper, against 1.6 for B0 and 0.4 for B1); how many of those are false alarms needs
the hand adjudication of S5, so the "no increase in false alarms on clean papers" half of
the target is not settled. The verifier verifies 180 of the 203 findings that match a
planted error, drops none, and cuts unlabelled flags by 35% at a cost of 3.1 points of
Macro-F1. The experiments cost $125.04 on the team's Vertex credits.

**Rescored on 11 November 2026.** The numbers here are the same runs scored with the
current code. The first version of this file (0.703 for the full system) was scored before
d2f93cd made method matching stricter (#76); B0 and B1 are unchanged. Coverage is now the
share of tasks checked by the parts that ran: the citation agent's placeholder tasks used
to be counted as checked (cf52612). `results/p4/set_c-rq1.*` hold the rescore; it needs
the 40 repository snapshots cached and, on another machine, `--remap
file:///home/dheeruys/Documents/Github/Research_Paper_auditor/=<this checkout as a file URL>/`
for the patched repositories. `set_c-failures.md` keeps the first scoring: its stages need
the runs' `tasks.jsonl`, which is not committed.

## What is new

| Task | What | Where |
|---|---|---|
| P4.5 | `auditor evaluate`: one scorer for every set and system (D3 matching, Macro-F1, precision, recall, false-positive rate on clean controls, flags per original paper and share of originals flagged (D14), unlabelled flags, review load, coverage, grounding, tokens, calls and cost). Runs of one system are summarised as mean ± sample standard deviation; each run's JSON lists what happened to every label. | `auditor.evaluation` |
| P4.7 | Every full-system run is also scored without its verifier (`full_no_verifier`): every candidate finding counts, as for B0 and B1. The math review's own drops (D21) still count, since that review belongs to the theory checker. | `RunData.predictions_without_verifier` |
| P4.3 | Verifier decisions on real runs: each finding's state, split by what its gold label says it is (real error, wrong class, clean control, unlabelled). | `verifier_decisions`, in the `evaluate` output |
| P4.4 | Audit report per paper, `report.md` and `report.json`, written by every full-system run: verified flags, items for human review with the verifier's reason, checks not done with their reasons (D6), and what the verifier set aside. Repeats (D20) are shown once. The wording describes differences, never causes. `auditor report` rebuilds it for a finished run. | `auditor.reporting` |
| P4.6 | Set C experiments: settings in `configs/experiments/set_c.env`, dataset `configs/experiments/set_c.jsonl` (paths relative to the file). | below |
| P4.8 | `scripts/p4_analysis.py`: where each planted error is lost, stability across runs, and the full system with uncertain findings counted. | below |

Smaller changes:

- `AUDITOR_SKIP_PARTS` (JSON list) leaves specialist parts out of a run on purpose. They
  are recorded as placeholders in the manifest, with their B1 tools, so the full system and
  B1 stay matched. Earlier set C runs left the citation agent out with an uncommitted local
  change.
- `scripts/verifier_replay.py` takes `--dataset`, `--online` and `--remap OLD=NEW`, for runs
  made on another machine. Its output is a complete run for `auditor evaluate`.
- Repository snapshots no longer download files the host lists as empty.
  raw.githubusercontent.com sometimes answers 501 for an empty `__init__.py`, which failed
  the whole snapshot (seen on six set C repositories).
- `NO_ERROR_REASON` in `auditor.agents.theory.review` names the math review's drop reason.
- `scripts/review_sheet.py` turns a D18 review sample into a local HTML page; its
  "Download" button saves the filled file for `scripts/set_c.py apply-review`.

```bash
uv run auditor evaluate datasets/set_c/labels.jsonl runs/p4-setc-full-r1 runs/p4-setc-b1-r1 \
    --dataset configs/experiments/set_c.jsonl --markdown out.md --json out.json
uv run auditor report runs/p4-setc-full-r1/2001.07645v3-injected --paper data/set_c/injected/2001.07645v3.json
uv run python scripts/p4_analysis.py datasets/set_c/labels.jsonl runs/p4-setc-*-r1
```

## Set C experiments (P4.6)

### Setup

- **Data.** Set C as frozen in `datasets/set_c/labels.jsonl`: 163 planted errors (100
  `method_mismatch`, 60 `setting_mismatch`, 3 `metric_mismatch`) and 72 clean controls, on
  80 paper versions (40 papers × injected and original). 47 of the labels (20%) have had
  their D18 second review; scoring can be redone at no cost if a label changes.
- **Systems.** Full system, B0, B1; the no-verifier ablation is scored from the full runs.
  Full and B0 ran three times, B1 twice (see Cost). B1's partial third run (10 of 80
  versions, `p4-setc-b1-r3`) is not scored.
- **Settings** (`configs/experiments/set_c.env`), as in the earlier RQ1 runs
  (`results/README.md`): Gemini 3.8 Flash on Vertex (`lmaproject6767`), for the main and
  the review model; method check in "rag" mode; citation agent left out
  (`AUDITOR_SKIP_PARTS=["citation"]`; set C labels no citation classes), and its planner
  pass off; up to 5 retries of a model call; no Vertex context caches. Batches answer from
  the LLM cache only within a run (D10).
- **Code.** All runs used the same auditor code (code hash `88b2408fee`, clean tree; git
  commits 0f20a51, f797ad5 and 5e1aacb differ only in scripts and docs).
- **Embeddings.** The method check ranks code with nomic-embed-text-v1.5. 54 of the 80
  versions were covered by the team's cached embeddings (LM Studio); for the other 26 (13
  papers) the same model was served locally on the CPU (fastembed, ONNX) and cached before
  the runs, so no paper fell back to BM25. On 30 queries over cached papers, the local model
  picks 9 of the same top-10 code chunks as the cached vectors on average, BM25 5.
- **Runs.** `runs/p4-setc-{full,b0,b1}-r{1,2,3}`, one repetition of all three systems at a
  time. Every version finished in every scored run; papers that hit a batch's spend limit
  were not started and ran in the next attempt.

### Results

Mean ± sample standard deviation over runs (`results/p4/set_c-rq1.md`, `.json`):

| | Full system | Full without verifier | B0 | B1 |
|---|---|---|---|---|
| Runs | 3 | 3 | 3 | 2 |
| **Macro-F1** | **0.636 ± 0.004** | 0.667 ± 0.003 | 0.048 ± 0.015 | 0.046 ± 0.007 |
| Precision (labelled items) | 1.000 | 1.000 | 1.000 | 1.000 |
| Recall | 0.368 ± 0.006 | 0.415 ± 0.004 | 0.029 ± 0.009 | 0.028 ± 0.004 |
| F1 method / metric / setting | 0.338 / 0.800 / 0.771 | 0.343 / 0.800 / 0.857 | 0 / 0 / 0.144 | 0 / 0 / 0.139 |
| False positives on 72 clean controls | 0 | 0 | 0 | 0 |
| Flags per original paper | 3.12 ± 0.04 | 4.71 ± 0.04 | 1.63 ± 0.25 | 0.39 ± 0.23 |
| Original papers with a flag | 94% | 95% | 76% | 21% |
| Unlabelled flags per run | 255 ± 12 | 391 ± 13 | 125 ± 19 | 29 ± 21 |
| Items for human review per paper | 1.68 | - | - | - |
| Tasks checked (coverage) of the parts that ran | 63% | 63% | - | - |
| Flags backed by evidence found in its source | 100% | 100% | 0% | 0% |
| Model calls per paper version | 13.2 | 13.2 | 1.0 | 20.1 |
| Cost per paper version | $0.079 | $0.079 | $0.047 | $0.542 |

**RQ1.** Against B1 (D14) the full system gains 59.0 points of Macro-F1, against B0 58.8.
Neither baseline finds a single method or metric error; both find 3 to 6 of the 60 changed
settings. The full system is also the cheaper of the two tool-using systems: B1 sends the
whole paper with every one of its 20 calls (about 670,000 input tokens per paper version).

**False alarms.** On clean control items all four are at zero, so the item-level part of
D14 holds. On the unchanged originals the full system flags 94% of papers, 3.1 flags each,
against 76% and 1.6 for B0 and 21% and 0.4 for B1. These flags match no label: some are real
discrepancies already in the papers (set C's SciCoQA pairs include reported real ones), the
rest false alarms. Until S5 adjudicates a sample, the paper-level part of the target ("no
increase in false alarms on clean papers") is open, and leans against the full system.
Most of those flags are one kind: in each run 68 to 71 of the 124 to 127 verified flags on
the originals are "Unfinished text", mostly sentences GROBID split or ran into table text.
Since c04359c (10 November) such findings are uncertain and go to human review, so these
runs overstate what the current code flags on clean papers.

**Flags without evidence.** Every verified flag of the full system rests on a quote of the
paper and code lines that were found again in their sources. B0 and B1 attach no evidence to
their flags.

### The verifier (P4.3, P4.7)

Every full-system finding over the three runs, by what its gold label says it is:

| Finding | Verified | Human review | Dropped |
|---|---|---|---|
| Matches a planted error | 180 | 23 | 0 |
| Matches a clean control | 0 | 0 | 0 |
| Matches no label | 740 | 380 | 0 |

- The verifier drops nothing: every piece of evidence the specialists gave was found again
  in its source. Every planted error it holds back is held because the code agent marked
  it uncertain.
- Taking it out raises Macro-F1 from 0.636 to 0.667 and unlabelled flags from 255 to 391 per
  run (flags per original paper from 3.1 to 4.7). So on set C it trades 3.1 points of
  Macro-F1 for 35% fewer flags that no label covers. Whether that improves precision depends
  on how many of the 136 held-back unlabelled flags per run are false alarms (S5).
- **D23's open question** (should uncertain findings count as flags): counting them gives
  Macro-F1 0.668, 0.668 and 0.663 instead of 0.641, 0.636 and 0.632, and 131 to 134 more
  unlabelled flags per run. This is a choice between recall and review load, to settle with
  Vijay.

### Where the full system loses planted errors (P4.8)

Each of the 163 planted errors in each of the 3 full runs, by the first stage it did not
pass (`results/p4/set_c-failures.md`):

| Stage | method | metric | setting | All |
|---|---|---|---|---|
| Found (verified) | 61 | 6 | 113 | 180 |
| Human review (marked uncertain) | 4 | 0 | 22 | 26 |
| Task finished, no matching finding | 235 | 0 | 0 | 235 |
| Task unverifiable | 0 | 3 | 45 | 48 |
| Total | 300 | 9 | 180 | 489 |

- **Method changes** are the main loss: 235 of 300 times the method check raised nothing
  that matches. P3's diagnosis of 57 method misses (`docs/p3_example_audit_report.md`): 26 were
  in code shown to the model but not reported, 14 were never retrieved, 7 were dropped by
  the evidence gate.
- **Settings** are lost at the "which experiment" step: 45 times every repository value of
  the setting (batch size 18, learning rate 15, epochs 9, dropout 3) was judged to belong to
  another experiment than the sentence's, so the task ended unverifiable rather than raising
  a mismatch. That is the conservative failure: it never flags a wrong value.
- **Metric**: one of the three planted metric changes is unverifiable in every run (the
  repository computes several metrics and the code does not say which one the paper
  reports).
- **Stability**: of the 163 planted errors, the full system finds 57 in all 3 runs, 2 in 2
  runs, 5 in 1 run and 99 in none. B0 finds 3 errors in all of its runs and 5 more once; B1
  finds 2 in both runs and 5 more once.

### Cost

| | Runs | Cost |
|---|---|---|
| Pilot (2 papers, all three systems) | 1 | $1.70 |
| Full system | 3 | $18.84 ($6.26 to $6.30 a run) |
| B0 | 3 | $11.35 ($3.68 to $3.93) |
| B1 | 2 | $86.78 ($42.59, $44.19) |
| B1, third run stopped at the spend limit (not scored) | 10 of 80 versions | $6.36 |
| **Total** (shared ledger `data/p4/ledger.jsonl`) | | **$125.04** |

Priced at Gemini 3.8 Flash list rates ($0.75 / $3.75 per million tokens); Google's bill may
differ slightly. B1 cost twice what the 2-paper pilot suggested ($0.54 against $0.31 a
version), because the pilot papers were short and B1 resends the growing context with each
call. The run was stopped by the $125 ledger limit, so B1 has two runs and the others three.

## Earlier: replayed runs (7 November)

Before the fresh runs, two earlier runs were replayed through the verifier at no cost
(`results/p4/set_c-verifier-ablation.md`, `results/p4/rq1-pilot.md`). On the same predictions
`auditor evaluate` gives the numbers `scripts/set_c.py score` gives (Macro-F1 0.600 before
and 0.521 after the verifier on set C, 0.576 and 0.375 on the RQ1 pilot, as in D23). That set
C run had no model for the method check, so all its method labels were unverifiable; the
fresh runs above replace it. Those replayed numbers come from before d2f93cd; their runs are
not in the repository, so they were not scored again.

## What is left

- **D23**: decide with Vijay whether uncertain findings count as flags (numbers above), then
  mark D23 agreed.
- **S5**: adjudicate a sample of the unlabelled flags on original papers, the remaining part
  of the false-alarm question. The audit reports (`report.md` in each paper folder) show each
  flag with its evidence.
- **Set B and the abstract-only ablation** (P2.9, P4.7): the set B papers are committed in
  `datasets/set_b` (11 November). The same `auditor batch` and `auditor evaluate` commands
  apply, with `AUDITOR_CITATION_ABSTRACT_ONLY=true` for the ablation. It needs new credits:
  Anurag's mini-run used about 60 model calls per paper pair. The verifier sends every
  `reference_not_found` finding to human review (it rests on a record not being found), and
  96 of set B's 113 planted errors are of that class. Since only verified flags count, the
  full system can score at most the other 17 unless D23 or that rule changes; B0 and B1
  count every flag.
- **B1's third run** would cost about $44 more if three B1 runs are wanted.
