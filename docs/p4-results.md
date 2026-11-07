# Part 4: evaluation runner, audit report and first verifier ablation

7 November 2026, Sudheera Y S. No model calls were made for anything in this note.

## What is new

| Task | What | Where |
|---|---|---|
| P4.5 | `auditor evaluate`: one scorer for every set and system (D3 matching, Macro-F1, precision, recall, false-positive rate on clean controls, flags per original paper and share of originals flagged (D14), unlabelled flags, review load, coverage, grounding, tokens, calls and cost). Runs of one system are summarised as mean ± sample standard deviation. | `auditor.evaluation` |
| P4.7 | Every full-system run is also scored without its verifier (`full_no_verifier`): every candidate finding counts, as for B0 and B1. The math review's own drops (D21) still count, since that review belongs to the theory checker. | `RunData.predictions_without_verifier` |
| P4.3 | Verifier decisions on real runs: each finding's state, split by what its gold label says it is (real error, wrong class, clean control, unlabelled). | `verifier_decisions`, in the `evaluate` output |
| P4.4 | Audit report per paper, `report.md` and `report.json`, written by every full-system run: verified flags, items for human review with the verifier's reason, checks not done with their reasons (D6), and what the verifier set aside. Repeats (D20) are shown once. The wording describes differences, never causes. `auditor report` rebuilds it for a finished run. | `auditor.reporting` |

Smaller changes:

- `scripts/verifier_replay.py` takes `--dataset` (a dataset file with this machine's paths), `--online` (fetch repository snapshots that are not cached; GitHub only) and `--remap OLD=NEW` (evidence from a local patched repository names it by its path, `file:///Users/...`, which differs per machine). Its output folder now has the run's results, usage and failures, and its manifest no longer lists the verifier as a placeholder, so `auditor evaluate` reads it as a complete run.
- Repository snapshots no longer download files the host lists as empty. raw.githubusercontent.com sometimes answers 501 for an empty `__init__.py`, which failed the whole snapshot (seen on six set C repositories during the replay). (`auditor.infra.snapshots.snapshot`, test added.)
- `NO_ERROR_REASON` in `auditor.agents.theory.review` names the math review's drop reason, so the ablation can tell those drops from the verifier's.

```bash
uv run auditor evaluate datasets/set_c/labels.jsonl runs/main-full-r1 runs/main-b0-r1 runs/main-b1-r1 \
    --dataset datasets/set_c/dataset.jsonl --markdown results/main.md --json results/main.json
uv run auditor report runs/main-full-r1/2001.07645v3-injected --paper data/set_c/injected/2001.07645v3.json
```

`--area code` (repeatable) scores one area's classes; `--strict` refuses runs with placeholder parts (D19); without `--strict` they are scored and marked.

## Check against the existing scorers

On the same predictions, `auditor evaluate` gives the numbers `scripts/set_c.py score` gives: Macro-F1 0.600 before and 0.521 after the verifier on set C, 0.576 and 0.375 on the RQ1 pilot (D23). Unlabelled flags differ by design: `set_c.py` counts only its seven code and data classes, `evaluate` every class unless `--area` is given.

The replay on this machine decides 94, 183, 0 (verified, review, dropped) where D23 reports 95, 182, 0. The scores are the same; the one finding was not traced.

## First results: the verifier on set C and the RQ1 pilot

Both runs are from before the verifier and were replayed through it (`results/p4/`). The citation checker was off in both (placeholder), so these are code and data classes only.

Set C, 80 paper versions (40 papers, injected and original), one run. That run
(`setc-rv-p3-code-fixes-full-r1`) had no model for the method check: all 100 method labels
ended "unverifiable: comparing the method with the code needs an LLM" (found with
`scripts/p4_analysis.py`), so method_mismatch scores 0 here by construction and the
numbers below are the settings and metric checks. The fresh runs below include it.

| | Full system | Without verifier |
|---|---|---|
| Macro-F1 | 0.521 | 0.600 |
| Precision, recall (labelled items) | 1.000, 0.239 | 1.000, 0.380 |
| False positives on 72 clean controls | 0 | 0 |
| Flags per original paper | 0.72 | 2.73 |
| Original papers with a flag | 50% | 80% |
| Unlabelled flags (unique) | 63 (31) | 241 (89) |
| Items for human review per paper | 2.29 | - |
| Flags backed by evidence found in its source | 94 of 94 | 272 of 277 |

Verifier decisions on set C: of 62 findings that match a planted error, 39 are verified and 23 go to review; none is dropped. No finding matches a clean control.

RQ1 pilot (5 pairs, one run each, labels `data/set_c/rq1/labels5.jsonl`):

| | Full | Full without verifier | B0 | B1 |
|---|---|---|---|---|
| Macro-F1 | 0.375 | 0.576 | 0.143 | 0.000 |
| Unlabelled flags | 20 | 49 | 7 | 1 |
| Model calls per paper | 24.6 | 24.6 | 1.2 | 28.0 |
| Cost per paper version | $0.117 | $0.117 | $0.061 | $0.271 |
| Paper versions finished, of 10 | 8 | 8 | 8 | 6 |

What this shows so far:

- The verifier drops nothing on these runs: every quote and code line was found in its source. It only moves findings to human review, mostly those the code agent marked uncertain. So it costs recall (Macro-F1 0.600 to 0.521) and cannot raise labelled precision, which is already 1.000 with no false positives on clean controls.
- What it does cut is the flags no label covers: 241 to 63 on set C, and flags per original paper from 2.73 to 0.72. Whether that is a precision gain depends on how many of those 241 are false alarms, which needs the hand adjudication of S5 (D14). Until then RQ1's precision side cannot be answered on set C.
- Whether uncertain findings should count as flags is the open question in D23. It decides most of the difference above.
- The pilot comparison is not fair yet: B1 finished 6 of 10 versions and B0 8 of 10 (rate limits), and `evaluate` counts their missing papers' labels as misses ("labels on papers not run" in the run table).

## Cost of the experiments still to run (P4.6)

From the pilot rates above (Gemini 3.8 Flash at $0.75 / $3.75 per million tokens), set C with all 80 versions:

| | One run | Three runs |
|---|---|---|
| Full system | about $9 | about $28 |
| B0 | about $5 | about $15 |
| B1 | about $22 | about $65 |
| All three | about $36 | about $108 |

Set B (citations, about 120 versions) adds to this; its per-paper cost needs a pilot of 5 papers first. The no-verifier ablation costs nothing extra. Abstract-only retrieval needs its own citation runs. Plan: one run of each system on set C first, check the ledger, then agree the rest with the team.

## Next for P4

- Agree D23's open question (do uncertain findings count) with Vijay, then mark D23 agreed.
- S5: adjudicate a sample of the unlabelled set C flags, so the verifier's effect on false alarms can be measured.
- P4.6: one run of B0, B1 and the full system on set C, then set B after a 5-paper pilot.
- P4.7: abstract-only retrieval on set B; the single-agent comparison is B1.
- P4.8: result tables from `auditor evaluate --json`, and failure analysis.
