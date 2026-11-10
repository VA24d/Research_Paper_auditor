# Raw results

Outputs of the runs behind the mid-submission report, copied from the git-ignored `data/` folder with the same relative paths. `results/set_c/runs/rq1-c5-full-r1` was `data/set_c/runs/rq1-c5-full-r1`.

| File | What it holds |
|---|---|
| `predictions.jsonl` | Findings per paper version, scored by `scripts/set_c.py score` and `scripts/spot.py` |
| `results.jsonl` | Each specialist task's output, with quoted evidence |
| `spot_grades.jsonl`, `second_grades.jsonl`, `third_grades.jsonl` | SPOT grades from the first, second and third judge |
| `llm_calls.jsonl` | Model, tokens and timing of each model call |
| `manifest.json`, `run.log` | Settings, code commit and log of each run |
| `ledger-*.jsonl` | Spend ledger |
| `set_c/rq1/` | Labels for the 5 RQ1 pairs and the prediction sets that were scored |

Left out, because they are large or can be rebuilt: PDFs and LaTeX sources, parsed papers, code snapshots, checkpoints, HTTP logs and caches.

The RQ1 full-system scores use `rq1-pilotb` and `rq1-c5`, with `rq1-fix2` replacing 2304.04740 and 2402.18396. B0 and B1 fill their gaps from `rq1-rb0` and `rq1-rb1`. Pair 2401.05363 is dropped.

`mid-submission-audit.md` lists what the audit of the report found.

## Part 4 (`results/p4/`, 8 November 2026)

| File | What it holds |
|---|---|
| `set_c-rq1.md`, `set_c-rq1.json` | `auditor evaluate` on the set C runs: full system and B0 three runs, B1 two, plus the full system without its verifier |
| `set_c-failures.md` | `scripts/p4_analysis.py`: where each planted error is lost, stability across runs, uncertain findings counted |
| `set_c-verifier-ablation.*`, `rq1-pilot.*` | Earlier runs replayed through the verifier (7 November) |
| `runs/p4-setc-*`, `runs/p4-pilot-*` | Per run: manifest, log, failures, model calls; per paper: predictions, task results, verifier decisions |
| `ledger-p4.jsonl` | Spend ledger of these runs ($125.04) |

`runs/p4-setc-b1-r3` stopped at the spend limit after 10 of 80 versions and is not scored. Left out: checkpoints, `tasks.jsonl` and the audit reports (`report.md`, `report.json`), which `auditor report` rebuilds. Settings: `configs/experiments/set_c.env`. Write-up: `docs/p4-results.md`.
