# RQ1 pilot, set C, 5 pairs (verifier replayed)

Only verified flags count for systems with a verifier; every flag counts for B0, B1 and `full_no_verifier` (D3). Values are the mean ± sample standard deviation over runs. Unlabelled flags match no gold label and are not scored (D14).

| System | Runs | Macro-F1 | Precision | Recall | FP rate (clean) | Flags / original | Originals flagged | Unlabelled | Review / paper | Coverage | Flags backed | Quotes grounded | LLM calls / paper | Cost / paper ($) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b0 | 1 | 0.143 | 1.000 | 0.050 | 0.000 | 0.50 | 0.50 | 7.0 | - | - | 0.00 | - | 1.2 | 0.061 |
| b1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | 0.00 | 1.0 | - | - | 0.00 | - | 28.0 | 0.271 |
| full | 1 | 0.375 | 1.000 | 0.200 | 0.000 | 2.50 | 1.00 | 20.0 | 3.50 | 0.96 | 1.00 | 1.00 | 24.6 | 0.117 |
| full_no_verifier | 1 | 0.576 | 1.000 | 0.350 | 0.000 | 5.50 | 1.00 | 49.0 | - | 0.96 | 1.00 | 1.00 | 24.6 | 0.117 |

## F1 per class

| System | method_mismatch | setting_mismatch |
|---|---|---|
| b0 | 0.000 | 0.286 |
| b1 | 0.000 | 0.000 |
| full | 0.250 | 0.500 |
| full_no_verifier | 0.353 | 0.800 |

## Verifier decisions

Every full-system finding, by what its gold label says it is, summed over runs. Real errors should be verified; findings on clean controls should be dropped.

| Finding | Verified | Human review | Dropped |
|---|---|---|---|
| real_error | 4 | 3 | 0 |
| wrong_class | 0 | 0 | 0 |
| clean_control | 0 | 0 | 0 |
| unlabelled | 20 | 25 | 0 |

## Runs

| Run | System | Papers | Macro-F1 | TP | FP | FN | Clean flagged | Unlabelled (unique) | LLM calls | Tokens in / out | Cost ($) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rq1-c5-b0-r1 | b0 | 8 | 0.143 | 1 | 0 | 19 | 0 of 6 | 7 (6) | 10 | 137,058 / 102,986 | 0.49 | 2 failed; 12 labels on papers not run |
| rq1-c5-b1-r1 | b1 | 6 | 0.000 | 0 | 0 | 20 | 0 of 6 | 1 (1) | 168 | 4,363,375 / 59,801 | 1.63 | mocked: citation tools; 4 failed; 8 labels on papers not run |
| rq1-c5-full-r1 | full | 8 | 0.375 | 4 | 0 | 16 | 0 of 6 | 20 (12) | 197 | 524,685 / 145,646 | 0.94 | mocked: citation; 2 failed; 5 labels on papers not run |
| rq1-c5-full-r1 | full_no_verifier | 8 | 0.576 | 7 | 0 | 13 | 0 of 6 | 49 (23) | 197 | 524,685 / 145,646 | 0.94 | mocked: citation; 2 failed; 5 labels on papers not run |
