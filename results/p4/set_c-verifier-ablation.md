# Set C, 80 paper versions: verifier ablation (replayed, no model calls)

Only verified flags count for systems with a verifier; every flag counts for B0, B1 and `full_no_verifier` (D3). Values are the mean ± sample standard deviation over runs. Unlabelled flags match no gold label and are not scored (D14).

| System | Runs | Macro-F1 | Precision | Recall | FP rate (clean) | Flags / original | Originals flagged | Unlabelled | Review / paper | Coverage | Flags backed | Quotes grounded | LLM calls / paper | Cost / paper ($) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 1 | 0.521 | 1.000 | 0.239 | 0.000 | 0.72 | 0.50 | 63.0 | 2.29 | 0.96 | 1.00 | 1.00 | - | - |
| full_no_verifier | 1 | 0.600 | 1.000 | 0.380 | 0.000 | 2.73 | 0.80 | 241.0 | - | 0.96 | 0.98 | 1.00 | - | - |

## F1 per class

| System | method_mismatch | metric_mismatch | setting_mismatch |
|---|---|---|---|
| full | 0.000 | 0.800 | 0.763 |
| full_no_verifier | 0.000 | 0.800 | 1.000 |

## Verifier decisions

Every full-system finding, by what its gold label says it is, summed over runs. Real errors should be verified; findings on clean controls should be dropped.

| Finding | Verified | Human review | Dropped |
|---|---|---|---|
| real_error | 39 | 23 | 0 |
| wrong_class | 0 | 0 | 0 |
| clean_control | 0 | 0 | 0 |
| unlabelled | 55 | 160 | 0 |

## Runs

| Run | System | Papers | Macro-F1 | TP | FP | FN | Clean flagged | Unlabelled (unique) | LLM calls | Tokens in / out | Cost ($) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| setc-rv-p3-code-fixes-full-r1 | full | 80 | 0.521 | 39 | 0 | 124 | 0 of 72 | 63 (31) | 0 | 0 / 0 | - | mocked: citation |
| setc-rv-p3-code-fixes-full-r1 | full_no_verifier | 80 | 0.600 | 62 | 0 | 101 | 0 of 72 | 241 (89) | 0 | 0 / 0 | - | mocked: citation |
