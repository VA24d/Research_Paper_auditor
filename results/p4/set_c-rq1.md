# Set C: full system, B0 and B1 (P4.6)

Only verified flags count for systems with a verifier; every flag counts for B0, B1 and `full_no_verifier` (D3). Values are the mean ± sample standard deviation over runs. Unlabelled flags match no gold label and are not scored (D14).

| System | Runs | Macro-F1 | Precision | Recall | FP rate (clean) | Flags / original | Originals flagged | Unlabelled | Review / paper | Coverage | Flags backed | Quotes grounded | LLM calls / paper | Cost / paper ($) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b0 | 3 | 0.048 ± 0.015 | 1.000 ± 0.000 | 0.029 ± 0.009 | 0.000 ± 0.000 | 1.63 ± 0.25 | 0.76 ± 0.03 | 125.3 ± 18.9 | - | - | 0.00 ± 0.00 | - | 1.0 ± 0.0 | 0.047 ± 0.002 |
| b1 | 2 | 0.046 ± 0.007 | 1.000 ± 0.000 | 0.028 ± 0.004 | 0.000 ± 0.000 | 0.39 ± 0.23 | 0.21 ± 0.05 | 28.5 ± 20.5 | - | - | 0.00 ± 0.00 | - | 20.1 ± 0.0 | 0.542 ± 0.014 |
| full | 3 | 0.703 ± 0.006 | 1.000 ± 0.000 | 0.468 ± 0.009 | 0.000 ± 0.000 | 3.12 ± 0.04 | 0.94 ± 0.01 | 238.3 ± 12.1 | 1.68 ± 0.01 | 0.95 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 13.2 ± 0.2 | 0.079 ± 0.000 |
| full_no_verifier | 3 | 0.736 ± 0.006 | 1.000 ± 0.000 | 0.521 ± 0.011 | 0.000 ± 0.000 | 4.71 ± 0.04 | 0.95 ± 0.00 | 374.0 ± 12.0 | - | 0.95 ± 0.00 | 1.00 ± 0.00 | 1.00 ± 0.00 | 13.2 ± 0.2 | 0.079 ± 0.000 |

## F1 per class

| System | method_mismatch | metric_mismatch | setting_mismatch |
|---|---|---|---|
| b0 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.144 ± 0.044 |
| b1 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.139 ± 0.020 |
| full | 0.536 ± 0.016 | 0.800 ± 0.000 | 0.771 ± 0.007 |
| full_no_verifier | 0.551 ± 0.018 | 0.800 ± 0.000 | 0.857 ± 0.000 |

## Verifier decisions

Every full-system finding, by what its gold label says it is, summed over runs. Real errors should be verified; findings on clean controls should be dropped.

| Finding | Verified | Human review | Dropped |
|---|---|---|---|
| real_error | 229 | 26 | 0 |
| wrong_class | 0 | 0 | 0 |
| clean_control | 0 | 0 | 0 |
| unlabelled | 691 | 377 | 0 |

## Runs

| Run | System | Papers | Macro-F1 | TP | FP | FN | Clean flagged | Unlabelled (unique) | LLM calls | Tokens in / out | Cost ($) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| p4-setc-full-r1 | full | 80 | 0.698 | 75 | 0 | 88 | 0 of 72 | 237 (142) | 1047 | 4,777,877 / 715,088 | 6.26 | mocked: citation |
| p4-setc-full-r1 | full_no_verifier | 80 | 0.729 | 83 | 0 | 80 | 0 of 72 | 374 (177) | 1047 | 4,777,877 / 715,088 | 6.26 | mocked: citation |
| p4-setc-full-r2 | full | 80 | 0.709 | 78 | 0 | 85 | 0 of 72 | 227 (139) | 1080 | 4,934,772 / 693,596 | 6.30 | mocked: citation |
| p4-setc-full-r2 | full_no_verifier | 80 | 0.739 | 86 | 0 | 77 | 0 of 72 | 362 (172) | 1080 | 4,934,772 / 693,596 | 6.30 | mocked: citation |
| p4-setc-full-r3 | full | 80 | 0.701 | 76 | 0 | 87 | 0 of 72 | 251 (155) | 1044 | 4,816,326 / 710,515 | 6.28 | mocked: citation |
| p4-setc-full-r3 | full_no_verifier | 80 | 0.739 | 86 | 0 | 77 | 0 of 72 | 386 (188) | 1044 | 4,816,326 / 710,515 | 6.28 | mocked: citation |
| p4-setc-b0-r1 | b0 | 80 | 0.051 | 5 | 0 | 158 | 0 of 72 | 104 (75) | 80 | 1,749,721 / 631,679 | 3.68 |  |
| p4-setc-b0-r2 | b0 | 80 | 0.061 | 6 | 0 | 157 | 0 of 72 | 140 (87) | 80 | 1,749,721 / 646,167 | 3.74 |  |
| p4-setc-b0-r3 | b0 | 80 | 0.032 | 3 | 0 | 160 | 0 of 72 | 132 (94) | 80 | 1,749,721 / 698,653 | 3.93 |  |
| p4-setc-b1-r1 | b1 | 80 | 0.042 | 4 | 0 | 159 | 0 of 72 | 43 (31) | 1610 | 53,683,046 / 619,601 | 42.59 | mocked: citation tools |
| p4-setc-b1-r2 | b1 | 80 | 0.051 | 5 | 0 | 158 | 0 of 72 | 14 (13) | 1610 | 55,727,153 / 638,238 | 44.19 | mocked: citation tools |
