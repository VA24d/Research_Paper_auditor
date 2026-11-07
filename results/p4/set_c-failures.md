# Failure analysis

## Where the full system loses each planted error

Each planted error in each full-system run, by the first stage it did not pass.

| Stage | method_mismatch | metric_mismatch | setting_mismatch | All |
|---|---|---|---|---|
| found | 110 | 6 | 113 | 229 |
| human review | 4 | 0 | 22 | 26 |
| no finding | 186 | 0 | 0 | 186 |
| unverifiable | 0 | 3 | 45 | 48 |
| total | 300 | 9 | 180 | 489 |

Most common reasons, human review:

- 26 x The evidence is in its sources, but the specialist marked it uncertain.

Most common reasons, unverifiable:

- 18 x every repository value of batch_size was judged to belong to something other than the experiment the sentence describes
- 15 x every repository value of learning_rate was judged to belong to something other than the experiment the sentence describes
- 9 x every repository value of epochs was judged to belong to something other than the experiment the sentence describes
- 3 x the repository computes several metrics (bleu, <x> matthews_correlation, pearson_correlation, spearman_correlation), and the code does not s
- 3 x every repository value of dropout was judged to belong to something other than the experiment the sentence describes

## Stability across runs

How many runs of each system found each planted error (verified flags only for the full system).

- b0 (3 runs): 3 of 3: 3, 2 of 3: 0, 1 of 3: 5, 0 of 3: 155
- b1 (2 runs): 2 of 2: 2, 1 of 2: 5, 0 of 2: 156
- full (3 runs): 3 of 3: 74, 2 of 3: 2, 1 of 3: 3, 0 of 3: 84

## If uncertain findings counted (D23)

| Run | Macro-F1 as run | Macro-F1 with uncertain counted | Unlabelled as run | Unlabelled with uncertain counted |
|---|---|---|---|---|
| p4-setc-full-r1 | 0.698 | 0.729 | 237 | 370 |
| p4-setc-full-r2 | 0.709 | 0.739 | 227 | 358 |
| p4-setc-full-r3 | 0.701 | 0.739 | 251 | 382 |
