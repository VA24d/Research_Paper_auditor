# Research Paper Auditor: example audit report

27 October 2026, Vijay Aravynthan. Shared copy: <https://claude.ai/artifact/BBSoqfCsHm9WTZRrDt7NLN>

## Summary

The auditor finds changed training settings almost always, finds about a third of method-level code mismatches, catches about a third to a half of known maths errors, and rarely catches a changed number in a results table. Every example below comes from a measured run in October 2026; the model is Gemini 3.8 Flash unless stated.

| Area | Test set | Found | False alarms |
| --- | --- | --- | --- |
| Training settings in code (learning rate, batch size, epochs, optimizer, dropout) | Set C: 60 planted changes in 40 papers with their official code | 60 of 60 | 0 of 72 clean controls |
| Evaluation metric in code | Set C: 3 planted changes | 2 of 3 | 0 |
| Method described vs method implemented | Set C: 57 SciCoQA synthetic code changes, 25 paper-code pairs | 17 of 57 | 25 findings, 3 judged not real |
| Maths errors in proofs and derivations | SPOT dev half: 13 annotated errors in 12 papers | 6 of 13 (one judge), 4 of 13 (both judges) | not measured |
| Changed numbers in prose vs tables | Set D: 10 fair number plants | 2 of 10 (no-model rule) | 0 on 254 clean papers |

## Detected: code and method mismatches

Each row is a finding the method check reported, with the code lines it cited. Most come from SciCoQA's synthetic code changes on set C; the last two are from the 27 October run on three hard papers.

| Paper | What the paper says | What the code does | Cited lines |
| --- | --- | --- | --- |
| 2001.07645 (SAUNet) | Total loss = cross entropy + Dice + edge loss | `DualLoss.forward` returns `ce + edge`; Dice is left out | `loss.py:149` |
| 2002.07128 (GAT) | The last layer averages the attention heads | `gat2` is built with `concat=True`, so heads are concatenated | `transduction/trans.py:285` |
| 2208.04654 (NGCC-PHAT) | Trained with Adam | Trained with SGD with momentum | `main.py:164` |
| 2004.10016 | Table I reports top-1 accuracy | `eval.py` computes top-5 (`topk = min(5, class_num)`) | `code/eval.py:83` |
| 2110.13653 | Height and age labels are standardised (mean, variance) | Min-max scaling | `TIMIT/dataset.py:89` |
| 2202.03772 (Particle Transformer) | No positional encoding, since particles are permutation invariant | Adds a linear 1D positional encoding | `networks/example_ParticleTransformer.py:21` |
| 2002.00718 (MiB) | New-class classifier initialised from the background classifier | Weights and bias set to zero | `segmentation_module.py:85` |
| 1910.10261 (QuartzNet) | Layer C1 uses stride 2 | First block configured with stride 1 | `_quartznet5x5_config` |
| 2006.06958 (Stable SGD) | No data shuffling | Every training `DataLoader` has `shuffle=True` (also in the original repository, so a real mismatch set C does not label) | `stable_sgd/data_utils.py:27` |

## Detected: maths errors

On the SPOT dev half (13 annotated errors, two looks per paper), all three judges agreed on these four. Each finding names the step that fails, three of them with a counterexample.

| Paper | Where | Annotated error | What the auditor reported |
| --- | --- | --- | --- |
| 2405.05635, Large bricks and join-irreducible torsionfree classes | Proposition 3.9 | A logical gap in the proof | A branching tree quiver such as V1 ← V0 → V2 has no common upper bound for V1 and V2, so its path category is not filtered and the colimit is not exact |
| 2406.01892, Triviality of unramified Iwasawa modules | Section 4.2.3 | A wrong step in the derivation | Extensions of trivial modules are classified by H¹(Γ, X); with a non-zero extension class the step in the proof of Proposition 4.2.1 fails |
| 2406.06431, CR functions at CR singularities | Theorem 4.2 | A logical gap in the proof | When M is not generic at CR points, local holomorphic extensions are not unique and need not continue along M; z₂/(z₁ − 2) extends 0 near z₁ = 0 but has a pole at z₁ = 2 |
| 2407.06913, Differentially private all-pairs shortest paths | Lemma 3.1 (Sections 3 and 4) | Assumes the shortest-path tree topology is public | With m edge-disjoint paths of length t, the minimum of m sums of Laplace noise drifts below t, so the perturbed distance is biased and the stated bound fails |

Four more errors, in 2401.06455, 2404.04502 and 2401.16191 (two errors), were matched by one or two of the three judges only.

## Detected: result and table-number errors

Two published papers contain a prose claim that its own table contradicts, and the auditor reported both.

| Paper | Claim in the text | What the numbers show | Which check found it |
| --- | --- | --- | --- |
| 2312.11580 | PlaNet-S (0.73) is higher than U-Net (0.78) | 0.73 is lower than 0.78, so the comparison is reversed | Results check |
| 2512.21706 | The simulation data gets the highest mean score, 8.93, ahead of GPT-4 (7.06) and Moshi (4.25) | Table 6 gives GPT-4 7.07, Moshi 6.85 and the simulation data 6.30, the lowest | The maths deep check, reading the table as text; the results check missed it |

## Missed: examples and why

Most method misses are reading failures: in a diagnosis run over the 57 method changes, 26 were in code the model was shown but did not report, 14 were never shown, and 7 were reported and then dropped by the evidence gate.

| Area | Paper | The error | Why it was missed |
| --- | --- | --- | --- |
| Method | 2003.10606 | The paper counts a grounding as correct at IoU ≥ 0.5; the evaluation code uses 0.4 | Shown to the model, not reported |
| Method | 1910.10261 | The paper trains with NovoGrad and cosine annealing; the code defaults to SGD with momentum | Shown to the model, not reported |
| Method | 2006.06958 | The paper says no data augmentation; the code adds random crops and flips for Split CIFAR-100 | The changed lines were never retrieved |
| Method | 2012.15000 | The paper computes mAP over the two event classes only; the code includes the background class | The changed lines were never retrieved |
| Metric | 2004.09297 | A planted change of the reported metric | The repository computes several metrics, so the agent cannot tell which one the paper reports and marks it unverifiable |
| Maths | 2401.09543 | Claim 7 states a linear kernel; only a quadratic one follows | The auditor found a different flaw in the same induction (the set J_m) but never named the kernel bound |
| Maths | 2406.11623 | The Green function of a geodesic ball is wrongly said to vanish on its boundary sphere | None of the three issues reported touch the Green function |
| Maths | 2407.09178 | An error in the Kähler twist construction | Reported a structural gap elsewhere in the proof |
| Maths | 2405.01133, 2408.16162 | Proposition 2.3 breaks Lemmas 3 and 4; an automorphism extension silently changes values on elements generated by Y | No matching issue reported |
| Table number | 2304.08467 | 50.8 changed to 46.0 in a sentence listing several results with "respectively" | The rule cannot pair each number with its row in a respectively list |
| Table number | 2002.00718 | 35.1 changed to 38.5, quoted as the result "on the overlapped" setting | The text names the column differently from the table header, so no cell matches |
| Table number | 2406.05992 | 4.1119 changed to 3.3797, written as "3.3797G FLOPs" | The number carries its unit, so it is not matched to the table cell |

## How to read these numbers

- **Set C** pairs 40 SciCoQA papers with their code at a fixed commit. Settings were planted only where the agent had already found the paper's value in the code, so 60 of 60 measures noticing a changed value, not finding settings in unfamiliar code.
- **Method changes** are SciCoQA's synthetic code edits. A hit is scored by description as well as by sentence, because the sentence-based scorer also credits a different finding on the labelled sentence.
- **SPOT** grades come from model judges; on the dev half the first judge matched 6 of 13 errors and the second 4, all four among the first judge's six, so both counts are given.
- **Set D** plants are changed numbers in prose that a table contradicts; only 10 are fair tests, too few to measure recall closely.
- Several detected items, such as the shuffling in 2006.06958 and the two published result errors, are real problems that no test set labels.
