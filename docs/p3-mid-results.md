# P3 mid-term results (2026-10-30)

All measured numbers for Part 3 (code and data agent, template checker, theory checker,
integrity and results checks) as of this date, on `main`. The decisions behind each check
and the full tables are in [decisions.md](decisions.md): D11 (code), D18 (labels), D21
(theory), D22 (integrity and results). Unless a row says otherwise the model is Gemini
3.8 Flash on Vertex AI, and costs come from the run's spend ledger.

## 1. What Part 3 checks

| Check | Error classes | Needs a model | Status |
|---|---|---|---|
| Code agent: paper settings against the repository (learning rate, batch size, epochs, optimizer, dropout, metric, dataset, splits) | `setting_mismatch`, `metric_mismatch`, `data_split_mismatch`, `dataset_size_mismatch` | no (a second look uses one for uncertain cases) | done |
| Data analysis mode: downloads the dataset files, counts rows, label balance, test rows also in training | `dataset_size_mismatch`, `label_distribution_mismatch`, `data_leakage` | no | done, opt-in |
| Method check: method section against the code that implements it | `method_mismatch` | yes | done ("rag" mode) |
| Claim loop: a tool-using agent checks claims about data and experiments in the repository | `method_mismatch`, data classes | yes | done, opt-in |
| Code defects: undefined names, syntax errors, missing local modules in experiment scripts | `code_defect` | no | done |
| Template checker: page limit, margins, font size, paper size, anonymity, sections | `template_violation` | no | done |
| Theory checker: proofs and derivations | `math_error` | yes | done, opt-in |
| Integrity screen and results check: placeholder text, GRIM-impossible means, prose claims a table contradicts | `unfinished_or_placeholder_content`, `result_inconsistency` | partly | done |

## 2. Datasets

| Set | What it is | Size | Labels |
|---|---|---|---|
| Set C | SciCoQA paper-code pairs with planted errors | 40 pairs, 80 paper versions, 235 gold labels: 63 setting and metric errors, 36 clean controls (72 labels), 92 method discrepancies (84 synthetic, 8 reported by users) | built; waiting for the 20% second review (D18) |
| Set D | Copies of the 40 set C papers with planted integrity and result errors | 192 plants (set D2: 10 changed-number plants instead of 7) | built from deterministic templates |
| SPOT | Published maths and CS papers with an annotated proof error | 26 papers, 29 errors, split into a dev and a test half of 13 papers each | external benchmark, LLM judge |
| Real cases | arXiv withdrawals and revisions, ICLR review cases | 33 + 30 papers | authors' own corrections |
| Template fixtures | Hand-built papers, one rule each | 15 cases | hand-built |

## 3. Results

### 3.1 Settings, metrics and code defects (no model)

Full set C, 80 paper versions, 21 s with the repository snapshots cached:

| Class | Precision | Recall |
|---|---|---|
| `setting_mismatch` (60 labels) | 1.00 | 1.00 (60 of 60) |
| `metric_mismatch` (3) | 1.00 | 0.67 (2 of 3) |
| clean controls flagged | | 0 of 72 |

The settings were planted only where the agent had already found the paper's value in the
code, so recall measures noticing a changed value, not finding settings in unfamiliar code.
The run also raises 152 flags that match no label (74 distinct across the two copies of a
paper); they are false alarms or real mismatches in the published papers and still need
adjudication. The missed metric is in 2004.09297, whose repository computes several metrics.

Code defects on set D: 26 of 26 planted undefined names and 13 of 13 planted syntax errors.
Template checker: 15 of 15 fixture cases (`scripts/template_eval.py`).

### 3.2 Method check

57 SciCoQA synthetic discrepancies in 21 papers of 25 pairs, findings counted after the
evidence gate, judged by 3.8 Flash:

| Setup | Found | Findings | Cost |
|---|---|---|---|
| "rag", 3.8 Flash, 60k characters of code (default) | 17 / 57 | 25 | about $1 |
| Gemini 3.1 Pro Preview | 19 / 57 | 57 | $3.60 |
| DeepSeek V4.1 Flash, thinking on | 19 / 57 | 63 | about $0.65 |
| union of three 3.8 runs | 21 / 57 | | about $3 |
| 3.5 Flash-Lite | 7 / 57 | 18 | $0.19 |

Where the misses come from (a replay of the cached answers): the changed line was not shown
14 times, shown but not reported 26 times, reported and then dropped by the evidence gate 7
times. Anchoring a quote on the sentence that holds its opening words took the earlier "rag"
run from 11 to 19 of 65 discrepancies. Reading, not retrieval, is now the main gap.

### 3.3 Theory checker (SPOT)

| Setup | Errors found | Papers | Cost |
|---|---|---|---|
| Test half, frozen defaults (triage, one look), first judge | 8 / 14 | 7 / 13 | $1.68 |
| the same, both judges agree | 5 / 14 | 4 / 13 (31%) | |
| Test half, two looks, both judges | 6 / 14 | 5 / 13 | $2.48 |
| Dev half, two looks, 3.8 deep checks, first judge | 6 / 13 | | $4.21 |

The published reference (PAT, arXiv 2606.28277, Gemini 3.1 Pro) reports 55.2% recall for a
single call and 89.7% for its full system. The test half has been scored for several setups,
so its numbers are less clean than a first look.

### 3.4 Integrity and results checks (set D)

| Plant | Found |
|---|---|
| placeholder or simulated-results sentence | 40 / 40 |
| GRIM-impossible mean | 40 / 40 |
| reversed comparison of two table rows | 28 / 33 |
| wrong size of a gain between two table rows | 21 / 33 |
| number the text quotes from a table, changed (set D2) | 2 / 10 |

The changed-number rule adds no finding on 254 clean papers. Tables are read from crops of
the PDF: on a 12-paper benchmark with LaTeX ground truth the crop text recalls 98% of table
numbers, against 51% for layout text with GROBID.

On real papers the results check found a reversed comparison in arXiv 2312.11580 ("PlaNet-S
(0.73) is higher than U-Net (0.78)"), and the theory checker found a mean score of 8.93 in
2512.21706 that its own Table 6 gives as 6.30.

## 4. Ablations

| Question | Variants | Outcome |
|---|---|---|
| How to find the code a method sentence describes | six files by word overlap; BM25 over function chunks; nomic embeddings; BM25 + embeddings; bge-base; nomic + ettin reranker | changed line shown 35, 48, 59, 57, 58, 57 of 84; nomic kept |
| How to read it | "whole", "pairs", "rag", grep windows, claim loop, claim loop told the sentence | on 10 pairs (28 discrepancies): 2, 0, 4, 0, 1, 2 found; "rag" kept |
| More code or a different order | 60k against 120k characters; reversed chunk order | 17, 16, 16 of 57; no gain |
| Which reading model | 3.8 Flash, 3.1 Pro, DeepSeek, Kimi K2, GLM-5, Flash-Lite | 17, 19, 19, 13, 11, 7 of 57; 3.8 kept on cost |
| Theory pipeline shape | cascade (screen then deep), one call, triage, pooled triage, two looks | 2/29, 4/29, 8/14, 7/14, 6/14 (both judges); triage with two looks kept |
| Claim-loop effort and limits | medium effort; low effort with a 540 s per-claim limit | 32 against 0 task timeouts on 3 papers, $0.63 against $0.47 |
| Claim-loop context pruning | full history against older tool output removed | no input saving; dropped |
| Table reading | GROBID, layout text, both, crop text, crop image | 57%, 43%, 72%, 96 to 98%, 96% of numbers |

## 5. Known issues

- The 152 unlabelled set C flags have not been adjudicated, so precision on unlabelled
  mismatches is unknown.
- Set C still needs its 20% second review before it is frozen (D18).
- SPOT grades come from model judges that agree on about half of their matches; no person has
  checked them.
- Changed table numbers in prose are rarely caught (2 of 10), and only 10 papers offer a fair
  plant.
- The claim loop finds little that the method check does not, at several times the calls.
- Citation checks make full runs network-bound; set C runs here turn them off.

## 6. What is next for P3

- Adjudicate a sample of the unlabelled set C flags, and finish the set C second review.
- Method check: report what the model is shown (26 of the misses), for example by asking for
  each method sentence separately with its own code.
- Theory: a third judge for the cases where the two disagree; a final run on the test half.
- Results: pair "respectively" lists and column names written differently from the header.
- Full set C and set D runs of the full system against B0 and B1 for RQ1.
