# Set C: paper-code consistency

Set C tests the code agent on 40 published papers with their official code. Each paper
appears twice in the experiment dataset: the original, and an injected copy whose text
(and sometimes repository) carries planted errors.

| Item | Count |
|---|---|
| Paper and code pairs | 40 (8 with a discrepancy reported by users) |
| Injected setting errors (`setting_mismatch`, `metric_mismatch`) | 63 |
| Clean controls (`none`), labelled in both the original and the injected copy | 36 (72 labels) |
| Method discrepancies (`method_mismatch`) | 94: 8 reported, 86 synthetic |
| Gold labels in `labels.jsonl` | 237 |

Injected settings by key: batch size 16, epochs 15, learning rate 11, optimizer 9,
dropout 6, metric 2, weight decay 2, training steps 2. Clean controls cover learning rate
(9), optimizer (9), epochs, dataset, maximum sequence length and batch size (3 each),
dropout and weight decay (2 each), metric and warmup steps (1 each).

No item tests `data_split_mismatch`, `dataset_size_mismatch`,
`label_distribution_mismatch` or `data_leakage`. None of the 169 candidate papers stated a
split, a dataset size or a class balance that its repository confirmed, so there was
nothing to inject into without guessing the ground truth.

## Source

The pairs come from SciCoQA (Baumgärtner et al., "SciCoQA: Quality Assurance for Scientific
Paper–Code Alignment", ACL 2026; data on Hugging Face as `UKPLab/scicoqa` at revision
`3ccd45dc5de772c3c82365c9ed47263aac2dd3c9`, CC BY 4.0). SciCoQA lists papers with their
repositories pinned to a commit, and discrepancies that GitHub issues, reproducibility
reports, or a synthetic code change describe. Of its pairs, the 169 with a versioned arXiv
PDF and a GitHub repository at a commit formed the pool (one pair per paper).

## How the labels were made

`scripts/set_c.py` does every step; the commands are in its docstring.

1. `collect` downloaded each arXiv PDF, parsed it with GROBID 0.9.1, and ran the code agent
   on every setting the planner found, against the repository at the SciCoQA commit.
2. `select` kept pairs where the code confirmed at least two settings, then pairs with
   SciCoQA discrepancies, up to 40. For each pair it chose settings the repository
   confirms (with the file and line as evidence), changed the paper's value for up to two
   of them and kept up to two others as clean controls. The choice is seeded by the
   arXiv id, so rerunning gives the same set.
3. An injected value was kept only if the planner reads it back at the same place in the
   injected copy and no repository value for that key equals it (D18). The new value
   keeps the paper's notation: 3e-4 becomes 3e-3, 25,000 becomes 50,000, AdamW becomes SGD.
4. SciCoQA's discrepancies became `method_mismatch` labels with a `document` anchor and
   target `method`. A reported one is in both copies, since the original code has it. A
   synthetic one is only in the injected copy, whose repository is a copy of the snapshot
   with SciCoQA's code change applied. Changes that did not apply cleanly were left out.

## Files

| File | Contents |
|---|---|
| `pairs.jsonl` | The 40 pairs: arXiv version, PDF SHA-256, repository and commit, SciCoQA ids |
| `labels.jsonl` | Gold labels (`GoldLabel`) |
| `injections.jsonl` | One record per injected setting: where, the original and the new value, and the evidence type |
| `method_issues.jsonl` | The SciCoQA discrepancies used, with SciCoQA's description, paper quotes and code changes |

PDFs, parsed papers, injected copies, patched repositories and the experiment dataset
file stay under `data/set_c/` (D13). To recreate them, run `collect` on the pool (or only
for the 40 pairs) and then `build`; `build` refuses a PDF whose hash differs from
`pairs.jsonl`, and re-applies each injection only where the original value is still found.

## Before freezing (D18)

- A second team member must review at least 20% of the labels. `review-sample` writes a
  fixed random 20% sample with the sentence, rationale and evidence for each label; fill in
  `reviewer` and `agree`, then run `apply-review`. Until then `validate` reports the
  missing reviews and the set is not frozen.
- The method discrepancies need an LLM: without one the code agent marks every method task
  unverifiable, which counts as a miss.
- Run the experiments with `--analyse-data` only if data labels are added later; none of
  the current labels needs it.

## Pilot run

The full system with the placeholder verifier and no LLM, over the 80 papers (originals and
injected copies), scored before the verifier with
`scripts/set_c.py score datasets/set_c/labels.jsonl <run> --before-verifier`:

| Class | Precision | Recall |
|---|---|---|
| `setting_mismatch` (61 labels) | 1.00 | 1.00 |
| `metric_mismatch` (2) | 1.00 | 1.00 |
| `method_mismatch` (102) | no findings | 0.00 |

No clean control was flagged. 228 flags matched no label; they may be false alarms or real
mismatches already in the papers, and a sample has to be adjudicated by hand (D14).

Read the setting scores with care. Injections were only made where the code agent had
already found the paper's value in the repository, so recall on them measures whether the
agent notices a changed value, not whether it can find settings in unfamiliar code. The
method class scores zero because no LLM was configured for the run.
