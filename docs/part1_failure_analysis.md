# Part 1 failure analysis (task P1.17)

Where parsing, planning and orchestration lose or distort what the checkers need. The data
is set C (40 papers, each original and injected, with 163 planted errors), its runs in
`results/p4/runs` (three full-system runs, three of B0, three of B1 and a pilot), and the
parser evaluation on ten arXiv papers ([parser_evaluation.md](parser_evaluation.md)).
Nothing here needed new model calls except the integrity re-run under "Sentence
splitting" ($0.09).

## Parsing

**Citation markers.** 765 of 797 citation markers (96.0%) in the ten evaluation papers are
linked to a bibliography entry, and the hand-checked sample of 100 links scores 100 of 100.
The unlinked 4% are mostly author-year markers in unusual forms.

**Sentence splitting.** GROBID ends a sentence at an abbreviation's full stop. In the 80
set C paper versions (30,886 sentences), 162 sentences end with "Fig.", "Figs.", "Eq.",
"e.g.", "i.e." or a similar abbreviation while the next one carries on in lower case, with
a digit or with a bracket ("The global parsing map is shown in Figs." / "5(b-c), while
..."). Figure, table and footnote text that GROBID places inside a paragraph cuts more
sentences in two. The integrity screen read these pieces as unfinished text: in
`p4-setc-full-r1`, 147 of the 152 `unfinished_or_placeholder_content` flags were of this
kind. Set C has no labels for the class, so none of them lowered the reported precision.
Since #85 such findings are uncertain and go to human review; re-run on the ten paper
versions with the most flags, 55 flags became 8 uncertain findings. The splitting itself is
not changed, because merging sentences renumbers the sentence ids that the gold labels of
sets B, C and D point to.

**Tables.** On the 39 set C papers with a LaTeX source, GROBID's tables hold 57% of the
numbers in the papers' tables, the PDF's layout text 43%, both together 72%, and the PDF
text inside each table's crop 96% (D22). The integrity checks read the crop tables. The
planner reads settings only from GROBID's tables; it skips the layout and crop tables so
that a setting is not read twice. A hyperparameter that only a garbled GROBID table holds
is therefore not checked. Set C cannot show how often this happens (see Planning).

## Planning

**Planted errors.** Planned again without a model on the 80 set C paper versions (13,020
tasks), every planted error has a task: all 60 setting errors and all 3 metric errors on
their labelled sentence, and a method task for each of the 100 method labels. This is so
by construction: set C planted a setting error only where the planner had found the
setting and reads the new value back ([set C README](../datasets/set_c/README.md)). It
shows that planning loses none of the planted errors; it does not measure how many real
settings the planner misses.

Where the planted errors are lost instead, over the three full-system runs (489 planted
error instances, [set_c-failures.md](../results/p4/set_c-failures.md)): 186 method
instances got no finding, 48 setting and metric instances were unverifiable (mostly "every
repository value of batch_size was judged to belong to something other than the
experiment"), and 26 went to human review because the checker marked them uncertain. All of
these are in the checkers, after planning.

**The model step (P1.12).** Run live on two papers, Gemini 2.5 Flash left the citation
markers out of most of its quotes, so 48 of 126 claims in 2005.11401 fell back to the whole
sentence. Since #71 a quote may leave out the markers; on the same answers 89 of 126 claims
are narrowed and 5 are refused (quotes that skip words with "...").

## Orchestration

**Full system.** Each of the three runs planned about 13,270 tasks over the 80 paper
versions. No task failed, none was retried, and no paper failed.

**Baselines.** Every paper failure in the batches came from model calls, not from the
pipeline:

| Run | Paper failures | Cause | Papers with predictions |
|---|---|---|---|
| `p4-pilot-b0-r1` | 1 | rate limit (HTTP 429) | 3 of 3 |
| `p4-setc-b1-r1` | 22 | 21 budget, 1 rate limit | 80 of 80 |
| `p4-setc-b1-r2` | 27 | 26 budget, 1 rate limit | 80 of 80 |
| `p4-setc-b1-r3` | 210 records, 70 papers | budget | 10 of 80 |

A failed paper was recorded in `failures.jsonl` and the batch went on; running the same
command again retried it, which is how the first two B1 runs reached all 80 versions. The
budget failures came from the run's $40 limit and the $125 ledger limit. B1 sends the
paper again on every turn and stops at 20 model calls; 77-82% of its input was read from
Vertex's prompt cache. Google documents a 90% discount on such input, which would have made
B1's first run $14.62 instead of $42.59, but the bill for 7 November (Rs 11,805.69 for all
the set C runs: 137M input tokens, 92M of them cached, 5.5M output) matches the full price
($123, about Rs 96 to the dollar at the repository's prices for 3.8 Flash) and not the
discounted one ($61, which would mean Rs 194 to the dollar). The limits count cached input in full, which the
bill bears out.

**Fixed after these runs.**
- One failed tool call (a rate-limited GitHub request) ended a B1 paper; the error now goes
  back to the agent as a tool message.
- An unusable answer in the verifier's second reading or the math challenge failed the
  paper; those findings now go to human review (#85).
- A paper that ran out of model calls was saved as complete and scored; it is now saved as
  `predictions.incomplete.jsonl` (#69).
- Calls sent together could all pass the dollar check before any was charged; each now
  reserves its most possible cost while it waits (#69).
- B0 and B1 answers that could not become predictions were only logged; a batch now keeps
  their reasons in `dropped.jsonl`.

## Known limits

- The planner does not read settings from the layout or crop tables (see Tables).
- Sentence splitting at abbreviations is left as GROBID does it (see Sentence splitting).
- Findings whose only evidence is the paper itself (math errors, result inconsistencies,
  unfinished content) are never merged as repeats (D20), so one problem reported by two
  tasks counts twice in the report.
- A rate-limited call still fails its paper once the provider's retries are spent; the
  batch has to be run again.
