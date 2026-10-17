# SPOT subset: errors in proofs and equations

This is the outside benchmark for the theory checker (decision D21). It holds the 29
equation and proof errors of SPOT's Mathematics and Computer Science papers, 26 papers in
all. Google's Paper Assistant Tool reports on the same subset (arXiv 2606.28277): 55.2%
recall for one Gemini 3.1 Pro call, 89.7% for PAT.

| Item | Count |
|---|---|
| Papers | 26 (20 Mathematics, 6 Computer Science) |
| Errors | 29 (22 led to a retraction, 7 to an erratum) |

## Source

SPOT, from Son et al., "When AI Co-Scientists Fail: SPOT, a Benchmark for Automated
Verification of Scientific Research" (arXiv 2505.11855). The data is on Hugging Face as
`amphora/SPOT` at revision `0641b6aea2495b7262a349856d24ebeea93a8e58`, CC BY 4.0. Each row
gives a paper as text pages with LaTeX, plus one error: its category, location, severity
and a description written by the annotators. `errors.jsonl` keeps those fields for the
subset. The paper text stays under `data/spot/` and is not committed (D13).

## Running it

```
uv run python scripts/spot.py fetch --data data/spot --out datasets/spot
AUDITOR_THEORY_CHECK_ENABLED=true AUDITOR_THEORY_MAX_SEGMENTS=12 \
  uv run auditor batch data/spot/dataset.jsonl --experiment spot --systems full b0 --runs 1
uv run python scripts/spot.py score runs/spot-full-r1 --errors datasets/spot/errors.jsonl \
  --data data/spot --before-verifier
```

`fetch` turns each text page into a section and each paragraph into a sentence, so the
planner's segments are runs of formal pages, as in PAT. With 12 segments allowed, every
formal page of every paper is checked once: 107 theory calls, about 1.5 million input
tokens. B0 adds one call per paper.

`score` asks an LLM judge, once per annotated error, which of the run's `math_error`
reports for that paper describe it. `--any-class` also shows the judge reports of other
classes, which is fairer to B0 if it files a proof error under another class. Repeats of
an earlier finding (D20) count once. The grades go to `spot_grades.jsonl` in the run
folder and are reused on the next `score`.

## Before reporting numbers

- Check every grade by hand and set `human_agrees`, as the PAT authors did. Setting it to
  `false` rejects the judge's match. For a match the judge missed, put the finding id in
  `matched` and set it to `true`. `score` prints how many grades are still unchecked.
- Our judge accepts a reworded description of the same flaw. SPOT's own grader needed
  keyword matches, so our numbers are comparable to PAT's but not to SPOT's.
- The text comes from SPOT's conversion of the PDFs, not from our GROBID parser. A paper
  audited from its PDF loses some formula text, so these numbers show what the checker
  does with clean text.
