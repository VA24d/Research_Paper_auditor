# Set B: claim–source support

Set B tests the citation agent on 60 recent arXiv papers (cs.LG/cs.CL, submitted January–
March 2025, picked deterministically — first 60 by submission date from an arXiv API
query; the ids and their versions are in `arxiv_ids.txt`). Every paper appears at most
twice in the experiment dataset (`dataset.jsonl`): the kept original, and an injected copy
whose claim sentences or bibliography carry planted errors (decision D1).

| Item | Count |
|---|---|
| Papers (arXiv ids in `arxiv_ids.txt`) | 60 |
| Gold labels in `labels.jsonl` | 158 |
| Injected errors | 112: `reference_not_found` 96, `claim_number_mismatch` 4, `reference_metadata_mismatch` 9, `claim_not_supported` 3 |
| Clean controls (`none`) | 46 |
| Bibliography entries fetched with their cited work's text | 2,743 (one JSON line per entry under `data/set_b/sources/`) |

## Files

- `arxiv_ids.txt` — the 60 versioned arXiv ids (fetch the PDFs with
  `scripts/fetch_arxiv.py datasets/set_b/arxiv_ids.txt`).
- `labels.jsonl` — one `GoldLabel` per injected error or clean control.
- `injections.jsonl` — one `InjectionRecord` per injected error: what was edited where,
  and the gold evidence passage of the cited work (text + exact char offsets), which the
  Recall@5 measurement uses.
- `dataset.jsonl` — the `auditor batch` experiment file (injected copies and originals;
  the parsed papers live in `data/set_b/papers/`, fetched not committed, D13 — rebuild
  the injected copies from `injections.jsonl` and a fresh parse with
  `scripts/set_b_build.py`).

## How it was built

`scripts/set_b_build.py`: proposes injections deterministically (number perturbations of
numbers the claim states and the source confirms; direction-swapped claims the source
contradicts; a bibliography year changed by +2 or +3; a fabricated title with cleared
identifiers), VERIFIES every proposal (the perturbed value must not be stated by the
recorded source; the source must state the claim's original direction; a metadata edit
must disagree with the work the index resolves beyond the checker's ±1 year tolerance; a
fabricated title must resolve to nothing), and applies only the verified ones.
Bibliography injections are capped at two per paper so the claim kinds are not drowned.
Clean controls are sampled from verified pairs that were not injected. Every injected
error and clean control was checked by `scripts/set_b.py validate datasets/set_b
--sources data/set_b/sources` — exit status 1 only because the D18 second review (20% of
labels, `reviewed_by`) has not happened yet.

After the build, three corrections were applied (all 2026-10-30):

- The year injections were moved from +1 to +3 (`runs/_regen_years.py`): one year of
  drift is normal in real bibliographies, so the checker tolerates ±1 and the injected
  year must be beyond that.
- Labels stacked on an entry that also carries a `reference_not_found` injection were
  dropped (`runs/_drop_collisions.py`, 32 labels): with a fabricated title on the entry,
  a metadata mismatch can never be reported and a claim citing it can never be judged,
  so those labels were unreachable by construction.
- Claim-number labels whose perturbed value sits in a parser-merged table fragment were
  dropped (`runs/_drop_table_junk.py`, 11 of 15): the builder had extracted numbers from
  the whole sentence text, and GROBID merges table rows into sentences, so the perturbed
  value was table data no claim states. `propose()` now skips sentences crowded with
  bare numbers.

## What is not done yet

- The D18 second review: 20% of the labels need a second person's `reviewed_by` before
  the set is frozen.
- The claim classes are thin (15 + 5 of 154): most claim sentences do not state a
  specific number the source confirms, and few contain a direction phrase the source
  contradicts. Growing them needs claim-level injections written by hand (the `inject`
  subcommand of `scripts/set_b.py` takes any claim text) or a larger paper pool.
- The support-judgement runs (`auditor batch --systems b1` full-text vs
  `AUDITOR_CITATION_ABSTRACT_ONLY=1`) need an LLM configured.
