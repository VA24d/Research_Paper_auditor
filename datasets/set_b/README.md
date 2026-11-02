# Set B: claim–source support

Set B tests the citation agent on 60 recent arXiv papers (cs.LG/cs.CL, submitted January–
March 2025, picked deterministically — first 60 by submission date from an arXiv API
query; the ids and their versions are in `arxiv_ids.txt`). Every paper appears at most
twice in the experiment dataset (`dataset.jsonl`): the kept original, and an injected copy
whose claim sentences or bibliography carry planted errors (decision D1).

| Item | Count |
|---|---|
| Papers (arXiv ids in `arxiv_ids.txt`) | 60 |
| Gold labels in `labels.jsonl` | 157 |
| Injected errors | 111: `reference_not_found` 96, `reference_metadata_mismatch` 9, `claim_number_mismatch` 3, `claim_not_supported` 3 |
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

## The D18 second review (2026-11-02, dataset FROZEN)

The independent second review (a P2 agent session, reviewer marker `p2-review-agent`,
seed `random.Random(20260101)`) re-verified 32 of the 157 labels (20.4%, stratified):
all 3 `claim_number_mismatch` + 3 `claim_not_supported` labels in full, 5/9
`reference_metadata_mismatch`, 12/96 `reference_not_found`, 9/46 clean controls. Every
sampled label was checked against the injected paper, the original paper and the recorded
source text. Resolutions:

- **One label dropped** (`b-632fd74c81b5`, claim_number_mismatch on 2501.09760 citing
  TSPLIB): the recorded source (TSPLIB's abstract) contains no numbers at all, so the
  rationale ("the source states 2000") was false and the mismatch is unverifiable from
  the source. Label and injection record removed; the injected sentence keeps its edit
  (an unlabelled injected error — a finding there scores as an unlabelled flag).
- **Two rationales amended** (the 2501.00742 `claim_number_mismatch` pair): the source's
  "1000" refers to training iterations, not neurons per layer; the amended rationale says
  so. Labels kept: the number the claim states was changed, and the cited work's
  experiments use 64-256 neurons per layer.
- **One disk drift fixed** (`datasets/set_b/papers/2501.00742-injected.json`, s53): the
  local paper copy said "1350neurons" where the committed record/label say the injection
  is 1000 -> 750; the copy was rewritten to match the record (same-length rewrite, no
  offsets moved).
- The other 29 sampled labels agreed exactly (agreement 32/32 on the final sample set).
- `scripts/set_b.py validate datasets/set_b --sources data/set_b/sources` now exits 0.

## What is not done yet

- The claim classes are thin (6 of 157): most claim sentences do not state a
  specific number the source confirms, and few contain a direction phrase the source
  contradicts. Growing them needs claim-level injections written by hand (the `inject`
  subcommand of `scripts/set_b.py` takes any claim text) or a larger paper pool.
- The support-judgement runs (`auditor batch --systems b1` full-text vs
  `AUDITOR_CITATION_ABSTRACT_ONLY=1`) need an LLM configured.
