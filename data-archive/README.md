# Data archives

Our `data/` folder is git-ignored. This folder backs it up as ten zstd-compressed tar files
in Git LFS. They hold the run outputs, parsed papers, PDFs, repository snapshots and the
model-call cache behind the reported numbers. The files come from arXiv, OpenReview and
GitHub under their own licences, so keep this repository private.

A plain `git pull` does not download them: `.lfsconfig` excludes `data-archive/**`. Fetch
the ones you need and unpack them from the repository root:

```sh
git lfs pull --include="data-archive/spot.tar.zst"     # one archive
git lfs pull --include="data-archive/*"                # all of them, 2.4 GB
zstd -dc data-archive/spot.tar.zst | tar -xf -         # recreates data/spot/
```

| Archive | Size | Holds |
|---|---|---|
| `set_c-runs` | 52 MB | `data/set_c/runs/`: every set C run (tasks, results, verified findings, predictions, manifests) |
| `set_c-pdfs` | 967 MB | `data/set_c/pdfs/`: the set C paper PDFs |
| `set_c-rest` | 162 MB | the rest of `data/set_c/`: parsed and injected papers, dataset files, RQ1 labels and logs |
| `spot` | 44 MB | `data/spot/`: SPOT papers, runs and judge grades |
| `set_d` | 15 MB | `data/set_d/` and `data/set_d2/`: set D plants and runs |
| `real_cases` | 676 MB | `data/real_cases/`: the OpenReview and retraction cases, with their runs |
| `cache` | 364 MB | `data/cache/` (HTTP, GROBID, LLM and snapshot caches) and `data/rag_cache.json` |
| `scratch` | 59 MB | `data/scratch/` |
| `retrieval_bench` | 68 MB | `data/retrieval_bench/` |
| `misc` | 0.6 MB | the loose files in `data/` (claim-check runs and their logs) |

Sizes are the compressed sizes. The set C runs were packed on 31 October 2026 and the rest
on 6 November 2026, so runs made after those dates are not included.

## local-extras (added 18 November 2026)

`local-extras.tar.zst` (0.3 MB) holds git-ignored local files that the other archives miss:
`data/runs/`, `data/ledger-2026-10-20.jsonl`, `.trail/learnings/`, `AGENTS.md`, `gemini.md`,
`.claude/settings*.json`, `.agents/`, `.gemini/`. `.env` is left out; its keys are listed in
`gemini.md`.

    git lfs pull --include="data-archive/local-extras.tar.zst"
    zstd -dc data-archive/local-extras.tar.zst | tar -xf -

## delta-2026-11-19 (added 19 November 2026)

`delta-2026-11-19.tar.zst` (12 MB, 555 files) holds every file in `data/` changed between
6 and 19 November 2026, which the archives above miss: new cache entries, the set B runs,
the UI demo files, the ledger, and the math-checker runs on OpenAI's withdrawn
Weil-classes paper (`data/openai_papers/`). Unpack it after the others, so its newer files
replace theirs.

    git lfs pull --include="data-archive/delta-2026-11-19.tar.zst"
    zstd -dc data-archive/delta-2026-11-19.tar.zst | tar -xf -
