# Demo reports

Saved reports for the read-only web demo (`AUDITOR_DEMO=1 AUDITOR_RUNS_DIR=demo/runs`).
No model calls were made to write them:

- `set-c-final-run/`: `auditor report` on four papers of the final set C run
  (`results/p4/runs/p4-setc-full-r1`).
- `stable-sgd-example/`: 2006.06958 (Stable SGD), from the earlier run
  `results/set_c/runs/hard3c-timeouts-full-r1` after `scripts/verifier_replay.py`, so the
  current verifier made its decisions. It holds the shuffling mismatch shown on the poster.

Run the demo locally:

    AUDITOR_DEMO=1 AUDITOR_RUNS_DIR=demo/runs uv run --extra ui streamlit run src/auditor/ui/app.py

The Hugging Face Space is `demo/space/` (Dockerfile and Space README) plus `pyproject.toml`, `uv.lock`, `src/` and `demo/` copied next to it.
