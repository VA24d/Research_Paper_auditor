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

On Streamlit Community Cloud (free): New app, repository `VA24d/Research_Paper_auditor`,
branch `main`, main file `demo/streamlit_app.py`, Python 3.12. It installs
`demo/requirements.txt` and imports the package from `src/`. Hugging Face Docker Spaces
need a paid plan, so `demo/space/` is kept only for that route.

The hosted demo is https://researchpaperauditor.streamlit.app. Its second tab runs a new
audit on one of the ten papers in `demo/papers/` (or an uploaded ParsedPaper JSON) with the
visitor's own model: a Gemini API key, or any OpenAI-compatible server (base URL, model
name, optional key). The key is passed to the audit process in `GOOGLE_API_KEY` or
`OPENAI_API_KEY` only, so it never enters the run manifest or the logs. A model on the
visitor's own machine works when the app runs locally (command above) or when the server is
reachable from the internet.
