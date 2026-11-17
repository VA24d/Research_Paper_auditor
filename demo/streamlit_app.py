"""Entry point for the read-only demo on Streamlit Community Cloud.

Set the app's main file to `demo/streamlit_app.py`. It turns on demo mode, points the
report list at `demo/runs` and runs the normal web UI from `src/`.
"""

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

os.environ.setdefault("AUDITOR_DEMO", "1")
os.environ.setdefault("AUDITOR_RUNS_DIR", str(HERE / "runs"))
os.environ.setdefault("AUDITOR_DATA_DIR", "/tmp/auditor-data")
sys.path.insert(0, str(ROOT / "src"))

runpy.run_path(str(ROOT / "src" / "auditor" / "ui" / "app.py"), run_name="__main__")
