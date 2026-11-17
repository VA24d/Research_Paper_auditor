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
os.environ.setdefault("AUDITOR_CACHE_DIR", "/tmp/auditor-data/cache")
sys.path.insert(0, str(ROOT / "src"))
# Audits run `python -m auditor.cli` in a child process, which needs src/ on its path too
# (on Streamlit Cloud the package is not installed).
os.environ["PYTHONPATH"] = os.pathsep.join(
    filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH")])
)
os.chdir(ROOT)

runpy.run_path(str(ROOT / "src" / "auditor" / "ui" / "app.py"), run_name="__main__")
