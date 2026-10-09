import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep each test away from the developer's AUDITOR_* variables and `.env` file."""
    for name in list(os.environ):
        if name.startswith("AUDITOR_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
