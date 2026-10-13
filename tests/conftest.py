import os
import sys
import types
from pathlib import Path

import pytest

from auditor.orchestration.registry import SPECIALIST_PACKAGES, VERIFIER_PACKAGE


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep each test away from the developer's AUDITOR_* variables and `.env` file."""
    for name in list(os.environ):
        if name.startswith("AUDITOR_"):
            monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def parts(monkeypatch: pytest.MonkeyPatch) -> dict[str, types.ModuleType]:
    """Empty stand-ins for the specialist, verifier and tool modules, so pipeline tests do not
    depend on which parts teammates have built. Set an attribute on one to simulate that part:
    `parts["auditor.agents.code"].check = check`."""
    packages = [*SPECIALIST_PACKAGES.values(), VERIFIER_PACKAGE]
    names = [*packages, *(f"{package}.tools" for package in SPECIALIST_PACKAGES.values())]
    modules = {name: types.ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return modules
