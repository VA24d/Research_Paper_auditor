"""Finds each part's entry points by convention, with mocks until they exist.

- `auditor.agents.citation`, `.code`, `.template` and `.theory` expose
  `check(task, ctx) -> TaskResult`.
- `auditor.agents.verifier` exposes `verify(results, ctx) -> list[VerifiedFinding]`.
- `auditor.agents.<name>.tools` exposes `make_tools(ctx) -> list[BaseTool]` for B1.

A missing entry point falls back to a mock and is reported, so runs record which parts
were real. An entry point that exists but fails to import is an error, not a fallback.
"""

import importlib
import logging
from collections.abc import Callable
from types import ModuleType
from typing import Any, cast

from langchain_core.tools import BaseTool

from auditor.infra.context import RunContext
from auditor.orchestration.graph import Specialists, VerifyFn
from auditor.orchestration.mocks import check_nothing, send_all_to_review
from auditor.orchestration.runner import CheckFn

logger = logging.getLogger(__name__)

SPECIALIST_PACKAGES = {
    "citation": "auditor.agents.citation",
    "code": "auditor.agents.code",
    "template": "auditor.agents.template",
    "theory": "auditor.agents.theory",
}
VERIFIER_PACKAGE = "auditor.agents.verifier"


def _optional_module(name: str) -> ModuleType | None:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name == name:
            return None
        raise


def _entry_point(package: str, attribute: str) -> Callable[..., Any] | None:
    module = _optional_module(package)
    function = getattr(module, attribute, None) if module is not None else None
    return function if callable(function) else None


def load_specialists() -> tuple[Specialists, list[str]]:
    """The specialists to run, and the names of those still replaced by mocks."""
    checks: dict[str, CheckFn] = {}
    mocked = []
    for name, package in SPECIALIST_PACKAGES.items():
        function = _entry_point(package, "check")
        if function is None:
            mocked.append(name)
            checks[name] = check_nothing
        else:
            checks[name] = cast(CheckFn, function)
    if mocked:
        logger.warning("using placeholder specialists for: %s", ", ".join(mocked))
    return Specialists(**checks), mocked


def load_verifier() -> tuple[VerifyFn, bool]:
    """The verifier, and whether it is the placeholder."""
    function = _entry_point(VERIFIER_PACKAGE, "verify")
    if function is None:
        logger.warning("using the placeholder verifier: every finding goes to human review")
        return send_all_to_review, True
    return cast(VerifyFn, function), False


def missing_tools() -> list[str]:
    """Specialist parts that do not expose B1 tools yet."""
    return [
        name
        for name, package in SPECIALIST_PACKAGES.items()
        if _entry_point(f"{package}.tools", "make_tools") is None
    ]


def load_tools(ctx: RunContext) -> list[BaseTool]:
    """Every tool the specialist parts expose for baseline B1 (decision D7)."""
    tools: list[BaseTool] = []
    for name, package in SPECIALIST_PACKAGES.items():
        make_tools = _entry_point(f"{package}.tools", "make_tools")
        if make_tools is None:
            logger.warning("no B1 tools from the %s part yet", name)
            continue
        tools.extend(make_tools(ctx))
    return tools
