"""Defects in the released code, found by static analysis (decisions D4 `code_defect`, D11).

Nothing in the repository runs. Python files are compiled to syntax trees in this process,
and Ruff reads them for names that are never defined; both only read the text. Only the
files the experiment uses are checked: the scripts that run as programs (they have an
`if __name__ == "__main__":` block) and every local module they import, directly or not.
Old or unused scripts elsewhere in the repository are not the paper's experiment.

Three kinds of defect are reported, each with its file and line:
- a file that does not compile;
- a name that is used but defined nowhere (Ruff F821, F822, F823);
- an import of a module of a local package that the repository does not contain.

Code written for Python 2 fails these checks by design, so such a repository is reported
as unverifiable instead.
"""

import ast
import asyncio
import importlib.util
import json
import logging
import re
import shutil
import subprocess
import sys
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from auditor.infra.context import RunContext
from auditor.infra.snapshots import (
    Snapshot,
    SnapshotUnavailableError,
    SourceError,
    fetch_snapshot,
)
from auditor.schemas import (
    AuditTask,
    CandidateFinding,
    DocumentAnchor,
    ErrorClass,
    Evidence,
    EvidenceKind,
    SourceKind,
    SourceRef,
    TaskOutcome,
    TaskResult,
)

logger = logging.getLogger(__name__)

MAX_FINDINGS = 20
RUFF_TIMEOUT_S = 120
_RUFF_RULES = "F821,F822,F823"
# Python 2 names that Python 3 lacks: undefined in Python 3, but not defects of that code.
_PYTHON2_NAMES = frozenset(
    {"xrange", "unicode", "basestring", "raw_input", "long", "unichr", "execfile", "file"}
)
_PYTHON2_SYNTAX = re.compile(
    r"Missing parentheses in call to|print statement|invalid syntax.*print|except .*,"
)


def ruff_available() -> bool:
    return importlib.util.find_spec("ruff") is not None or shutil.which("ruff") is not None


def _ruff_command() -> list[str]:
    if importlib.util.find_spec("ruff") is not None:
        return [sys.executable, "-m", "ruff"]
    return ["ruff"]


def _module_name(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _parse(text: str, path: str) -> ast.Module | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(text, filename=path)
    except (SyntaxError, ValueError, RecursionError):
        return None


_MAIN_GUARD = re.compile(r"^if\s+__name__\s*==\s*['\"]__main__['\"]", re.MULTILINE)


def _is_entry_point(tree: ast.Module) -> bool:
    for node in tree.body:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ):
            return True
    return False


@dataclass(frozen=True)
class _Import:
    module: str
    line: int
    # False for `from pkg import x`'s `pkg.x`, which may be an attribute, not a module.
    certain: bool = True


def _imports(tree: ast.Module, path: str) -> list[_Import]:
    """Absolute module names a file imports (relative imports resolved)."""
    package = _module_name(path)
    if not path.endswith("__init__.py"):
        package = package.rpartition(".")[0]
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [_Import(alias.name, node.lineno) for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".") if package else []
                if node.level > 1:
                    base = base[: len(base) - (node.level - 1)]
                module = ".".join([*base, node.module] if node.module else base)
            else:
                module = node.module or ""
            if module:
                found.append(_Import(module, node.lineno))
                found += [
                    _Import(f"{module}.{alias.name}", node.lineno, certain=False)
                    for alias in node.names
                ]
    return found


def _candidates(module: str, importer: str, has_src: bool = False) -> list[str]:
    """The dotted names an import may resolve to: from the repository root, and from the
    importing script's own folder (scripts are often run from where they live), or under
    src/ when the repository uses a src/ layout."""
    folder = ".".join(PurePosixPath(importer).parent.parts)
    cands = [module]
    if folder:
        cands.append(f"{folder}.{module}")
    if (
        has_src or folder.startswith("src.") or folder == "src" or importer.startswith("src/")
    ) and not module.startswith("src."):
        cands.append(f"src.{module}")
    return list(dict.fromkeys(cands))


def experiment_files(sources: Mapping[str, str]) -> list[str]:
    """The entry-point scripts and every local module they import, sorted."""
    has_src = any(p.startswith("src/") for p in sources)
    by_module = {_module_name(path): path for path in sources}
    trees = {path: _parse(text, path) for path, text in sources.items()}
    # A script that does not compile still runs as one: its main guard is in the text.
    entries = [
        p
        for p, tree in trees.items()
        if (_is_entry_point(tree) if tree is not None else _MAIN_GUARD.search(sources[p]))
    ]
    if not entries:
        # Without an entry point, every file is a candidate; nothing narrows.
        entries = list(sources)
    seen: set[str] = set()
    queue = list(entries)
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        tree = trees.get(path)
        if tree is None:
            continue
        for item in _imports(tree, path):
            for name in _candidates(item.module, path, has_src=has_src):
                parts = name.split(".")
                for depth in range(1, len(parts) + 1):
                    local = by_module.get(".".join(parts[:depth]))
                    if local is not None and local not in seen:
                        queue.append(local)
    return sorted(seen)


def _missing_modules(
    files: list[str], sources: Mapping[str, str], known: set[str]
) -> list[tuple[str, int, str]]:
    """Imports of a module inside a local package that the repository does not have.
    `known` holds the module names of every file, downloaded or skipped."""
    has_src = any(p.startswith("src/") for p in sources) or any(k.startswith("src.") for k in known)
    packages = {name.rpartition(".")[0] for name in known if "." in name} | {
        _module_name(p) for p in sources if p.endswith("__init__.py")
    }
    if has_src:
        packages |= {p.removeprefix("src.") for p in packages if p.startswith("src.")}

    def present(name: str) -> bool:
        return name in known or any(k.startswith(name + ".") for k in known)

    missing = []
    for path in files:
        tree = _parse(sources[path], path)
        if tree is None:
            continue
        for item in _imports(tree, path):
            if not item.certain:
                continue
            names = _candidates(item.module, path, has_src=has_src)
            if any(present(name) for name in names):
                continue
            # Missing only when the import starts inside the repository: its first part
            # is a local package, and the rest is not there.
            first = item.module.split(".")[0]
            for name in names:
                head = name[: len(name) - len(item.module)] + first
                if head != name and head in packages:
                    missing.append((path, item.line, name))
                    break
    return missing


def _code_evidence(snapshot: Snapshot, path: str, line: int) -> Evidence:
    lines = snapshot.read_text(path).splitlines()
    text = lines[line - 1] if 0 < line <= len(lines) else ""
    return Evidence(
        kind=EvidenceKind.CODE,
        source=SourceRef(
            kind=SourceKind.REPOSITORY, id=snapshot.source_id, url=snapshot.manifest.url
        ),
        text=text.strip() or f"{path}:{line}",
        path=path,
        line_start=max(1, line),
    )


def _relative(snapshot: Snapshot, filename: str) -> str:
    """Ruff's file name (absolute, with the OS's separators) as a snapshot path."""
    try:
        return Path(filename).resolve().relative_to(snapshot.root.resolve()).as_posix()
    except ValueError:
        return Path(filename).as_posix()


def _ruff(snapshot: Snapshot, files: list[str]) -> tuple[list[dict[str, object]], bool]:
    """Ruff's reports, and whether it timed out (then there are none)."""
    command = [
        *_ruff_command(),
        "check",
        "--isolated",
        "--no-cache",
        "--select",
        _RUFF_RULES,
        "--output-format",
        "json",
        "--exit-zero",
        *files,
    ]
    try:
        done = subprocess.run(
            command,
            cwd=snapshot.root,
            capture_output=True,
            text=True,
            timeout=RUFF_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Ruff timed out after %ds on %s", RUFF_TIMEOUT_S, snapshot.source_id)
        return [], True
    reports = json.loads(done.stdout or "[]")
    return (reports if isinstance(reports, list) else []), False


async def check_defects(task: AuditTask, ctx: RunContext) -> TaskResult:
    repository = task.repository
    if repository is None:
        raise ValueError(f"{task.task_id} has no repository")
    try:
        snapshot = await fetch_snapshot(repository, ctx.http, ctx.settings)
    except (SnapshotUnavailableError, SourceError) as exc:
        return TaskResult.for_task(
            task, TaskOutcome.UNVERIFIABLE, reason=f"the repository could not be fetched: {exc}"
        )
    manifest = snapshot.manifest
    sources = {f.path: snapshot.read_text(f.path) for f in manifest.files if f.path.endswith(".py")}
    if not sources:
        return TaskResult.for_task(
            task, TaskOutcome.UNVERIFIABLE, reason="the repository has no Python files"
        )
    files = experiment_files(sources)
    defects: list[tuple[str, int, str]] = []
    undefined: list[tuple[str, int, str]] = []
    python2 = False
    for path in files:
        try:
            with warnings.catch_warnings():
                # Invalid escape sequences are common in repository code and do not stop it.
                warnings.simplefilter("ignore", SyntaxWarning)
                compile(sources[path], path, "exec", dont_inherit=True)
        except SyntaxError as exc:
            if _PYTHON2_SYNTAX.search(str(exc)) or _PYTHON2_SYNTAX.search(exc.text or ""):
                python2 = True
                continue
            defects.append((path, exc.lineno or 1, f"{path} does not compile: {exc.msg}"))
        except (ValueError, RecursionError):
            continue
    if python2:
        return TaskResult.for_task(
            task,
            TaskOutcome.UNVERIFIABLE,
            reason="the experiment code is written for Python 2, which these checks do not cover",
        )
    broken = {path for path, _, _ in defects}
    ruff_timed_out = False
    if ruff_available():
        reports, ruff_timed_out = await asyncio.to_thread(_ruff, snapshot, files)
        # Grouped per file: the names it uses but never defines, each at its first use.
        names: dict[str, dict[str, int]] = {}
        for report in reports:
            if report.get("code") not in _RUFF_RULES.split(","):
                continue
            message = str(report.get("message", ""))
            name = message.split("`")[1] if message.count("`") >= 2 else message
            path = _relative(snapshot, str(report.get("filename", "")))
            if name in _PYTHON2_NAMES or path in broken:
                continue
            location = report.get("location")
            line = int(location.get("row", 1)) if isinstance(location, dict) else 1
            names.setdefault(path, {}).setdefault(name, line)
        for path, used in sorted(names.items()):
            first = min(used.values())
            listed = ", ".join(
                f"{n} (line {line})" for n, line in sorted(used.items(), key=lambda i: i[1])
            )
            undefined.append((path, first, f"{path} uses names defined nowhere: {listed}"))
    # A module counts as present when it was downloaded or skipped for size or kind.
    known = {_module_name(p) for p in sources} | {
        _module_name(s.path) for s in manifest.skipped if s.path.endswith(".py")
    }
    complete = not manifest.incomplete and manifest.skipped_count == len(manifest.skipped)
    if complete:
        for path, line, name in _missing_modules(files, sources, known):
            defects.append(
                (path, line, f"{path}:{line} imports {name}, which the repository does not contain")
            )
    # Syntax errors and missing modules stop a program at once; an undefined name only on
    # the path that uses it, which may never run, so those are uncertain.
    ranked = [(d, False) for d in defects] + [(d, True) for d in undefined]
    findings = [
        CandidateFinding.for_task(
            task,
            number,
            error_class=ErrorClass.CODE_DEFECT,
            anchor=DocumentAnchor(),
            summary=summary,
            explanation="Found by static analysis; nothing in the repository was run.",
            evidence=[_code_evidence(snapshot, path, line)],
            uncertain=uncertain,
        )
        for number, ((path, line, summary), uncertain) in enumerate(ranked[:MAX_FINDINGS])
    ]
    reason = None
    if not ruff_available():
        reason = "undefined names were not checked: Ruff is not installed"
    elif ruff_timed_out:
        reason = "Ruff timed out: only syntax and compile checks ran"
    return TaskResult.for_task(
        task,
        TaskOutcome.CHECKED if reason is None or findings else TaskOutcome.UNVERIFIABLE,
        reason=reason if (not findings or ruff_timed_out) else None,
        findings=findings,
    )
