from pathlib import Path

import pytest

from auditor.agents.code import check
from auditor.agents.code.defects import experiment_files, ruff_available
from auditor.planner.tasks import DEFECTS_KEY, defect_tasks
from auditor.schemas import (
    AuditTask,
    DocumentAnchor,
    ErrorClass,
    EvidenceKind,
    RepositoryRef,
    SettingTarget,
    TaskOutcome,
)
from auditor.schemas.examples import example_paper
from auditor.testing import offline_context

MAIN = """\
from models.net import build
from utils import load

if __name__ == "__main__":
    model = build()
    data = load("train.csv")
    print(accuracy(model, data))
"""
UTILS = "def load(path):\n    return open(path).read()\n"
NET = "def build():\n    return None\n"
BROKEN_UNUSED = "def old(:\n    pass\n"


def repo(tmp_path: Path, files: dict[str, str]) -> str:
    root = tmp_path / "repo"
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root.resolve().as_uri()


def task(url: str) -> AuditTask:
    (only,) = defect_tasks(example_paper(), RepositoryRef(url=url))
    return only


def test_one_defect_task_per_paper_with_a_repository() -> None:
    (only,) = defect_tasks(example_paper(), RepositoryRef(url="https://github.com/o/r"))
    assert only.target == SettingTarget(key=DEFECTS_KEY)
    assert defect_tasks(example_paper(), None) == []


def test_the_experiment_is_the_entry_points_and_what_they_import() -> None:
    files = {
        "main.py": MAIN,
        "utils.py": UTILS,
        "models/__init__.py": "",
        "models/net.py": NET,
        "scripts/old.py": BROKEN_UNUSED,
    }
    sources = dict(files)
    assert experiment_files(sources) == [
        "main.py",
        "models/__init__.py",
        "models/net.py",
        "utils.py",
    ]


@pytest.mark.skipif(not ruff_available(), reason="ruff is not installed")
async def test_undefined_names_in_experiment_files_are_defects(tmp_path: Path) -> None:
    files = {
        "main.py": MAIN,
        "utils.py": UTILS,
        "models/__init__.py": "",
        "models/net.py": NET,
        "scripts/old.py": BROKEN_UNUSED,  # broken, but not part of the experiment
    }
    url = repo(tmp_path, files)
    result = await check(task(url), offline_context(tmp_path))
    assert result.outcome is TaskOutcome.CHECKED
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.CODE_DEFECT
    assert finding.anchor == DocumentAnchor()
    assert "accuracy" in finding.summary
    (code,) = finding.evidence
    assert code.kind is EvidenceKind.CODE
    assert (code.path, code.line_start) == ("main.py", 7)


async def test_a_syntax_error_in_an_experiment_file_is_a_defect(tmp_path: Path) -> None:
    files = {"train.py": 'if __name__ == "__main__":\n    run(\n', "README.md": "# r\n"}
    result = await check(task(repo(tmp_path, files)), offline_context(tmp_path))
    syntax = [f for f in result.findings if "does not compile" in f.summary]
    assert len(syntax) == 1
    assert syntax[0].evidence[0].path == "train.py"


def test_a_script_that_does_not_compile_is_still_part_of_the_experiment() -> None:
    broken = 'def main(args)\n    pass\n\n\nif __name__ == "__main__":\n    main(1)\n'
    sources = {
        "main.py": MAIN,
        "utils.py": UTILS,
        "eval.py": broken,
        "scripts/old.py": BROKEN_UNUSED,
    }
    assert "eval.py" in experiment_files(sources)
    assert "scripts/old.py" not in experiment_files(sources)


async def test_a_missing_module_of_a_local_package_is_a_defect(tmp_path: Path) -> None:
    files = {
        "main.py": 'from models.resnet import build\n\nif __name__ == "__main__":\n    build()\n',
        "models/__init__.py": "",
        "models/net.py": NET,
    }
    result = await check(task(repo(tmp_path, files)), offline_context(tmp_path))
    missing = [f for f in result.findings if "models.resnet" in f.summary]
    assert len(missing) == 1
    assert missing[0].evidence[0].line_start == 1


async def test_python_2_code_is_not_reported_as_broken(tmp_path: Path) -> None:
    files = {"main.py": 'print "hello"\nfor i in xrange(3):\n    pass\n'}
    result = await check(task(repo(tmp_path, files)), offline_context(tmp_path))
    assert result.findings == ()
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "Python 2" in result.reason


async def test_a_clean_repository_has_no_defects(tmp_path: Path) -> None:
    files = {"main.py": 'from utils import load\n\nif __name__ == "__main__":\n    load("x")\n'}
    files["utils.py"] = UTILS
    result = await check(task(repo(tmp_path, files)), offline_context(tmp_path))
    assert result.outcome is TaskOutcome.CHECKED
    assert result.findings == ()


def test_src_layout_experiment_files() -> None:
    sources = {
        "main.py": 'from mypkg.net import build\n\nif __name__ == "__main__":\n    build()\n',
        "src/mypkg/__init__.py": "",
        "src/mypkg/net.py": "def build(): return 1\n",
    }
    assert experiment_files(sources) == [
        "main.py",
        "src/mypkg/__init__.py",
        "src/mypkg/net.py",
    ]


async def test_src_layout_missing_submodule_is_defect(tmp_path: Path) -> None:
    files = {
        "main.py": 'from mypkg.missing import build\n\nif __name__ == "__main__":\n    build()\n',
        "src/mypkg/__init__.py": "",
        "src/mypkg/net.py": "def build(): return 1\n",
    }
    result = await check(task(repo(tmp_path, files)), offline_context(tmp_path))
    missing = [f for f in result.findings if "mypkg.missing" in f.summary]
    assert len(missing) == 1


async def test_ruff_timeout_records_in_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    from typing import Any

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="ruff", timeout=120)

    monkeypatch.setattr(subprocess, "run", fake_run)
    files = {"main.py": 'from utils import load\n\nif __name__ == "__main__":\n    load("x")\n'}
    files["utils.py"] = UTILS
    result = await check(task(repo(tmp_path, files)), offline_context(tmp_path))
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "Ruff timed out" in result.reason


def test_candidates_deduplicates_and_avoids_double_src() -> None:
    from auditor.agents.code.defects import _candidates

    # When module already starts with src., it should not produce src.src.
    cands = _candidates("src.pkg.mod", "src/pkg/runner.py", has_src=True)
    assert "src.src.pkg.mod" not in cands
    assert cands[0] == "src.pkg.mod"

    # From root, has_src adds src. prefix once
    cands_root = _candidates("pkg.mod", "main.py", has_src=True)
    assert cands_root == ["pkg.mod", "src.pkg.mod"]


async def test_src_layout_nested_relative_import(tmp_path: Path) -> None:
    files = {
        "src/mypkg/__init__.py": "",
        "src/mypkg/sub/__init__.py": "",
        "src/mypkg/sub/helper.py": "def helper_fn(): return 42\n",
        "src/mypkg/sub/worker.py": (
            'from .helper import helper_fn\n\nif __name__ == "__main__":\n    helper_fn()\n'
        ),
    }
    result = await check(task(repo(tmp_path, files)), offline_context(tmp_path))
    assert result.outcome is TaskOutcome.CHECKED
    assert result.findings == ()


async def test_ruff_timeout_with_syntax_errors_retains_findings_and_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    from typing import Any

    def fake_run(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="ruff", timeout=120)

    monkeypatch.setattr(subprocess, "run", fake_run)
    files = {"broken.py": "def broken_syntax(:\n"}
    result = await check(task(repo(tmp_path, files)), offline_context(tmp_path))
    assert result.outcome is TaskOutcome.CHECKED
    assert len(result.findings) >= 1
    assert result.reason is not None
    assert "Ruff timed out" in result.reason


@pytest.mark.skipif(not ruff_available(), reason="ruff is not installed")
async def test_a_ruff_module_in_the_repository_is_not_run(tmp_path: Path) -> None:
    # `python -m ruff` started in the snapshot would import the repository's own ruff.py.
    marker = tmp_path / "ran"
    files = {"main.py": MAIN, "utils.py": UTILS, "models/__init__.py": "", "models/net.py": NET}
    files["ruff.py"] = f"open({str(marker)!r}, 'w').close()\n"
    result = await check(task(repo(tmp_path, files)), offline_context(tmp_path))
    assert not marker.exists()
    assert any("accuracy" in f.summary for f in result.findings)


@pytest.mark.skipif(not ruff_available(), reason="ruff is not installed")
async def test_a_file_named_like_an_option_is_checked_as_a_file(tmp_path: Path) -> None:
    files = {"--output-file=out.py": 'if __name__ == "__main__":\n    print(accuracy)\n'}
    result = await check(task(repo(tmp_path, files)), offline_context(tmp_path))
    assert any("accuracy" in f.summary for f in result.findings)
