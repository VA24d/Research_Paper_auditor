import pytest

from auditor.infra.snapshots import (
    DownloadLimits,
    RemoteFile,
    core_file_reason,
    safe_relative_path,
    select_files,
)


@pytest.mark.parametrize(
    "path",
    [
        "train.py",
        "configs/base.yaml",
        "configs/base.yml",
        "conf/model.json",
        "pyproject.toml",
        "setup.cfg",
        "scripts/run_glue.sh",
        "configs/model.gin",
        "notebooks/explore.ipynb",
        "README.md",
        "data/splits/train.txt",
        "Makefile",
    ],
)
def test_core_files_are_kept(path: str) -> None:
    assert core_file_reason(path) is None


@pytest.mark.parametrize(
    ("path", "reason"),
    [
        ("data/train.csv", "not a core file type"),
        ("checkpoints/model.pt", "not a core file type"),
        ("weights/model.safetensors", "not a core file type"),
        ("data/train.jsonl", "not a core file type"),
        ("assets/figure.png", "not a core file type"),
        (".git/config", "excluded directory"),
        (".github/workflows/ci.yml", "excluded directory"),
        ("node_modules/pkg/package.json", "excluded directory"),
        ("src/__pycache__/x.py", "excluded directory"),
        ("venv/lib/site.py", "excluded directory"),
        ("wandb/run-1/config.yaml", "excluded directory"),
        ("pkg.egg-info/PKG-INFO.txt", "excluded directory"),
    ],
)
def test_other_files_are_skipped_with_a_reason(path: str, reason: str) -> None:
    assert core_file_reason(path) == reason


@pytest.mark.parametrize(
    ("raw", "safe"),
    [
        ("a/b.py", "a/b.py"),
        ("./a//b.py", "a/b.py"),
        ("../evil.py", None),
        ("a/../../evil.py", None),
        ("/etc/passwd", None),
        ("C:/windows/x.py", None),
        ("a\\b.py", None),
        ("con/x.py", None),
        ("a/aux.py", None),
        ("a/b.py ", None),
        ("a/b?.py", None),
        ("a/\x00.py", None),
        ("", None),
    ],
)
def test_only_portable_relative_paths_are_safe(raw: str, safe: str | None) -> None:
    assert safe_relative_path(raw) == safe


async def _never(max_bytes: int) -> bytes:
    raise AssertionError("selection must not download")


def remote(path: str, size: int | None = 10) -> RemoteFile:
    return RemoteFile(path=path, size=size, load=_never)


def test_selection_keeps_core_files_and_explains_the_rest() -> None:
    selection = select_files(
        [
            remote("README.md"),
            remote("train.py"),
            remote("data/train.csv"),
            remote("configs/huge.json", size=5_000),
            remote("../evil.py"),
            remote("Train.py"),
        ],
        DownloadLimits(max_file_bytes=1_000),
    )
    assert [f.path for f in selection.chosen] == ["train.py", "README.md"]
    reasons = {s.path: s.reason for s in selection.skipped}
    assert reasons == {
        "data/train.csv": "not a core file type",
        "configs/huge.json": "larger than the per-file limit",
        "../evil.py": "unsafe path",
        "Train.py": "same path as another file when case is ignored",
    }
    assert selection.incomplete == ()


def test_settings_files_come_first_when_the_budget_runs_out() -> None:
    selection = select_files(
        [remote("README.md", 10), remote("configs/deep/base.yaml", 10), remote("train.py", 10)],
        DownloadLimits(max_total_bytes=20),
    )
    assert [f.path for f in selection.chosen] == ["train.py", "configs/deep/base.yaml"]
    assert [s.reason for s in selection.skipped] == ["total download limit reached"]
    assert selection.incomplete == ("total download limit reached",)


def test_file_count_limit_marks_the_snapshot_incomplete() -> None:
    selection = select_files(
        [remote("a.py"), remote("b.py"), remote("c.py")], DownloadLimits(max_files=2)
    )
    assert [f.path for f in selection.chosen] == ["a.py", "b.py"]
    assert selection.incomplete == ("file count limit reached",)


def test_files_of_unknown_size_are_kept_for_download_time() -> None:
    selection = select_files([remote("a.py", None)], DownloadLimits(max_total_bytes=1))
    assert [f.path for f in selection.chosen] == ["a.py"]


def test_tests_and_docs_come_after_the_experiment_code() -> None:
    selection = select_files(
        [remote("tests/test_train.py"), remote("docs/conf.py"), remote("src/a/b/train.py")],
        DownloadLimits(max_files=1),
    )
    assert [f.path for f in selection.chosen] == ["src/a/b/train.py"]


def test_default_limits_keep_snapshots_small() -> None:
    limits = DownloadLimits()
    assert limits.max_total_bytes <= 10_000_000
    assert limits.max_files <= 500
