import json
from pathlib import Path

import pytest

from auditor.baselines.common import ReportedErrors
from auditor.cli import main
from auditor.infra.llm import StructuredLLM
from auditor.schemas import PaperVariant, RunManifest
from auditor.schemas.examples import example_paper
from auditor.testing import ScriptedChatModel, tool_reply


def write_dataset(tmp_path: Path, *variants: PaperVariant) -> Path:
    folder = tmp_path / "dataset"
    folder.mkdir()
    lines = []
    for variant in variants:
        paper = example_paper().model_copy(update={"variant": variant})
        name = f"paper-{variant.value}.json"
        (folder / name).write_text(paper.model_dump_json(), encoding="utf-8")
        lines.append(json.dumps({"paper": name}))
    dataset = folder / "dataset.jsonl"
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dataset


def predictions(tmp_path: Path, run_id: str, variant: str) -> Path:
    return tmp_path / "runs" / run_id / f"example-paper-{variant}" / "predictions.jsonl"


def test_batch_runs_every_system_and_repetition(tmp_path: Path) -> None:
    dataset = write_dataset(tmp_path, PaperVariant.ORIGINAL, PaperVariant.INJECTED)
    code = main(["batch", str(dataset), "--experiment", "exp", "--systems", "full", "--runs", "2"])
    assert code == 0
    for run_id in ("exp-full-r1", "exp-full-r2"):
        for variant in ("original", "injected"):
            assert predictions(tmp_path, run_id, variant).is_file()
        manifest = RunManifest.model_validate_json(
            (tmp_path / "runs" / run_id / "manifest.json").read_bytes()
        )
        assert manifest.finished_at is not None
        assert manifest.llm_cache is False
        assert manifest.config["experiment"] == "exp"
        assert manifest.config["papers"] == 2


def test_batch_resumes_and_skips_finished_papers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = write_dataset(tmp_path, PaperVariant.ORIGINAL, PaperVariant.INJECTED)
    argv = ["batch", str(dataset), "--experiment", "exp", "--systems", "full", "--runs", "1"]
    assert main(argv) == 0
    finished = predictions(tmp_path, "exp-full-r1", "injected")
    finished.unlink()
    capsys.readouterr()
    assert main(argv) == 0
    assert "1 done, 1 already done, 0 failed" in capsys.readouterr().out
    assert finished.is_file()


def test_batch_records_failures_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = write_dataset(tmp_path, PaperVariant.ORIGINAL, PaperVariant.INJECTED)
    model = ScriptedChatModel(
        replies=[tool_reply(ReportedErrors, {"errors": []})]  # only one paper gets an answer
    )
    monkeypatch.setattr(
        StructuredLLM,
        "from_settings",
        classmethod(lambda cls, settings: cls(model, name="test:scripted", max_attempts=1)),
    )
    code = main(["batch", str(dataset), "--experiment", "exp", "--systems", "b0", "--runs", "1"])
    assert code == 1
    assert predictions(tmp_path, "exp-b0-r1", "original").is_file()
    failures = (tmp_path / "runs" / "exp-b0-r1" / "failures.jsonl").read_text(encoding="utf-8")
    assert "example-paper" in failures
    assert "1 done, 0 already done, 1 failed" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--strict"], "still mocks"),
        (["--systems", "b0"], "Baselines need an LLM"),
    ],
)
def test_batch_refuses_unusable_setups(
    tmp_path: Path, extra: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = write_dataset(tmp_path, PaperVariant.ORIGINAL)
    assert main(["batch", str(dataset), "--experiment", "exp", *extra]) == 2
    assert message in capsys.readouterr().err


def test_batch_rejects_duplicate_papers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset = write_dataset(tmp_path, PaperVariant.ORIGINAL)
    line = dataset.read_text(encoding="utf-8")
    dataset.write_text(line + line, encoding="utf-8")
    assert main(["batch", str(dataset), "--experiment", "exp", "--systems", "full"]) == 2
    assert "listed twice" in capsys.readouterr().err
