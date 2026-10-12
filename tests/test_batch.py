import json
from pathlib import Path

import pytest

from auditor import experiments
from auditor.baselines.common import ReportedErrors
from auditor.cli import main
from auditor.experiments import InputError, missing_parts, resolve_repository
from auditor.infra.llm import StructuredLLM
from auditor.orchestration import registry
from auditor.schemas import AuditTask, PaperVariant, RepositoryRef, RunManifest, SystemName
from auditor.schemas.examples import example_paper
from auditor.testing import ScriptedChatModel, tool_reply


def write_dataset(tmp_path: Path, *variants: PaperVariant, **fields: str) -> Path:
    folder = tmp_path / "dataset"
    folder.mkdir()
    lines = []
    for variant in variants:
        paper = example_paper().model_copy(update={"variant": variant})
        name = f"paper-{variant.value}.json"
        (folder / name).write_text(paper.model_dump_json(), encoding="utf-8")
        lines.append(json.dumps({"paper": name, **fields}))
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


def test_batch_checks_code_against_the_given_commit(tmp_path: Path) -> None:
    dataset = write_dataset(
        tmp_path,
        PaperVariant.ORIGINAL,
        repository="https://github.com/example/auditor-demo",
        repository_commit="0123abc",
    )
    assert main(["batch", str(dataset), "--experiment", "exp", "--systems", "full"]) == 0
    folder = predictions(tmp_path, "exp-full-r1", "original").parent
    lines = (folder / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
    code = [t for t in map(AuditTask.model_validate_json, lines) if t.repository]
    assert code
    assert {t.repository for t in code} == {
        RepositoryRef(url="https://github.com/example/auditor-demo", commit="0123abc")
    }


def test_repository_commit_from_the_url_or_alone() -> None:
    paper = example_paper()
    from_url = resolve_repository("https://github.com/a/b/tree/0123abc", paper)
    assert from_url == RepositoryRef(url="https://github.com/a/b", commit="0123abc")
    found = resolve_repository(None, paper, "0123abc")
    assert found is not None
    assert found.commit == "0123abc"
    with pytest.raises(InputError, match="names 0123abc"):
        resolve_repository("https://github.com/a/b/commit/0123abc", paper, "fff0000")
    without_links = paper.model_copy(update={"links": ()})
    with pytest.raises(InputError, match="no repository"):
        resolve_repository(None, without_links, "0123abc")


def only_tools_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def entry_point(package: str, attribute: str) -> object:
        return None if attribute == "make_tools" else (lambda *args: None)

    monkeypatch.setattr(registry, "_entry_point", entry_point)


def test_each_system_lists_the_parts_it_still_lacks(monkeypatch: pytest.MonkeyPatch) -> None:
    only_tools_missing(monkeypatch)
    assert missing_parts(SystemName.FULL) == []
    assert missing_parts(SystemName.B0) == []
    assert missing_parts(SystemName.B1) == ["citation tools", "code tools", "template tools"]


def test_strict_refuses_b1_without_the_specialists_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    only_tools_missing(monkeypatch)
    dataset = write_dataset(tmp_path, PaperVariant.ORIGINAL)
    argv = ["batch", str(dataset), "--experiment", "exp", "--systems", "b1", "--strict"]
    assert main(argv) == 2
    assert "b1: citation tools, code tools, template tools" in capsys.readouterr().err


def test_continuing_a_run_with_different_inputs_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = write_dataset(tmp_path, PaperVariant.ORIGINAL)
    argv = ["batch", str(dataset), "--experiment", "exp", "--systems", "full"]
    assert main(argv) == 0
    assert main(argv) == 0

    monkeypatch.setenv("AUDITOR_LLM_OPTIONS", '{"max_output_tokens": 16}')
    capsys.readouterr()
    assert main(argv) == 2
    assert "exp-full-r1 was started with different llm_settings" in capsys.readouterr().err
    monkeypatch.delenv("AUDITOR_LLM_OPTIONS")

    paper = example_paper()
    changed = paper.sentences[0].model_copy(update={"text": "A changed sentence."})
    changed_paper = paper.model_copy(update={"sentences": (changed, *paper.sentences[1:])})
    (dataset.parent / "paper-original.json").write_text(
        changed_paper.model_dump_json(), encoding="utf-8"
    )
    assert main(argv) == 2
    assert "different papers" in capsys.readouterr().err


def test_b1_without_a_final_answer_is_a_failure_not_an_empty_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = write_dataset(tmp_path, PaperVariant.ORIGINAL)
    model = ScriptedChatModel(replies=[tool_reply("lookup", {"q": str(i)}) for i in range(50)])
    monkeypatch.setattr(
        StructuredLLM,
        "from_settings",
        classmethod(lambda cls, settings: cls(model, name="test:scripted")),
    )
    monkeypatch.setattr(experiments, "create_chat_model", lambda settings: model)
    monkeypatch.setattr(experiments, "model_id", lambda settings: "test:scripted")
    argv = ["batch", str(dataset), "--experiment", "exp", "--systems", "b1", "--runs", "1"]
    assert main(argv) == 1
    assert not predictions(tmp_path, "exp-b1-r1", "original").exists()
    failures = (tmp_path / "runs" / "exp-b1-r1" / "failures.jsonl").read_text(encoding="utf-8")
    assert "IncompleteAnswerError" in failures
    assert "0 done, 0 already done, 1 failed" in capsys.readouterr().out
