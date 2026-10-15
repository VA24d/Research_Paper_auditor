import json
import types
from pathlib import Path

import pytest
from langchain_core.tools import tool

from auditor.baselines.common import ReportedErrors
from auditor.cli import main
from auditor.infra.context import RunContext
from auditor.infra.llm import StructuredLLM
from auditor.orchestration import registry
from auditor.orchestration.mocks import check_nothing, send_all_to_review
from auditor.orchestration.outputs import RunDirectory, predictions_from_verified, write_jsonl
from auditor.schemas import (
    AuditTask,
    Prediction,
    RunManifest,
    SystemName,
    TaskResult,
    VerificationState,
)
from auditor.schemas.examples import example_paper, example_verified_findings
from auditor.testing import ScriptedChatModel, checked_result, offline_context, tool_reply

FIXTURES = Path(__file__).parent / "fixtures" / "schemas"
PAPER_JSON = FIXTURES / "parsed_paper.json"
VENUE_JSON = FIXTURES / "venue_profile.json"

# These tests are about the registry and the CLI, not about whichever parts exist yet.
pytestmark = pytest.mark.usefixtures("parts")


def lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_missing_entry_points_fall_back_to_mocks() -> None:
    specialists, mocked = registry.load_specialists()
    assert mocked == ["citation", "code", "template"]
    assert specialists.code is check_nothing
    verifier, is_mock = registry.load_verifier()
    assert verifier is send_all_to_review
    assert is_mock


def test_real_entry_points_are_used(parts: dict[str, types.ModuleType]) -> None:
    async def check(task: AuditTask, ctx: RunContext) -> TaskResult:
        return checked_result(task)

    parts["auditor.agents.code"].check = check  # type: ignore[attr-defined]
    specialists, mocked = registry.load_specialists()
    assert specialists.code is check
    assert mocked == ["citation", "template"]


def test_tools_are_collected_from_each_part(
    parts: dict[str, types.ModuleType], tmp_path: Path
) -> None:
    @tool
    def find_reference(title: str) -> str:
        """Find a cited work by title."""
        return title

    parts["auditor.agents.citation.tools"].make_tools = lambda ctx: [find_reference]  # type: ignore[attr-defined]
    assert registry.load_tools(offline_context(tmp_path)) == [find_reference]


def test_broken_imports_are_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(name: str) -> types.ModuleType:
        raise ModuleNotFoundError("No module named 'missing_dependency'", name="missing_dependency")

    monkeypatch.setattr("importlib.import_module", fail)
    with pytest.raises(ModuleNotFoundError):
        registry.load_specialists()


def test_predictions_keep_the_verifier_state() -> None:
    verified = example_verified_findings()
    predictions = predictions_from_verified(verified, run_id="r1", system=SystemName.FULL)
    assert [p.state for p in predictions] == [
        VerificationState.VERIFIED_FLAG,
        VerificationState.NEEDS_HUMAN_REVIEW,
    ]
    assert [p.is_positive for p in predictions] == [True, False]
    assert predictions[0].evidence == verified[0].accepted_evidence


def test_write_jsonl_uses_lf(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "x.jsonl", example_verified_findings())
    assert b"\r\n" not in path.read_bytes()
    assert len(lines(path)) == 2


def test_audit_command_writes_a_complete_run(tmp_path: Path) -> None:
    code = main(
        ["audit", str(PAPER_JSON), "--no-llm", "--run-id", "t1", "--venue", str(VENUE_JSON)]
    )
    assert code == 0
    run = tmp_path / "runs" / "t1"
    manifest = RunManifest.model_validate_json((run / "manifest.json").read_bytes())
    assert manifest.finished_at is not None
    assert manifest.config["mocked_parts"] == ["citation", "code", "template", "verifier"]
    assert manifest.config["repository"] == "https://github.com/example/auditor-demo"
    paper_dir = run / "example-paper-original"
    assert len(lines(paper_dir / "tasks.jsonl")) == 12
    assert {r["outcome"] for r in lines(paper_dir / "results.jsonl")} == {"checked"}
    assert lines(paper_dir / "predictions.jsonl") == []
    assert (run / "checkpoints.sqlite").exists()
    assert (run / "run.log").exists()


def test_audit_refuses_to_reuse_a_run_id(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["audit", str(PAPER_JSON), "--no-llm", "--run-id", "t2"]) == 0
    assert main(["audit", str(PAPER_JSON), "--no-llm", "--run-id", "t2"]) == 2
    assert "already exists" in capsys.readouterr().err
    assert main(["audit", str(PAPER_JSON), "--no-llm", "--run-id", "t2", "--resume"]) == 0
    assert main(["audit", str(PAPER_JSON), "--no-llm", "--run-id", "t9", "--resume"]) == 2
    assert "no run to resume" in capsys.readouterr().err


def test_resume_with_different_inputs_is_refused_and_keeps_the_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["audit", str(PAPER_JSON), "--no-llm", "--run-id", "t3"]
    assert main([*argv, "--repo", "https://github.com/example/original"]) == 0
    manifest = tmp_path / "runs" / "t3" / "manifest.json"
    before = manifest.read_bytes()
    changed = [*argv, "--resume", "--repo", "https://github.com/example/changed"]
    assert main(changed) == 2
    assert "t3 was started with different papers" in capsys.readouterr().err
    assert manifest.read_bytes() == before


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["audit", str(PAPER_JSON), "--resume"], "--resume needs the --run-id"),
        (["audit", "missing.json"], "No such file"),
        (["audit", str(PAPER_JSON), "--repo", "https://example.org/x"], "not a supported"),
        (["audit", str(PAPER_JSON), "--venue", "missing.json"], "No such venue profile"),
        (["baseline", "b0", str(PAPER_JSON)], "Baselines need an LLM"),
    ],
)
def test_command_errors_are_reported(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["batch", "d.jsonl", "--experiment", "e", "--runs", "0"], "at least 1"),
        (["audit", str(PAPER_JSON), "--max-concurrency", "x"], "at least 1"),
        (["audit", str(PAPER_JSON), "--run-id", "review:1"], "use letters, digits"),
        (["batch", "d.jsonl", "--experiment", "../up"], "use letters, digits"),
    ],
)
def test_bad_argument_values_are_refused(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert message in capsys.readouterr().err


def test_baseline_b0_command(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    answer = {
        "errors": [
            {
                "error_class": "claim_number_mismatch",
                "anchor_kind": "sentence",
                "anchor_id": "s2",
                "target_kind": "bibliography",
                "target": "b1",
                "summary": "The number is not in the cited work.",
            }
        ]
    }
    model = ScriptedChatModel(replies=[tool_reply(ReportedErrors, answer)])
    monkeypatch.setattr(
        StructuredLLM,
        "from_settings",
        classmethod(lambda cls, settings: cls(model, name="test:scripted")),
    )
    assert main(["baseline", "b0", str(PAPER_JSON), "--run-id", "b0run"]) == 0
    path = tmp_path / "runs" / "b0run" / "example-paper-original" / "predictions.jsonl"
    (prediction,) = [Prediction.model_validate(line) for line in lines(path)]
    assert prediction.system is SystemName.B0
    assert prediction.state is None
    assert len(lines(tmp_path / "runs" / "b0run" / "usage" / "llm_calls.jsonl")) == 1


def test_run_directory_paths(tmp_path: Path) -> None:
    run = RunDirectory(tmp_path, "r1")
    assert run.paper_dir(example_paper()) == tmp_path / "r1" / "example-paper-original"
