"""scripts/p4_analysis.py: where the full system loses each planted error (P4.8)."""

import importlib.util
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

from pydantic import BaseModel

from auditor.evaluation.metrics import score
from auditor.evaluation.runs import load_run
from auditor.orchestration.outputs import predictions_from_verified
from auditor.schemas import (
    AuditTask,
    CandidateFinding,
    DocumentAnchor,
    ErrorClass,
    Evidence,
    EvidenceKind,
    ExtractedSetting,
    GoldLabel,
    PaperSource,
    PaperVariant,
    RepositoryRef,
    RunManifest,
    SentenceAnchor,
    SettingTarget,
    SourceKind,
    SourceRef,
    SystemName,
    TaskKind,
    TaskOutcome,
    TaskResult,
    VerificationState,
    VerifiedFinding,
)

ROOT = Path(__file__).parent.parent
PID = "p1"
QUOTE = Evidence(
    kind=EvidenceKind.QUOTE, source=SourceRef(kind=SourceKind.SUBMISSION, id=PID), text="x"
)


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "p4_analysis", ROOT / "scripts" / "p4_analysis.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["p4_analysis"] = module
    spec.loader.exec_module(module)
    return module


def write(path: Path, records: Sequence[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")


def task(sentence: str, key: str) -> AuditTask:
    anchor = SentenceAnchor(sentence_id=sentence)
    return AuditTask.create(
        paper_id=PID,
        variant=PaperVariant.INJECTED,
        kind=TaskKind.CODE,
        anchor=anchor,
        target=SettingTarget(key=key),
        setting=ExtractedSetting(
            key=key, raw_name=key, raw_value="1", source=PaperSource(anchor=anchor)
        ),
        repository=RepositoryRef(url="https://github.com/o/r", commit="abc"),
    )


def label(label_id: str, error_class: ErrorClass, sentence: str | None, key: str) -> GoldLabel:
    return GoldLabel(
        label_id=label_id,
        paper_id=PID,
        variant=PaperVariant.INJECTED,
        error_class=error_class,
        anchor=SentenceAnchor(sentence_id=sentence) if sentence else DocumentAnchor(),
        target=SettingTarget(key=key),
        rationale="hand-made",
        evidence=(QUOTE,),
    )


def decision(t: AuditTask, state: VerificationState, reason: str) -> VerifiedFinding:
    finding = CandidateFinding.for_task(
        t, 0, error_class=ErrorClass.SETTING_MISMATCH, summary="differs", evidence=(QUOTE,)
    )
    return VerifiedFinding(
        finding=finding,
        state=state,
        reason=reason,
        accepted_evidence=(QUOTE,) if state is VerificationState.VERIFIED_FLAG else (),
    )


def run_folder(root: Path, uncertain_reason: str) -> Path:
    lr, bs, ep = task("s1", "learning_rate"), task("s2", "batch_size"), task("s3", "epochs")
    method = task("s9", "method")
    run = root / "x-full-r1"
    write(
        run / "manifest.json",
        [
            RunManifest(
                run_id="x-full-r1",
                system=SystemName.FULL,
                started_at=datetime(2026, 11, 1, tzinfo=UTC),
                python_version="3.13",
                platform="test",
            )
        ],
    )
    folder = run / f"{PID}-injected"
    write(folder / "tasks.jsonl", [lr, bs, ep, method])
    write(
        folder / "results.jsonl",
        [
            TaskResult.for_task(lr, TaskOutcome.CHECKED),
            TaskResult.for_task(bs, TaskOutcome.CHECKED),
            TaskResult.for_task(ep, TaskOutcome.CHECKED),
            TaskResult.for_task(method, TaskOutcome.UNVERIFIABLE, reason="needs an LLM"),
        ],
    )
    verified = [
        decision(lr, VerificationState.VERIFIED_FLAG, "shown"),
        decision(bs, VerificationState.NEEDS_HUMAN_REVIEW, uncertain_reason),
    ]
    write(folder / "verified.jsonl", verified)
    write(
        folder / "predictions.jsonl",
        predictions_from_verified(verified, run_id="x-full-r1", system=SystemName.FULL),
    )
    return run


def test_each_label_is_placed_at_the_stage_where_it_was_lost(tmp_path: Path) -> None:
    script = _script()
    run = load_run(run_folder(tmp_path, script.UNCERTAIN_ONLY))
    labels = {
        "found": label("a", ErrorClass.SETTING_MISMATCH, "s1", "learning_rate"),
        "review": label("b", ErrorClass.SETTING_MISMATCH, "s2", "batch_size"),
        "nothing": label("c", ErrorClass.SETTING_MISMATCH, "s3", "epochs"),
        "unplanned": label("d", ErrorClass.SETTING_MISMATCH, "s4", "dropout"),
        # A document-level method label is checked by a sentence-anchored task.
        "method": label("e", ErrorClass.METHOD_MISMATCH, None, "method"),
    }
    outcomes = score(list(labels.values()), run.predictions()).outcomes
    stages = {
        lid: script.stage_of(lbl, run, outcomes.get(lbl.label_id))[0] for lid, lbl in labels.items()
    }
    assert stages == {
        "found": "found",
        "review": "human review",
        "nothing": "no finding",
        "unplanned": "not planned",
        "method": "unverifiable",
    }


def test_uncertain_findings_can_be_counted_as_flags(tmp_path: Path) -> None:
    script = _script()
    run = load_run(run_folder(tmp_path, script.UNCERTAIN_ONLY))
    states = [p.state for p in script.uncertain_counted(run)]
    assert states == [VerificationState.VERIFIED_FLAG, VerificationState.VERIFIED_FLAG]
    other = load_run(run_folder(tmp_path / "other", "Not settled: no evidence."))
    assert script.uncertain_counted(other)[1].state is VerificationState.NEEDS_HUMAN_REVIEW
