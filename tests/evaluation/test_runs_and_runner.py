"""Reading run folders and scoring them end to end (team plan P4.5 and P4.7)."""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, JsonValue

from auditor.agents.theory.review import NO_ERROR_REASON
from auditor.cli import main
from auditor.config import Settings
from auditor.evaluation.grounding import grounding, load_inputs
from auditor.evaluation.runner import evaluate, render_markdown, to_json
from auditor.evaluation.runs import find_runs, load_run
from auditor.orchestration.outputs import predictions_from_verified
from auditor.schemas import (
    CandidateFinding,
    ErrorClass,
    Evidence,
    EvidenceKind,
    GoldLabel,
    LLMCallRecord,
    PaperVariant,
    Prediction,
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
from auditor.schemas.examples import example_paper

PAPER = example_paper()
PID = PAPER.paper_id
S4 = "We train with a learning rate of 3e-4 and a batch size of 256."
LR = SettingTarget(key="learning_rate")


def write(path: Path, records: Sequence[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8")


def manifest(run_id: str, system: SystemName, **config: JsonValue) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        system=system,
        started_at=datetime(2026, 11, 1, tzinfo=UTC),
        python_version="3.13",
        platform="test",
        config={"llm_prices": {"test-model": [1.0, 10.0]}, **config},
    )


def quote(text: str) -> Evidence:
    return Evidence(
        kind=EvidenceKind.QUOTE, source=SourceRef(kind=SourceKind.SUBMISSION, id=PID), text=text
    )


def finding(number: int, sentence: str, error_class: ErrorClass, text: str) -> CandidateFinding:
    return CandidateFinding(
        finding_id=f"t{number}-f0",
        task_id=f"t{number}",
        paper_id=PID,
        variant=PaperVariant.INJECTED,
        error_class=error_class,
        anchor=SentenceAnchor(sentence_id=sentence),
        target=LR,
        summary="suspected",
        evidence=(quote(text),),
    )


def gold(label_id: str, error_class: ErrorClass | str, sentence: str) -> GoldLabel:
    return GoldLabel(
        label_id=label_id,
        paper_id=PID,
        variant=PaperVariant.INJECTED,
        error_class=error_class,  # type: ignore[arg-type]
        anchor=SentenceAnchor(sentence_id=sentence),
        target=LR,
        rationale="hand-made",
        evidence=(quote(S4),),
    )


LABELS = [gold("l0", ErrorClass.SETTING_MISMATCH, "s4"), gold("l1", ErrorClass.MATH_ERROR, "s3")]


def full_run(root: Path) -> Path:
    """A full-system run: one verified flag (true), one finding the verifier sent to review,
    and one the math review dropped; one paper version ran out of model calls, one failed
    once and then finished, one failed for good."""
    run = root / "exp-full-r1"
    write(
        run / "manifest.json", [manifest("exp-full-r1", SystemName.FULL, mocked_parts=["citation"])]
    )
    verified = [
        VerifiedFinding(
            finding=finding(0, "s4", ErrorClass.SETTING_MISMATCH, S4),
            state=VerificationState.VERIFIED_FLAG,
            reason="shown",
            accepted_evidence=(quote(S4),),
        ),
        VerifiedFinding(
            finding=finding(1, "s5", ErrorClass.SETTING_MISMATCH, "a quote that is not there"),
            state=VerificationState.NEEDS_HUMAN_REVIEW,
            reason="uncertain",
        ),
        VerifiedFinding(
            finding=finding(2, "s3", ErrorClass.MATH_ERROR, "Retrieval also improves factual"),
            state=VerificationState.DROPPED,
            reason=f"{NO_ERROR_REASON}: the paper is right",
        ),
    ]
    folder = run / f"{PID}-injected"
    write(folder / "verified.jsonl", verified)
    write(
        folder / "predictions.jsonl",
        predictions_from_verified(verified, run_id="exp-full-r1", system=SystemName.FULL),
    )
    write(
        folder / "results.jsonl",
        [
            TaskResult(
                task_id=f"t{n}",
                paper_id=PID,
                variant=PaperVariant.INJECTED,
                kind=TaskKind.CODE,
                outcome=outcome,
                reason=None if outcome is TaskOutcome.CHECKED else "no repository",
            )
            for n, outcome in enumerate([TaskOutcome.CHECKED] * 3 + [TaskOutcome.UNVERIFIABLE])
        ],
    )
    write(run / "other-original" / "predictions.incomplete.jsonl", [])
    (run / "failures.jsonl").write_text(
        json.dumps({"paper_id": PID, "variant": "injected", "error": "429"})
        + "\n"
        + json.dumps({"paper_id": "gone", "variant": "original", "error": "boom"})
        + "\n",
        encoding="utf-8",
    )
    calls = [
        LLMCallRecord(
            run_id="exp-full-r1",
            model="google_genai:test-model",
            input_tokens=1_000_000,
            output_tokens=100_000,
            started_at=datetime(2026, 11, 1, tzinfo=UTC),
            duration_s=1.0,
            status="ok",  # type: ignore[arg-type]
            cache_hit=hit,
        )
        for hit in (False, True)
    ]
    write(run / "usage" / "llm_calls.jsonl", calls)
    return run


def b0_run(root: Path) -> Path:
    run = root / "exp-b0-r1"
    write(run / "manifest.json", [manifest("exp-b0-r1", SystemName.B0)])
    write(
        run / f"{PID}-injected" / "predictions.jsonl",
        [
            Prediction(
                run_id="exp-b0-r1",
                system=SystemName.B0,
                paper_id=PID,
                variant=PaperVariant.INJECTED,
                error_class=ErrorClass.SETTING_MISMATCH,
                anchor=SentenceAnchor(sentence_id="s4"),
                target=LR,
                summary="lr differs",
            )
        ],
    )
    return run


def dataset(root: Path) -> Path:
    paper = PAPER.model_copy(update={"variant": PaperVariant.INJECTED})
    (root / "paper.json").write_text(paper.model_dump_json(), encoding="utf-8")
    path = root / "dataset.jsonl"
    path.write_text(json.dumps({"paper": "paper.json"}) + "\n", encoding="utf-8")
    return path


def test_a_run_folder_is_read_with_its_incomplete_papers_failures_and_cost(tmp_path: Path) -> None:
    run = load_run(full_run(tmp_path))
    assert run.system is SystemName.FULL
    assert run.mocked_parts == ["citation"]
    assert [p.complete for p in run.papers] == [True, False]
    # The failure of a paper that finished on a later attempt is not counted.
    assert run.failed == ["gone-original"]
    # 1M input tokens at $1 and 100k output at $10 per million; the cached call is free.
    assert run.usage.llm_calls == 1
    assert run.usage.cached_llm_calls == 1
    assert run.usage.cost_usd == 2.0
    assert run.task_outcomes()[TaskOutcome.CHECKED] == 3


def test_coverage_leaves_out_the_tasks_of_placeholder_parts(tmp_path: Path) -> None:
    # Runs made before placeholders said so recorded a left-out part's tasks as checked.
    run_dir = full_run(tmp_path)
    results = next(run_dir.glob("*/results.jsonl"))
    placeholder = TaskResult(
        task_id="c0",
        paper_id=PID,
        variant=PaperVariant.INJECTED,
        kind=TaskKind.REFERENCE,
        outcome=TaskOutcome.CHECKED,
    )
    with results.open("a", encoding="utf-8") as handle:
        handle.write(placeholder.model_dump_json() + "\n")
    outcomes = load_run(run_dir).task_outcomes()
    assert outcomes[TaskOutcome.CHECKED] == 3
    assert sum(outcomes.values()) == 4


def test_without_the_verifier_every_finding_counts_but_math_review_drops(tmp_path: Path) -> None:
    predictions = load_run(full_run(tmp_path)).predictions_without_verifier()
    assert [p.finding_id for p in predictions] == ["t0-f0", "t1-f0"]
    assert all(p.system is SystemName.FULL_NO_VERIFIER and p.is_positive for p in predictions)


async def test_grounding_finds_paper_quotes_and_baselines_have_no_backing(tmp_path: Path) -> None:
    inputs = load_inputs(dataset(tmp_path))
    full = load_run(full_run(tmp_path))
    stats = await grounding(full.predictions_without_verifier(), inputs, Settings())
    assert (stats.flags, stats.backed, stats.found, stats.not_found) == (2, 1, 1, 1)
    assert stats.rate == 0.5
    b0 = await grounding(load_run(b0_run(tmp_path)).predictions(), inputs, Settings())
    assert (b0.flags, b0.backed, b0.backed_share) == (1, 0, 0.0)


async def test_evaluate_scores_each_system_and_the_no_verifier_ablation(tmp_path: Path) -> None:
    full_run(tmp_path)
    b0_run(tmp_path)
    runs = [load_run(p) for p in find_runs([tmp_path])]
    scores = await evaluate(runs, LABELS, inputs=load_inputs(dataset(tmp_path)))
    by_system = {s.system: s for s in scores}
    assert set(by_system) == {"b0", "full", "full_no_verifier"}
    full = by_system["full"]
    assert full.scores.total.true_positives == 1
    assert full.scores.per_class[ErrorClass.MATH_ERROR].false_negatives == 1
    assert full.review_load == 1
    assert full.coverage == 0.75
    assert full.decisions is not None
    assert full.decisions["real_error"]["verified_flag"] == 1
    assert by_system["full_no_verifier"].scores.unlabelled == 1
    assert by_system["full_no_verifier"].review_load is None
    markdown = render_markdown(scores)
    assert "| full_no_verifier | 1 |" in markdown
    assert "mocked: citation" in markdown
    report = to_json(scores)
    assert report["summary"]["full"]["Macro-F1"]["mean"] == 0.5


def test_the_evaluate_command_writes_markdown_and_json(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    full_run(tmp_path / "runs")
    b0_run(tmp_path / "runs")
    labels = tmp_path / "labels.jsonl"
    write(labels, LABELS)
    out_md, out_json = tmp_path / "eval.md", tmp_path / "eval.json"
    code = main(
        [
            "evaluate",
            str(labels),
            str(tmp_path / "runs"),
            "--dataset",
            str(dataset(tmp_path)),
            "--markdown",
            str(out_md),
            "--json",
            str(out_json),
        ]
    )
    assert code == 0
    assert "## Verifier decisions" in out_md.read_text(encoding="utf-8")
    numbers = json.loads(out_json.read_text(encoding="utf-8"))
    assert {r["system"] for r in numbers["runs"]} == {"b0", "full", "full_no_verifier"}
    assert "Macro-F1" in capsys.readouterr().out


def test_strict_refuses_runs_with_placeholder_parts(tmp_path: Path) -> None:
    full_run(tmp_path / "runs")
    labels = tmp_path / "labels.jsonl"
    write(labels, LABELS)
    assert main(["evaluate", str(labels), str(tmp_path / "runs"), "--strict"]) == 2
