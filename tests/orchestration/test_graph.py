import asyncio
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from auditor.infra.context import RunContext
from auditor.infra.tools import tracked_tool
from auditor.infra.usage import UsageLog
from auditor.orchestration.graph import (
    AuditPipeline,
    NothingToResumeError,
    PlanFn,
    Specialists,
    ThreadInUseError,
    run_audit,
    thread_id,
)
from auditor.orchestration.outputs import RunDirectory, predictions_from_verified
from auditor.orchestration.runner import CheckFn, TaskPolicy
from auditor.orchestration.synthesis import synthesise
from auditor.schemas import (
    AuditTask,
    BibTarget,
    PaperVariant,
    ParsedPaper,
    Prediction,
    SentenceAnchor,
    SystemName,
    TaskKind,
    TaskOutcome,
    TaskResult,
    VerificationState,
    VerifiedFinding,
)
from auditor.schemas.examples import example_paper, example_tasks, example_verified_findings
from auditor.testing import checked_result, offline_context

FAST = TaskPolicy(max_attempts=1, timeout_s=5.0, backoff_base_s=0.0)


def citation_task(index: int) -> AuditTask:
    return AuditTask.create(
        paper_id="example-paper",
        variant=PaperVariant.ORIGINAL,
        kind=TaskKind.CITATION,
        anchor=SentenceAnchor(sentence_id=f"s{index}"),
        target=BibTarget(bib_id="b0"),
        claim=f"claim {index}",
    )


def planner_for(tasks: list[AuditTask]) -> PlanFn:
    async def plan(paper: ParsedPaper, ctx: RunContext) -> list[AuditTask]:
        return list(tasks)

    return plan


class Recorder:
    """Specialist and verifier stand-ins that remember how they were called."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.verify_calls: list[list[TaskResult]] = []

    def check(self, name: str) -> CheckFn:
        async def run(task: AuditTask, ctx: RunContext) -> TaskResult:
            self.calls[name] += 1
            return checked_result(task)

        return run

    async def verify(self, results: Sequence[TaskResult], ctx: RunContext) -> list[VerifiedFinding]:
        self.verify_calls.append(list(results))
        return []


def pipeline(tmp_path: Path, tasks: list[AuditTask], **overrides: object) -> AuditPipeline:
    recorder = overrides.pop("recorder", None)
    assert recorder is None or isinstance(recorder, Recorder)
    defaults: dict[str, object] = {"policy": FAST}
    if recorder is not None:
        defaults["specialists"] = Specialists(
            citation=recorder.check("citation"),
            code=recorder.check("code"),
            template=recorder.check("template"),
        )
        defaults["verifier"] = recorder.verify
    return AuditPipeline(
        ctx=offline_context(tmp_path),
        planner=planner_for(tasks),
        **{**defaults, **overrides},  # type: ignore[arg-type]
    )


async def test_tasks_are_routed_by_kind_and_verified_once(tmp_path: Path) -> None:
    recorder = Recorder()
    tasks = example_tasks()
    outcome = await run_audit(
        pipeline(tmp_path, tasks, recorder=recorder),
        example_paper(),
        run_id="r1",
        system=SystemName.FULL,
    )
    assert recorder.calls == {"citation": 3, "code": 1, "template": 1}
    assert len(recorder.verify_calls) == 1
    assert [r.task_id for r in outcome.results] == [t.task_id for t in tasks]
    assert all(r.outcome is TaskOutcome.CHECKED for r in outcome.results)


async def test_paper_with_no_tasks_still_reaches_the_verifier(tmp_path: Path) -> None:
    recorder = Recorder()
    outcome = await run_audit(
        pipeline(tmp_path, [], recorder=recorder),
        example_paper(),
        run_id="r1",
        system=SystemName.FULL,
    )
    assert outcome.results == []
    assert recorder.verify_calls == [[]]


async def test_only_one_kind_of_task(tmp_path: Path) -> None:
    recorder = Recorder()
    await run_audit(
        pipeline(tmp_path, [citation_task(1), citation_task(2)], recorder=recorder),
        example_paper(),
        run_id="r1",
        system=SystemName.FULL,
    )
    assert recorder.calls == {"citation": 2}
    assert len(recorder.verify_calls[0]) == 2


async def test_failing_task_does_not_stop_parallel_tasks(tmp_path: Path) -> None:
    recorder = Recorder()

    async def broken(task: AuditTask, ctx: RunContext) -> TaskResult:
        await asyncio.sleep(0.01)
        raise RuntimeError("repository could not be read")

    async def slow_timeout(task: AuditTask, ctx: RunContext) -> TaskResult:
        await asyncio.sleep(10)
        return checked_result(task)

    specialists = Specialists(
        citation=recorder.check("citation"), code=broken, template=slow_timeout
    )
    outcome = await run_audit(
        pipeline(
            tmp_path,
            example_tasks(),
            specialists=specialists,
            verifier=recorder.verify,
            policy=TaskPolicy(max_attempts=1, timeout_s=0.2, backoff_base_s=0.0),
        ),
        example_paper(),
        run_id="r1",
        system=SystemName.FULL,
    )
    by_kind = {r.kind: r for r in outcome.results}
    assert by_kind[TaskKind.CODE].outcome is TaskOutcome.FAILED
    assert by_kind[TaskKind.TEMPLATE].outcome is TaskOutcome.FAILED
    assert by_kind[TaskKind.REFERENCE].outcome is TaskOutcome.CHECKED
    assert recorder.calls["citation"] == 3
    assert len(recorder.verify_calls[0]) == 5


async def test_concurrency_cap_is_respected(tmp_path: Path) -> None:
    running = 0
    peak = 0

    async def check(task: AuditTask, ctx: RunContext) -> TaskResult:
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        running -= 1
        return checked_result(task)

    tasks = [citation_task(i) for i in range(12)]
    outcome = await run_audit(
        pipeline(tmp_path, tasks, specialists=Specialists(citation=check), max_concurrency=3),
        example_paper(),
        run_id="r1",
        system=SystemName.FULL,
    )
    assert len(outcome.results) == 12
    assert peak == 3


async def test_usage_is_attributed_to_tasks(tmp_path: Path) -> None:
    @tracked_tool()
    async def lookup(title: str) -> str:
        return title

    async def check(task: AuditTask, ctx: RunContext) -> TaskResult:
        await lookup(task.task_id)
        return checked_result(task)

    usage = UsageLog()
    tasks = [citation_task(i) for i in range(3)]
    await run_audit(
        pipeline(tmp_path, tasks, specialists=Specialists(citation=check)),
        example_paper(),
        run_id="r7",
        system=SystemName.FULL,
        usage=usage,
    )
    assert sorted(r.task_id or "" for r in usage.tool_calls) == sorted(t.task_id for t in tasks)
    assert {r.run_id for r in usage.tool_calls} == {"r7"}


def test_thread_id_separates_runs_systems_and_variants() -> None:
    paper = example_paper()
    injected = paper.model_copy(update={"variant": PaperVariant.INJECTED})
    ids = {
        thread_id("r1", SystemName.FULL, paper),
        thread_id("r2", SystemName.FULL, paper),
        thread_id("r1", SystemName.B1, paper),
        thread_id("r1", SystemName.FULL, injected),
    }
    assert len(ids) == 4


async def test_reusing_a_thread_for_a_fresh_run_is_refused(tmp_path: Path) -> None:
    saver = InMemorySaver()
    audit = pipeline(tmp_path, [citation_task(1)])
    await run_audit(audit, example_paper(), run_id="r1", system=SystemName.FULL, checkpointer=saver)
    with pytest.raises(ThreadInUseError):
        await run_audit(
            audit, example_paper(), run_id="r1", system=SystemName.FULL, checkpointer=saver
        )
    second = await run_audit(
        audit, example_paper(), run_id="r2", system=SystemName.FULL, checkpointer=saver
    )
    assert len(second.results) == 1


class Crash(BaseException):
    """Stands in for the process dying mid-run (not an Exception, so nothing catches it)."""


async def test_interrupted_run_resumes_without_redoing_finished_tasks(tmp_path: Path) -> None:
    tasks = example_tasks()
    calls: Counter[str] = Counter()
    crash = True

    def counting(name: str) -> CheckFn:
        async def run(task: AuditTask, ctx: RunContext) -> TaskResult:
            calls[name] += 1
            return checked_result(task)

        return run

    async def code_check(task: AuditTask, ctx: RunContext) -> TaskResult:
        calls["code"] += 1
        await asyncio.sleep(0.05)
        if crash:
            raise Crash
        return checked_result(task)

    audit = pipeline(
        tmp_path,
        tasks,
        specialists=Specialists(
            citation=counting("citation"), code=code_check, template=counting("template")
        ),
    )
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "checkpoints.sqlite")) as saver:
        with pytest.raises(Crash):
            await run_audit(
                audit, example_paper(), run_id="r1", system=SystemName.FULL, checkpointer=saver
            )
        assert calls == {"citation": 3, "code": 1, "template": 1}

        crash = False
        outcome = await run_audit(
            audit,
            example_paper(),
            run_id="r1",
            system=SystemName.FULL,
            checkpointer=saver,
            resume=True,
        )
    assert calls == {"citation": 3, "code": 2, "template": 1}
    assert len(outcome.results) == 5
    assert all(r.outcome is TaskOutcome.CHECKED for r in outcome.results)


async def test_resuming_a_thread_without_state_is_refused(tmp_path: Path) -> None:
    with pytest.raises(NothingToResumeError):
        await run_audit(
            pipeline(tmp_path, [citation_task(1)]),
            example_paper(),
            run_id="never-ran",
            system=SystemName.FULL,
            checkpointer=InMemorySaver(),
            resume=True,
        )


async def test_verifier_decisions_reach_the_output_files(tmp_path: Path) -> None:
    decisions = example_verified_findings()

    async def verify(results: Sequence[TaskResult], ctx: RunContext) -> list[VerifiedFinding]:
        return list(decisions)

    outcome = await run_audit(
        pipeline(tmp_path, example_tasks(), verifier=verify),
        example_paper(),
        run_id="r1",
        system=SystemName.FULL,
    )
    # The synthesis step rates every decision; the second repeats the first.
    decisions = synthesise(decisions)
    assert [d.duplicate_of for d in decisions] == [None, decisions[0].finding.finding_id]
    assert outcome.verified == decisions
    predictions = predictions_from_verified(outcome.verified, run_id="r1", system=SystemName.FULL)
    folder = RunDirectory(tmp_path / "runs", "r1").save_audit(example_paper(), outcome, predictions)

    def read(name: str) -> list[str]:
        return (folder / name).read_text(encoding="utf-8").splitlines()

    assert [
        VerifiedFinding.model_validate_json(line) for line in read("verified.jsonl")
    ] == decisions
    written = [Prediction.model_validate_json(line) for line in read("predictions.jsonl")]
    assert [(p.finding_id, p.state, p.evidence, p.duplicate_of) for p in written] == [
        (d.finding.finding_id, d.state, d.accepted_evidence, d.duplicate_of) for d in decisions
    ]
    assert {d.state for d in decisions} == {
        VerificationState.VERIFIED_FLAG,
        VerificationState.NEEDS_HUMAN_REVIEW,
    }
