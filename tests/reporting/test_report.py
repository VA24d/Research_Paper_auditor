"""The audit report: three sections, evidence with its source, advisory wording (team plan
P4.4)."""

import json
from pathlib import Path

from auditor.cli import main
from auditor.reporting import AuditReport, build_report, render_markdown, write_report
from auditor.reporting.report import CLASS_TITLES, NOTICE, short_repository
from auditor.schemas import (
    ErrorClass,
    Evidence,
    EvidenceKind,
    Severity,
    SourceKind,
    SourceRef,
    VerificationState,
    VerifiedFinding,
)
from auditor.schemas.examples import (
    example_finding,
    example_paper,
    example_task_results,
    example_tasks,
)

CODE = Evidence(
    kind=EvidenceKind.CODE,
    source=SourceRef(kind=SourceKind.REPOSITORY, id="https://github.com/example/demo@0123abcdef"),
    text="lr = 1e-3",
    path="train.py",
    line_start=12,
)


def decisions() -> list[VerifiedFinding]:
    finding = example_finding()
    return [
        VerifiedFinding(
            finding=finding,
            state=VerificationState.VERIFIED_FLAG,
            reason="Every side of the finding is shown.",
            accepted_evidence=finding.evidence,
            severity=Severity.HIGH,
        ),
        VerifiedFinding(
            finding=finding.model_copy(update={"finding_id": f"{finding.task_id}-f1"}),
            state=VerificationState.VERIFIED_FLAG,
            reason="Every side of the finding is shown.",
            accepted_evidence=finding.evidence,
            duplicate_of=finding.finding_id,
        ),
        VerifiedFinding(
            finding=finding.model_copy(
                update={
                    "finding_id": f"{finding.task_id}-f2",
                    "error_class": ErrorClass.SETTING_MISMATCH,
                    "evidence": (CODE,),
                    "uncertain": True,
                }
            ),
            state=VerificationState.NEEDS_HUMAN_REVIEW,
            reason="The evidence is in its sources, but the specialist marked it uncertain.",
        ),
        VerifiedFinding(
            finding=finding.model_copy(update={"finding_id": f"{finding.task_id}-f3"}),
            state=VerificationState.DROPPED,
            reason="Not in its source: the quote was not found in the cited work.",
        ),
    ]


def report() -> AuditReport:
    return build_report(
        example_paper(), example_tasks(), example_task_results(), decisions(), run_id="r1"
    )


def test_findings_go_to_their_sections_and_repeats_are_shown_once() -> None:
    built = report()
    assert [i.finding_id for i in built.verified] == [example_finding().finding_id]
    assert built.verified[0].also_reported_as == (f"{example_finding().task_id}-f1",)
    assert built.verified[0].severity == "high"
    assert [i.error_class for i in built.review] == [ErrorClass.SETTING_MISMATCH]
    assert len(built.dropped) == 1


def test_items_show_the_papers_text_and_their_evidence() -> None:
    flag = report().verified[0]
    assert flag.place.sentence_id == "s2"
    assert flag.place.text == "Prior work reports 92.1% exact match on SQuAD [2, 3]."
    assert flag.evidence[0].source_kind is SourceKind.CITED_WORK
    review = report().review[0]
    assert review.uncertain
    assert review.evidence[0].location == "train.py:12"


def test_unfinished_checks_are_listed_with_their_reasons() -> None:
    not_checked = report().not_checked
    assert {i.outcome for i in not_checked} == {"unverifiable", "failed"}
    assert any("not linked to any bibliography entry" in i.reason for i in not_checked)
    assert report().task_outcomes == {"checked": 2, "unverifiable": 1, "failed": 1}


def test_markdown_is_advisory_and_has_the_three_sections() -> None:
    markdown = render_markdown(report())
    assert NOTICE in markdown
    for heading in (
        "## 1. Verified flags",
        "## 2. Items for human review",
        "## 3. Items not checked",
    ):
        assert heading in markdown
    assert "Repository `example/demo@0123abc`, train.py:12" in markdown
    assert "**Why it is here:**" in markdown
    lowered = markdown.lower()
    for word in ("fraud", "deliberate", "intentional", "misconduct", "fabricat"):
        assert word not in lowered


def test_every_error_class_has_a_neutral_title() -> None:
    assert set(CLASS_TITLES) == set(ErrorClass)


def test_short_repository_names() -> None:
    assert short_repository("https://github.com/owner/repo@0123456789") == "owner/repo@0123456"
    assert short_repository("file:///data/patched/2001.0001v1@content-abcdef123") == (
        "2001.0001v1@abcdef1"
    )


def test_write_report_and_the_report_command(tmp_path: Path) -> None:
    folder = tmp_path / "runs" / "r1" / "example-paper-original"
    folder.mkdir(parents=True)
    write_report(folder, report())
    assert json.loads((folder / "report.json").read_text(encoding="utf-8"))["paper_id"] == (
        "example-paper"
    )
    for name, records in (
        ("tasks.jsonl", example_tasks()),
        ("results.jsonl", example_task_results()),
        ("verified.jsonl", decisions()),
    ):
        (folder / name).write_text(
            "".join(r.model_dump_json() + "\n" for r in records), encoding="utf-8"
        )
    paper = tmp_path / "paper.json"
    paper.write_text(example_paper().model_dump_json(), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    assert main(["report", str(folder), "--paper", str(paper), "-o", str(out)]) == 0
    assert "## 1. Verified flags" in (out / "report.md").read_text(encoding="utf-8")
