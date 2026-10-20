"""Offline tests for the reference check (P2.5): canned resolutions, no network."""

import dataclasses
from pathlib import Path

from auditor.agents.citation.reference import check_reference
from auditor.retrieval.clients import Work
from auditor.retrieval.resolve import Resolution
from auditor.schemas import (
    AuditTask,
    BibliographyAnchor,
    BibTarget,
    ErrorClass,
    EvidenceKind,
    ParsedPaper,
    TaskKind,
    TaskOutcome,
)
from auditor.schemas.examples import example_paper
from auditor.testing import offline_context

MATCHING_WORK = Work(
    id="doi:10.18653/v1/d16-1264",
    source="crossref",
    title="SQuAD: 100,000+ Questions for Machine Comprehension of Text",
    authors=("Pranav Rajpurkar", "Jian Zhang"),
    year=2016,
    venue="EMNLP",
    doi="10.18653/v1/d16-1264",
)


def reference_task(bib_id: str = "b1") -> AuditTask:
    paper = example_paper()
    return AuditTask.create(
        paper_id=paper.paper_id,
        variant=paper.variant,
        kind=TaskKind.REFERENCE,
        anchor=BibliographyAnchor(bib_id=bib_id),
        target=BibTarget(bib_id=bib_id),
    )


async def test_a_clean_reference_is_checked_without_findings(tmp_path: Path) -> None:
    result = await check_reference(
        reference_task(),
        offline_context(tmp_path, paper=example_paper()),
        Resolution(work=MATCHING_WORK, queries=(("q", "found"),)),
    )
    assert result.outcome is TaskOutcome.CHECKED
    assert result.findings == ()


async def test_an_unresolved_reference_is_a_finding(tmp_path: Path) -> None:
    resolution = Resolution(reason="no work with this title was found", queries=(("q", "0 hits"),))
    result = await check_reference(
        reference_task(), offline_context(tmp_path, paper=example_paper()), resolution
    )
    assert result.outcome is TaskOutcome.CHECKED
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.REFERENCE_NOT_FOUND
    assert "no work with this title" in finding.summary
    (lookup,) = finding.evidence
    assert lookup.kind is EvidenceKind.LOOKUP
    assert "not found" in lookup.text


async def test_metadata_mismatches_are_reported_field_by_field(tmp_path: Path) -> None:
    work = dataclasses.replace(
        MATCHING_WORK, title="A Different Title Entirely", year=2019, venue="ACL"
    )
    result = await check_reference(
        reference_task(),
        offline_context(tmp_path, paper=example_paper()),
        Resolution(work=work, queries=(("q", "found"),)),
    )
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.REFERENCE_METADATA_MISMATCH
    for field in ("title", "year", "venue"):
        assert field in finding.summary
    (lookup,) = finding.evidence
    assert lookup.kind is EvidenceKind.LOOKUP


async def test_metadata_that_the_index_does_not_give_is_not_a_mismatch(tmp_path: Path) -> None:
    work = Work(id="w1", source="s2", title="SQuAD: 100,000+ Questions for Machine Comprehension")
    result = await check_reference(
        reference_task(),
        offline_context(tmp_path, paper=example_paper()),
        Resolution(work=work, queries=(("q", "found"),)),
    )
    assert result.findings == ()


async def test_an_entry_missing_from_the_paper_is_unverifiable(tmp_path: Path) -> None:
    paper = example_paper()
    without_b1 = ParsedPaper.model_validate(
        {
            **paper.model_dump(),
            "bibliography": [b.model_dump() for b in paper.bibliography if b.id != "b1"],
            "citations": [c.model_dump() for c in paper.citations if "b1" not in c.bib_ids],
        }
    )
    result = await check_reference(
        reference_task(),
        offline_context(tmp_path, paper=without_b1),
        Resolution(reason="no work with this title was found"),
    )
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "not in the paper" in result.reason
