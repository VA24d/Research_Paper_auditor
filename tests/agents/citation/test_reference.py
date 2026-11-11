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


async def test_a_venue_alias_is_not_a_mismatch(tmp_path: Path) -> None:
    work = dataclasses.replace(
        MATCHING_WORK, venue="Annual Meeting of the Association for Computational Linguistics"
    )
    entry_with_short_venue = example_paper().model_copy(
        update={
            "bibliography": tuple(
                b.model_copy(update={"venue": "ACL"}) if b.id == "b1" else b
                for b in example_paper().bibliography
            )
        }
    )
    task = AuditTask.create(
        paper_id=entry_with_short_venue.paper_id,
        variant=entry_with_short_venue.variant,
        kind=TaskKind.REFERENCE,
        anchor=BibliographyAnchor(bib_id="b1"),
        target=BibTarget(bib_id="b1"),
    )
    result = await check_reference(
        task,
        offline_context(tmp_path, paper=entry_with_short_venue),
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


async def test_an_author_the_work_does_not_have_is_a_mismatch(tmp_path: Path) -> None:
    """A placeholder or fabricated author the index does not know is reported."""
    work = dataclasses.replace(MATCHING_WORK, authors=("Jian Zhang", "Robin Li"))
    result = await check_reference(
        reference_task(),
        offline_context(tmp_path, paper=example_paper()),
        Resolution(work=work, queries=(("q", "found"),)),
    )
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.REFERENCE_METADATA_MISMATCH
    assert "author" in finding.summary
    assert "rajpurkar" in finding.summary


async def test_a_silently_truncated_author_list_is_a_mismatch(tmp_path: Path) -> None:
    """An entry listing 2 of the work's 5 authors without 'et al.' is reported."""
    work = dataclasses.replace(
        MATCHING_WORK,
        authors=(
            "Pranav Rajpurkar",
            "Jian Zhang",
            "Robin Li",
            "Percy Liang",
            "Christopher Manning",
        ),
    )
    result = await check_reference(
        reference_task(),
        offline_context(tmp_path, paper=example_paper()),
        Resolution(work=work, queries=(("q", "found"),)),
    )
    (finding,) = result.findings
    assert "2 of the work's 5 authors" in finding.summary


async def test_a_truncation_mark_is_not_a_mismatch(tmp_path: Path) -> None:
    work = dataclasses.replace(
        MATCHING_WORK,
        authors=(
            "Pranav Rajpurkar",
            "Jian Zhang",
            "Robin Li",
            "Percy Liang",
            "Christopher Manning",
        ),
    )
    paper = example_paper().model_copy(
        update={
            "bibliography": tuple(
                b.model_copy(update={"authors": ("Pranav Rajpurkar et al.",)})
                if b.id == "b1"
                else b
                for b in example_paper().bibliography
            )
        }
    )
    task = AuditTask.create(
        paper_id=paper.paper_id,
        variant=paper.variant,
        kind=TaskKind.REFERENCE,
        anchor=BibliographyAnchor(bib_id="b1"),
        target=BibTarget(bib_id="b1"),
    )
    result = await check_reference(
        task,
        offline_context(tmp_path, paper=paper),
        Resolution(work=work, queries=(("q", "found"),)),
    )
    assert result.findings == ()


async def test_swapped_authors_are_a_mismatch(tmp_path: Path) -> None:
    work = dataclasses.replace(MATCHING_WORK, authors=("Jian Zhang", "Pranav Rajpurkar"))
    result = await check_reference(
        reference_task(),
        offline_context(tmp_path, paper=example_paper()),
        Resolution(work=work, queries=(("q", "found"),)),
    )
    (finding,) = result.findings
    assert "first author" in finding.summary


async def test_a_near_miss_title_is_a_mismatch(tmp_path: Path) -> None:
    """A one-letter title corruption still resolves (fuzzy match) and is reported."""
    paper = example_paper().model_copy(
        update={
            "bibliography": tuple(
                b.model_copy(
                    update={"title": "SQuAD: 100,000+ Questions for Machine Comprehension of Tex"}
                )
                if b.id == "b1"
                else b
                for b in example_paper().bibliography
            )
        }
    )
    task = AuditTask.create(
        paper_id=paper.paper_id,
        variant=paper.variant,
        kind=TaskKind.REFERENCE,
        anchor=BibliographyAnchor(bib_id="b1"),
        target=BibTarget(bib_id="b1"),
    )
    result = await check_reference(
        task,
        offline_context(tmp_path, paper=paper),
        Resolution(work=MATCHING_WORK, queries=(("q", "found"),)),
    )
    (finding,) = result.findings
    assert "near-miss" in finding.summary


async def test_a_title_that_lost_its_hyphen_is_not_a_near_miss(tmp_path: Path) -> None:
    # PDF text drops hyphens ("energyefficient" for "energy-efficient" in set B 2501.00742).
    entry_title = "SQuAD: 100,000+ Questions for MachineComprehension of Text"
    paper = example_paper().model_copy(
        update={
            "bibliography": tuple(
                b.model_copy(update={"title": entry_title}) if b.id == "b1" else b
                for b in example_paper().bibliography
            )
        }
    )
    work = dataclasses.replace(
        MATCHING_WORK, title="SQuAD: 100,000+ Questions for Machine-Comprehension of Text"
    )
    result = await check_reference(
        reference_task(),
        offline_context(tmp_path, paper=paper),
        Resolution(work=work, queries=(("q", "found"),)),
    )
    assert result.findings == ()
