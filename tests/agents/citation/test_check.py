"""Offline tests for the citation agent's check() entry point (P2.5).

The resolution and retrieval are monkeypatched, so the tests cover the wiring (which
resolution path each target takes, and how the abstract-only setting is passed on) without
any network.
"""

import dataclasses
from pathlib import Path

from auditor.agents.citation import check
from auditor.retrieval.clients import Work
from auditor.retrieval.fulltext import SourceLevel, SourceText
from auditor.retrieval.resolve import Resolution
from auditor.schemas import (
    AuditTask,
    BibliographyAnchor,
    BibTarget,
    MarkerTarget,
    ParsedPaper,
    SentenceAnchor,
    TaskKind,
    TaskOutcome,
)
from auditor.schemas.examples import example_paper
from auditor.testing import offline_context

WORK = Work(id="arxiv:1706.03762", source="arxiv", title="Attention Is All You Need")


def reference_task() -> AuditTask:
    paper = example_paper()
    return AuditTask.create(
        paper_id=paper.paper_id,
        variant=paper.variant,
        kind=TaskKind.REFERENCE,
        anchor=BibliographyAnchor(bib_id="b0"),
        target=BibTarget(bib_id="b0"),
    )


def citation_task(target: BibTarget | MarkerTarget) -> AuditTask:
    paper = example_paper()
    return AuditTask.create(
        paper_id=paper.paper_id,
        variant=paper.variant,
        kind=TaskKind.CITATION,
        anchor=SentenceAnchor(sentence_id="s2"),
        target=target,
        claim=paper.sentence("s2").text,
    )


def patch_resolution(monkeypatch, resolution: Resolution, calls: list) -> None:
    async def fake_resolve(http, entry):
        calls.append(entry)
        return resolution

    monkeypatch.setattr("auditor.agents.citation.resolve_entry", fake_resolve)


def patch_marker_resolution(monkeypatch, resolution: Resolution, calls: list) -> None:
    async def fake_resolve_marker(http, marker, entries):
        calls.append(marker)
        return resolution

    monkeypatch.setattr("auditor.agents.citation.resolve_marker", fake_resolve_marker)


def patch_retrieval(monkeypatch, calls: list) -> None:
    async def fake_retrieve(http, work, settings, *, grobid=None, abstract_only=False):
        calls.append(abstract_only)
        return SourceText(work=work, level=SourceLevel.ABSTRACT, text="The abstract text.")

    monkeypatch.setattr("auditor.agents.citation.retrieve_source", fake_retrieve)


async def test_a_reference_task_resolves_its_entry(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    patch_resolution(monkeypatch, Resolution(work=WORK), calls)
    result = await check(reference_task(), offline_context(tmp_path))
    assert result.outcome is TaskOutcome.CHECKED
    assert [entry.id for entry in calls] == ["b0"]


async def test_a_citation_task_with_a_bib_target_resolves_and_retrieves(
    tmp_path: Path, monkeypatch
) -> None:
    resolved: list = []
    retrieved: list = []
    patch_resolution(monkeypatch, Resolution(work=WORK), resolved)
    patch_retrieval(monkeypatch, retrieved)
    result = await check(citation_task(BibTarget(bib_id="b1")), offline_context(tmp_path))
    assert result.outcome is TaskOutcome.UNVERIFIABLE  # no LLM configured
    assert [entry.id for entry in resolved] == ["b1"]
    assert retrieved == [False]  # full text by default


async def test_the_abstract_only_setting_is_passed_to_retrieval(
    tmp_path: Path, monkeypatch
) -> None:
    retrieved: list = []
    patch_resolution(monkeypatch, Resolution(work=WORK), [])
    patch_retrieval(monkeypatch, retrieved)
    ctx = offline_context(tmp_path)
    settings = ctx.settings.model_copy(update={"citation_abstract_only": True})
    await check(citation_task(BibTarget(bib_id="b1")), dataclasses.replace(ctx, settings=settings))
    assert retrieved == [True]


async def test_an_unresolvable_citation_is_unverifiable_with_the_reason(
    tmp_path: Path, monkeypatch
) -> None:
    patch_resolution(monkeypatch, Resolution(reason="no work with this title was found"), [])
    result = await check(citation_task(BibTarget(bib_id="b1")), offline_context(tmp_path))
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason == "no work with this title was found"


async def test_an_unlinked_marker_goes_through_resolve_marker(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    patch_marker_resolution(monkeypatch, Resolution(reason="no bibliography entry matches"), calls)
    result = await check(citation_task(MarkerTarget(marker="[7]")), offline_context(tmp_path))
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert calls == ["[7]"]


async def test_a_resolved_marker_retrieves_the_source(tmp_path: Path, monkeypatch) -> None:
    retrieved: list = []
    patch_marker_resolution(monkeypatch, Resolution(work=WORK), [])
    patch_retrieval(monkeypatch, retrieved)
    result = await check(citation_task(MarkerTarget(marker="[7]")), offline_context(tmp_path))
    assert result.outcome is TaskOutcome.UNVERIFIABLE  # no LLM configured
    assert retrieved == [False]


async def test_an_entry_missing_from_the_paper_is_unverifiable(tmp_path: Path, monkeypatch) -> None:
    paper = example_paper()
    without_b1 = ParsedPaper.model_validate(
        {
            **paper.model_dump(),
            "bibliography": [b.model_dump() for b in paper.bibliography if b.id != "b1"],
            "citations": [c.model_dump() for c in paper.citations if "b1" not in c.bib_ids],
        }
    )
    result = await check(
        citation_task(BibTarget(bib_id="b1")), offline_context(tmp_path, paper=without_b1)
    )
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "not in the paper" in result.reason
