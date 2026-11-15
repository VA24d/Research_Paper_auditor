"""Offline tests for the citation tools for B1 (P2.6): canned API replies, no network."""

import dataclasses
import json
from pathlib import Path

import httpx
from langchain_core.tools import BaseTool

from auditor.agents.citation.tools import MAX_TOOL_CHARS, make_tools
from auditor.config import Settings
from auditor.infra.context import RunContext
from auditor.infra.http import HttpClient
from auditor.infra.scope import Scope, enter_scope
from auditor.infra.usage import UsageLog
from auditor.schemas import BibEntry
from auditor.testing import offline_context

CROSSREF_HIT = {
    "message": {
        "DOI": "10.18653/V1/D16-1264",
        "title": ["SQuAD: 100,000+ Questions for Machine Comprehension of Text"],
        "container-title": ["EMNLP"],
        "author": [{"given": "Pranav", "family": "Rajpurkar"}],
        "issued": {"date-parts": [[2016]]},
        "abstract": "We present a reading comprehension dataset. " * 400,
    }
}

ENTRY = BibEntry(
    id="b1",
    raw="Rajpurkar et al. SQuAD. EMNLP 2016.",
    title="SQuAD: 100,000+ Questions for Machine Comprehension of Text",
    authors=("Pranav Rajpurkar",),
    year=2016,
    doi="10.18653/v1/D16-1264",
)


async def _no_sleep(_seconds: float) -> None:
    return None


def context(tmp_path: Path, routes: dict[str, httpx.Response] | None = None) -> RunContext:
    def handler(request: httpx.Request) -> httpx.Response:
        for pattern, response in (routes or {}).items():
            if pattern in str(request.url):
                return response
        return httpx.Response(404)

    settings = Settings(cache_dir=tmp_path / "cache", scratch_dir=tmp_path / "scratch")
    http = HttpClient(settings, transport=httpx.MockTransport(handler), sleep=_no_sleep)
    return offline_context(tmp_path, http=http)


def tools_by_name(ctx: RunContext) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in make_tools(ctx)}


async def test_make_tools_gives_the_three_citation_tools(tmp_path: Path) -> None:
    assert set(tools_by_name(context(tmp_path))) == {
        "lookup_reference",
        "retrieve_cited_text",
        "find_supporting_passages",
    }


async def test_lookup_reference_answers_the_resolved_work(tmp_path: Path) -> None:
    routes = {"api.crossref.org": httpx.Response(200, json=CROSSREF_HIT)}
    tools = tools_by_name(context(tmp_path, routes))
    lookup = await tools["lookup_reference"].ainvoke({"entry_json": ENTRY.model_dump_json()})
    record = json.loads(lookup)
    assert record["found"] is True
    assert record["source"] == "crossref"
    assert record["doi"] == "10.18653/v1/d16-1264"
    assert record["queries"] == ["crossref doi:10.18653/v1/d16-1264"]


async def test_lookup_reference_says_why_nothing_was_found(tmp_path: Path) -> None:
    tools = tools_by_name(context(tmp_path))
    lookup = await tools["lookup_reference"].ainvoke({"entry_json": ENTRY.model_dump_json()})
    record = json.loads(lookup)
    assert record["found"] is False
    assert record["reason"] is not None


async def test_retrieve_cited_text_prefers_the_abstract_when_nothing_else_exists(
    tmp_path: Path,
) -> None:
    routes = {"api.crossref.org": httpx.Response(200, json=CROSSREF_HIT)}
    tools = tools_by_name(context(tmp_path, routes))
    text = await tools["retrieve_cited_text"].ainvoke({"entry_json": ENTRY.model_dump_json()})
    assert text.startswith("source level: abstract")
    assert "reading comprehension dataset" in text


async def test_retrieve_cited_text_clips_long_text(tmp_path: Path) -> None:
    routes = {"api.crossref.org": httpx.Response(200, json=CROSSREF_HIT)}
    tools = tools_by_name(context(tmp_path, routes))
    text = await tools["retrieve_cited_text"].ainvoke({"entry_json": ENTRY.model_dump_json()})
    assert len(text) <= MAX_TOOL_CHARS + 100
    assert "clipped" in text


async def test_find_supporting_passages_numbers_the_passages(tmp_path: Path) -> None:
    routes = {"api.crossref.org": httpx.Response(200, json=CROSSREF_HIT)}
    tools = tools_by_name(context(tmp_path, routes))
    text = await tools["find_supporting_passages"].ainvoke(
        {"entry_json": ENTRY.model_dump_json(), "claim": "reading comprehension dataset"}
    )
    assert text.startswith("source level: abstract")
    assert "[1]" in text


async def test_an_unresolvable_entry_is_reported_not_raised(tmp_path: Path) -> None:
    tools = tools_by_name(context(tmp_path))
    text = await tools["retrieve_cited_text"].ainvoke({"entry_json": ENTRY.model_dump_json()})
    assert text.startswith("Could not resolve the reference")


async def test_tool_calls_are_counted(tmp_path: Path) -> None:
    routes = {"api.crossref.org": httpx.Response(200, json=CROSSREF_HIT)}
    ctx = context(tmp_path, routes)
    tools = tools_by_name(ctx)
    usage = UsageLog()
    with enter_scope(Scope(run_id="r", usage=usage)):
        await tools["lookup_reference"].ainvoke({"entry_json": ENTRY.model_dump_json()})
    assert [record.tool for record in usage.tool_calls] == ["lookup_reference"]


ARXIV_ENTRY = BibEntry(
    id="b0",
    raw="Vaswani et al. Attention is all you need. NeurIPS 2017.",
    title="Attention Is All You Need",
    authors=("Ashish Vaswani",),
    year=2017,
    arxiv_id="1706.03762",
)

ARXIV_ROUTES = {
    "export.arxiv.org": httpx.Response(
        200,
        text=(
            '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
            "<entry><id>http://arxiv.org/abs/1706.03762</id>"
            "<title>Attention Is All You Need</title>"
            "<summary>The dominant sequence models are based on recurrence.</summary>"
            "<author><name>Ashish Vaswani</name></author>"
            "<published>2017-06-12T00:00:00Z</published></entry></feed>"
        ),
    ),
    "arxiv.org/html/1706.03762": httpx.Response(
        200,
        text="<html><body><p>"
        + "Self-attention replaces recurrence entirely. " * 80
        + "</p></body></html>",
    ),
}


async def test_the_abstract_only_setting_reaches_the_tools(tmp_path: Path) -> None:
    """The tools follow the setting unless the call passes its own value (shortcoming 9)."""
    base = context(tmp_path, ARXIV_ROUTES)
    settings = base.settings.model_copy(update={"citation_abstract_only": True})
    ctx = dataclasses.replace(base, settings=settings)
    text = await tools_by_name(ctx)["retrieve_cited_text"].ainvoke(
        {"entry_json": ARXIV_ENTRY.model_dump_json()}
    )
    assert text.startswith("source level: abstract")

    # An explicit argument overrides the setting, both ways.
    full = await tools_by_name(ctx)["retrieve_cited_text"].ainvoke(
        {"entry_json": ARXIV_ENTRY.model_dump_json(), "abstract_only": False}
    )
    assert full.startswith("source level: full_html")

    default_ctx = context(tmp_path, ARXIV_ROUTES)
    text = await tools_by_name(default_ctx)["retrieve_cited_text"].ainvoke(
        {"entry_json": ARXIV_ENTRY.model_dump_json()}
    )
    assert text.startswith("source level: full_html")


def test_a_partial_text_is_named_as_such_for_the_agent() -> None:
    """B1 must know a publisher page may hold only the abstract, as the agent does."""
    from auditor.agents.citation.tools import _level_line
    from auditor.retrieval.clients import Work
    from auditor.retrieval.fulltext import SourceLevel, SourceText

    work = Work(id="doi:10.1/x", source="openalex", title="T")
    page = SourceText(work=work, level=SourceLevel.FULL_HTML, text="t", partial=True)
    assert "partial" in _level_line(page)
    full = SourceText(work=work, level=SourceLevel.FULL_HTML, text="t")
    assert _level_line(full) == "source level: full_html"
