"""Offline tests for the citation tools for B1 (P2.6): canned API replies, no network."""

import json
from pathlib import Path

import httpx

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


def tools_by_name(ctx: RunContext) -> dict[str, object]:
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
    lookup = await tools["lookup_reference"].ainvoke({"entry_json": ENTRY.model_dump_json()})  # type: ignore[union-attr]
    record = json.loads(lookup)
    assert record["found"] is True
    assert record["source"] == "crossref"
    assert record["doi"] == "10.18653/v1/d16-1264"
    assert record["queries"] == ["crossref doi:10.18653/v1/d16-1264"]


async def test_lookup_reference_says_why_nothing_was_found(tmp_path: Path) -> None:
    tools = tools_by_name(context(tmp_path))
    lookup = await tools["lookup_reference"].ainvoke({"entry_json": ENTRY.model_dump_json()})  # type: ignore[union-attr]
    record = json.loads(lookup)
    assert record["found"] is False
    assert record["reason"] is not None


async def test_retrieve_cited_text_prefers_the_abstract_when_nothing_else_exists(
    tmp_path: Path,
) -> None:
    routes = {"api.crossref.org": httpx.Response(200, json=CROSSREF_HIT)}
    tools = tools_by_name(context(tmp_path, routes))
    text = await tools["retrieve_cited_text"].ainvoke({"entry_json": ENTRY.model_dump_json()})  # type: ignore[union-attr]
    assert text.startswith("source level: abstract")
    assert "reading comprehension dataset" in text


async def test_retrieve_cited_text_clips_long_text(tmp_path: Path) -> None:
    routes = {"api.crossref.org": httpx.Response(200, json=CROSSREF_HIT)}
    tools = tools_by_name(context(tmp_path, routes))
    text = await tools["retrieve_cited_text"].ainvoke({"entry_json": ENTRY.model_dump_json()})  # type: ignore[union-attr]
    assert len(text) <= MAX_TOOL_CHARS + 100
    assert "clipped" in text


async def test_find_supporting_passages_numbers_the_passages(tmp_path: Path) -> None:
    routes = {"api.crossref.org": httpx.Response(200, json=CROSSREF_HIT)}
    tools = tools_by_name(context(tmp_path, routes))
    text = await tools["find_supporting_passages"].ainvoke(
        {"entry_json": ENTRY.model_dump_json(), "claim": "reading comprehension dataset"}
    )  # type: ignore[union-attr]
    assert text.startswith("source level: abstract")
    assert "[1]" in text


async def test_an_unresolvable_entry_is_reported_not_raised(tmp_path: Path) -> None:
    tools = tools_by_name(context(tmp_path))
    text = await tools["retrieve_cited_text"].ainvoke({"entry_json": ENTRY.model_dump_json()})  # type: ignore[union-attr]
    assert text.startswith("Could not resolve the reference")


async def test_tool_calls_are_counted(tmp_path: Path) -> None:
    routes = {"api.crossref.org": httpx.Response(200, json=CROSSREF_HIT)}
    ctx = context(tmp_path, routes)
    tools = tools_by_name(ctx)
    usage = UsageLog()
    with enter_scope(Scope(run_id="r", usage=usage)):
        await tools["lookup_reference"].ainvoke({"entry_json": ENTRY.model_dump_json()})  # type: ignore[union-attr]
    assert [record.tool for record in usage.tool_calls] == ["lookup_reference"]
