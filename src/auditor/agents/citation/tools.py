"""Citation agent tools (decision P2.6): what baseline B1 and the agent share.

B1 gets the same capabilities the citation agent uses: resolve a reference to a work,
retrieve the work's text at the best level available, and find the passages of that text
relevant to a claim. Each shared function is counted as a tool call (decision D8).
"""

import json

from langchain_core.tools import BaseTool

from auditor.infra.context import RunContext
from auditor.infra.tools import tracked_tool
from auditor.retrieval.embeddings import get_embedder
from auditor.retrieval.fulltext import SourceLevel, SourceText, grobid_for, retrieve_source
from auditor.retrieval.passages import rank_passages
from auditor.retrieval.resolve import Resolution, resolve_entry
from auditor.schemas import BibEntry

# A tool's answer is capped, so B1's prompt does not fill up with one work's text.
MAX_TOOL_CHARS = 6000


def resolution_record(resolution: Resolution) -> dict[str, object]:
    """The lookup part of a tool's answer, as a small JSON-ready dict."""
    work = resolution.work
    if work is None:
        return {"found": False, "reason": resolution.reason}
    out = {
        "found": True,
        "id": work.id,
        "source": work.source,
        "title": work.title,
        "authors": list(work.authors),
        "year": work.year,
        "venue": work.venue,
        "doi": work.doi,
        "arxiv_id": work.arxiv_id,
        "queries": [q for q, _ in resolution.queries],
    }
    if work.pdf_url:
        out["pdf_url"] = work.pdf_url
    if work.html_url:
        out["html_url"] = work.html_url
    return out


def _clip(text: str, limit: int = MAX_TOOL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n...(clipped at {limit} of {len(text)} characters)"


@tracked_tool()
async def lookup_reference(ctx: RunContext, entry: BibEntry) -> str:
    """The work a bibliography entry most plausibly cites, or why none was found."""
    resolution = await resolve_entry(ctx.http, entry)
    return json.dumps(resolution_record(resolution), indent=1)


def _level_line(source: SourceText) -> str:
    """What the text is, so the agent knows how much a missing value means."""
    if source.partial:
        return (
            f"source level: {source.level.value} (partial: a publisher page that may hold "
            "only the abstract; a value missing here may still be in the paper)"
        )
    return f"source level: {source.level.value}"


@tracked_tool()
async def retrieve_cited_text(
    ctx: RunContext, entry: BibEntry, abstract_only: bool | None = None
) -> str:
    """The text of the work an entry cites, at the best level available."""
    resolution = await resolve_entry(ctx.http, entry)
    if not resolution.found or resolution.work is None:
        return f"Could not resolve the reference: {resolution.reason}"
    source = await retrieve_source(
        ctx.http,
        resolution.work,
        ctx.settings,
        grobid=grobid_for(ctx.settings, ctx.http),
        abstract_only=(
            ctx.settings.citation_abstract_only if abstract_only is None else abstract_only
        ),
    )
    if source.level is SourceLevel.NONE or not source.text:
        return f"No text of the cited work was available ({source.reason})."
    return _clip(f"{_level_line(source)}\n\n{source.text}")


@tracked_tool()
async def find_supporting_passages(
    ctx: RunContext, entry: BibEntry, claim: str, abstract_only: bool | None = None
) -> str:
    """The passages of a cited work's text most relevant to a claim."""
    resolution = await resolve_entry(ctx.http, entry)
    if not resolution.found or resolution.work is None:
        return f"Could not resolve the reference: {resolution.reason}"
    source = await retrieve_source(
        ctx.http,
        resolution.work,
        ctx.settings,
        grobid=grobid_for(ctx.settings, ctx.http),
        abstract_only=(
            ctx.settings.citation_abstract_only if abstract_only is None else abstract_only
        ),
    )
    if source.level is SourceLevel.NONE or not source.text:
        return f"No text of the cited work was available ({source.reason})."
    passages = rank_passages(
        source.text, claim, embedder=await get_embedder(ctx.settings, ctx.http)
    )
    if not passages:
        return "The retrieved text has no passages to judge the claim against."
    block = "\n\n".join(
        f"[{number}] {passage.text}" for number, passage in enumerate(passages, start=1)
    )
    return f"{_level_line(source)}\n\n{block}"


def make_tools(ctx: RunContext) -> list[BaseTool]:
    """The citation agent's tools for B1, in structured-output wrappers."""
    from langchain_core.tools import StructuredTool

    async def lookup(entry_json: str) -> str:
        return await lookup_reference(ctx, BibEntry.model_validate_json(entry_json))

    async def retrieve(entry_json: str, abstract_only: bool | None = None) -> str:
        return await retrieve_cited_text(
            ctx, BibEntry.model_validate_json(entry_json), abstract_only=abstract_only
        )

    async def passages(entry_json: str, claim: str, abstract_only: bool | None = None) -> str:
        return await find_supporting_passages(
            ctx, BibEntry.model_validate_json(entry_json), claim, abstract_only=abstract_only
        )

    return [
        StructuredTool.from_function(
            coroutine=lookup,
            name="lookup_reference",
            description=(
                "Look up one bibliography entry (as JSON with id, raw, title, authors, year, "
                "venue, doi, arxiv_id, url) in the scholarly indexes. Returns the work's "
                "metadata, or why nothing was found."
            ),
        ),
        StructuredTool.from_function(
            coroutine=retrieve,
            name="retrieve_cited_text",
            description=(
                "Download the text of a cited work (best level available: arXiv HTML, PDF, "
                "or the abstract). Takes the bibliography entry as JSON."
            ),
        ),
        StructuredTool.from_function(
            coroutine=passages,
            name="find_supporting_passages",
            description=(
                "The passages of a cited work's text most relevant to a claim, with the "
                "level of the text they came from. Takes the bibliography entry as JSON "
                "and the claim."
            ),
        ),
    ]
