"""Offline tests for the raw citation lookup (P2.1/P2.2): canned API replies, no network."""

from pathlib import Path

import httpx

from auditor.agents.citation.lookup import lookup_citation
from auditor.config import Settings
from auditor.infra.http import HttpClient

CROSSREF_HIT = {
    "message": {
        "DOI": "10.18653/V1/D16-1264",
        "title": ["SQuAD: 100,000+ Questions for Machine Comprehension of Text"],
        "author": [{"given": "Pranav", "family": "Rajpurkar"}],
        "issued": {"date-parts": [[2016]]},
    }
}
OPENALEX_HITS = {
    "results": [
        {
            "id": "https://openalex.org/W123",
            "title": "SQuAD: 100,000+ Questions for Machine Comprehension of Text",
            "publication_year": 2016,
            "authorships": [{"author": {"display_name": "Pranav Rajpurkar"}}],
        }
    ]
}
ARXIV_FEED = (
    '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
    "<entry><id>http://arxiv.org/abs/1706.03762</id>"
    "<title>Attention Is All You Need</title>"
    "<author><name>Ashish Vaswani</name></author>"
    "<published>2017-06-12T00:00:00Z</published></entry></feed>"
)


async def _no_sleep(_seconds: float) -> None:
    return None


def api_client(tmp_path: Path, routes: dict[str, httpx.Response]) -> HttpClient:
    def handler(request: httpx.Request) -> httpx.Response:
        for pattern, response in routes.items():
            if pattern in str(request.url):
                return response
        return httpx.Response(404)

    settings = Settings(cache_dir=tmp_path / "cache")
    return HttpClient(settings, transport=httpx.MockTransport(handler), sleep=_no_sleep)


async def test_a_doi_in_the_citation_wins(tmp_path: Path) -> None:
    routes = {"api.crossref.org": httpx.Response(200, json=CROSSREF_HIT)}
    result = await lookup_citation(
        api_client(tmp_path, routes), "SQuAD paper. doi:10.18653/v1/D16-1264"
    )
    assert result.found
    assert result.work is not None
    assert result.work.source == "crossref"
    assert result.queries[0] == "crossref doi:10.18653/v1/d16-1264"


async def test_a_doi_miss_falls_back_to_openalex(tmp_path: Path) -> None:
    routes = {"api.openalex.org": httpx.Response(200, json=OPENALEX_HITS)}
    result = await lookup_citation(
        api_client(tmp_path, routes), "SQuAD. https://doi.org/10.18653/v1/D16-1264"
    )
    assert result.found
    assert result.work is not None
    assert result.work.source == "openalex"


async def test_an_arxiv_url_is_looked_up_by_id(tmp_path: Path) -> None:
    routes = {"export.arxiv.org": httpx.Response(200, text=ARXIV_FEED)}
    result = await lookup_citation(
        api_client(tmp_path, routes), "Vaswani et al., https://arxiv.org/abs/1706.03762"
    )
    assert result.found
    assert result.work is not None
    assert result.work.arxiv_id == "1706.03762"


async def test_a_title_search_tries_the_indexes_in_order(tmp_path: Path) -> None:
    routes = {"api.openalex.org": httpx.Response(200, json=OPENALEX_HITS)}
    result = await lookup_citation(
        api_client(tmp_path, routes), "SQuAD: 100,000+ Questions for Machine Comprehension"
    )
    assert result.found
    assert result.work is not None
    assert result.work.source == "openalex"
    assert result.queries == ("openalex title search (1 hits)",)


async def test_nothing_found_says_why(tmp_path: Path) -> None:
    result = await lookup_citation(api_client(tmp_path, {}), "A Title Nobody Wrote At All Ever")
    assert not result.found
    assert result.reason == "no work with this title was found in the scholarly indexes"
    assert len(result.queries) == 4


async def test_a_citation_with_nothing_to_look_up(tmp_path: Path) -> None:
    # Only a year: the title part is empty, so there is nothing to search for.
    result = await lookup_citation(api_client(tmp_path, {}), "2019.")
    assert not result.found
    assert result.reason == "the citation gives no DOI, arXiv id or title to look up"
    assert result.queries == ()


async def test_an_arxiv_doi_is_looked_up_on_arxiv(tmp_path: Path) -> None:
    """arXiv's DOIs live in DataCite, so Crossref and OpenAlex do not know them."""
    routes = {
        "api.crossref.org": httpx.Response(404),
        "api.openalex.org": httpx.Response(404),
        "export.arxiv.org": httpx.Response(200, text=ARXIV_FEED),
    }
    result = await lookup_citation(
        api_client(tmp_path, routes), "Attention Is All You Need. doi:10.48550/arXiv.1706.03762"
    )
    assert result.found
    assert result.work is not None
    assert result.work.arxiv_id == "1706.03762"
    assert "arxiv id from doi:10.48550/arxiv.1706.03762" in result.queries


async def test_a_mojibake_title_is_folded_before_the_search(tmp_path: Path) -> None:
    """A title with a mojibake character finds nothing as written; folded it does."""
    routes = {
        "api.openalex.org": httpx.Response(200, json=OPENALEX_HITS),
    }
    result = await lookup_citation(
        api_client(tmp_path, routes), "SQuAD: 100,000+ Que\u0398stions for Machine Comprehension"
    )
    assert result.found
    assert result.work is not None
    assert result.work.source == "openalex"


async def test_a_long_title_gets_a_shortened_retry(tmp_path: Path) -> None:
    routes = {
        "api.openalex.org": httpx.Response(200, json=OPENALEX_HITS),
    }
    long_citation = (
        "A Very Long Made-Up Title With Many Words Before The Real SQuAD Question Answering "
        "Dataset Finally Appears"
    )
    result = await lookup_citation(api_client(tmp_path, routes), long_citation)
    assert not result.found  # the shortened retry does not match a different work
    assert any(q.startswith("shortened title retry") for q in result.queries)


async def test_a_dead_index_is_skipped(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "api.openalex.org" in str(request.url):
            raise httpx.ConnectError("connection refused", request=request)
        if "api.crossref.org" in str(request.url):
            return httpx.Response(200, json={"message": {"items": [CROSSREF_HIT["message"]]}})
        return httpx.Response(404)

    settings = Settings(cache_dir=tmp_path / "cache")
    client = HttpClient(settings, transport=httpx.MockTransport(handler), sleep=_no_sleep)
    result = await lookup_citation(client, "SQuAD: 100,000+ Questions for Machine Comprehension")
    assert result.found
    assert result.queries[0] == "openalex title search failed (HttpRequestError)"
    assert result.queries[1] == "crossref title search (1 hits)"
