"""Offline tests for the scholarly API clients (P2.1): canned API replies, no network."""

from pathlib import Path

import httpx

from auditor.config import Settings
from auditor.infra.http import HttpClient
from auditor.retrieval.clients import (
    arxiv_by_id,
    arxiv_id_from_doi,
    arxiv_search,
    ascii_fold,
    crossref_by_doi,
    crossref_search,
    is_arxiv_id,
    is_doi,
    normalize_doi,
    openalex_by_doi,
    openalex_search,
    s2_by_identifier,
    s2_search,
    short_title,
)

ARXIV_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v2</id>
    <title>  Attention Is
    All You Need </title>
    <summary>  The dominant sequence
    transfer models.  </summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <published>2017-06-12T17:57:34Z</published>
    <link href="https://arxiv.org/pdf/1706.03762v2" rel="related" type="application/pdf"/>
  </entry>
</feed>
"""

CROSSREF_WORK = {
    "message": {
        "DOI": "10.18653/V1/D16-1264",
        "title": ["SQuAD: 100,000+ Questions for Machine Comprehension of Text"],
        "container-title": ["EMNLP"],
        "author": [
            {"given": "Pranav", "family": "Rajpurkar"},
            {"given": "Jian", "family": "Zhang"},
        ],
        "issued": {"date-parts": [[2016, 11]]},
        "abstract": "<jats:p>We present a reading comprehension dataset.</jats:p>",
    }
}

OPENALEX_WORK = {
    "results": [
        {
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.18653/v1/D16-1264",
            "title": "SQuAD: 100,000+ Questions for Machine Comprehension of Text",
            "publication_year": 2016,
            "authorships": [{"author": {"display_name": "Pranav Rajpurkar"}}],
            "primary_location": {
                "source": {"display_name": "EMNLP"},
                "pdf_url": "https://example.org/squad.pdf",
                "landing_page_url": "https://example.org/squad",
            },
            "abstract_inverted_index": {"reading": [0], "comprehension": [1], "dataset": [2]},
        }
    ]
}

S2_WORK = {
    "paperId": "abc123",
    "title": "SQuAD: 100,000+ Questions for Machine Comprehension of Text",
    "authors": [{"name": "Pranav Rajpurkar"}, {"name": "Jian Zhang"}],
    "year": 2016,
    "venue": "EMNLP",
    "externalIds": {"DOI": "10.18653/v1/D16-1264", "ArXiv": "1606.05250"},
    "openAccessPdf": {"url": "https://example.org/squad.pdf"},
    "abstract": "We present a reading comprehension dataset.",
}

CROSSREF_SEARCH = {"message": {"items": [CROSSREF_WORK["message"]]}}
S2_SEARCH = {"data": [S2_WORK]}


async def _no_sleep(_seconds: float) -> None:
    return None


def api_client(
    tmp_path: Path, routes: dict[str, httpx.Response], requested: list[str]
) -> HttpClient:
    """A client answering the first route whose pattern appears in the URL; anything else
    is a 404. Every URL requested is recorded."""

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        for pattern, response in routes.items():
            if pattern in str(request.url):
                return response
        return httpx.Response(404)

    settings = Settings(cache_dir=tmp_path / "cache")
    return HttpClient(
        settings, transport=httpx.MockTransport(handler), sleep=_no_sleep, min_intervals={}
    )


def test_normalize_doi_strips_prefixes_and_case() -> None:
    assert normalize_doi(" 10.18653/V1/D16-1264 ") == "10.18653/v1/d16-1264"
    assert normalize_doi("https://doi.org/10.1234/x") == "10.1234/x"
    assert normalize_doi("https://dx.doi.org/10.1234/x") == "10.1234/x"
    assert normalize_doi("doi:10.1234/x") == "10.1234/x"


def test_is_doi_and_is_arxiv_id() -> None:
    assert is_doi("10.18653/v1/D16-1264")
    assert is_doi("doi:10.1234/x")
    assert not is_doi("1706.03762")
    assert not is_doi("not a doi")
    assert is_arxiv_id("1706.03762")
    assert is_arxiv_id("1706.03762v2")
    assert is_arxiv_id(" 2401.09543 ")
    assert not is_arxiv_id("10.18653/v1/D16-1264")


async def test_arxiv_by_id_parses_the_feed(tmp_path: Path) -> None:
    requested: list[str] = []
    routes = {"export.arxiv.org": httpx.Response(200, text=ARXIV_FEED)}
    work = await arxiv_by_id(api_client(tmp_path, routes, requested), "1706.03762v2")
    assert work is not None
    assert work.id == "arxiv:1706.03762v2"
    assert work.source == "arxiv"
    assert work.title == "Attention Is All You Need"
    assert work.authors == ("Ashish Vaswani", "Noam Shazeer")
    assert work.year == 2017
    assert work.abstract == "The dominant sequence transfer models."
    assert work.pdf_url == "https://arxiv.org/pdf/1706.03762v2"
    assert "id_list=1706.03762v2" in requested[0]


async def test_arxiv_by_id_returns_none_for_an_empty_or_bad_feed(tmp_path: Path) -> None:
    empty = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'
    http = api_client(tmp_path, {"export.arxiv.org": httpx.Response(200, text=empty)}, [])
    assert await arxiv_by_id(http, "9999.99999") is None
    http = api_client(tmp_path, {"export.arxiv.org": httpx.Response(200, text="<feed>")}, [])
    assert await arxiv_by_id(http, "1706.03762") is None


async def test_arxiv_search_returns_every_entry(tmp_path: Path) -> None:
    routes = {"export.arxiv.org": httpx.Response(200, text=ARXIV_FEED)}
    works = await arxiv_search(api_client(tmp_path, routes, []), "attention")
    assert [w.arxiv_id for w in works] == ["1706.03762v2"]


async def test_openalex_by_doi_rebuilds_the_abstract(tmp_path: Path) -> None:
    requested: list[str] = []
    routes = {"api.openalex.org": httpx.Response(200, json=OPENALEX_WORK)}
    work = await openalex_by_doi(api_client(tmp_path, routes, requested), "10.18653/v1/D16-1264")
    assert work is not None
    assert work.id == "https://openalex.org/W123"
    assert work.source == "openalex"
    assert work.doi == "10.18653/v1/d16-1264"
    assert work.authors == ("Pranav Rajpurkar",)
    assert work.venue == "EMNLP"
    assert work.abstract == "reading comprehension dataset"
    assert work.pdf_url == "https://example.org/squad.pdf"
    assert work.html_url == "https://example.org/squad"
    assert httpx.URL(requested[0]).params["filter"] == "doi:10.18653/v1/d16-1264"


async def test_openalex_search(tmp_path: Path) -> None:
    routes = {"api.openalex.org": httpx.Response(200, json=OPENALEX_WORK)}
    works = await openalex_search(api_client(tmp_path, routes, []), "SQuAD questions")
    assert [w.title for w in works] == [OPENALEX_WORK["results"][0]["title"]]


async def test_crossref_by_doi_strips_jats_tags(tmp_path: Path) -> None:
    requested: list[str] = []
    routes = {"api.crossref.org": httpx.Response(200, json=CROSSREF_WORK)}
    work = await crossref_by_doi(api_client(tmp_path, routes, requested), "10.18653/v1/D16-1264")
    assert work is not None
    assert work.id == "doi:10.18653/v1/d16-1264"
    assert work.source == "crossref"
    assert work.title == "SQuAD: 100,000+ Questions for Machine Comprehension of Text"
    assert work.authors == ("Pranav Rajpurkar", "Jian Zhang")
    assert work.year == 2016
    assert work.venue == "EMNLP"
    assert work.abstract == "We present a reading comprehension dataset."
    assert requested[0].endswith("/works/10.18653/v1/d16-1264")


async def test_crossref_search(tmp_path: Path) -> None:
    routes = {"api.crossref.org": httpx.Response(200, json=CROSSREF_SEARCH)}
    works = await crossref_search(api_client(tmp_path, routes, []), "SQuAD questions")
    assert [w.doi for w in works] == ["10.18653/v1/d16-1264"]


async def test_s2_by_identifier_accepts_doi_and_arxiv_id(tmp_path: Path) -> None:
    requested: list[str] = []
    routes = {"api.semanticscholar.org": httpx.Response(200, json=S2_WORK)}
    http = api_client(tmp_path, routes, requested)
    work = await s2_by_identifier(http, doi="10.18653/v1/D16-1264")
    assert work is not None
    assert work.id == "s2:abc123"
    assert work.source == "s2"
    assert work.doi == "10.18653/v1/d16-1264"
    assert work.arxiv_id == "1606.05250"
    assert work.year == 2016
    assert work.venue == "EMNLP"
    assert work.pdf_url == "https://example.org/squad.pdf"
    assert "DOI:10.18653/v1/d16-1264" in requested[0]
    work = await s2_by_identifier(http, arxiv_id="1606.05250")
    assert work is not None
    assert "arXiv:1606.05250" in requested[1]
    assert await s2_by_identifier(http) is None


async def test_s2_search(tmp_path: Path) -> None:
    routes = {"api.semanticscholar.org": httpx.Response(200, json=S2_SEARCH)}
    works = await s2_search(api_client(tmp_path, routes, []), "SQuAD questions")
    assert [w.id for w in works] == ["s2:abc123"]


async def test_http_errors_give_none_or_empty(tmp_path: Path) -> None:
    http = api_client(tmp_path, {}, [])
    assert await arxiv_by_id(http, "1706.03762") is None
    assert await arxiv_search(http, "attention") == []
    assert await openalex_by_doi(http, "10.1234/x") is None
    assert await openalex_search(http, "title") == []
    assert await crossref_by_doi(http, "10.1234/x") is None
    assert await crossref_search(http, "title") == []
    assert await s2_by_identifier(http, doi="10.1234/x") is None
    assert await s2_search(http, "title") == []

    assert not is_arxiv_id("1706.037")


def test_arxiv_id_from_doi_extracts_the_arxiv_id() -> None:
    # arXiv's DOIs live in DataCite, not Crossref/OpenAlex; the id behind them is usable.
    assert arxiv_id_from_doi("10.48550/arXiv.2602.12204v1") == "2602.12204"
    assert arxiv_id_from_doi("10.48550/arxiv.1706.03762") == "1706.03762"
    assert arxiv_id_from_doi("10.18653/v1/D16-1264") is None
    assert arxiv_id_from_doi("https://doi.org/10.48550/arXiv.1706.03762") == "1706.03762"


def test_ascii_fold_and_short_title_make_searchable_queries() -> None:
    # Accents decompose; typography and letters an English query would not carry drop.
    assert ascii_fold("\u00c9tude sur l'apprentissage") == "Etude sur l'apprentissage"
    assert ascii_fold("Na\u00efve \u201cbayes\u201d \u2014 a survey") == 'Naive "bayes" - a survey'
    assert ascii_fold("Demoir\u0398ing") == "Demoiring"  # a mojibake theta drops out
    assert short_title(
        "Real-Time Image Demoir\u0398ing on Mobile Devices with Deep Features", words=5
    ) == ("Real Time Image Demoiring on")
    assert short_title("One Two Three Four Five Six Seven Eight Nine Ten", words=8) == (
        "One Two Three Four Five Six Seven Eight"
    )
