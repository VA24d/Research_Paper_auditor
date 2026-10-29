"""Offline tests for reference resolution (P2.2): canned API replies, no network."""

from pathlib import Path

import httpx

from auditor.config import Settings
from auditor.infra.http import HttpClient
from auditor.retrieval.resolve import (
    first_surname,
    match_marker_to_bibliography,
    parse_marker,
    resolve_entry,
    resolve_marker,
    title_tokens,
    titles_match,
    venues_match,
)
from auditor.schemas import BibEntry

CROSSREF_HIT = {
    "message": {
        "DOI": "10.18653/V1/D16-1264",
        "title": ["SQuAD: 100,000+ Questions for Machine Comprehension of Text"],
        "container-title": ["EMNLP"],
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


def test_venues_match_normalises_proceedings_names() -> None:
    assert venues_match("NeurIPS", "Advances in Neural Information Processing Systems")
    assert venues_match("NIPS", "NeurIPS")
    assert venues_match("EMNLP", "Conference on Empirical Methods in Natural Language Processing")
    assert venues_match("CVPR", "IEEE/CVF Conference on Computer Vision and Pattern Recognition")
    assert venues_match("JMLR", "Journal of Machine Learning Research")
    # Crossref writes the year in front and the short name in brackets behind.
    assert venues_match(
        "CVPR", "2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)"
    )
    assert venues_match("AAAI", "Proceedings of the AAAI Conference on Artificial Intelligence")
    # An arXiv preprint record's venue says nothing about the publication venue.
    assert venues_match("ICML", "arXiv (Cornell University)")
    assert venues_match("NeurIPS", "arXiv")
    assert venues_match(
        "empirical methods natural language processing",
        "conference on empirical methods in natural language processing",
    )
    assert venues_match(None, "NeurIPS") is False
    assert venues_match("NeurIPS", None) is False
    # Different venues never match, however similar the wording.
    assert not venues_match("ICML", "International Conference on Learning Representations")
    assert not venues_match("NeurIPS", "ICLR")
    assert not venues_match("ACL", "EMNLP")


def test_titles_match_ignores_case_punctuation_and_stopwords() -> None:
    assert titles_match("Attention Is All You Need", "attention is all you need!")
    assert titles_match("A Study of Transformers", "Study Transformers")
    assert not titles_match("Attention Is All You Need", "BERT: Pre-training of Transformers")
    assert not titles_match("Attention", None)
    assert not titles_match(None, None)


def test_title_tokens_drop_stopwords_and_single_letters() -> None:
    assert title_tokens("A Study of the Transformer") == frozenset({"study", "transformer"})


def test_first_surname_handles_both_name_orders() -> None:
    assert first_surname("Vaswani, Ashish") == "vaswani"
    assert first_surname("Ashish Vaswani") == "vaswani"
    assert first_surname("  ") == ""


def test_titles_match_accepts_a_main_title_the_index_shortened() -> None:
    # An index record sometimes keeps only the part before the colon.
    assert titles_match(
        "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis", "NeRF"
    )
    # A different work is still refused.
    assert not titles_match(
        "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
        "BERT: Pre-training of Deep Bidirectional Transformers",
    )


def test_venues_match_ignores_repository_copies() -> None:
    # OpenAlex sometimes holds an institutional repository's record instead of the
    # proceedings one; its venue says nothing about where the work was published.
    assert venues_match("NeurIPS", "UCL Discovery (University College London)")
    assert venues_match("ICML", "eScholarship")


def test_parse_marker_pulls_surnames_and_year() -> None:
    parsed = parse_marker("(Smith et al., 2020; Jones and Lee, 2019a)")
    assert parsed.surnames == ("smith", "jones", "lee")
    assert parsed.year == 2020
    numeric = parse_marker("[7]")
    assert numeric.surnames == ()
    assert numeric.year is None


def test_match_marker_to_bibliography_prefers_the_matching_year() -> None:
    entries = (
        BibEntry(id="b0", raw="Smith 2018", authors=("Adam Smith",), year=2018),
        BibEntry(id="b1", raw="Smith 2020", authors=("Jane Smith",), year=2020),
    )
    matched = match_marker_to_bibliography(entries, "(Smith et al., 2020)")
    assert matched is not None
    assert matched.id == "b1"
    # A year that matches no entry within one year rejects the marker.
    assert match_marker_to_bibliography(entries, "(Smith et al., 2001)") is None


async def test_resolve_entry_tries_the_doi_first(tmp_path: Path) -> None:
    routes = {"api.crossref.org": httpx.Response(200, json=CROSSREF_HIT)}
    entry = BibEntry(id="b1", title="SQuAD", doi="10.18653/v1/D16-1264")
    resolution = await resolve_entry(api_client(tmp_path, routes), entry)
    assert resolution.found
    assert resolution.work is not None
    assert resolution.work.source == "crossref"
    assert resolution.queries == (("crossref doi:10.18653/v1/d16-1264", "found"),)


async def test_resolve_entry_falls_back_to_openalex_on_a_doi_miss(tmp_path: Path) -> None:
    routes = {"api.openalex.org": httpx.Response(200, json=OPENALEX_HITS)}
    entry = BibEntry(id="b1", title="SQuAD", doi="10.18653/v1/D16-1264")
    resolution = await resolve_entry(api_client(tmp_path, routes), entry)
    assert resolution.found
    assert resolution.work is not None
    assert resolution.work.source == "openalex"
    assert resolution.queries[0] == ("crossref doi:10.18653/v1/d16-1264", "not found")
    assert resolution.queries[1] == ("openalex doi:10.18653/v1/d16-1264", "found")


async def test_resolve_entry_uses_the_arxiv_id_when_the_doi_is_unknown(tmp_path: Path) -> None:
    feed = (
        '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
        "<entry><id>http://arxiv.org/abs/1706.03762</id>"
        "<title>Attention Is All You Need</title></entry></feed>"
    )
    routes = {"export.arxiv.org": httpx.Response(200, text=feed)}
    entry = BibEntry(id="b0", title="Attention Is All You Need", arxiv_id="1706.03762")
    resolution = await resolve_entry(api_client(tmp_path, routes), entry)
    assert resolution.found
    assert resolution.work is not None
    assert resolution.work.arxiv_id == "1706.03762"


async def test_resolve_entry_searches_by_title(tmp_path: Path) -> None:
    routes = {"api.openalex.org": httpx.Response(200, json=OPENALEX_HITS)}
    entry = BibEntry(
        id="b1",
        title="SQuAD: 100,000+ Questions for Machine Comprehension of Text",
        authors=("Pranav Rajpurkar",),
        year=2016,
    )
    resolution = await resolve_entry(api_client(tmp_path, routes), entry)
    assert resolution.found
    assert resolution.work is not None
    assert resolution.work.id == "https://openalex.org/W123"
    assert resolution.queries[0] == ("openalex title search", "1 hits")


async def test_resolve_entry_stops_at_the_first_index_with_a_match(tmp_path: Path) -> None:
    """The remaining indexes are not queried once one has found a candidate."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "api.openalex.org" in str(request.url):
            return httpx.Response(200, json=OPENALEX_HITS)
        raise AssertionError(f"unexpected request to {request.url}")

    settings = Settings(cache_dir=tmp_path / "cache")
    client = HttpClient(settings, transport=httpx.MockTransport(handler), sleep=_no_sleep)
    entry = BibEntry(
        id="b1",
        title="SQuAD: 100,000+ Questions for Machine Comprehension of Text",
        authors=("Pranav Rajpurkar",),
        year=2016,
    )
    resolution = await resolve_entry(client, entry)
    assert resolution.found
    assert resolution.queries == (("openalex title search", "1 hits"),)


async def test_resolve_entry_skips_a_dead_index(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "api.openalex.org" in str(request.url):
            raise httpx.ConnectError("connection refused", request=request)
        if "api.crossref.org" in str(request.url):
            return httpx.Response(200, json={"message": {"items": [CROSSREF_HIT["message"]]}})
        return httpx.Response(404)

    settings = Settings(cache_dir=tmp_path / "cache")
    client = HttpClient(settings, transport=httpx.MockTransport(handler), sleep=_no_sleep)
    entry = BibEntry(
        id="b1",
        title="SQuAD: 100,000+ Questions for Machine Comprehension of Text",
        authors=("Pranav Rajpurkar",),
        year=2016,
    )
    resolution = await resolve_entry(client, entry)
    assert resolution.found
    assert resolution.queries[0] == ("openalex title search", "failed (HttpRequestError)")
    assert resolution.queries[1] == ("crossref title search", "1 hits")


async def test_resolve_entry_routes_an_arxiv_doi_to_arxiv(tmp_path: Path) -> None:
    """arXiv's DOIs live in DataCite, so Crossref and OpenAlex do not know them; the
    entry must not be called fabricated."""
    feed = (
        '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
        "<entry><id>http://arxiv.org/abs/2602.12204v1</id>"
        "<title>Online Elicitation of Necessarily Optimal Matchings</title></entry></feed>"
    )
    routes = {
        "export.arxiv.org": httpx.Response(200, text=feed),
        "api.crossref.org": httpx.Response(404),
        "api.openalex.org": httpx.Response(404),
    }
    entry = BibEntry(
        id="b2",
        title="Online Elicitation of Necessarily Optimal Matchings",
        doi="10.48550/arXiv.2602.12204v1",
    )
    resolution = await resolve_entry(api_client(tmp_path, routes), entry)
    assert resolution.found
    assert resolution.work is not None
    assert resolution.work.arxiv_id == "2602.12204v1"
    assert not resolution.doi_unresolved


async def test_resolve_entry_reports_why_nothing_was_found(tmp_path: Path) -> None:
    entry = BibEntry(id="b9", title="No Such Work Anywhere At All")
    resolution = await resolve_entry(api_client(tmp_path, {}), entry)
    assert not resolution.found
    assert resolution.reason == "no work with this title was found in the scholarly indexes"
    assert len(resolution.queries) == 4  # openalex, crossref, arxiv and s2 were all tried


async def test_resolve_entry_without_any_identifier(tmp_path: Path) -> None:
    resolution = await resolve_entry(api_client(tmp_path, {}), BibEntry(id="b9"))
    assert not resolution.found
    assert resolution.reason == "the entry has no DOI, arXiv id or title to look up"


async def test_resolve_marker_matches_the_bibliography_first(tmp_path: Path) -> None:
    routes = {"api.crossref.org": httpx.Response(200, json=CROSSREF_HIT)}
    entries = (
        BibEntry(id="b1", title="SQuAD", doi="10.18653/v1/D16-1264", authors=("Pranav Rajpurkar",)),
    )
    resolution = await resolve_marker(
        api_client(tmp_path, routes), "(Rajpurkar et al., 2016)", entries
    )
    assert resolution.found

    # A numeric marker cannot be resolved this way, and says so.
    resolution = await resolve_marker(api_client(tmp_path, routes), "[7]", entries)
    assert not resolution.found
    assert resolution.reason is not None
    assert "author names" in resolution.reason

    # An author-year marker with no matching entry says so too.
    resolution = await resolve_marker(
        api_client(tmp_path, routes), "(Nobody et al., 2020)", entries
    )
    assert not resolution.found
    assert resolution.reason is not None
    assert "no bibliography entry matches" in resolution.reason


def test_match_marker_to_bibliography_falls_back_to_the_raw_text() -> None:
    entries = (BibEntry(id="b0", raw="J. Smith and K. Lee 2019"),)
    assert match_marker_to_bibliography(entries, "Smith and Lee 2019") is not None
    assert match_marker_to_bibliography(entries, "[7]") is None
