"""Offline tests for full-text retrieval (P2.3): canned pages, no network."""

from pathlib import Path

import httpx

from auditor.config import Settings
from auditor.infra.http import HttpClient
from auditor.retrieval.clients import Work
from auditor.retrieval.fulltext import SourceLevel, _store, html_to_text, retrieve_source

HTML_PAGE = (
    "<html><head><style>body { color: red }</style></head><body>"
    "<h1>Attention Is All You&nbsp;Need</h1>"
    "<p>The  Transformer,  the first sequence model based solely on attention.</p>"
    "<script>var x = 1;</script>"
    "<p>It reaches 28.4 BLEU on WMT 2014.</p>"
    # Enough body text that the page is not treated as a stub (the stub check is 2000).
    + ("<p>Self-attention computes a weighted sum of value vectors.</p>" * 40)
    + "</body></html>"
)


async def _no_sleep(_seconds: float) -> None:
    return None


def refusing_client(tmp_path: Path) -> HttpClient:
    """A client that fails the test if anything tries to use the network."""

    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request to {request.url}")

    settings = Settings(cache_dir=tmp_path / "cache", scratch_dir=tmp_path / "scratch")
    return HttpClient(settings, transport=httpx.MockTransport(refuse), sleep=_no_sleep)


def page_client(tmp_path: Path, routes: dict[str, httpx.Response]) -> HttpClient:
    def handler(request: httpx.Request) -> httpx.Response:
        for pattern, response in routes.items():
            if pattern in str(request.url):
                return response
        return httpx.Response(404)

    settings = Settings(cache_dir=tmp_path / "cache", scratch_dir=tmp_path / "scratch")
    return HttpClient(settings, transport=httpx.MockTransport(handler), sleep=_no_sleep)


def work(**overrides: object) -> Work:
    fields: dict[str, object] = {
        "id": "arxiv:1706.03762",
        "source": "arxiv",
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "abstract": "The dominant sequence transfer models are based on recurrence.",
    }
    fields.update(overrides)
    return Work(**fields)  # type: ignore[arg-type]


def test_html_to_text_drops_scripts_and_keeps_line_structure() -> None:
    text = html_to_text(HTML_PAGE)
    lines = text.splitlines()
    assert lines[:3] == [
        "Attention Is All You Need",
        "The Transformer, the first sequence model based solely on attention.",
        "It reaches 28.4 BLEU on WMT 2014.",
    ]
    # The style's and script's contents are dropped, and the filler paragraphs follow.
    assert "color: red" not in text
    assert "var x = 1" not in text
    assert lines[3:] == ["Self-attention computes a weighted sum of value vectors."] * 40


async def test_full_html_is_retrieved_and_cached(tmp_path: Path) -> None:
    routes = {"arxiv.org/html/1706.03762": httpx.Response(200, text=HTML_PAGE)}
    source = await retrieve_source(
        page_client(tmp_path, routes),
        work(),
        Settings(cache_dir=tmp_path / "cache", scratch_dir=tmp_path / "scratch"),
    )
    assert source.level is SourceLevel.FULL_HTML
    assert "28.4 BLEU" in source.text
    assert source.reason is None
    # A second run reads the same text from the disk cache, without any request.
    again = await retrieve_source(
        refusing_client(tmp_path),
        work(),
        Settings(cache_dir=tmp_path / "cache", scratch_dir=tmp_path / "scratch"),
    )
    assert again == source


async def test_stub_html_pages_fall_back_to_the_abstract(tmp_path: Path) -> None:
    routes = {"arxiv.org/html/1706.03762": httpx.Response(200, text="<p>no html</p>")}
    settings = Settings(cache_dir=tmp_path / "cache", scratch_dir=tmp_path / "scratch")
    source = await retrieve_source(page_client(tmp_path, routes), work(), settings)
    assert source.level is SourceLevel.ABSTRACT
    assert source.text.startswith("The dominant sequence")
    assert source.reason == "no full text was available"


async def test_abstract_only_never_fetches_the_full_text(tmp_path: Path) -> None:
    settings = Settings(cache_dir=tmp_path / "cache", scratch_dir=tmp_path / "scratch")
    source = await retrieve_source(refusing_client(tmp_path), work(), settings, abstract_only=True)
    assert source.level is SourceLevel.ABSTRACT
    assert source.reason is None


async def test_no_text_at_all_is_recorded_as_none(tmp_path: Path) -> None:
    settings = Settings(cache_dir=tmp_path / "cache", scratch_dir=tmp_path / "scratch")
    source = await retrieve_source(
        refusing_client(tmp_path),
        work(arxiv_id=None, abstract=None),
        settings,
    )
    assert source.level is SourceLevel.NONE
    assert source.text == ""
    assert source.reason == "the work has no open full text and no abstract"


async def test_a_pdf_without_grobid_falls_back_to_the_abstract(tmp_path: Path) -> None:
    settings = Settings(cache_dir=tmp_path / "cache", scratch_dir=tmp_path / "scratch")
    source = await retrieve_source(
        refusing_client(tmp_path),
        work(arxiv_id=None, pdf_url="https://arxiv.org/pdf/1706.03762"),
        settings,
    )
    assert source.level is SourceLevel.ABSTRACT
    assert source.reason == "no full text was available"


def test_grobid_for_builds_a_client_when_a_service_is_configured(tmp_path: Path) -> None:
    from auditor.parser.grobid import GrobidClient
    from auditor.retrieval.fulltext import grobid_for

    settings = Settings(cache_dir=tmp_path / "cache", grobid_url="http://localhost:8070")
    client = grobid_for(settings, refusing_client(tmp_path))
    assert isinstance(client, GrobidClient)
    # No service configured: the agent stays on HTML and abstracts.
    assert (
        grobid_for(Settings(cache_dir=tmp_path / "cache", grobid_url=""), refusing_client(tmp_path))
        is None
    )


async def test_a_dead_grobid_skips_the_pdf_and_falls_back(tmp_path: Path) -> None:
    """GROBID being down is a fallback, not a failure: no download, straight to abstract."""
    from auditor.retrieval.fulltext import grobid_for

    settings = Settings(cache_dir=tmp_path / "cache", scratch_dir=tmp_path / "scratch")
    http = page_client(tmp_path, {"isalive": httpx.Response(500)})
    source = await retrieve_source(
        http,
        work(arxiv_id=None, pdf_url="https://example.org/hidden.pdf"),
        settings,
        grobid=grobid_for(settings, http),
    )
    assert source.level is SourceLevel.ABSTRACT
    assert source.reason == "no full text was available"


LOGIN_PAGE = (
    "<html><body><h1>Sign in to continue</h1>"
    + "<p>We use cookies to improve your experience. Manage your account settings.</p>" * 60
    + "</body></html>"
)

PUBLISHER_WORK = {
    "id": "doi:10.1000/example",
    "source": "openalex",
    "arxiv_id": None,
    "html_url": "https://www.nature.com/articles/example",
}


async def test_a_publisher_page_that_does_not_show_the_work_is_not_used(tmp_path: Path) -> None:
    """A long login or cookie page is not the paper: the abstract is used instead."""
    settings = Settings(cache_dir=tmp_path / "cache", scratch_dir=tmp_path / "scratch")
    routes = {"nature.com/articles/example": httpx.Response(200, text=LOGIN_PAGE)}
    source = await retrieve_source(page_client(tmp_path, routes), work(**PUBLISHER_WORK), settings)
    assert source.level is SourceLevel.ABSTRACT
    assert source.text == work().abstract


async def test_a_publisher_page_is_a_partial_text(tmp_path: Path) -> None:
    """A landing page that shows the work is read, but may hold the abstract alone."""
    settings = Settings(cache_dir=tmp_path / "cache", scratch_dir=tmp_path / "scratch")
    routes = {"nature.com/articles/example": httpx.Response(200, text=HTML_PAGE)}
    source = await retrieve_source(page_client(tmp_path, routes), work(**PUBLISHER_WORK), settings)
    assert source.level is SourceLevel.FULL_HTML
    assert source.partial
    # arXiv's own rendering is the paper itself.
    routes = {"arxiv.org/html/1706.03762": httpx.Response(200, text=HTML_PAGE)}
    arxiv = await retrieve_source(page_client(tmp_path / "arxiv", routes), work(), settings)
    assert arxiv.level is SourceLevel.FULL_HTML
    assert not arxiv.partial


async def test_a_cached_page_that_does_not_show_the_work_is_not_used(tmp_path: Path) -> None:
    """Texts cached before the check came in are checked too."""
    settings = Settings(cache_dir=tmp_path / "cache", scratch_dir=tmp_path / "scratch")
    _store(settings, work(**PUBLISHER_WORK), SourceLevel.FULL_HTML, html_to_text(LOGIN_PAGE))
    source = await retrieve_source(refusing_client(tmp_path), work(**PUBLISHER_WORK), settings)
    assert source.level is SourceLevel.ABSTRACT


async def test_the_identity_check_needs_a_distinctive_title(tmp_path: Path) -> None:
    """A title as short as "On" is on every page; accents written two ways are the same
    title; titles in other scripts count."""
    from auditor.retrieval.fulltext import _shows_the_work

    cookie_page = "Sign in to continue. We use cookies on this site. " * 60
    assert not _shows_the_work(cookie_page, work(**PUBLISHER_WORK, title="On", abstract=None))
    combining = "Café Models for Coffee Shop Recommendation, the full article. " * 30
    assert _shows_the_work(
        combining, work(**PUBLISHER_WORK, title="Café Models for Coffee Shop Recommendation")
    )
    cjk_title = "基于深度学习的研究方法综述与展望"
    assert _shows_the_work(f"{cjk_title} 正文 " * 40, work(**PUBLISHER_WORK, title=cjk_title))
    assert not _shows_the_work(
        "\u03b2-learning for robots everywhere " * 40,
        work(**PUBLISHER_WORK, title="\u03b1-learning for robots everywhere", abstract=None),
    )
