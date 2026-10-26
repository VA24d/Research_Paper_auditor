"""Offline tests for full-text retrieval (P2.3): canned pages, no network."""

from pathlib import Path

import httpx

from auditor.config import Settings
from auditor.infra.http import HttpClient
from auditor.retrieval.clients import Work
from auditor.retrieval.fulltext import SourceLevel, html_to_text, retrieve_source

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
