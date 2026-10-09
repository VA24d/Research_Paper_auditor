from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from auditor.cli import main
from auditor.config import Settings
from auditor.infra.http import HttpClient
from auditor.parser.grobid import GrobidClient, GrobidError
from auditor.parser.pipeline import ParseError, parse_pdf, tei_cache_path
from auditor.schemas import LinkSource

FIXTURES = Path(__file__).parents[1] / "fixtures"
SAMPLE_TEI = (FIXTURES / "tei" / "sample.tei.xml").read_bytes()
PROPOSAL = Path(__file__).parents[2] / "LMA group project proposal.pdf"
GROBID_URL = "http://localhost:8070"


async def no_sleep(_: float) -> None:
    return None


def grobid_with(handler: Callable[[httpx.Request], httpx.Response], tmp_path: Path) -> GrobidClient:
    settings = Settings(cache_dir=tmp_path / "cache", http_max_attempts=2)
    http = HttpClient(settings, transport=httpx.MockTransport(handler), sleep=no_sleep)
    return GrobidClient(http, GROBID_URL)


def tei_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=SAMPLE_TEI, headers={"content-type": "application/xml"})


async def test_fulltext_request_sends_the_agreed_options(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return tei_response(request)

    tei = await grobid_with(handler, tmp_path).process_fulltext(b"%PDF-1.5", "paper.pdf")
    assert tei.startswith("<?xml")
    (request,) = seen
    assert request.url.path == "/api/processFulltextDocument"
    body = request.content
    for name, value in [
        (b"segmentSentences", b"1"),
        (b"consolidateHeader", b"0"),
        (b"consolidateCitations", b"0"),
        (b"includeRawCitations", b"1"),
    ]:
        assert b'name="' + name + b'"\r\n\r\n' + value in body
    assert body.count(b'name="teiCoordinates"') == 7
    assert b'filename="paper.pdf"' in body


@pytest.mark.parametrize(
    ("status", "message"),
    [(204, "no content"), (500, "HTTP 500: NO_BLOCKS"), (503, "busy")],
)
async def test_grobid_errors_are_reported(tmp_path: Path, status: int, message: str) -> None:
    grobid = grobid_with(lambda _: httpx.Response(status, text="NO_BLOCKS"), tmp_path)
    with pytest.raises(GrobidError, match=message):
        await grobid.process_fulltext(b"%PDF", "x.pdf")


async def test_is_alive(tmp_path: Path) -> None:
    assert await grobid_with(lambda _: httpx.Response(200, text="true"), tmp_path).is_alive()
    assert not await grobid_with(lambda _: httpx.Response(200, text="false"), tmp_path).is_alive()

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert not await grobid_with(refuse, tmp_path).is_alive()


async def test_parse_pdf_combines_tei_and_pdf_layout(tmp_path: Path) -> None:
    paper = await parse_pdf(PROPOSAL, grobid=grobid_with(tei_response, tmp_path))
    assert paper.paper_id == PROPOSAL.stem
    assert paper.layout is not None
    assert paper.layout.page_count == 5
    assert paper.layout.fonts
    assert paper.parser.pdf_sha256 is not None
    assert len(paper.parser.pdf_sha256) == 64
    assert any(link.source is LinkSource.TEXT for link in paper.links)


async def test_parse_pdf_uses_the_cache(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return tei_response(request)

    grobid = grobid_with(handler, tmp_path)
    cache_dir = tmp_path / "cache"
    first = await parse_pdf(PROPOSAL, grobid=grobid, cache_dir=cache_dir)
    second = await parse_pdf(PROPOSAL, grobid=grobid, cache_dir=cache_dir, paper_id="renamed")
    assert calls == 1
    assert second.paper_id == "renamed"
    assert second.sentences == first.sentences
    await parse_pdf(PROPOSAL, grobid=grobid, cache_dir=cache_dir, refresh=True)
    assert calls == 2
    assert first.parser.pdf_sha256 is not None
    assert tei_cache_path(cache_dir, first.parser.pdf_sha256).read_bytes() == SAMPLE_TEI


async def test_corrupt_cache_entry_is_parsed_again(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return tei_response(request)

    grobid = grobid_with(handler, tmp_path)
    cache_dir = tmp_path / "cache"
    paper = await parse_pdf(PROPOSAL, grobid=grobid, cache_dir=cache_dir)
    assert paper.parser.pdf_sha256 is not None
    tei_cache_path(cache_dir, paper.parser.pdf_sha256).write_text("{}", encoding="utf-8")
    again = await parse_pdf(PROPOSAL, grobid=grobid, cache_dir=cache_dir)
    assert again.sentences == paper.sentences
    assert calls == 2


async def test_grobid_failure_becomes_parse_error(tmp_path: Path) -> None:
    grobid = grobid_with(lambda _: httpx.Response(500, text="TIMEOUT"), tmp_path)
    with pytest.raises(ParseError, match="TIMEOUT"):
        await parse_pdf(PROPOSAL, grobid=grobid)


def test_cli_reports_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["parse", "missing.pdf"]) == 2
    assert "No such file" in capsys.readouterr().err


def test_cli_reports_unreachable_grobid(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Port 9 (discard) is closed on developer machines and CI runners.
    monkeypatch.setenv("AUDITOR_GROBID_URL", "http://127.0.0.1:9")
    assert main(["parse", str(PROPOSAL)]) == 2
    assert "docker compose up -d" in capsys.readouterr().err
