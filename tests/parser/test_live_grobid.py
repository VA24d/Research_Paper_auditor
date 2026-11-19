"""End-to-end parsing against a running GROBID. Run with: uv run pytest -m grobid"""

from pathlib import Path

import pytest

from auditor.cli import main
from auditor.config import Settings
from auditor.infra.http import HttpClient
from auditor.parser.grobid import GrobidClient
from auditor.parser.pipeline import parse_pdf
from auditor.schemas import ParsedPaper, SentenceKind

PROPOSAL = Path(__file__).parents[2] / "docs" / "proposal" / "proposal.pdf"


@pytest.fixture
async def grobid(tmp_path: Path):  # type: ignore[no-untyped-def]
    settings = Settings(cache_dir=tmp_path / "cache")
    async with HttpClient(settings) as http:
        client = GrobidClient(http, settings.grobid_url)
        if not await client.is_alive():
            pytest.fail(f"GROBID is not running at {settings.grobid_url}")
        yield client


@pytest.mark.grobid
async def test_proposal_pdf_parses(grobid: GrobidClient) -> None:
    paper = await parse_pdf(PROPOSAL, grobid=grobid)
    assert paper.layout is not None
    assert paper.layout.page_count == 5
    assert len(paper.sentences) > 20
    assert "Why the problem matters" in [s.title for s in paper.sections]


@pytest.mark.grobid
def test_cli_writes_parsed_paper(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "proposal.json"
    assert main(["parse", str(PROPOSAL), "-o", str(output)]) == 0
    paper = ParsedPaper.model_validate_json(output.read_bytes())
    assert paper.paper_id == PROPOSAL.stem
    assert "5 pages" in capsys.readouterr().out


@pytest.mark.grobid
@pytest.mark.network
async def test_arxiv_paper_citations_are_linked(grobid: GrobidClient, tmp_path: Path) -> None:
    settings = Settings(cache_dir=tmp_path / "cache")
    async with HttpClient(settings) as http:
        response = await http.get("https://arxiv.org/pdf/2310.06825v1")
    response.raise_for_status()
    pdf = tmp_path / "2310.06825.pdf"
    pdf.write_bytes(response.content)

    paper = await parse_pdf(pdf, grobid=grobid)
    linked = [c for c in paper.citations if c.bib_ids]
    assert len(paper.bibliography) >= 25
    assert len(linked) / len(paper.citations) >= 0.95
    assert any(s.kind is SentenceKind.ABSTRACT for s in paper.sentences)
    assert any("github.com/mistralai" in link.url for link in paper.links)
    assert paper.layout is not None
    assert paper.layout.pages[0].text_left is not None
    assert paper.layout.pages[0].text_left > 72
