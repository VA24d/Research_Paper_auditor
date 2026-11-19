from pathlib import Path

import pytest

from auditor.parser.layout import LayoutError, extract_layout

# A PDF the team wrote, so it can live in the repository (decision D13).
PROPOSAL = Path(__file__).parents[2] / "docs" / "proposal" / "proposal.pdf"


def test_layout_of_the_proposal_pdf() -> None:
    layout, links = extract_layout(PROPOSAL)
    assert layout.page_count == 5
    for page in layout.pages:
        assert page.width > 0
        assert page.height > 0
        assert page.text_left is not None
        assert page.text_right is not None
        assert page.text_top is not None
        assert page.text_bottom is not None
        assert 0 <= page.text_left < page.text_right <= page.width
        assert 0 <= page.text_top < page.text_bottom <= page.height
    sizes = [font.size for font in layout.fonts]
    counts = [font.chars for font in layout.fonts]
    assert sizes
    assert counts == sorted(counts, reverse=True)
    assert all("+" not in font.name for font in layout.fonts)
    assert all(link.url.startswith("http") for link in links)


def test_non_pdf_raises_layout_error(tmp_path: Path) -> None:
    fake = tmp_path / "paper.pdf"
    fake.write_bytes(b"this is not a pdf")
    with pytest.raises(LayoutError):
        extract_layout(fake)
