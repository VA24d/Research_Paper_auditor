from pathlib import Path

import pytest

from auditor.parser.tei import TeiError, normalise_arxiv_id, parse_coords, tei_to_paper
from auditor.schemas import (
    Box,
    Layout,
    Link,
    LinkMethod,
    LinkSource,
    PageLayout,
    ParsedPaper,
    SentenceKind,
)

SAMPLE = Path(__file__).parents[1] / "fixtures" / "tei" / "sample.tei.xml"


@pytest.fixture
def tei() -> str:
    return SAMPLE.read_text(encoding="utf-8")


@pytest.fixture
def paper(tei: str) -> ParsedPaper:
    return tei_to_paper(tei, paper_id="sample", pdf_sha256="a" * 64)


def test_sentences_follow_document_order_and_kind(paper: ParsedPaper) -> None:
    assert [(s.kind, s.section_id) for s in paper.sentences] == [
        (SentenceKind.ABSTRACT, None),
        (SentenceKind.BODY, "sec0"),
        (SentenceKind.BODY, "sec0"),
        (SentenceKind.BODY, "sec0"),
        (SentenceKind.BODY, "sec0"),
        (SentenceKind.BODY, "sec1"),
        (SentenceKind.CAPTION, None),
        (SentenceKind.FOOTNOTE, None),
        (SentenceKind.BODY, "sec2"),
        (SentenceKind.BODY, "sec3"),
    ]


def test_sections_keep_titles_and_numbers(paper: ParsedPaper) -> None:
    assert [(s.title, s.number) for s in paper.sections] == [
        ("Introduction", "1"),
        ("Method", "2"),
        ("Acknowledgements", None),
        ("Hyperparameters", "A"),
    ]


def test_sentence_text_is_trimmed_and_flattened(paper: ParsedPaper) -> None:
    assert paper.sentence("s3").text == "Earlier results [9] are disputed."
    assert paper.sentence("s5").text.startswith("A paragraph that GROBID did not split")


def test_sentence_boxes_come_from_coords(paper: ParsedPaper) -> None:
    assert paper.sentence("s1").boxes == (Box(page=1, x=100.0, y=270.0, width=400.0, height=9.0),)
    assert paper.sentence("s5").boxes == ()


def test_citation_markers_are_linked_with_exact_spans(paper: ParsedPaper) -> None:
    found = {c.marker: (c.bib_ids, c.link_method) for c in paper.citations}
    assert found == {
        "[1]": (("b0",), LinkMethod.PARSER),
        "[2,3]": (("b1", "b2"), LinkMethod.PARSER),
        "[9]": ((), LinkMethod.NONE),
        "(2020)": (("b2",), LinkMethod.PARSER),
    }
    for mention in paper.citations:
        text = paper.sentence(mention.sentence_id).text
        assert text[mention.start : mention.end] == mention.marker


def test_links_to_unknown_entries_are_dropped(tei: str) -> None:
    paper = tei_to_paper(tei.replace('target="#b0"', 'target="#b77"'), paper_id="x")
    mention = next(c for c in paper.citations if c.marker == "[1]")
    assert mention.bib_ids == ()
    assert mention.link_method is LinkMethod.NONE


def test_split_markers_with_mixed_resolution_stay_separate(tei: str) -> None:
    changed = tei.replace('<ref type="bibr" target="#b2">3]</ref>', '<ref type="bibr">3]</ref>')
    paper = tei_to_paper(changed, paper_id="x")
    markers = [(c.marker, c.bib_ids) for c in paper.citations if c.sentence_id == "s2"]
    assert markers == [("[2,", ("b1",)), ("3]", ())]


def test_links_from_url_refs_and_plain_text(paper: ParsedPaper) -> None:
    assert [(link.url, link.page, link.sentence_id) for link in paper.links] == [
        ("https://github.com/example/claims", 1, "s4"),
        ("https://example.org/data", None, "s5"),
        ("https://github.com/example/claims", 2, "s7"),
    ]
    assert all(link.source is LinkSource.TEXT for link in paper.links)


def test_extra_links_are_added_without_duplicates(tei: str) -> None:
    extra = [
        Link(url="https://github.com/example/claims", page=1, source=LinkSource.ANNOTATION),
        Link(url="https://github.com/other/repo", page=2, source=LinkSource.ANNOTATION),
    ]
    paper = tei_to_paper(tei, paper_id="x", extra_links=extra)
    urls = [(link.url, link.page) for link in paper.links]
    assert urls.count(("https://github.com/example/claims", 1)) == 1
    assert ("https://github.com/other/repo", 2) in urls


def test_bibliography_fields(paper: ParsedPaper) -> None:
    article, book, journal = paper.bibliography
    assert (article.title, article.authors, article.year) == (
        "Attention is all you need",
        ("Ashish Vaswani", "Noam Shazeer"),
        2017,
    )
    assert article.venue == "Advances in Neural Information Processing Systems"
    assert article.arxiv_id == "1706.03762"
    assert article.raw is not None
    assert article.raw.startswith("Vaswani, A.")
    assert article.boxes[0].page == 2
    assert (book.title, book.authors, book.year, book.venue) == (
        "A Book on Retrieval",
        ("María Núñez",),
        2019,
        None,
    )
    assert book.url == "https://example.org/book"
    assert (journal.venue, journal.doi) == ("Journal of Retrieval", "10.1000/xyz123")


def test_metadata_and_parser_info(paper: ParsedPaper) -> None:
    assert paper.metadata.title == "Checking Claims in Research Papers"
    assert paper.metadata.authors == ("Ada Example", "Alan B Sample")
    assert paper.metadata.year == 2026
    assert paper.metadata.arxiv_id == "2605.01234"
    assert (paper.parser.name, paper.parser.version) == ("grobid", "0.9.1")
    assert paper.parser.pdf_sha256 == "a" * 64


def test_page_sizes_come_from_facsimile_without_pdf_layout(paper: ParsedPaper) -> None:
    assert paper.layout is not None
    assert [(p.width, p.height) for p in paper.layout.pages] == [(612.0, 792.0)] * 2


def test_pdf_layout_takes_precedence(tei: str) -> None:
    layout = Layout(pages=tuple(PageLayout(number=n, width=595.0, height=842.0) for n in (1, 2)))
    paper = tei_to_paper(tei, paper_id="x", layout=layout)
    assert paper.layout == layout


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ("<TEI><unclosed></TEI>", "well-formed"),
        ('<html xmlns="http://www.tei-c.org/ns/1.0"/>', "expected a TEI"),
        ('<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body/></text></TEI>', "no text"),
    ],
)
def test_unusable_documents_raise(document: str, message: str) -> None:
    with pytest.raises(TeiError, match=message):
        tei_to_paper(document, paper_id="x")


def test_parse_coords_skips_malformed_parts() -> None:
    boxes = parse_coords("1,10,20,30,5;bad;2,x,1,1,1;0,1,1,1,1;3,1.5,2.5,10,4")
    assert boxes == (
        Box(page=1, x=10.0, y=20.0, width=30.0, height=5.0),
        Box(page=3, x=1.5, y=2.5, width=10.0, height=4.0),
    )
    assert parse_coords(None) == ()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2310.06825v1[cs.CL]", "2310.06825"),
        ("arXiv:2305.13245", "2305.13245"),
        ("arXiv:hep-th/9901001v2", "hep-th/9901001"),
        ("not an id", None),
    ],
)
def test_normalise_arxiv_id(value: str, expected: str | None) -> None:
    assert normalise_arxiv_id(value) == expected


def test_urls_broken_at_line_ends_are_repaired_from_annotations(tei: str) -> None:
    broken = tei.replace(
        "We use a learning rate of 3e-4.",
        "Code is available at https://github.com/ example/vit-code.",
    )
    annotation = Link(
        url="https://github.com/example/vit-code", page=2, source=LinkSource.ANNOTATION
    )
    paper = tei_to_paper(broken, paper_id="x", extra_links=[annotation])
    repaired = [link for link in paper.links if link.sentence_id == "s9"]
    assert [link.url for link in repaired] == ["https://github.com/example/vit-code"]
    assert [link.url for link in paper.links].count("https://github.com/example/vit-code") == 1


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("2022a. PromptChainer: Chaining prompts", "PromptChainer: Chaining prompts"),
        ("b. AI chains: Transparent interaction", "AI chains: Transparent interaction"),
        ("A. Very short title", "A. Very short title"),
        ("2020 Vision benchmarks", "2020 Vision benchmarks"),
    ],
)
def test_year_labels_are_removed_from_titles(tei: str, title: str, expected: str) -> None:
    changed = tei.replace(">Dense retrieval at scale<", f">{title}<")
    paper = tei_to_paper(changed, paper_id="x")
    assert paper.bib_entry("b2").title == expected


def test_captions_and_footnotes_without_paragraphs_are_kept(tei: str) -> None:
    changed = tei.replace(
        '<figDesc><div><p><s coords="2,100.00,400.00,300.00,8.00">'
        "Figure 1: Accuracy against model size.</s></p></div></figDesc>",
        "<figDesc>Figure 1: Accuracy against model size.</figDesc>",
    ).replace(
        '<p><s coords="2,100.00,700.00,200.00,7.00">https://github.com/example/claims.</s></p>',
        "Code: https://github.com/example/claims",
    )
    paper = tei_to_paper(changed, paper_id="x")
    texts = {s.kind: s.text for s in paper.sentences if s.kind is not SentenceKind.BODY}
    assert texts[SentenceKind.CAPTION] == "Figure 1: Accuracy against model size."
    assert texts[SentenceKind.FOOTNOTE] == "Code: https://github.com/example/claims"


def test_appendix_tables_captions_and_footnotes_are_kept(tei: str) -> None:
    appendix = (
        "<p><s>We use a learning rate of 3e-4.</s></p>"
        '<figure type="table" xml:id="tab_9"><head>Table 9:</head>'
        "<figDesc>Fine-tuning settings.</figDesc>"
        "<table><row><cell>batch size</cell><cell>512</cell></row></table></figure>"
        '<note place="foot">Run on 8 GPUs.</note>'
    )
    changed = tei.replace(
        '<p><s coords="2,100.00,500.00,300.00,9.00">We use a learning rate of 3e-4.</s></p>',
        appendix,
    )
    assert changed != tei
    paper = tei_to_paper(changed, paper_id="x")
    (table,) = paper.tables
    assert table.label == "Table 9"
    assert [paper.sentence(r).text for r in table.row_ids] == ["batch size | 512"]
    caption = next(
        s for s in paper.sentences if s.kind is SentenceKind.CAPTION and "settings" in s.text
    )
    appendix_section = next(s for s in paper.sections if s.title == "Hyperparameters")
    assert caption.section_id == appendix_section.id
    assert any(
        s.kind is SentenceKind.FOOTNOTE and s.text == "Run on 8 GPUs." for s in paper.sentences
    )
