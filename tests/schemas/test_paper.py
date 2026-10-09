from typing import Any

import pytest
from pydantic import ValidationError

from auditor.schemas import (
    BibliographyAnchor,
    CitationMention,
    DocumentAnchor,
    LinkMethod,
    PageAnchor,
    ParsedPaper,
    SentenceAnchor,
)
from auditor.schemas.examples import example_paper


def rebuild(**changes: Any) -> ParsedPaper:
    data = example_paper().model_dump()
    data.update(changes)
    return ParsedPaper.model_validate(data)


def test_example_paper_lookups() -> None:
    paper = example_paper()
    assert paper.sentence("s1").text.startswith("Transformers")
    assert paper.bib_entry("b1").doi == "10.18653/v1/D16-1264"
    assert [m.marker for m in paper.mentions_in("s2")] == ["[2, 3]"]
    assert paper.layout is not None
    assert paper.layout.page_count == 2


def test_unknown_lookup_raises_key_error() -> None:
    with pytest.raises(KeyError):
        example_paper().sentence("missing")


def test_paper_is_immutable() -> None:
    paper = example_paper()
    with pytest.raises(ValidationError):
        paper.paper_id = "other"  # type: ignore[misc]


def test_unknown_fields_are_rejected() -> None:
    data = example_paper().model_dump()
    data["comments"] = "not part of the contract"
    with pytest.raises(ValidationError, match="Extra inputs"):
        ParsedPaper.model_validate(data)


def test_duplicate_sentence_ids_are_rejected() -> None:
    sentences = list(example_paper().model_dump()["sentences"])
    sentences.append({**sentences[0]})
    with pytest.raises(ValidationError, match="duplicate sentence ids"):
        rebuild(sentences=sentences)


def test_sentence_with_unknown_section_is_rejected() -> None:
    sentences = example_paper().model_dump()["sentences"]
    sentences[1]["section_id"] = "sec9"
    with pytest.raises(ValidationError, match="unknown section"):
        rebuild(sentences=sentences)


def test_citation_span_must_match_marker() -> None:
    citations = example_paper().model_dump()["citations"]
    citations[0]["start"] += 1
    citations[0]["end"] += 1
    with pytest.raises(ValidationError, match="does not match its marker"):
        rebuild(citations=citations)


def test_editing_sentence_text_without_fixing_spans_is_caught() -> None:
    sentences = example_paper().model_dump()["sentences"]
    sentences[1]["text"] = "Recurrent networks " + sentences[1]["text"]
    with pytest.raises(ValidationError, match="does not match its marker"):
        rebuild(sentences=sentences)


def test_citation_to_unknown_entry_is_rejected() -> None:
    citations = example_paper().model_dump()["citations"]
    citations[0]["bib_ids"] = ["b99"]
    with pytest.raises(ValidationError, match="unknown entries"):
        rebuild(citations=citations)


@pytest.mark.parametrize(
    ("bib_ids", "method"),
    [(("b0",), LinkMethod.NONE), ((), LinkMethod.PARSER), ((), LinkMethod.HEURISTIC)],
)
def test_link_method_must_agree_with_bib_ids(bib_ids: tuple[str, ...], method: LinkMethod) -> None:
    with pytest.raises(ValidationError, match="link_method"):
        CitationMention(
            id="c9",
            sentence_id="s1",
            start=0,
            end=3,
            marker="[1]",
            bib_ids=bib_ids,
            link_method=method,
        )


def test_layout_pages_must_be_numbered_in_order() -> None:
    layout = example_paper().model_dump()["layout"]
    layout["pages"][1]["number"] = 3
    with pytest.raises(ValidationError, match="numbered"):
        rebuild(layout=layout)


def test_box_past_last_page_is_rejected() -> None:
    sentences = example_paper().model_dump()["sentences"]
    sentences[0]["boxes"][0]["page"] = 3
    with pytest.raises(ValidationError, match="past the last page"):
        rebuild(sentences=sentences)


def test_paper_without_layout_is_valid() -> None:
    assert rebuild(layout=None).layout is None


def test_pdf_hash_must_be_hex() -> None:
    parser = example_paper().model_dump()["parser"]
    parser["pdf_sha256"] = "not-a-hash"
    with pytest.raises(ValidationError):
        rebuild(parser=parser)


def test_validate_anchor_accepts_valid_anchors() -> None:
    paper = example_paper()
    paper.validate_anchor(SentenceAnchor(sentence_id="s1", start=0, end=5))
    paper.validate_anchor(BibliographyAnchor(bib_id="b2"))
    paper.validate_anchor(PageAnchor(pages=(2,)))
    paper.validate_anchor(DocumentAnchor())


@pytest.mark.parametrize(
    "anchor",
    [
        SentenceAnchor(sentence_id="s99"),
        SentenceAnchor(sentence_id="s1", start=0, end=500),
        BibliographyAnchor(bib_id="b99"),
        PageAnchor(pages=(3,)),
    ],
)
def test_validate_anchor_rejects_invalid_anchors(
    anchor: SentenceAnchor | BibliographyAnchor | PageAnchor,
) -> None:
    with pytest.raises(ValueError, match=r"unknown|past"):
        example_paper().validate_anchor(anchor)
