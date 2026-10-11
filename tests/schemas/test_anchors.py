import pytest
from pydantic import TypeAdapter, ValidationError

from auditor.schemas import (
    Anchor,
    BibliographyAnchor,
    DocumentAnchor,
    PageAnchor,
    SentenceAnchor,
    anchors_match,
)

ANCHOR = TypeAdapter[Anchor](Anchor)


def test_sentence_span_needs_both_ends() -> None:
    with pytest.raises(ValidationError):
        SentenceAnchor(sentence_id="s1", start=3)


def test_sentence_span_must_not_be_empty() -> None:
    with pytest.raises(ValidationError):
        SentenceAnchor(sentence_id="s1", start=4, end=4)


def test_pages_are_sorted_and_deduplicated() -> None:
    assert PageAnchor(pages=(3, 1, 3)).pages == (1, 3)


@pytest.mark.parametrize("pages", [(), (0,), (2, -1)])
def test_invalid_pages_are_rejected(pages: tuple[int, ...]) -> None:
    with pytest.raises(ValidationError):
        PageAnchor(pages=pages)


def test_json_is_parsed_into_the_right_kind() -> None:
    anchor = ANCHOR.validate_json('{"kind": "page", "pages": [2]}')
    assert anchor == PageAnchor(pages=(2,))


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ANCHOR.validate_json('{"kind": "table", "id": "t1"}')


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        (SentenceAnchor(sentence_id="s1"), SentenceAnchor(sentence_id="s1"), True),
        (SentenceAnchor(sentence_id="s1"), SentenceAnchor(sentence_id="s2"), False),
        (
            SentenceAnchor(sentence_id="s1"),
            SentenceAnchor(sentence_id="s1", start=0, end=5),
            True,
        ),
        (
            SentenceAnchor(sentence_id="s1", start=0, end=5),
            SentenceAnchor(sentence_id="s1", start=4, end=9),
            True,
        ),
        (
            SentenceAnchor(sentence_id="s1", start=0, end=5),
            SentenceAnchor(sentence_id="s1", start=5, end=9),
            False,
        ),
        (BibliographyAnchor(bib_id="b1"), BibliographyAnchor(bib_id="b1"), True),
        (BibliographyAnchor(bib_id="b1"), BibliographyAnchor(bib_id="b2"), False),
        (PageAnchor(pages=(1, 2)), PageAnchor(pages=(2, 3)), True),
        (PageAnchor(pages=(1,)), PageAnchor(pages=(2,)), False),
        (DocumentAnchor(), DocumentAnchor(), True),
        (DocumentAnchor(), PageAnchor(pages=(1,)), False),
        (SentenceAnchor(sentence_id="b1"), BibliographyAnchor(bib_id="b1"), False),
    ],
)
def test_anchors_match(a: Anchor, b: Anchor, expected: bool) -> None:
    assert anchors_match(a, b) is expected
    assert anchors_match(b, a) is expected
