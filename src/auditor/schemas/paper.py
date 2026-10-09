"""The parsed form of a submission that every stage after parsing reads (decision D1)."""

from collections import Counter
from collections.abc import Iterable
from enum import StrEnum
from functools import cached_property
from typing import Self

from pydantic import Field, model_validator

from auditor.schemas.anchors import (
    Anchor,
    BibliographyAnchor,
    DocumentAnchor,
    PageAnchor,
    SentenceAnchor,
)
from auditor.schemas.base import SCHEMA_VERSION, NonEmptyStr, Record, SchemaVersion


class PaperVariant(StrEnum):
    ORIGINAL = "original"
    INJECTED = "injected"


class SentenceKind(StrEnum):
    ABSTRACT = "abstract"
    BODY = "body"
    CAPTION = "caption"
    FOOTNOTE = "footnote"


class LinkMethod(StrEnum):
    """How a citation mention was linked to the bibliography."""

    PARSER = "parser"
    HEURISTIC = "heuristic"
    NONE = "none"


class LinkSource(StrEnum):
    ANNOTATION = "annotation"
    TEXT = "text"


class Box(Record):
    """A rectangle on a page, in PDF points with the origin at the top left."""

    page: int = Field(ge=1)
    x: float
    y: float
    width: float = Field(ge=0)
    height: float = Field(ge=0)


class Section(Record):
    id: NonEmptyStr
    title: str
    number: str | None = None


class Sentence(Record):
    id: NonEmptyStr
    section_id: str | None = None
    kind: SentenceKind = SentenceKind.BODY
    text: NonEmptyStr
    boxes: tuple[Box, ...] = ()

    @property
    def pages(self) -> tuple[int, ...]:
        return tuple(sorted({box.page for box in self.boxes}))


class BibEntry(Record):
    id: NonEmptyStr
    raw: str | None = None
    title: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    boxes: tuple[Box, ...] = ()


class CitationMention(Record):
    """An in-text citation marker. `start`/`end` index into the sentence's text."""

    id: NonEmptyStr
    sentence_id: NonEmptyStr
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    marker: NonEmptyStr
    bib_ids: tuple[str, ...] = ()
    link_method: LinkMethod

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be greater than start")
        if (self.link_method is LinkMethod.NONE) != (not self.bib_ids):
            raise ValueError("link_method must be 'none' exactly when bib_ids is empty")
        return self

    @property
    def is_linked(self) -> bool:
        return bool(self.bib_ids)


class Link(Record):
    url: NonEmptyStr
    page: int | None = Field(default=None, ge=1)
    sentence_id: str | None = None
    source: LinkSource


class FontUsage(Record):
    name: str
    size: float = Field(gt=0)
    chars: int = Field(ge=0)


class PageLayout(Record):
    """Page size and the bounding box of all text on the page (None for a page without text)."""

    number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    text_left: float | None = None
    text_top: float | None = None
    text_right: float | None = None
    text_bottom: float | None = None


class Layout(Record):
    pages: tuple[PageLayout, ...] = Field(min_length=1)
    fonts: tuple[FontUsage, ...] = ()

    @model_validator(mode="after")
    def _check_numbering(self) -> Self:
        numbers = [page.number for page in self.pages]
        if numbers != list(range(1, len(numbers) + 1)):
            raise ValueError("pages must be numbered 1..n in order")
        return self

    @property
    def page_count(self) -> int:
        return len(self.pages)


class ParserInfo(Record):
    name: NonEmptyStr
    version: NonEmptyStr
    pdf_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class PaperMetadata(Record):
    title: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None


class ParsedPaper(Record):
    schema_version: SchemaVersion = SCHEMA_VERSION
    paper_id: NonEmptyStr
    variant: PaperVariant = PaperVariant.ORIGINAL
    parser: ParserInfo
    metadata: PaperMetadata = PaperMetadata()
    sections: tuple[Section, ...] = ()
    sentences: tuple[Sentence, ...] = Field(min_length=1)
    bibliography: tuple[BibEntry, ...] = ()
    citations: tuple[CitationMention, ...] = ()
    links: tuple[Link, ...] = ()
    layout: Layout | None = None

    @model_validator(mode="after")
    def _check_references(self) -> Self:
        _require_unique("section", (s.id for s in self.sections))
        _require_unique("sentence", (s.id for s in self.sentences))
        _require_unique("bibliography entry", (b.id for b in self.bibliography))
        _require_unique("citation mention", (c.id for c in self.citations))

        section_ids = {s.id for s in self.sections}
        sentences = {s.id: s for s in self.sentences}
        bib_ids = {b.id for b in self.bibliography}

        for sentence in self.sentences:
            if sentence.section_id is not None and sentence.section_id not in section_ids:
                raise ValueError(f"sentence {sentence.id} refers to unknown section")
        for mention in self.citations:
            cited_in = sentences.get(mention.sentence_id)
            if cited_in is None:
                raise ValueError(f"citation {mention.id} refers to unknown sentence")
            if cited_in.text[mention.start : mention.end] != mention.marker:
                raise ValueError(f"citation {mention.id} span does not match its marker")
            unknown = set(mention.bib_ids) - bib_ids
            if unknown:
                raise ValueError(f"citation {mention.id} links unknown entries {sorted(unknown)}")
        for link in self.links:
            if link.sentence_id is not None and link.sentence_id not in sentences:
                raise ValueError(f"link {link.url} refers to unknown sentence")

        if self.layout is not None:
            page_count = self.layout.page_count
            boxes = [box for s in self.sentences for box in s.boxes]
            boxes += [box for b in self.bibliography for box in b.boxes]
            if any(box.page > page_count for box in boxes):
                raise ValueError("a box points past the last page")
            if any(link.page is not None and link.page > page_count for link in self.links):
                raise ValueError("a link points past the last page")
        return self

    @cached_property
    def _sentences_by_id(self) -> dict[str, Sentence]:
        return {s.id: s for s in self.sentences}

    @cached_property
    def _bibliography_by_id(self) -> dict[str, BibEntry]:
        return {b.id: b for b in self.bibliography}

    def sentence(self, sentence_id: str) -> Sentence:
        return self._sentences_by_id[sentence_id]

    def bib_entry(self, bib_id: str) -> BibEntry:
        return self._bibliography_by_id[bib_id]

    def mentions_in(self, sentence_id: str) -> tuple[CitationMention, ...]:
        return tuple(c for c in self.citations if c.sentence_id == sentence_id)

    def validate_anchor(self, anchor: Anchor) -> None:
        """Raise ValueError if the anchor does not point at something in this paper."""
        match anchor:
            case SentenceAnchor():
                sentence = self._sentences_by_id.get(anchor.sentence_id)
                if sentence is None:
                    raise ValueError(f"unknown sentence {anchor.sentence_id}")
                if anchor.end is not None and anchor.end > len(sentence.text):
                    raise ValueError(f"span runs past the end of sentence {anchor.sentence_id}")
            case BibliographyAnchor():
                if anchor.bib_id not in self._bibliography_by_id:
                    raise ValueError(f"unknown bibliography entry {anchor.bib_id}")
            case PageAnchor():
                if self.layout is not None and anchor.pages[-1] > self.layout.page_count:
                    raise ValueError("page anchor points past the last page")
            case DocumentAnchor():
                pass


def _require_unique(label: str, ids: Iterable[str]) -> None:
    duplicates = sorted(i for i, n in Counter(ids).items() if n > 1)
    if duplicates:
        raise ValueError(f"duplicate {label} ids: {duplicates}")
