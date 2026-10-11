"""Where in a paper a task, finding or gold label points (decision D3)."""

from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from auditor.schemas.base import NonEmptyStr, Record


class SentenceAnchor(Record):
    """A sentence, optionally narrowed to a character span within its text."""

    kind: Literal["sentence"] = "sentence"
    sentence_id: NonEmptyStr
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        if (self.start is None) != (self.end is None):
            raise ValueError("start and end must be given together")
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("end must be greater than start")
        return self

    @property
    def has_span(self) -> bool:
        return self.start is not None


class BibliographyAnchor(Record):
    kind: Literal["bibliography"] = "bibliography"
    bib_id: NonEmptyStr


class PageAnchor(Record):
    kind: Literal["page"] = "page"
    pages: tuple[int, ...] = Field(min_length=1)

    @field_validator("pages")
    @classmethod
    def _normalise_pages(cls, pages: tuple[int, ...]) -> tuple[int, ...]:
        if any(page < 1 for page in pages):
            raise ValueError("page numbers start at 1")
        return tuple(sorted(set(pages)))


class DocumentAnchor(Record):
    kind: Literal["document"] = "document"


Anchor = Annotated[
    SentenceAnchor | BibliographyAnchor | PageAnchor | DocumentAnchor,
    Field(discriminator="kind"),
]


def anchors_match(a: Anchor, b: Anchor) -> bool:
    """Whether two anchors point at the same place, using the rules in decision D3."""
    match a, b:
        case SentenceAnchor(), SentenceAnchor():
            if a.sentence_id != b.sentence_id:
                return False
            if a.start is None or a.end is None or b.start is None or b.end is None:
                return True
            return a.start < b.end and b.start < a.end
        case BibliographyAnchor(), BibliographyAnchor():
            return a.bib_id == b.bib_id
        case PageAnchor(), PageAnchor():
            return not set(a.pages).isdisjoint(b.pages)
        case DocumentAnchor(), DocumentAnchor():
            return True
        case _:
            return False
