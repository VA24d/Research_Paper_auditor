"""Hand-built papers for the template checker: one passing and one violating case per rule.

The cases are used by the tests and by `scripts/template_eval.py`, which reports how many
of them the checker gets right per rule (P3.9).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import JsonValue

from auditor.schemas import (
    BibEntry,
    Box,
    DocumentAnchor,
    FontUsage,
    Layout,
    Link,
    LinkSource,
    PageAnchor,
    PageLayout,
    PaperMetadata,
    ParsedPaper,
    ParserInfo,
    Section,
    Sentence,
    SentenceKind,
    TaskOutcome,
    TaskResult,
    VenueRule,
)

# US Letter with the NeurIPS text block: 5.5 x 9 inches, 1.5 inch left and 1 inch top margin.
LETTER = (612.0, 792.0)
A4 = (595.0, 842.0)
TEXT_BOX = (108.0, 72.0, 504.0, 720.0)

# (title, number, pages holding one body sentence each)
SectionSpec = tuple[str, str | None, Sequence[int]]

MAIN_SECTIONS: tuple[SectionSpec, ...] = (
    ("Introduction", "1", (1, 2)),
    ("Method", "2", (3, 4, 5)),
    ("Experiments", "3", (6, 7, 8)),
    ("Conclusion", "4", (9,)),
)
APPENDIX: tuple[SectionSpec, ...] = (("Proofs", "A", (11, 12)),)
BODY_FONTS = (("Times-Roman", 10.0, 60000), ("Times-Roman", 8.0, 9000), ("CMR7", 7.0, 800))


def build_paper(
    *,
    page_count: int = 12,
    sections: Sequence[SectionSpec] = (*MAIN_SECTIONS, *APPENDIX),
    bibliography_pages: Sequence[int] = (10,),
    fonts: Sequence[tuple[str, float, int]] = BODY_FONTS,
    page_size: tuple[float, float] = LETTER,
    text_box: tuple[float, float, float, float] = TEXT_BOX,
    page_overrides: Mapping[int, Mapping[str, float | None]] | None = None,
    with_layout: bool = True,
    with_boxes: bool = True,
    authors: Sequence[str] = (),
    links: Sequence[str] = (),
) -> ParsedPaper:
    """A paper with one abstract sentence on page 1 and one body sentence per listed page.

    `page_overrides` replaces fields of single pages' layout, e.g. `{3: {"text_left": 60}}`.
    """
    sentences: list[Sentence] = [
        Sentence(id="s0", kind=SentenceKind.ABSTRACT, text="We study things.", boxes=_box(1))
    ]
    section_records = []
    for index, (title, number, pages) in enumerate(sections, start=1):
        section_id = f"sec{index}"
        section_records.append(Section(id=section_id, title=title, number=number))
        for page in pages:
            sentences.append(
                Sentence(
                    id=f"s{len(sentences)}",
                    section_id=section_id,
                    text=f"Text of {title} on page {page}.",
                    boxes=_box(page) if with_boxes else (),
                )
            )
    if not with_boxes:
        sentences[0] = sentences[0].model_copy(update={"boxes": ()})
    bibliography = tuple(
        BibEntry(
            id=f"b{index}",
            title=f"Cited work {index}",
            boxes=_box(page) if with_boxes else (),
        )
        for index, page in enumerate(bibliography_pages)
    )
    layout = None
    if with_layout:
        overrides = page_overrides or {}
        left, top, right, bottom = text_box
        page_layouts: list[PageLayout] = []
        for page_number in range(1, page_count + 1):
            values: dict[str, float | None] = {
                "width": page_size[0],
                "height": page_size[1],
                "text_left": left,
                "text_top": top,
                "text_right": right,
                "text_bottom": bottom,
            }
            values.update(overrides.get(page_number, {}))
            page_layouts.append(PageLayout.model_validate({"number": page_number, **values}))
        layout = Layout(
            pages=tuple(page_layouts),
            fonts=tuple(FontUsage(name=n, size=s, chars=c) for n, s, c in fonts),
        )
    return ParsedPaper(
        paper_id="template-fixture",
        parser=ParserInfo(name="fixture", version="1"),
        metadata=PaperMetadata(title="A Fixture Paper", authors=tuple(authors)),
        sections=tuple(section_records),
        sentences=tuple(sentences),
        bibliography=bibliography,
        links=tuple(Link(url=url, source=LinkSource.TEXT) for url in links),
        layout=layout,
    )


def _box(page: int) -> tuple[Box, ...]:
    return (Box(page=page, x=110.0, y=100.0, width=380.0, height=10.0),)


Expected = Literal["pass", "violation", "unverifiable"]


@dataclass(frozen=True)
class TemplateCase:
    """A paper, a rule and what the checker should conclude about them."""

    name: str
    rule: VenueRule
    paper: ParsedPaper
    expected: Expected
    # The pages a violation should be anchored on; empty for a document-level finding.
    pages: tuple[int, ...] = field(default=())


def rule(check: str, params: Mapping[str, JsonValue] | None = None) -> VenueRule:
    return VenueRule(
        id=check.replace("_", "-"),
        description=f"Fixture rule for {check}.",
        check=check,
        params=dict(params or {}),
    )


def template_cases() -> list[TemplateCase]:
    page_limit = rule("max_pages", {"max_pages": 9})
    font = rule("min_body_font_size", {"min_size": 10.0})
    size = rule("paper_size", {"width": 612, "height": 792})
    margins = rule("margins", {"min_left": 108, "min_right": 108, "min_top": 72, "min_bottom": 72})
    limitations = rule("required_sections", {"sections": ["limitations"]})
    no_acknowledgements = rule("forbidden_sections", {"sections": ["acknowledg"]})
    references = rule("references_present")
    long_main = (*MAIN_SECTIONS[:3], ("Conclusion", "4", (9, 10)))
    with_limitations = (*MAIN_SECTIONS, ("Limitations", None, (9,)), *APPENDIX)
    with_acknowledgements = (*MAIN_SECTIONS, ("Acknowledgments", None, (9,)), *APPENDIX)
    return [
        TemplateCase("page limit kept", page_limit, build_paper(), "pass"),
        TemplateCase(
            "page limit exceeded",
            page_limit,
            build_paper(page_count=13, sections=(*long_main, *APPENDIX), bibliography_pages=(11,)),
            "violation",
        ),
        TemplateCase("body font 10 pt", font, build_paper(), "pass"),
        TemplateCase(
            "body font 9 pt",
            font,
            build_paper(fonts=(("Times-Roman", 9.0, 60000), ("Times-Roman", 10.0, 2000))),
            "violation",
        ),
        TemplateCase("US Letter pages", size, build_paper(), "pass"),
        TemplateCase(
            "A4 page in a Letter paper",
            size,
            build_paper(page_overrides={4: {"width": A4[0], "height": A4[1]}}),
            "violation",
            pages=(4,),
        ),
        TemplateCase("text inside the margins", margins, build_paper(), "pass"),
        TemplateCase(
            "text in the left margin",
            margins,
            build_paper(page_overrides={3: {"text_left": 90.0}, 7: {"text_bottom": 760.0}}),
            "violation",
            pages=(3, 7),
        ),
        TemplateCase(
            "limitations section present",
            limitations,
            build_paper(sections=with_limitations),
            "pass",
        ),
        TemplateCase("limitations section missing", limitations, build_paper(), "violation"),
        TemplateCase("no acknowledgements", no_acknowledgements, build_paper(), "pass"),
        TemplateCase(
            "acknowledgements in the submission",
            no_acknowledgements,
            build_paper(sections=with_acknowledgements),
            "violation",
        ),
        TemplateCase("references present", references, build_paper(), "pass"),
        TemplateCase("no references", references, build_paper(bibliography_pages=()), "violation"),
        TemplateCase(
            "anonymity needs visual inspection",
            rule("anonymity"),
            build_paper(),
            "unverifiable",
        ),
    ]


def observed(result: TaskResult) -> Expected:
    """What a result concluded, in the terms of `TemplateCase.expected`."""
    if result.outcome is not TaskOutcome.CHECKED:
        return "unverifiable"
    return "violation" if result.findings else "pass"


def judge(case: TemplateCase, result: TaskResult) -> bool:
    """Whether the checker got the case right: same conclusion and, for a violation, one
    finding on the expected pages (or the document)."""
    if observed(result) != case.expected:
        return False
    if case.expected != "violation":
        return True
    expected = PageAnchor(pages=case.pages) if case.pages else DocumentAnchor()
    return len(result.findings) == 1 and result.findings[0].anchor == expected
