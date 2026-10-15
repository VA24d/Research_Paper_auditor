"""Deterministic venue rule checks over a ParsedPaper. No LLM calls.

Each check is a pure function of the paper and the rule's `params`, which are validated
against a small Pydantic model first. A check returns a `RuleReport`: what it measured, the
violations it found, or why it could not decide (decision D6).

Margins use the same measurements as the paper rendering for B0 and B1
(`auditor.baselines.common`): left = `text_left`, top = `text_top`,
right = `width - text_right`, bottom = `height - text_bottom`, in PDF points.
"""

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_validator,
    model_validator,
)

from auditor.schemas import Layout, PageLayout, ParsedPaper

# Section titles (regular expressions, matched case-insensitively) that do not count toward
# a page limit unless a rule says otherwise.
DEFAULT_EXCLUDED_SECTIONS = (
    "references",
    "bibliography",
    "appendix",
    "appendices",
    "acknowledg",
    "checklist",
    "supplementary material",
)
SIDES = ("left", "right", "top", "bottom")
FONT_TOLERANCE = 0.1
POINT_TOLERANCE = 2.0

_LETTER_NUMBER = re.compile(r"^[A-Z](\.\d+)*\.?$")
_DIGIT_NUMBER = re.compile(r"^\d")
# Links that name an account on a code host, e.g. github.com/<owner>.
_ACCOUNT_LINK = re.compile(
    r"https?://(?:www\.)?(github\.com|gitlab\.com|huggingface\.co|bitbucket\.org)/([\w.-]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Violation:
    """One way a rule is broken. `pages` is empty for a document-level problem."""

    summary: str
    measured: str
    pages: tuple[int, ...] = ()
    uncertain: bool = False
    explanation: str = ""


@dataclass(frozen=True)
class RuleReport:
    """What a check measured, what it found, and why it could not decide (if it could not)."""

    measured: str
    violations: tuple[Violation, ...] = ()
    unverifiable: str | None = None

    @property
    def passed(self) -> bool:
        return self.unverifiable is None and not self.violations


class RuleParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _compile_all(patterns: tuple[str, ...]) -> tuple[str, ...]:
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"{pattern!r} is not a valid regular expression: {exc}") from exc
    return patterns


class MaxPagesParams(RuleParams):
    max_pages: int = Field(gt=0)
    exclude_sections: tuple[str, ...] = DEFAULT_EXCLUDED_SECTIONS

    @field_validator("exclude_sections")
    @classmethod
    def _valid_patterns(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        return _compile_all(patterns)


class FontSizeParams(RuleParams):
    min_size: float = Field(gt=0)
    tolerance: float = Field(default=FONT_TOLERANCE, ge=0)


class PaperSizeParams(RuleParams):
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    tolerance: float = Field(default=POINT_TOLERANCE, ge=0)
    allow_landscape: bool = True


class MarginParams(RuleParams):
    min_left: float | None = Field(default=None, ge=0)
    min_right: float | None = Field(default=None, ge=0)
    min_top: float | None = Field(default=None, ge=0)
    min_bottom: float | None = Field(default=None, ge=0)
    tolerance: float = Field(default=POINT_TOLERANCE, ge=0)

    @model_validator(mode="after")
    def _some_side(self) -> Self:
        if all(self.minimum(side) is None for side in SIDES):
            raise ValueError("give at least one of min_left, min_right, min_top, min_bottom")
        return self

    def minimum(self, side: str) -> float | None:
        value: float | None = getattr(self, f"min_{side}")
        return value


class SectionParams(RuleParams):
    sections: tuple[str, ...] = Field(min_length=1)

    @field_validator("sections")
    @classmethod
    def _valid_patterns(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        return _compile_all(patterns)


class NoParams(RuleParams):
    pass


# --- measurements -------------------------------------------------------------------


@dataclass(frozen=True)
class MainText:
    """How many pages the main text spans and where it was found to end."""

    pages: int
    boundary: str
    located: bool


def _matches(title: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, title, re.IGNORECASE) for pattern in patterns)


def _excluded_sections(paper: ParsedPaper, patterns: tuple[str, ...]) -> dict[str, str]:
    """Ids and titles of sections that do not count. Sections numbered with a letter (A,
    B.2) are appendices when an earlier section has a digit number, which keeps Roman
    numerals such as "I" and "V" in the main text."""
    excluded: dict[str, str] = {}
    seen_digit = False
    for section in paper.sections:
        number = (section.number or "").strip()
        lettered = bool(_LETTER_NUMBER.match(number)) and seen_digit
        if _matches(section.title, patterns) or lettered:
            excluded[section.id] = section.title or number
        seen_digit = seen_digit or bool(_DIGIT_NUMBER.match(number))
    return excluded


def main_text_pages(
    paper: ParsedPaper, exclude_sections: tuple[str, ...] = DEFAULT_EXCLUDED_SECTIONS
) -> MainText | None:
    """The last page holding text that counts toward a page limit, or None when no
    sentence has a page position.

    The main text is every sentence before the first sentence of an excluded section
    (references, appendix, ...), in document order. Pages after the first page of the
    reference list do not count either, which catches appendices the parser did not label.
    """
    if not any(sentence.boxes for sentence in paper.sentences):
        return None
    excluded = _excluded_sections(paper, exclude_sections)
    bib_pages = [box.page for entry in paper.bibliography for box in entry.boxes]
    references_start = min(bib_pages) if bib_pages else None
    boundary: str | None = None
    last_page = 0
    for sentence in paper.sentences:
        if sentence.section_id in excluded:
            boundary = f"before section '{excluded[sentence.section_id]}'"
            break
        for box in sentence.boxes:
            if references_start is None or box.page <= references_start:
                last_page = max(last_page, box.page)
    if references_start is not None:
        note = f"the references start on page {references_start}"
        boundary = f"{boundary}; {note}" if boundary else f"before the references ({note})"
    located = boundary is not None
    if boundary is None:
        boundary = "no references or excluded sections were found, so every page with text counts"
    return MainText(pages=last_page, boundary=boundary, located=located)


def dominant_font_size(layout: Layout) -> tuple[float, float] | None:
    """The font size with the most characters and its share of all characters."""
    by_size: Counter[float] = Counter()
    for font in layout.fonts:
        by_size[round(font.size, 1)] += font.chars
    total = sum(by_size.values())
    if not total:
        return None
    size, chars = by_size.most_common(1)[0]
    return size, chars / total


def page_margin(page: PageLayout, side: str) -> float | None:
    """Distance from the page edge to the nearest text, or None for a page without text."""
    match side:
        case "left":
            return page.text_left
        case "right":
            return None if page.text_right is None else page.width - page.text_right
        case "top":
            return page.text_top
        case _:
            return None if page.text_bottom is None else page.height - page.text_bottom


def _pages(pages: Iterable[int]) -> str:
    numbers = sorted(set(pages))
    return ("page " if len(numbers) == 1 else "pages ") + ", ".join(str(n) for n in numbers)


# --- checks ---------------------------------------------------------------------------

NO_LAYOUT = "The paper has no layout measurements (it was not parsed from the PDF)."


def check_max_pages(paper: ParsedPaper, params: MaxPagesParams) -> RuleReport:
    limit = params.max_pages
    main = main_text_pages(paper, params.exclude_sections)
    if main is not None:
        measured = f"{main.pages} main-text pages (the main text ends {main.boundary})"
        if main.pages <= limit:
            return RuleReport(measured=measured)
        violation = Violation(
            summary=f"The main text runs to {main.pages} pages; the limit is {limit}.",
            measured=measured,
            uncertain=not main.located,
            explanation=""
            if main.located
            else "The end of the main text could not be located, so pages after it may "
            "have been counted.",
        )
        return RuleReport(measured=measured, violations=(violation,))
    if paper.layout is None:
        return RuleReport(
            measured="",
            unverifiable="Sentences have no page positions and the paper has no layout, "
            "so the pages cannot be counted.",
        )
    total = paper.layout.page_count
    measured = f"{total} pages in total (sentences have no page positions)"
    if total <= limit:
        return RuleReport(measured=measured)
    return RuleReport(
        measured=measured,
        unverifiable=f"The PDF has {total} pages, more than the limit of {limit}, but "
        "sentences have no page positions, so the main text cannot be told apart from "
        "references and appendices.",
    )


def check_min_body_font_size(paper: ParsedPaper, params: FontSizeParams) -> RuleReport:
    assert paper.layout is not None
    dominant = dominant_font_size(paper.layout)
    if dominant is None:
        return RuleReport(measured="", unverifiable="The layout has no font measurements.")
    size, share = dominant
    measured = f"body font size {size:g} pt ({100 * share:.0f}% of characters)"
    if size >= params.min_size - params.tolerance:
        return RuleReport(measured=measured)
    violation = Violation(
        summary=f"The body text is set in {size:g} pt; the minimum is {params.min_size:g} pt.",
        measured=measured,
    )
    return RuleReport(measured=measured, violations=(violation,))


def check_paper_size(paper: ParsedPaper, params: PaperSizeParams) -> RuleReport:
    assert paper.layout is not None

    def fits(page: PageLayout, width: float, height: float) -> bool:
        tolerance = params.tolerance
        return abs(page.width - width) <= tolerance and abs(page.height - height) <= tolerance

    wrong = [
        page
        for page in paper.layout.pages
        if not fits(page, params.width, params.height)
        and not (params.allow_landscape and fits(page, params.height, params.width))
    ]
    expected = f"{params.width:g} x {params.height:g}"
    if not wrong:
        return RuleReport(measured=f"every page is {expected} pt")
    sizes = sorted({(round(p.width), round(p.height)) for p in wrong})
    measured = (
        f"{_pages(p.number for p in wrong)} measure "
        + ", ".join(f"{w} x {h}" for w, h in sizes)
        + f" pt instead of {expected} pt"
    )
    violation = Violation(
        summary=f"{len(wrong)} page(s) are not the required {expected} pt paper size.",
        measured=measured,
        pages=tuple(p.number for p in wrong),
    )
    return RuleReport(measured=measured, violations=(violation,))


def check_margins(paper: ParsedPaper, params: MarginParams) -> RuleReport:
    assert paper.layout is not None
    problems: dict[int, list[str]] = {}
    for page in paper.layout.pages:
        for side in SIDES:
            minimum = params.minimum(side)
            margin = page_margin(page, side)
            if minimum is None or margin is None:
                continue
            if margin < minimum - params.tolerance:
                problems.setdefault(page.number, []).append(f"{side} {margin:.1f} < {minimum:g}")
    if not problems:
        return RuleReport(measured="no text crosses the margins")
    measured = "; ".join(f"page {n}: {', '.join(found)}" for n, found in problems.items())
    violation = Violation(
        summary=f"Text crosses the required margins on {_pages(problems)}.",
        measured=measured,
        pages=tuple(problems),
    )
    return RuleReport(measured=measured, violations=(violation,))


def check_required_sections(paper: ParsedPaper, params: SectionParams) -> RuleReport:
    if not paper.sections:
        return RuleReport(
            measured="", unverifiable="The parser found no section headings in the paper."
        )
    titles = [section.title for section in paper.sections]
    missing = [p for p in params.sections if not any(_matches(t, (p,)) for t in titles)]
    if not missing:
        return RuleReport(measured="every required section is present")
    measured = "no section title matches " + ", ".join(f"'{p}'" for p in missing)
    violation = Violation(
        summary="The paper lacks a required section: " + ", ".join(missing) + ".",
        measured=measured,
    )
    return RuleReport(measured=measured, violations=(violation,))


def check_forbidden_sections(paper: ParsedPaper, params: SectionParams) -> RuleReport:
    found = [s for s in paper.sections if _matches(s.title, params.sections)]
    if not found:
        return RuleReport(measured="no forbidden section is present")
    measured = "sections " + ", ".join(f"'{s.title}'" for s in found)
    violation = Violation(
        summary="The paper contains a section the venue does not allow: "
        + ", ".join(s.title for s in found)
        + ".",
        measured=measured,
    )
    return RuleReport(measured=measured, violations=(violation,))


def check_references_present(paper: ParsedPaper, params: NoParams) -> RuleReport:
    if paper.bibliography:
        return RuleReport(measured=f"{len(paper.bibliography)} bibliography entries")
    if any(_matches(s.title, ("references", "bibliography")) for s in paper.sections):
        return RuleReport(measured="a references section without parsed entries")
    measured = "no bibliography entries and no references section"
    violation = Violation(summary="The paper has no reference list.", measured=measured)
    return RuleReport(measured=measured, violations=(violation,))


def check_anonymity(paper: ParsedPaper, params: NoParams) -> RuleReport:
    """Always unverifiable: anonymity needs a person to look. Author names in the header
    and links to accounts on code hosts are kept as uncertain hints."""
    hints: list[Violation] = []
    authors = [a for a in paper.metadata.authors if "anonymous" not in a.lower()]
    if authors:
        hints.append(
            Violation(
                summary="The paper header appears to name its authors.",
                measured="authors " + ", ".join(authors),
                uncertain=True,
            )
        )
    texts = [link.url for link in paper.links] + [s.text for s in paper.sentences]
    accounts = sorted({f"{m[1].lower()}/{m[2]}" for t in texts for m in _ACCOUNT_LINK.finditer(t)})
    if accounts:
        hints.append(
            Violation(
                summary="The paper links to accounts on code hosts, which may identify "
                "the authors.",
                measured="links to " + ", ".join(accounts),
                uncertain=True,
            )
        )
    measured = "; ".join(h.measured for h in hints) or "no author names or account links found"
    return RuleReport(
        measured=measured,
        violations=tuple(hints),
        unverifiable="Anonymity needs visual inspection of the title page, acknowledgements, "
        "self-citations and linked material.",
    )


# --- dispatch -------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleCheck:
    params: type[RuleParams]
    run: Callable[[ParsedPaper, Any], RuleReport]
    needs_layout: bool = False


RULE_CHECKS: dict[str, RuleCheck] = {
    "max_pages": RuleCheck(MaxPagesParams, check_max_pages),
    "min_body_font_size": RuleCheck(FontSizeParams, check_min_body_font_size, True),
    "paper_size": RuleCheck(PaperSizeParams, check_paper_size, True),
    "margins": RuleCheck(MarginParams, check_margins, True),
    "required_sections": RuleCheck(SectionParams, check_required_sections),
    "forbidden_sections": RuleCheck(SectionParams, check_forbidden_sections),
    "references_present": RuleCheck(NoParams, check_references_present),
    "anonymity": RuleCheck(NoParams, check_anonymity),
}


def _parse(check: str, params: Mapping[str, JsonValue]) -> tuple[RuleCheck, RuleParams]:
    rule_check = RULE_CHECKS.get(check)
    if rule_check is None:
        known = ", ".join(sorted(RULE_CHECKS))
        raise ValueError(f"No deterministic check named '{check}' exists (known: {known}).")
    try:
        return rule_check, rule_check.params.model_validate(dict(params))
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in error['loc']) or 'params'}: {error['msg']}"
            for error in exc.errors()
        )
        raise ValueError(f"Invalid params for check '{check}': {problems}") from exc


def params_error(check: str, params: Mapping[str, JsonValue]) -> str | None:
    """Why a rule definition cannot be run, or None when it can."""
    try:
        _parse(check, params)
    except ValueError as exc:
        return str(exc)
    return None


def evaluate(paper: ParsedPaper, check: str, params: Mapping[str, JsonValue]) -> RuleReport:
    """Run one check. Unknown checks, bad params and missing layout are unverifiable."""
    try:
        rule_check, parsed = _parse(check, params)
    except ValueError as exc:
        return RuleReport(measured="", unverifiable=str(exc))
    if rule_check.needs_layout and paper.layout is None:
        return RuleReport(measured="", unverifiable=NO_LAYOUT)
    return rule_check.run(paper, parsed)


def describe_layout(paper: ParsedPaper) -> str:
    """The measurements the checks read, as text for B1's `layout_summary` tool."""
    lines: list[str] = []
    layout = paper.layout
    if layout is None:
        lines.append(f"No layout measurements. {NO_LAYOUT}")
    else:
        lines.append(f"Pages: {layout.page_count}")
        sizes = sorted({(round(p.width), round(p.height)) for p in layout.pages})
        lines.append("Page sizes (pt): " + ", ".join(f"{w} x {h}" for w, h in sizes))
        margins = []
        for side in SIDES:
            measured = [
                (margin, page.number)
                for page in layout.pages
                if (margin := page_margin(page, side)) is not None
            ]
            if measured:
                value, number = min(measured)
                margins.append(f"{side} {value:.1f} (p. {number})")
        if margins:
            lines.append("Smallest margins (pt): " + ", ".join(margins))
        dominant = dominant_font_size(layout)
        if dominant is not None:
            size, share = dominant
            lines.append(f"Body font size: {size:g} pt ({100 * share:.0f}% of characters)")
    main = main_text_pages(paper)
    if main is not None:
        lines.append(f"Main text: {main.pages} pages (ends {main.boundary})")
    titles = [" ".join(p for p in (s.number, s.title) if p) for s in paper.sections]
    lines.append("Sections: " + ("; ".join(titles) if titles else "none found"))
    lines.append(f"Bibliography entries: {len(paper.bibliography)}")
    return "\n".join(lines)
