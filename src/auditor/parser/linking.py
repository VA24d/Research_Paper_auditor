"""Links citation markers GROBID left unresolved, where a simple rule is safe.

Two rules, both conservative:
- Numeric markers ("[12]", "[3-5]"), when the markers GROBID did resolve show that entry
  `b{n-1}` is reference number n.
- Author-year markers ("Smith et al., 2020"), when exactly one entry has that first author
  surname and year.
Mentions linked here are marked `LinkMethod.HEURISTIC`.
"""

import re
import unicodedata

from auditor.schemas import BibEntry, CitationMention, LinkMethod, ParsedPaper

# Ranges may use an en dash and surnames a typographic apostrophe, hence the escapes.
_NUMERIC_MARKER = re.compile(r"^[\[(]?\s*\d+(?:\s*[-\u2013,;]\s*\d+)*\s*[\])]?$")
_NUMBER_OR_RANGE = re.compile(r"(\d+)\s*[-\u2013]\s*(\d+)|(\d+)")
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})[a-z]?(?!\d)")
_SURNAME = re.compile(r"[^\W\d_][^\W\d_'\u2019-]*(?:[-'\u2019][^\W\d_]+)*")
_MAX_RANGE = 20
_MIN_EVIDENCE = 3
_MIN_AGREEMENT = 0.9


def link_unresolved(paper: ParsedPaper) -> ParsedPaper:
    unresolved = [m for m in paper.citations if not m.bib_ids]
    if not unresolved:
        return paper
    known = {entry.id for entry in paper.bibliography}
    numeric_ok = _numbering_matches_ids(paper.citations)
    updated: dict[str, CitationMention] = {}
    for mention in unresolved:
        bib_ids: tuple[str, ...] = ()
        if _NUMERIC_MARKER.match(mention.marker):
            if numeric_ok:
                numbers = _numbers(mention.marker)
                candidates = tuple(f"b{n - 1}" for n in numbers)
                if numbers and all(c in known for c in candidates):
                    bib_ids = candidates
        else:
            match = _author_year_match(mention.marker, paper.bibliography)
            if match is not None:
                bib_ids = (match,)
        if bib_ids:
            updated[mention.id] = mention.model_copy(
                update={"bib_ids": bib_ids, "link_method": LinkMethod.HEURISTIC}
            )
    if not updated:
        return paper
    citations = [updated.get(m.id, m) for m in paper.citations]
    data = paper.model_dump()
    data["citations"] = [c.model_dump() for c in citations]
    return ParsedPaper.model_validate(data)


def _numbers(marker: str) -> list[int]:
    numbers: list[int] = []
    for match in _NUMBER_OR_RANGE.finditer(marker):
        if match.group(3):
            numbers.append(int(match.group(3)))
            continue
        low, high = int(match.group(1)), int(match.group(2))
        if not 0 < high - low <= _MAX_RANGE:
            return []
        numbers.extend(range(low, high + 1))
    return [n for n in numbers if n >= 1]


def _numbering_matches_ids(citations: tuple[CitationMention, ...]) -> bool:
    """Whether resolved single-number markers say that reference n is entry b{n-1}."""
    agree = total = 0
    for mention in citations:
        if mention.link_method is not LinkMethod.PARSER or len(mention.bib_ids) != 1:
            continue
        if not _NUMERIC_MARKER.match(mention.marker):
            continue
        numbers = _numbers(mention.marker)
        if len(numbers) != 1:
            continue
        total += 1
        agree += mention.bib_ids[0] == f"b{numbers[0] - 1}"
    return total >= _MIN_EVIDENCE and agree / total >= _MIN_AGREEMENT


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()


def _first_author_surname(entry: BibEntry) -> str | None:
    if not entry.authors:
        return None
    parts = entry.authors[0].split()
    return _fold(parts[-1]) if parts else None


def _author_year_match(marker: str, bibliography: tuple[BibEntry, ...]) -> str | None:
    year_match = _YEAR.search(marker)
    surname_match = _SURNAME.search(marker)
    if year_match is None or surname_match is None:
        return None
    year = int(year_match.group(1))
    surname = _fold(surname_match.group(0))
    candidates = [
        entry.id
        for entry in bibliography
        if entry.year == year and _first_author_surname(entry) == surname
    ]
    return candidates[0] if len(candidates) == 1 else None
