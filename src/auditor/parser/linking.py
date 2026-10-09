"""Links citation markers GROBID left unresolved, where a simple rule is safe.

Two rules, both conservative:
- Numeric markers ("[12]", "[3-5]"), when the markers GROBID did resolve show that entry
  `b{n-1}` is reference number n.
- Author-year markers ("Smith et al., 2020", "(Yao et al., 2020; Huang et al., 2022a)"),
  when every citation in the marker matches exactly one entry by first-author surname and
  year. A year with a letter ("2022a") must appear with that letter in the entry's raw
  reference text.
Mentions linked here are marked `LinkMethod.HEURISTIC`.
"""

import re
import unicodedata

from auditor.schemas import BibEntry, CitationMention, LinkMethod, ParsedPaper

# Ranges may use an en dash and surnames a typographic apostrophe, hence the escapes.
_NUMERIC_MARKER = re.compile(r"^[\[(]?\s*\d+(?:\s*[-\u2013,;]\s*\d+)*\s*[\])]?$")
_NUMBER_OR_RANGE = re.compile(r"(\d+)\s*[-\u2013]\s*(\d+)|(\d+)")
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})([a-z])?(?![\da-z])")
_SURNAME = re.compile(r"[^\W\d_][^\W\d_'\u2019-]*(?:[-'\u2019][^\W\d_]+)*")
_SURNAME_BEFORE_ETAL = re.compile(
    r"([^\W\d_][^\W\d_'\u2019-]*)\s+(?:et\s+al\b|and\b|&)", re.IGNORECASE
)
_LETTER_LIST = re.compile(r"((?:19|20)\d{2}[a-z]|;[a-z])\s*,\s*([a-z])\b")
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
            bib_ids = _author_year_links(mention.marker, paper.bibliography)
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


def _surname_in(text: str) -> str | None:
    """The surname just before "et al." / "and", else the last capitalised word."""
    named = _SURNAME_BEFORE_ETAL.search(text)
    if named:
        return named.group(1)
    words = _SURNAME.findall(text)
    return words[-1] if words else None


def _parts(marker: str) -> list[tuple[str, int, str | None]] | None:
    """Split "(Yao et al., 2020; Huang et al., 2022a;b)" into (surname, year, suffix)."""
    parts: list[tuple[str, int, str | None]] = []
    # "2022a,b" cites two works; rewrite it as "2022a;b" so each letter becomes a part.
    marker = _LETTER_LIST.sub(lambda m: m.group(1) + ";" + m.group(2), marker)
    marker = _LETTER_LIST.sub(lambda m: m.group(1) + ";" + m.group(2), marker)
    for raw_part in marker.split(";"):
        part = raw_part.strip(" ()[],")
        if not part:
            continue
        if re.fullmatch(r"[a-z]", part) and parts:
            previous_surname, previous_year, _ = parts[-1]
            parts.append((previous_surname, previous_year, part))
            continue
        year_match = _YEAR.search(part)
        if year_match is None:
            return None
        surname = _surname_in(part[: year_match.start()])
        if surname is None:
            return None
        parts.append((surname, int(year_match.group(1)), year_match.group(2)))
    return parts or None


def _match_part(
    surname: str, year: int, suffix: str | None, bibliography: tuple[BibEntry, ...]
) -> str | None:
    folded = _fold(surname)
    same_author = [e for e in bibliography if _first_author_surname(e) == folded]
    if suffix is not None:
        # "2018a" only links to an entry whose own reference text says "2018a". Parsed
        # years drop the letter, so matching on the year alone could pick the 2018b paper.
        label = re.compile(rf"\b{year}{suffix}\b")
        candidates = [e for e in same_author if e.raw and label.search(e.raw)]
    else:
        lettered = re.compile(rf"\b{year}[a-z]\b")
        candidates = [
            e for e in same_author if e.year == year and not (e.raw and lettered.search(e.raw))
        ]
    return candidates[0].id if len(candidates) == 1 else None


def _author_year_links(marker: str, bibliography: tuple[BibEntry, ...]) -> tuple[str, ...]:
    """Links for every part of the marker, or none if any part is unclear."""
    parts = _parts(marker)
    if parts is None:
        return ()
    links = []
    for surname, year, suffix in parts:
        bib_id = _match_part(surname, year, suffix, bibliography)
        if bib_id is None:
            return ()
        links.append(bib_id)
    return tuple(dict.fromkeys(links))
