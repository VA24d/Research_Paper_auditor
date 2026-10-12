"""Citation-link accuracy against a hand-checked sample (task P1.10).

Gold items name the expected reference by first-author surname, year as printed and the
start of the title, so they stay valid when the parser's entry ids change. A sampled marker
the current parse no longer contains counts as a miss, so losing citations lowers recall.
"""

import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

from auditor.schemas import BibEntry, ParsedPaper

CONTEXT = 60


def context(paper: ParsedPaper, sentence_id: str, start: int, end: int) -> str:
    text = paper.sentence(sentence_id).text
    return text[max(0, start - CONTEXT) : end + CONTEXT]


def identity(entry: BibEntry) -> dict[str, str]:
    """First-author surname, year as printed (with any letter) and the title's first words."""
    surname = entry.authors[0].split()[-1] if entry.authors else ""
    printed = re.search(r"\b((?:19|20)\d{2}[a-z]?)\b", entry.raw or "")
    year = printed.group(1) if printed else str(entry.year or "")
    title = " ".join((entry.title or "").split()[:4])
    return {"surname": surname, "year": year, "title": title}


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    letters = "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()
    return re.sub(r"[^a-z0-9]+", " ", letters).strip()


def _is(entry: BibEntry, expected: Mapping[str, str]) -> bool:
    fields = [entry.raw or "", entry.title or "", " ".join(entry.authors), str(entry.year or "")]
    haystack = _fold(" ".join(fields))
    return all(_fold(expected[key]) in haystack for key in ("surname", "year", "title"))


def _find(paper: ParsedPaper, item: Mapping[str, Any]) -> list[str] | None:
    """Linked entry ids of the sampled marker in the current parse, or None if not found."""
    for mention in paper.citations:
        if mention.marker != item["marker"]:
            continue
        found = context(paper, mention.sentence_id, mention.start, mention.end)
        if _fold(found) == _fold(item["context"]) or _fold(item["context"]) in _fold(found):
            return list(mention.bib_ids)
    return None


def _correct(entries: list[BibEntry], expected: Sequence[Mapping[str, str]] | None) -> bool:
    """Correct only when the linked entries are exactly the expected references."""
    if not expected:
        return not entries
    return all(any(_is(e, x) for x in expected) for e in entries) and all(
        any(_is(e, x) for e in entries) for x in expected
    )


def score(papers: Mapping[str, ParsedPaper], items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Precision over linked markers, and recall over every marker that cites a reference."""
    linked = correct_links = positives = 0
    wrong: list[str] = []
    unlinked: list[str] = []
    not_found: list[str] = []
    for item in items:
        name = f"{item['paper_id']} {item['marker']!r}"
        expected = item["expected"]
        positives += bool(expected)
        paper = papers[item["paper_id"]]
        bib_ids = _find(paper, item)
        if bib_ids is None:
            not_found.append(name)
            continue
        if not bib_ids:
            if expected:
                unlinked.append(name)
            continue
        linked += 1
        if _correct([paper.bib_entry(b) for b in bib_ids], expected):
            correct_links += 1
        else:
            wrong.append(name)
    return {
        "items": len(items),
        "citing_a_reference": positives,
        "linked": linked,
        "correct_links": correct_links,
        "precision": round(correct_links / linked, 3) if linked else None,
        "recall": round(correct_links / positives, 3) if positives else None,
        "wrong": wrong,
        "unlinked": unlinked,
        "not_found": not_found,
    }
