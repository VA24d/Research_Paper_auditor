"""Hand-checked citation links: draw a sample to annotate, and score the parser against it.

    uv run python scripts/citation_sample.py draw data/papers sample.jsonl
    uv run python scripts/citation_sample.py score data/papers \
        datasets/parser_fixtures/citation_gold.jsonl

`draw` picks markers at random (fixed seed) and writes, for each, the marker, its context
and the reference strings an annotator needs; the annotator fills `expected` with the
correct reference (first author surname, year as printed, start of the title) or null if the
marker cites nothing in the list. `score` finds each marker in the current parse and checks
whether its links point at the expected reference.
"""

import argparse
import asyncio
import json
import random
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

from auditor.config import Settings
from auditor.infra.http import HttpClient
from auditor.parser.grobid import GrobidClient
from auditor.parser.pipeline import parse_pdf
from auditor.schemas import BibEntry, ParsedPaper

PER_PAPER = 10
SEED = 20261011
CONTEXT = 60


async def parse_all(pdfs: list[Path]) -> dict[str, ParsedPaper]:
    settings = Settings()
    papers = {}
    async with HttpClient(settings) as http:
        grobid = GrobidClient(http, settings.grobid_url)
        for pdf in pdfs:
            papers[pdf.stem] = await parse_pdf(pdf, grobid=grobid, cache_dir=settings.cache_dir)
    return papers


def context(paper: ParsedPaper, sentence_id: str, start: int, end: int) -> str:
    text = paper.sentence(sentence_id).text
    return text[max(0, start - CONTEXT) : end + CONTEXT]


def draw(papers: dict[str, ParsedPaper], output: Path) -> None:
    rng = random.Random(SEED)
    lines = []
    for paper_id, paper in papers.items():
        mentions = list(paper.citations)
        for mention in sorted(
            rng.sample(mentions, min(PER_PAPER, len(mentions))), key=lambda m: m.id
        ):
            lines.append(
                {
                    "paper_id": paper_id,
                    "marker": mention.marker,
                    "context": context(paper, mention.sentence_id, mention.start, mention.end),
                    "expected": None,
                    "references": [entry.raw or entry.title for entry in paper.bibliography],
                }
            )
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")


def identity(entry: BibEntry) -> dict[str, str]:
    """First-author surname, year as printed (with any letter) and the title's first words."""
    surname = entry.authors[0].split()[-1] if entry.authors else ""
    printed = re.search(r"\b((?:19|20)\d{2}[a-z]?)\b", entry.raw or "")
    year = printed.group(1) if printed else str(entry.year or "")
    title = " ".join((entry.title or "").split()[:4])
    return {"surname": surname, "year": year, "title": title}


def freeze(papers: dict[str, ParsedPaper], annotated: Path, output: Path) -> None:
    """Turn reference positions chosen by the annotator into parser-independent answers."""
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for line in annotated.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            bibliography = papers[item["paper_id"]].bibliography
            expected = [identity(bibliography[position - 1]) for position in item["answer"]]
            record = {
                "paper_id": item["paper_id"],
                "marker": item["marker"],
                "context": item["context"],
                "expected": expected or None,
            }
            if item.get("note"):
                record["note"] = item["note"]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    letters = "".join(c for c in decomposed if not unicodedata.combining(c)).casefold()
    return re.sub(r"[^a-z0-9]+", " ", letters).strip()


def _is(entry: BibEntry, expected: dict[str, str]) -> bool:
    haystack = _fold(" ".join([entry.raw or "", entry.title or "", " ".join(entry.authors)]))
    return all(_fold(expected[key]) in haystack for key in ("surname", "year", "title"))


def _find(paper: ParsedPaper, item: dict[str, Any]) -> list[str] | None:
    """Linked entry ids of the sampled marker in the current parse, or None if not found."""
    for mention in paper.citations:
        if mention.marker != item["marker"]:
            continue
        found = context(paper, mention.sentence_id, mention.start, mention.end)
        if _fold(found) == _fold(item["context"]) or _fold(item["context"]) in _fold(found):
            return list(mention.bib_ids)
    return None


def score(papers: dict[str, ParsedPaper], gold: Path) -> dict[str, Any]:
    items = [json.loads(line) for line in gold.read_text(encoding="utf-8").splitlines() if line]
    linked = correct = missing = not_found = 0
    wrong: list[str] = []
    for item in items:
        paper = papers[item["paper_id"]]
        bib_ids = _find(paper, item)
        if bib_ids is None:
            not_found += 1
            continue
        expected = item["expected"]
        if not bib_ids:
            missing += 1
            continue
        linked += 1
        entries = [paper.bib_entry(b) for b in bib_ids]
        # Correct only when the linked entries are exactly the expected references.
        if (
            expected
            and all(any(_is(e, x) for x in expected) for e in entries)
            and all(any(_is(e, x) for e in entries) for x in expected)
        ):
            correct += 1
        else:
            wrong.append(f"{item['paper_id']} {item['marker']!r}")
    total = len(items) - not_found
    return {
        "items": len(items),
        "found_in_parse": total,
        "linked": linked,
        "correct_links": correct,
        "unlinked": missing,
        "precision": round(correct / linked, 3) if linked else None,
        "recall": round(correct / total, 3) if total else None,
        "wrong": wrong,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["draw", "freeze", "score"])
    parser.add_argument("papers", type=Path)
    parser.add_argument("file", type=Path)
    parser.add_argument("--out", type=Path, help="freeze: where to write the gold file")
    args = parser.parse_args()
    papers = asyncio.run(parse_all(sorted(args.papers.glob("*.pdf"))))
    if args.action == "draw":
        draw(papers, args.file)
    elif args.action == "freeze":
        if args.out is None:
            parser.error("freeze needs --out")
        freeze(papers, args.file, args.out)
    else:
        print(json.dumps(score(papers, args.file), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
