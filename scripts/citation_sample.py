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
import sys
from pathlib import Path

from auditor.config import Settings
from auditor.infra.http import HttpClient
from auditor.parser.accuracy import context, identity, score
from auditor.parser.grobid import GrobidClient
from auditor.parser.pipeline import parse_pdf
from auditor.schemas import ParsedPaper

PER_PAPER = 10
SEED = 20261011


async def parse_all(pdfs: list[Path]) -> dict[str, ParsedPaper]:
    settings = Settings()
    papers = {}
    async with HttpClient(settings) as http:
        grobid = GrobidClient(http, settings.grobid_url)
        for pdf in pdfs:
            papers[pdf.stem] = await parse_pdf(pdf, grobid=grobid, cache_dir=settings.cache_dir)
    return papers


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
        lines = args.file.read_text(encoding="utf-8").splitlines()
        print(json.dumps(score(papers, [json.loads(line) for line in lines if line]), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
