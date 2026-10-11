"""Parse every PDF in a directory and print parser statistics as a Markdown table.

These are automatic measures (how much was extracted and linked). Correctness of the links
still needs the hand-checked sample described in task P1.10.

    uv run python scripts/parser_report.py data/papers
"""

import argparse
import asyncio
import sys
from pathlib import Path

from auditor.config import Settings
from auditor.infra.http import HttpClient
from auditor.parser.grobid import GrobidClient
from auditor.parser.pipeline import ParseError, parse_pdf
from auditor.planner.repository import find_repository
from auditor.planner.tables import paper_settings
from auditor.schemas import LinkMethod, ParsedPaper

HEADER = (
    "| Paper | Pages | Sentences | References | With title / year | Markers | "
    "Linked (parser + heuristic) | Settings | Repository |\n"
    "|---|---|---|---|---|---|---|---|---|"
)


def _percent(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "n/a"


def row(paper: ParsedPaper) -> str:
    entries = paper.bibliography
    titled = sum(1 for e in entries if e.title)
    dated = sum(1 for e in entries if e.year)
    markers = paper.citations
    by_parser = sum(1 for c in markers if c.link_method is LinkMethod.PARSER)
    by_rule = sum(1 for c in markers if c.link_method is LinkMethod.HEURISTIC)
    repository = find_repository(paper)
    return (
        f"| {paper.paper_id} | {paper.layout.page_count if paper.layout else '?'} "
        f"| {len(paper.sentences)} | {len(entries)} "
        f"| {_percent(titled, len(entries))} / {_percent(dated, len(entries))} "
        f"| {len(markers)} | {_percent(by_parser + by_rule, len(markers))} "
        f"({by_parser} + {by_rule}) | {len(paper_settings(paper))} "
        f"| {repository.url if repository else '-'} |"
    )


async def report(pdfs: list[Path], settings: Settings) -> int:
    failures = 0
    print(HEADER)
    async with HttpClient(settings) as http:
        grobid = GrobidClient(http, settings.grobid_url)
        if not await grobid.is_alive():
            print(f"GROBID is not reachable at {settings.grobid_url}", file=sys.stderr)
            return 1
        for pdf in pdfs:
            try:
                paper = await parse_pdf(pdf, grobid=grobid, cache_dir=settings.cache_dir)
            except ParseError as exc:
                print(f"| {pdf.stem} | parse failed: {exc} |")
                failures += 1
                continue
            print(row(paper))
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    pdfs = sorted(args.directory.glob("*.pdf"))
    return 1 if asyncio.run(report(pdfs, Settings())) else 0


if __name__ == "__main__":
    sys.exit(main())
