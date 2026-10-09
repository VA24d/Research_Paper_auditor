"""Command-line entry point: `uv run auditor --help`."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from auditor.config import Settings
from auditor.infra.files import atomic_write
from auditor.infra.http import HttpClient
from auditor.infra.log import configure_logging
from auditor.parser.grobid import GrobidClient
from auditor.parser.pipeline import ParseError, parse_pdf
from auditor.schemas import LinkMethod, ParsedPaper


def _summary(paper: ParsedPaper) -> str:
    linked = sum(1 for c in paper.citations if c.bib_ids)
    heuristic = sum(1 for c in paper.citations if c.link_method is LinkMethod.HEURISTIC)
    pages = paper.layout.page_count if paper.layout else "?"
    return (
        f"{paper.paper_id}: {pages} pages, {len(paper.sections)} sections, "
        f"{len(paper.sentences)} sentences, {len(paper.bibliography)} references, "
        f"{linked}/{len(paper.citations)} citation markers linked ({heuristic} by heuristic), "
        f"{len(paper.links)} links"
    )


async def _parse(args: argparse.Namespace, settings: Settings) -> int:
    async with HttpClient(settings) as http:
        grobid = GrobidClient(http, settings.grobid_url)
        if not await grobid.is_alive():
            print(
                f"GROBID is not reachable at {settings.grobid_url}. "
                "Start it with: docker compose up -d",
                file=sys.stderr,
            )
            return 2
        try:
            paper = await parse_pdf(
                args.pdf,
                grobid=grobid,
                cache_dir=settings.cache_dir,
                paper_id=args.paper_id,
                refresh=args.refresh,
            )
        except ParseError as exc:
            print(f"Could not parse: {exc}", file=sys.stderr)
            return 1
    if args.output is not None:
        atomic_write(args.output, (paper.model_dump_json(indent=1) + "\n").encode("utf-8"))
    print(_summary(paper))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="auditor", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    parse = commands.add_parser("parse", help="Parse a PDF into ParsedPaper JSON (needs GROBID)")
    parse.add_argument("pdf", type=Path)
    parse.add_argument("-o", "--output", type=Path, help="Write the ParsedPaper JSON here")
    parse.add_argument("--paper-id", help="Paper id to use (default: the file name)")
    parse.add_argument("--refresh", action="store_true", help="Ignore the parse cache")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    configure_logging(settings.log_level)
    if args.command == "parse":
        if not args.pdf.is_file():
            print(f"No such file: {args.pdf}", file=sys.stderr)
            return 2
        return asyncio.run(_parse(args, settings))
    return 2


if __name__ == "__main__":
    sys.exit(main())
