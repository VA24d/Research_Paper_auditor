"""Fetch the cited works' texts for every parsed paper of set B (P2.8).

    uv run python scripts/fetch_sources.py --papers data/set_b/papers \
        --out data/set_b/sources [--concurrency 3] [--limit 0]

For every paper, one JSON line per bibliography entry in
`<out>/<paper_id>.jsonl`: {paper_id, bib_id, level, source_text} (and, when nothing
resolved, a reason instead of text). Entries whose file already lists them are skipped,
so the run is resumable. The HTTP cache makes a re-run cheap. The per-host rate limits
of the shared client apply inside each paper's loop; `--concurrency` papers run at once.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from auditor.config import Settings
from auditor.infra.context import RunContext
from auditor.infra.http import HttpClient
from auditor.parser.grobid import GrobidClient
from auditor.retrieval.fulltext import retrieve_source
from auditor.retrieval.resolve import resolve_entry
from auditor.schemas import ParsedPaper


async def _fetch_paper(
    paper_path: Path,
    out_dir: Path,
    http: HttpClient,
    grobid: GrobidClient,
    settings: Settings,
    *,
    with_pdf: bool,
) -> tuple[str, int, int]:
    paper = ParsedPaper.model_validate_json(
        await asyncio.to_thread(paper_path.read_bytes)
    )
    out_path = out_dir / f"{paper.paper_id}.jsonl"
    done: dict[str, dict[str, object]] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[row["bib_id"]] = row
    ctx = RunContext(
        settings=settings,
        paper=paper,
        http=http,
        llm=None,
        logger=None,  # type: ignore[arg-type]
    )
    rows: list[dict[str, object]] = []
    number = 0
    for entry in paper.bibliography:
        if entry.id in done:
            continue
        number += 1
        if number % 5 == 0:
            print(f"{paper.paper_id}: {number} fetched", flush=True)
        resolution = await resolve_entry(ctx.http, entry)
        if not resolution.found or resolution.work is None:
            rows.append(
                {
                    "paper_id": paper.paper_id,
                    "bib_id": entry.id,
                    "level": "none",
                    "reason": resolution.reason,
                }
            )
            continue
        if not with_pdf and resolution.work.pdf_url is not None:
            # The PDF level downloads and GROBID-parses the whole PDF, far too slow for
            # a build; HTML and the abstract cover most arXiv works. Recorded as-is.
            work = resolution.work.__class__(**{**resolution.work.__dict__, "pdf_url": None})
            source = await retrieve_source(ctx.http, work, settings, grobid=grobid)
        else:
            source = await retrieve_source(
                ctx.http, resolution.work, settings, grobid=grobid
            )
        rows.append(
            {
                "paper_id": paper.paper_id,
                "bib_id": entry.id,
                "level": source.level.value,
                "source_text": source.text,
            }
        )
    if rows:
        with out_path.open("a", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return paper.paper_id, len(rows), len(done)


async def _main(args: argparse.Namespace) -> int:
    settings = Settings()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    gate = asyncio.Semaphore(args.concurrency)
    async with HttpClient(settings) as http:
        grobid = GrobidClient(http, settings.grobid_url)

        async def one(paper_path: Path) -> None:
            async with gate:
                paper_id, new, done = await _fetch_paper(
                    paper_path, out_dir, http, grobid, settings, with_pdf=args.with_pdf
                )
                print(f"{paper_id}: {new} new, {done} skipped", flush=True)

        papers = sorted(args.papers.glob("*.json"))
        if args.limit:
            papers = papers[: args.limit]
        await asyncio.gather(*(one(paper) for paper in papers))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--with-pdf",
        action="store_true",
        help="also try the PDF level (slow: full download + GROBID parse per work)",
    )
    args = parser.parse_args()
    return asyncio.run(_main(args))


if __name__ == "__main__":
    sys.exit(main())
