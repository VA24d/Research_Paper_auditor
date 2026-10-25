"""One-off: sample the metadata mismatches HALLMARK VALID entries get."""

import asyncio
import json
from pathlib import Path

from auditor.agents.citation.hallmark import bib_entry_for, check_entry, load_split
from auditor.config import Settings
from auditor.infra.context import RunContext
from auditor.infra.http import HttpClient
from auditor.schemas import BibliographyAnchor, BibTarget, TaskKind
from auditor.schemas.tasks import AuditTask


async def main() -> None:
    entries = load_split(Path("data/hallmark/test_public.jsonl"))
    flagged_keys = {
        row["bibtex_key"]
        for row in (
            json.loads(line)
            for line in Path("runs/hallmark/results.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row["label"] == "VALID" and row["flag"] == "reference_metadata_mismatch"
    }
    flagged = [e for e in entries if e.bibtex_key in flagged_keys][:12]
    settings = Settings()
    async with HttpClient(settings) as http:
        ctx = RunContext(settings=settings, paper=None, http=http, llm=None, logger=None)  # type: ignore[arg-type]
        for entry in flagged:
            from auditor.retrieval.resolve import resolve_entry

            resolution = await resolve_entry(ctx.http, bib_entry_for(entry))
            if resolution.work is None:
                print(entry.bibtex_key, "no work")
                continue
            work = resolution.work
            fields = []
            if entry.field("title") and work.title:
                fields.append(f"title entry={entry.field('title')[:60]!r} work={work.title[:60]!r}")
            if entry.field("year") and work.year:
                fields.append(f"year {entry.field('year')} vs {work.year}")
            if (entry.field("booktitle") or entry.field("journal")) and work.venue:
                fields.append(
                    f"venue {entry.field('booktitle') or entry.field('journal')!r} "
                    f"vs {work.venue!r}"
                )
            print(entry.bibtex_key, "work:", work.source, "|", "; ".join(fields) or "?")


asyncio.run(main())
