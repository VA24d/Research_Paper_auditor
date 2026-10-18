"""Download arXiv PDFs listed in a manifest into data/papers/ (decision D13).

Each manifest line gives a versioned arXiv id and, optionally, the SHA-256 of the PDF. A
file is saved as `<id without version>.pdf`, the paper id the parser and the gold sample
use. Existing files are checked against the hash rather than trusted, and a download whose
hash differs is not saved.

Requests go through the shared HTTP client, so arXiv's limit of one request every three
seconds and the response cache apply.

    uv run python scripts/fetch_arxiv.py datasets/parser_fixtures/arxiv_ids.txt
    uv run python scripts/fetch_arxiv.py ids.txt --source --out data/sources

With `--source`, the LaTeX e-print is fetched instead and unpacked into
`<out>/<paper id>/` (auditor.parser.latex.unpack_eprint); the main .tex file it prints is
an input for `auditor audit` that keeps the formulas (plan D7).
"""

import argparse
import asyncio
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from auditor.config import Settings
from auditor.infra.files import atomic_write
from auditor.infra.http import HttpClient
from auditor.parser.latex import LatexError, find_main, unpack_eprint

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class Entry:
    arxiv_id: str
    sha256: str | None

    @property
    def paper_id(self) -> str:
        return re.sub(r"v\d+$", "", self.arxiv_id).replace("/", "_")


def read_manifest(manifest: Path) -> list[Entry]:
    entries = []
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        sha256 = fields[1] if len(fields) > 1 and _SHA256.fullmatch(fields[1]) else None
        if not re.search(r"v\d+$", fields[0]):
            print(f"note: line {number}: {fields[0]} has no version, so arXiv serves the latest")
        entries.append(Entry(fields[0], sha256))
    return entries


def problem(entry: Entry, content: bytes) -> str | None:
    """Why the content is not the PDF the manifest describes, or None if it is."""
    if not content.startswith(b"%PDF"):
        return "not a PDF"
    if entry.sha256 is not None and hashlib.sha256(content).hexdigest() != entry.sha256:
        return "SHA-256 differs from the manifest"
    return None


async def fetch_sources(entries: list[Entry], out: Path, settings: Settings) -> int:
    failures = 0
    async with HttpClient(settings) as http:
        for entry in entries:
            folder = out / entry.paper_id
            if folder.exists():
                try:
                    print(f"have    {entry.arxiv_id}: {find_main(folder)}")
                    continue
                except LatexError:
                    pass
            response = await http.get(f"https://arxiv.org/e-print/{entry.arxiv_id}")
            if not response.ok:
                print(f"FAILED  {entry.arxiv_id}: HTTP {response.status_code}")
                failures += 1
                continue
            try:
                main = unpack_eprint(response.content, folder)
            except LatexError as exc:
                print(f"FAILED  {entry.arxiv_id}: {exc}")
                failures += 1
                continue
            print(f"fetched {entry.arxiv_id}: {main}")
    return failures


async def fetch(wanted: list[tuple[Entry, Path]], settings: Settings) -> int:
    failures = 0
    async with HttpClient(settings) as http:
        for entry, target in wanted:
            response = await http.get(f"https://arxiv.org/pdf/{entry.arxiv_id}")
            reason = f"HTTP {response.status_code}" if not response.ok else None
            reason = reason or problem(entry, response.content)
            if reason is not None:
                print(f"FAILED  {entry.arxiv_id}: {reason}")
                failures += 1
                continue
            atomic_write(target, response.content)
            print(f"fetched {entry.arxiv_id} ({len(response.content) // 1024} KB)")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, help="Directory for the PDFs (default: data/papers)")
    parser.add_argument("--source", action="store_true", help="Fetch the LaTeX e-print")
    args = parser.parse_args()
    settings = Settings()
    if args.source:
        out = args.out or settings.data_dir / "sources"
        out.mkdir(parents=True, exist_ok=True)
        return 1 if asyncio.run(fetch_sources(read_manifest(args.manifest), out, settings)) else 0
    out = args.out or settings.data_dir / "papers"
    out.mkdir(parents=True, exist_ok=True)
    wanted = []
    failures = 0
    for entry in read_manifest(args.manifest):
        target = out / f"{entry.paper_id}.pdf"
        if not target.exists():
            wanted.append((entry, target))
            continue
        reason = problem(entry, target.read_bytes())
        if reason is None:
            print(f"have    {entry.arxiv_id}")
        else:
            print(f"BAD     {target}: {reason}; delete it to download again")
            failures += 1
    failures += asyncio.run(fetch(wanted, settings))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
