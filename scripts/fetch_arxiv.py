"""Download arXiv PDFs listed in a manifest into data/papers/ (decision D13).

Requests go through the shared HTTP client, so arXiv's limit of one request every three
seconds and the response cache apply.

    uv run python scripts/fetch_arxiv.py datasets/parser_fixtures/arxiv_ids.txt
"""

import argparse
import asyncio
import sys
from pathlib import Path

from auditor.config import Settings
from auditor.infra.files import atomic_write
from auditor.infra.http import HttpClient


def read_ids(manifest: Path) -> list[str]:
    ids = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("#", 1)[0].split()
        if fields:
            ids.append(fields[0])
    return ids


async def fetch(wanted: list[tuple[str, Path]], settings: Settings) -> int:
    failures = 0
    async with HttpClient(settings) as http:
        for arxiv_id, target in wanted:
            response = await http.get(f"https://arxiv.org/pdf/{arxiv_id}")
            if not response.ok or not response.content.startswith(b"%PDF"):
                print(f"FAILED  {arxiv_id}: HTTP {response.status_code}")
                failures += 1
                continue
            atomic_write(target, response.content)
            print(f"fetched {arxiv_id} ({len(response.content) // 1024} KB)")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, help="Directory for the PDFs (default: data/papers)")
    args = parser.parse_args()
    settings = Settings()
    out = args.out or settings.data_dir / "papers"
    out.mkdir(parents=True, exist_ok=True)
    wanted = []
    for arxiv_id in read_ids(args.manifest):
        target = out / f"{arxiv_id.replace('/', '_')}.pdf"
        if target.exists():
            print(f"have    {arxiv_id}")
        else:
            wanted.append((arxiv_id, target))
    failures = asyncio.run(fetch(wanted, settings))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
