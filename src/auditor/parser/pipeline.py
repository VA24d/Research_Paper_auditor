"""PDF to ParsedPaper: GROBID for structure, pdfplumber for layout.

Only GROBID's TEI output is cached (by PDF hash), because that is the slow step. The
conversion and the citation linking run on every call, so improving them never needs a
cache bump; bump GROBID_REQUEST_VERSION only when the request sent to GROBID changes.
"""

import asyncio
import hashlib
import logging
from pathlib import Path

from pydantic import ValidationError

from auditor.infra.files import atomic_write
from auditor.parser.grobid import GrobidClient, GrobidError
from auditor.parser.layout import LayoutError, extract_layout
from auditor.parser.linking import link_unresolved
from auditor.parser.tei import TeiError, tei_to_paper
from auditor.schemas import Layout, Link, ParsedPaper

logger = logging.getLogger(__name__)

GROBID_REQUEST_VERSION = "1"


class ParseError(RuntimeError):
    """A PDF could not be turned into a ParsedPaper."""


def tei_cache_path(cache_dir: Path, pdf_sha256: str) -> Path:
    return cache_dir / "grobid" / f"{pdf_sha256}-r{GROBID_REQUEST_VERSION}.tei.xml"


async def _tei_for(
    path: Path, data: bytes, digest: str, grobid: GrobidClient, cache_file: Path | None
) -> tuple[str, bool]:
    """The TEI for a PDF, and whether it came from the cache."""
    if cache_file is not None:
        try:
            return await asyncio.to_thread(cache_file.read_text, encoding="utf-8"), True
        except FileNotFoundError:
            pass
    try:
        tei = await grobid.process_fulltext(data, path.name)
    except GrobidError as exc:
        raise ParseError(f"{path.name}: {exc}") from exc
    if cache_file is not None:
        atomic_write(cache_file, tei.encode("utf-8"))
    return tei, False


async def parse_pdf(
    path: Path,
    *,
    grobid: GrobidClient,
    cache_dir: Path | None = None,
    paper_id: str | None = None,
    refresh: bool = False,
) -> ParsedPaper:
    data = await asyncio.to_thread(path.read_bytes)
    digest = hashlib.sha256(data).hexdigest()
    cache_file = tei_cache_path(cache_dir, digest) if cache_dir is not None else None
    if refresh and cache_file is not None:
        cache_file.unlink(missing_ok=True)

    tei, cached = await _tei_for(path, data, digest, grobid, cache_file)

    layout: Layout | None = None
    links: list[Link] = []
    try:
        layout, links = await asyncio.to_thread(extract_layout, path)
    except LayoutError as exc:
        logger.warning("%s; falling back to page sizes from GROBID", exc)

    def convert(document: str) -> ParsedPaper:
        return tei_to_paper(
            document,
            paper_id=paper_id or path.stem,
            pdf_sha256=digest,
            layout=layout,
            extra_links=links,
        )

    try:
        paper = convert(tei)
    except (TeiError, ValidationError) as exc:
        if not cached or cache_file is None:
            raise ParseError(f"{path.name}: {exc}") from exc
        logger.warning("cached GROBID output for %s is unusable; parsing again", path.name)
        cache_file.unlink(missing_ok=True)
        tei, _ = await _tei_for(path, data, digest, grobid, cache_file)
        try:
            paper = convert(tei)
        except (TeiError, ValidationError) as retry_exc:
            raise ParseError(f"{path.name}: {retry_exc}") from retry_exc
    return link_unresolved(paper)
