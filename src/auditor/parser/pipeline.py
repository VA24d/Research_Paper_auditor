"""PDF to ParsedPaper: GROBID for structure, pdfplumber for layout, cached by content hash."""

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

# Bump when the conversion changes, so cached results from older code are not reused.
PARSER_VERSION = "1"


class ParseError(RuntimeError):
    """A PDF could not be turned into a ParsedPaper."""


def cache_path(cache_dir: Path, pdf_sha256: str) -> Path:
    return cache_dir / "parsed" / f"{pdf_sha256}-v{PARSER_VERSION}.json"


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
    wanted_id = paper_id or path.stem
    cached_file = cache_path(cache_dir, digest) if cache_dir is not None else None

    if cached_file is not None and not refresh:
        cached = _load(cached_file)
        if cached is not None:
            logger.info("using cached parse of %s", path.name)
            return cached.model_copy(update={"paper_id": wanted_id})

    try:
        tei = await grobid.process_fulltext(data, path.name)
    except GrobidError as exc:
        raise ParseError(f"{path.name}: {exc}") from exc

    layout: Layout | None = None
    links: list[Link] = []
    try:
        layout, links = await asyncio.to_thread(extract_layout, path)
    except LayoutError as exc:
        logger.warning("%s; falling back to page sizes from GROBID", exc)

    try:
        paper = tei_to_paper(
            tei, paper_id=wanted_id, pdf_sha256=digest, layout=layout, extra_links=links
        )
    except (TeiError, ValidationError) as exc:
        raise ParseError(f"{path.name}: {exc}") from exc
    paper = link_unresolved(paper)

    if cached_file is not None:
        atomic_write(cached_file, paper.model_dump_json(indent=1).encode("utf-8"))
    return paper


def _load(path: Path) -> ParsedPaper | None:
    try:
        return ParsedPaper.model_validate_json(path.read_bytes())
    except FileNotFoundError:
        return None
    except (OSError, ValidationError):
        logger.warning("ignoring unreadable parse cache %s", path)
        return None
