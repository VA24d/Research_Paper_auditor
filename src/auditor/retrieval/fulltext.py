"""Full-text retrieval for cited works (decision P2.3).

A work is retrieved at the best level available: the arXiv HTML full text, else its PDF
through the Part 1 parser (GROBID), else the abstract from the lookup, else nothing. The
level used is recorded, because the abstract-only ablation and the evaluation need to know
what the agent actually read. Retrieved text is cached on disk, so repeated runs of the
same paper do not re-download the sources.
"""

import hashlib
import html.parser
import logging
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from auditor.config import Settings
from auditor.infra.http import HttpClient
from auditor.parser.grobid import GrobidClient
from auditor.parser.pipeline import parse_pdf
from auditor.retrieval.clients import Work

logger = logging.getLogger(__name__)

MAX_HTML_BYTES = 5_000_000
MAX_PDF_BYTES = 50_000_000


class SourceLevel(StrEnum):
    """How much of a cited work the agent could read."""

    FULL_HTML = "full_html"
    FULL_PDF = "full_pdf"
    ABSTRACT = "abstract"
    NONE = "none"


@dataclass(frozen=True)
class SourceText:
    """The text of one cited work, and how much of it was read."""

    work: Work
    level: SourceLevel
    text: str = ""
    reason: str | None = None  # why a lower level was used, when it was
    # The text may not be the whole work: a publisher's landing page often holds only
    # the abstract, the references and the site's own text. What it lacks says nothing
    # about what the work states.
    partial: bool = False


class _TextExtractor(html.parser.HTMLParser):
    """HTML to plain text with the standard library only.

    Block-level tags start a new line, script and style contents are dropped, and
    whitespace runs are collapsed. Character offsets in the result are stable for a given
    input, which the passage offsets rely on.
    """

    _BLOCK = frozenset(
        [
            "address",
            "article",
            "aside",
            "blockquote",
            "br",
            "dd",
            "div",
            "dl",
            "dt",
            "fieldset",
            "figcaption",
            "figure",
            "footer",
            "form",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "header",
            "hr",
            "li",
            "main",
            "nav",
            "ol",
            "p",
            "pre",
            "section",
            "table",
            "tbody",
            "td",
            "tfoot",
            "th",
            "thead",
            "tr",
            "ul",
        ]
    )
    _SKIP = frozenset({"script", "style", "head", "noscript", "template"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        raw = "".join(self._parts)
        # Collapse whitespace inside lines (including &nbsp;), keep line structure.
        lines = (re.sub(r"[ \t\r\f\v\u00a0]+", " ", line).strip() for line in raw.split("\n"))
        return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html)
    extractor.close()
    return extractor.text()


def _cache_path(settings: Settings, work: Work, level: SourceLevel) -> Path:
    key = hashlib.sha256(f"{work.id}:{level.value}".encode()).hexdigest()[:24]
    return settings.cache_dir / "citation" / f"{key}.txt"


def _cached(settings: Settings, work: Work, level: SourceLevel) -> str | None:
    try:
        return _cache_path(settings, work, level).read_text(encoding="utf-8") or None
    except FileNotFoundError:
        return None


def _store(settings: Settings, work: Work, level: SourceLevel, text: str) -> None:
    from auditor.infra.files import atomic_write

    path = _cache_path(settings, work, level)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, text.encode("utf-8"))
    except OSError:
        logger.warning("could not cache the text of %s", work.id)


def _arxiv_html_url(work: Work) -> str | None:
    """The arXiv HTML full text of a work, when the work is on arXiv.

    The API's alternate link is the abstract page; the HTML rendering of the paper lives
    at /html/<id> (ar5iv-style, served by arXiv since 2023).
    """
    if work.arxiv_id:
        base = re.sub(r"v\d+$", "", work.arxiv_id)
        return f"https://arxiv.org/html/{base}"
    return work.html_url


async def _html_full_text(http: HttpClient, work: Work) -> str | None:
    url = _arxiv_html_url(work)
    if url is None:
        return None
    try:
        response = await http.get(url, max_bytes=MAX_HTML_BYTES)
    except Exception as exc:
        logger.info("HTML full text of %s unavailable: %s", work.id, exc)
        return None
    if not response.ok:
        return None
    text = html_to_text(response.text)
    # A stub page (no HTML rendering available) is short; require real content.
    return text if len(text) > 2000 else None


def _letters(text: str) -> str:
    """The letters and digits of `text` in any script, with accents composed the same way
    ("Café" written with a combining accent is the same title)."""
    return "".join(re.findall(r"\w", unicodedata.normalize("NFKC", text).casefold())).replace(
        "_", ""
    )


# A title shorter than this (in letters) is too common to identify a page on its own
# ("On", "BERT"): the abstract must be on the page too.
MIN_IDENTIFYING_TITLE = 12


def _is_publisher_page(work: Work) -> bool:
    """Whether the work's HTML is a publisher or repository page rather than arXiv's
    rendering of the paper itself."""
    return not work.arxiv_id


def _shows_the_work(text: str, work: Work) -> bool:
    """Whether a publisher page is about the work at all: it gives the work's title or
    the start of its abstract. A login, cookie or error page gives neither."""
    page = _letters(text)
    title = _letters(work.title or "")
    abstract = _letters(work.abstract or "")[:120]
    if len(abstract) >= 40 and abstract in page:
        return True
    return len(title) >= MIN_IDENTIFYING_TITLE and title in page


def grobid_for(settings: Settings, http: HttpClient) -> GrobidClient | None:
    """A GROBID client for the run's settings, or None when none is configured.

    Specialists and B1 tools build this per call: the client is stateless (an HTTP client
    plus a base URL) and the run context carries no parser state, so nothing needs to be
    threaded through the orchestrator."""
    if not settings.grobid_url:
        return None
    return GrobidClient(http, settings.grobid_url)


async def _pdf_full_text(
    http: HttpClient, work: Work, settings: Settings, grobid: GrobidClient | None
) -> str | None:
    if work.pdf_url is None or grobid is None:
        return None
    if not await grobid.is_alive():
        logger.info("PDF of %s skipped: GROBID is not running", work.id)
        return None
    digest = hashlib.sha256(work.pdf_url.encode()).hexdigest()[:24]
    destination = settings.scratch_dir / "citation" / f"{digest}.pdf"
    try:
        await http.download_to(work.pdf_url, destination, max_bytes=MAX_PDF_BYTES)
    except Exception as exc:
        logger.info("PDF of %s could not be downloaded: %s", work.id, exc)
        return None
    try:
        paper = await parse_pdf(destination, grobid=grobid, cache_dir=settings.cache_dir)
    except Exception as exc:
        logger.info("PDF of %s could not be parsed: %s", work.id, exc)
        return None
    return "\n".join(sentence.text for sentence in paper.sentences)


async def retrieve_source(
    http: HttpClient,
    work: Work,
    settings: Settings,
    *,
    grobid: GrobidClient | None = None,
    abstract_only: bool = False,
) -> SourceText:
    """The text of `work` at the best level available (or the abstract only).

    `abstract_only` is the ablation mode: the full text is never fetched even when it
    exists, so the evaluation can compare the two.
    """
    if not abstract_only:

        async def fetch_html() -> str | None:
            return await _html_full_text(http, work)

        async def fetch_pdf() -> str | None:
            return await _pdf_full_text(http, work, settings, grobid)

        for level, fetch in (
            (SourceLevel.FULL_HTML, fetch_html),
            (SourceLevel.FULL_PDF, fetch_pdf),
        ):
            # A publisher page is used only when it shows the work, and then as a partial
            # text: it may be a landing page with the abstract alone.
            publisher = level is SourceLevel.FULL_HTML and _is_publisher_page(work)
            cached = _cached(settings, work, level)
            if cached is not None:
                if publisher and not _shows_the_work(cached, work):
                    continue
                return SourceText(work=work, level=level, text=cached, partial=publisher)
            text = await fetch()
            if text and publisher and not _shows_the_work(text, work):
                logger.info("the page of %s does not show the work; not used", work.id)
                text = None
            if text:
                _store(settings, work, level, text)
                return SourceText(work=work, level=level, text=text, partial=publisher)

    if work.abstract:
        return SourceText(
            work=work,
            level=SourceLevel.ABSTRACT,
            text=work.abstract,
            reason=None if abstract_only else "no full text was available",
        )
    return SourceText(
        work=work,
        level=SourceLevel.NONE,
        reason="the work has no open full text and no abstract",
    )
