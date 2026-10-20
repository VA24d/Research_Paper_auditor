"""Citation lookup (decisions P2.1, P2.2): resolve a citation to a work and its metadata.

The lookup goes through the shared HTTP client, so the allowlist, rate limits and request
log apply. A citation is resolved by identifier first (DOI, arXiv id), then by title and
authors against the scholarly indexes. A lookup that finds nothing says why, so the agent
can report a clear "not found" instead of guessing.
"""

import re
from dataclasses import dataclass

from auditor.infra.http import HttpClient
from auditor.retrieval.clients import (
    Work,
    arxiv_by_id,
    arxiv_search,
    crossref_by_doi,
    crossref_search,
    normalize_doi,
    openalex_by_doi,
    openalex_search,
    s2_by_identifier,
    s2_search,
)

_DOI = re.compile(r"10\.\d{4,9}/\S+", re.IGNORECASE)
_ARXIV = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})(v\d+)?", re.IGNORECASE)


@dataclass(frozen=True)
class LookupResult:
    """The outcome of one citation lookup."""

    work: Work | None
    reason: str | None = None  # why nothing was found, when nothing was
    queries: tuple[str, ...] = ()  # what was looked up, for the evidence record

    @property
    def found(self) -> bool:
        return self.work is not None


def _extract_identifiers(text: str) -> tuple[str | None, str | None]:
    """The DOI and arXiv id a raw citation string carries, when it gives any."""
    doi_match = _DOI.search(text)
    doi = normalize_doi(doi_match.group(0)) if doi_match else None
    arxiv_match = _ARXIV.search(text)
    arxiv_id = arxiv_match.group(1) if arxiv_match else None
    return doi, arxiv_id


def _title_of(text: str) -> str:
    """The searchable part of a raw citation: the text before the first year or URL."""
    cut = re.split(r"https?://|\b(19|20)\d{2}\b", text)[0]
    return cut.strip(" .,;:()[]")


async def lookup_citation(http: HttpClient, citation: str) -> LookupResult:
    """Resolve one raw citation string (a marker, a reference entry, or a title).

    Identifiers win: a DOI or arXiv id in the text is looked up directly. Otherwise the
    text's title part is searched. Nothing found is a result, not an error.
    """
    doi, arxiv_id = _extract_identifiers(citation)
    queries: list[str] = []

    if doi:
        queries.append(f"crossref doi:{doi}")
        work = await crossref_by_doi(http, doi)
        if work is None:
            queries.append(f"openalex doi:{doi}")
            work = await openalex_by_doi(http, doi)
        if work is not None:
            return LookupResult(work=work, queries=tuple(queries))

    if arxiv_id:
        queries.append(f"arxiv id:{arxiv_id}")
        work = await arxiv_by_id(http, arxiv_id)
        if work is None:
            queries.append(f"s2 arxiv:{arxiv_id}")
            work = await s2_by_identifier(http, arxiv_id=arxiv_id)
        if work is not None:
            return LookupResult(work=work, queries=tuple(queries))

    title = _title_of(citation)
    if not title:
        return LookupResult(
            work=None,
            reason="the citation gives no DOI, arXiv id or title to look up",
            queries=tuple(queries),
        )

    for name, found in (
        ("openalex", await openalex_search(http, title)),
        ("crossref", await crossref_search(http, title)),
        ("arxiv", await arxiv_search(http, title)),
        ("s2", await s2_search(http, title)),
    ):
        queries.append(f"{name} title search ({len(found)} hits)")
        if found:
            return LookupResult(work=found[0], queries=tuple(queries))

    return LookupResult(
        work=None,
        reason="no work with this title was found in the scholarly indexes",
        queries=tuple(queries),
    )
