"""Clients for the scholarly APIs the citation agent uses (decision P2.1).

Every request goes through the shared HTTP client (`auditor.infra.http`), so the host
allowlist, per-host rate limits, retries and the request log apply to all of them. Each
client returns a `Work` or None; network failures raise, so the caller decides what a
failed lookup means for the task at hand. None (or no hits) means the index answered and
has no such record; an index that could not answer (refused, rate-limited, down, or sent
a reply it does not serve) raises `LookupUnavailableError`, because that says nothing
about whether the work exists.
"""

import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

from auditor.infra.http import HttpClient, HttpError, HttpResponse

ARXIV_API = "https://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org/works"
CROSSREF_API = "https://api.crossref.org/works"
S2_API = "https://api.semanticscholar.org/graph/v1/paper"

# The DOI prefix arXiv assigns (DataCite, not Crossref/OpenAlex), e.g.
# 10.48550/arXiv.2310.09173 or .../arXiv.2310.09173v2.
_ARXIV_DOI = re.compile(r"^10\.48550/arxiv\.(\d{4}\.\d{4,5})(v\d+)?$", re.IGNORECASE)

_ARXIV_NS = "{http://www.w3.org/2005/Atom}"
_ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
_DOI = re.compile(r"^10\.\d{4,9}/\S+$")

# Statuses an index gives for a record it does not have (or an id it cannot parse). Any
# other failure (401, 403, 429, 5xx after the retries) means it could not answer.
_NO_RECORD = frozenset({400, 404, 410})


class LookupUnavailableError(HttpError):
    """An index could not answer the lookup, so its silence is not a "not found"."""


def _answered(response: HttpResponse) -> bool:
    """True when the index answered with a record, False when it answered that it has
    none; raises when it could not answer."""
    if response.ok:
        return True
    if response.status_code in _NO_RECORD:
        return False
    raise LookupUnavailableError(f"HTTP {response.status_code} for {response.url}")


def _json(response: HttpResponse) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise LookupUnavailableError(f"{response.url} sent a reply that is not JSON") from exc
    if not isinstance(data, dict):
        raise LookupUnavailableError(f"{response.url} sent JSON that is not an object")
    return data


def _part[T](data: dict[str, Any], key: str, kind: type[T], response: HttpResponse) -> T:
    """A part every reply of an index has; a reply without it is an error page, not an
    answer, so it says nothing about the work."""
    value = data.get(key)
    if not isinstance(value, kind):
        raise LookupUnavailableError(f"{response.url} sent a reply without its '{key}'")
    return value


def _records(items: list[Any]) -> list[dict[str, Any]]:
    return [item for item in items if isinstance(item, dict)]


@dataclass(frozen=True)
class Work:
    """One resolved cited work, however it was found."""

    id: str
    source: str  # "arxiv" | "openalex" | "crossref" | "s2"
    title: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    abstract: str | None = None
    pdf_url: str | None = None
    html_url: str | None = None
    # Extra identifiers the resolver may use to match against a bibliography entry.
    extra: dict[str, str] = field(default_factory=dict)


def normalize_doi(doi: str) -> str:
    """Lowercase, strip a leading URL and `doi:`."""
    doi = doi.strip().lower()
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi)
    return doi


def is_doi(text: str) -> bool:
    return _DOI.match(normalize_doi(text)) is not None


def is_arxiv_id(text: str) -> bool:
    return _ARXIV_ID.match(text.strip()) is not None


def arxiv_id_from_doi(doi: str) -> str | None:
    """The arXiv id behind an arXiv-assigned DOI (10.48550/arXiv.<id>), or None.

    These DOIs live in DataCite, so Crossref and OpenAlex do not know them; a resolver
    that only tries Crossref would call a real reference fabricated."""
    return match.group(1) if (match := _ARXIV_DOI.match(normalize_doi(doi))) else None


# Typography an index query should not carry: some bibliographies hold mojibake or
# accented letters that the search APIs do not match (a title with a Greek capital theta
# where an accented e belongs finds nothing).
_TYPOGRAPHY = str.maketrans(
    {
        "\u2019": "'",  # right single quotation mark
        "\u2018": "'",  # left single quotation mark
        "\u201c": '"',  # left double quotation mark
        "\u201d": '"',  # right double quotation mark
        "\u2013": "-",  # en dash
        "\u2014": "-",  # em dash
        "\u2026": "...",  # horizontal ellipsis
    }
)


def ascii_fold(text: str) -> str:
    """A search-query form of `text`: accents decomposed, typography and any letters an
    English query would not carry dropped."""
    folded = unicodedata.normalize("NFKD", text.translate(_TYPOGRAPHY))
    return "".join(c for c in folded if ord(c) < 128)


def short_title(text: str, *, words: int = 8) -> str:
    """The first `words` alphanumeric words of a folded title, for a retry query."""
    return " ".join(re.findall(r"[A-Za-z0-9]+", ascii_fold(text))[:words])


def _clean(text: str | None) -> str | None:
    if text is None:
        return None
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def _first_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


# --- arXiv -------------------------------------------------------------------


def _arxiv_entry_id(entry_id: str) -> str | None:
    match = re.search(r"arxiv\.org/abs/([^\s/?#]+)", entry_id)
    return match.group(1) if match else None


def _arxiv_links(entry: ET.Element) -> tuple[str | None, str | None]:
    pdf_url = html_url = None
    for link in entry.findall(f"{_ARXIV_NS}link"):
        rel = link.get("rel", "")
        href = link.get("href")
        if href is None:
            continue
        if rel == "related" and link.get("type") == "application/pdf":
            pdf_url = href
        elif rel == "alternate" and link.get("type") == "text/html":
            html_url = href
    return pdf_url, html_url


def _parse_arxiv_feed(xml_text: str) -> list[Work]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise LookupUnavailableError("arXiv sent a reply that is not an Atom feed") from exc
    if root.tag != f"{_ARXIV_NS}feed":
        raise LookupUnavailableError("arXiv sent a reply that is not an Atom feed")
    works = []
    for entry in root.findall(f"{_ARXIV_NS}entry"):
        entry_id = _arxiv_entry_id(entry.findtext(f"{_ARXIV_NS}id") or "")
        if entry_id is None:
            continue
        title = _clean(entry.findtext(f"{_ARXIV_NS}title"))
        abstract = _clean(entry.findtext(f"{_ARXIV_NS}summary"))
        authors = tuple(
            _clean(name.text) or "" for name in entry.findall(f"{_ARXIV_NS}author/{_ARXIV_NS}name")
        )
        authors = tuple(a for a in authors if a)
        published = entry.findtext(f"{_ARXIV_NS}published") or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        pdf_url, html_url = _arxiv_links(entry)
        works.append(
            Work(
                id=f"arxiv:{entry_id}",
                source="arxiv",
                title=title,
                authors=authors,
                year=year,
                arxiv_id=entry_id,
                abstract=abstract,
                pdf_url=pdf_url or f"https://arxiv.org/pdf/{entry_id}",
                html_url=html_url,
            )
        )
    return works


async def arxiv_by_id(http: HttpClient, arxiv_id: str) -> Work | None:
    """Look up one arXiv paper by its id (new or old form, version suffix allowed)."""
    response = await http.get(ARXIV_API, params={"id_list": arxiv_id.strip()})
    if not _answered(response):
        return None
    works = _parse_arxiv_feed(response.text)
    return works[0] if works else None


async def arxiv_search(http: HttpClient, query: str, *, limit: int = 3) -> list[Work]:
    """Search arXiv by title words; results are in relevance order."""
    response = await http.get(
        ARXIV_API,
        params={
            "search_query": f"ti:{query}",
            "start": 0,
            "max_results": limit,
        },
    )
    if not _answered(response):
        return []
    return _parse_arxiv_feed(response.text)


# --- OpenAlex ----------------------------------------------------------------


def _openalex_abstract(inverted: dict[str, list[int]] | None) -> str | None:
    """Rebuild the abstract from OpenAlex's inverted index."""
    if not inverted:
        return None
    words: list[str | None] = []
    for word, positions in inverted.items():
        for position in positions:
            while len(words) <= position:
                words.append(None)
            words[position] = word
    text = " ".join(w for w in words if w is not None)
    return _clean(text)


def _openalex_work(record: dict[str, object]) -> Work | None:
    work_id = record.get("id")
    if not isinstance(work_id, str) or not work_id:
        return None
    doi = record.get("doi")
    doi = normalize_doi(doi) if isinstance(doi, str) else None
    if doi and not is_doi(doi):
        doi = None
    authorships = record.get("authorships")
    authors: tuple[str, ...] = ()
    if isinstance(authorships, list):
        names = []
        for authorship in authorships:
            if not isinstance(authorship, dict):
                continue
            author = authorship.get("author")
            name = author.get("display_name") if isinstance(author, dict) else None
            if isinstance(name, str) and name:
                names.append(name)
        authors = tuple(names)
    location = record.get("primary_location")
    venue = pdf_url = html_url = None
    if isinstance(location, dict):
        source = location.get("source")
        if isinstance(source, dict) and isinstance(source.get("display_name"), str):
            venue = source["display_name"] or None
        pdf = location.get("pdf_url")
        landing = location.get("landing_page_url")
        if isinstance(pdf, str):
            pdf_url = pdf or None
        if isinstance(landing, str):
            html_url = landing or None
    raw_title = record.get("title")
    inverted = record.get("abstract_inverted_index")
    return Work(
        id=work_id,
        source="openalex",
        title=_clean(raw_title) if isinstance(raw_title, str) else None,
        authors=authors,
        year=_first_int(record.get("publication_year")),
        venue=venue,
        doi=doi,
        abstract=_openalex_abstract(inverted if isinstance(inverted, dict) else None),
        pdf_url=pdf_url,
        html_url=html_url,
    )


async def openalex_by_doi(http: HttpClient, doi: str) -> Work | None:
    response = await http.get(OPENALEX_API, params={"filter": f"doi:{normalize_doi(doi)}"})
    if not _answered(response):
        return None
    results = _records(_part(_json(response), "results", list, response))
    return _openalex_work(results[0]) if results else None


async def openalex_search(http: HttpClient, title: str, *, limit: int = 3) -> list[Work]:
    response = await http.get(
        OPENALEX_API,
        params={"filter": f"title.search:{title}", "per-page": limit},
    )
    if not _answered(response):
        return []
    results = _records(_part(_json(response), "results", list, response))
    return [w for r in results if (w := _openalex_work(r))]


# --- Crossref ----------------------------------------------------------------


def _crossref_abstract(raw: str | None) -> str | None:
    """Strip the JATS tags Crossref puts around abstracts."""
    if not raw:
        return None
    text = re.sub(r"<[^>]+>", " ", raw)
    return _clean(text)


def _crossref_work(record: dict[str, object]) -> Work | None:
    doi = record.get("DOI")
    if not isinstance(doi, str) or not doi:
        return None
    titles = record.get("title")
    title = (
        titles[0] if isinstance(titles, list) and titles and isinstance(titles[0], str) else None
    )
    containers = record.get("container-title")
    venue = (
        containers[0]
        if isinstance(containers, list) and containers and isinstance(containers[0], str)
        else None
    )
    authors: list[str] = []
    raw_authors = record.get("author")
    if isinstance(raw_authors, list):
        for author in raw_authors:
            if not isinstance(author, dict):
                continue
            name = " ".join(
                part
                for part in (author.get("given"), author.get("family"))
                if isinstance(part, str) and part
            )
            if name:
                authors.append(name)
    year = None
    issued = record.get("issued")
    if isinstance(issued, dict):
        parts = issued.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            year = _first_int(parts[0][0])
    raw_abstract = record.get("abstract")
    return Work(
        id=f"doi:{normalize_doi(doi)}",
        source="crossref",
        title=_clean(title),
        authors=tuple(authors),
        year=year,
        venue=_clean(venue),
        doi=normalize_doi(doi),
        abstract=_crossref_abstract(raw_abstract if isinstance(raw_abstract, str) else None),
    )


async def crossref_by_doi(http: HttpClient, doi: str) -> Work | None:
    response = await http.get(f"{CROSSREF_API}/{normalize_doi(doi)}")
    if not _answered(response):
        return None
    return _crossref_work(_part(_json(response), "message", dict, response))


async def crossref_search(
    http: HttpClient, title: str, authors: tuple[str, ...] = (), *, limit: int = 3
) -> list[Work]:
    params: dict[str, object] = {
        "query.bibliographic": title,
        "rows": limit,
        "select": "DOI,title,author,issued,container-title,abstract",
    }
    if authors:
        params["query.author"] = " ".join(authors[:3])
    response = await http.get(CROSSREF_API, params=params)
    if not _answered(response):
        return []
    message = _part(_json(response), "message", dict, response)
    items = _records(_part(message, "items", list, response))
    return [w for r in items if isinstance(r, dict) and (w := _crossref_work(r))]


# --- Semantic Scholar --------------------------------------------------------


_S2_FIELDS = "title,authors,year,venue,abstract,externalIds,openAccessPdf,publicationVenue"


def _s2_work(record: dict[str, object]) -> Work | None:
    paper_id = record.get("paperId")
    if not isinstance(paper_id, str) or not paper_id:
        return None
    external = record.get("externalIds")
    doi = arxiv_id = None
    if isinstance(external, dict):
        raw_doi = external.get("DOI")
        if isinstance(raw_doi, str) and is_doi(raw_doi):
            doi = normalize_doi(raw_doi)
        raw_arxiv = external.get("ArXiv")
        if isinstance(raw_arxiv, str) and is_arxiv_id(raw_arxiv):
            arxiv_id = raw_arxiv
    authors: list[str] = []
    raw_authors = record.get("authors")
    if isinstance(raw_authors, list):
        for author in raw_authors:
            if isinstance(author, dict) and isinstance(author.get("name"), str):
                authors.append(author["name"])
    pdf_url = None
    open_access = record.get("openAccessPdf")
    if isinstance(open_access, dict) and isinstance(open_access.get("url"), str):
        pdf_url = open_access["url"] or None
    venue = record.get("venue")
    if not isinstance(venue, str) or not venue:
        publication_venue = record.get("publicationVenue")
        if isinstance(publication_venue, dict) and isinstance(publication_venue.get("name"), str):
            venue = publication_venue["name"]
    raw_title = record.get("title")
    raw_abstract = record.get("abstract")
    return Work(
        id=f"s2:{paper_id}",
        source="s2",
        title=_clean(raw_title) if isinstance(raw_title, str) else None,
        authors=tuple(authors),
        year=_first_int(record.get("year")),
        venue=_clean(venue) if isinstance(venue, str) else None,
        doi=doi,
        arxiv_id=arxiv_id,
        abstract=_clean(raw_abstract) if isinstance(raw_abstract, str) else None,
        pdf_url=pdf_url,
    )


async def s2_by_identifier(
    http: HttpClient, *, doi: str | None = None, arxiv_id: str | None = None
) -> Work | None:
    """Look up one paper by DOI or arXiv id (S2 accepts both as the path id)."""
    if doi:
        path_id = f"DOI:{normalize_doi(doi)}"
    elif arxiv_id:
        path_id = f"arXiv:{arxiv_id}"
    else:
        return None
    response = await http.get(f"{S2_API}/{path_id}", params={"fields": _S2_FIELDS})
    if not _answered(response):
        return None
    data = _json(response)
    _part(data, "paperId", str, response)
    return _s2_work(data)


async def s2_search(http: HttpClient, title: str, *, limit: int = 3) -> list[Work]:
    response = await http.get(
        f"{S2_API}/search",
        params={"query": title, "limit": limit, "fields": _S2_FIELDS},
    )
    if not _answered(response):
        return []
    reply = _json(response)
    if "data" not in reply and isinstance(reply.get("total"), int):
        return []  # a search with no hits gives its total and no "data"
    return [w for r in _records(_part(reply, "data", list, response)) if (w := _s2_work(r))]
