"""Reference resolution (decision P2.2): from a bibliography entry or a marker to a work.

The order follows the identifiers the entry carries: a DOI is authoritative, an arXiv id
is next, and a title with authors is matched against the scholarly indexes. A resolution
that finds nothing says why, so the caller can report it instead of guessing.
"""

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from auditor.infra.http import HttpClient
from auditor.retrieval.clients import (
    Work,
    arxiv_by_id,
    arxiv_id_from_doi,
    arxiv_search,
    ascii_fold,
    crossref_by_doi,
    crossref_search,
    is_arxiv_id,
    is_doi,
    normalize_doi,
    openalex_by_doi,
    openalex_search,
    s2_search,
)
from auditor.schemas import BibEntry

logger = logging.getLogger(__name__)

# Words ignored when comparing titles.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
        "via",
        "using",
    ]
)


def title_tokens(title: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", title.lower())
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 1)


def title_text(title: str) -> str:
    """A lowercase alphanumeric form of a title, for containment checks."""
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))


def titles_match(a: str | None, b: str | None) -> bool:
    """Whether two titles are the same work, ignoring case, punctuation and stopwords."""
    if not a or not b:
        return False
    tokens_a, tokens_b = title_tokens(a), title_tokens(b)
    if not tokens_a or not tokens_b:
        return False
    overlap = len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))
    if overlap >= 0.7:
        return True
    # An index sometimes keeps only the main title and drops the subtitle ("NeRF" for
    # "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"): the
    # longer title starting with the shorter one is the same work.
    shorter, longer = sorted((title_text(a), title_text(b)), key=len)
    return len(shorter) >= 4 and longer.startswith(shorter)


def first_surname(author: str) -> str:
    """The surname of one author name, as written in a bibliography or an API reply.

    Tolerates both name orders, a trailing initial ("Vaswani A") and a trailing
    "et al." — none of them is the surname."""
    author = author.strip()
    if "," in author:
        return author.split(",", 1)[0].strip().lower()
    parts = [p for p in author.split() if p]
    while len(parts) > 1 and (
        parts[-1].strip(".").lower() in {"et", "al", "others"} or len(parts[-1].strip(".")) <= 1
    ):
        parts.pop()
    return parts[-1].strip(".").lower() if parts else ""


# Venue spellings that name the same venue. An index usually gives the proceedings' full
# name and a bibliography its short name, so an exact comparison would report a mismatch
# that is not one. Keys are the group's ids; every value lists acceptable spellings.
_VENUE_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    (
        "neurips",
        frozenset(
            {
                "neurips",
                "nips",
                "neural information processing systems",
                "advances in neural information processing systems",
                "conference on neural information processing systems",
                "advances in neural information processing systems 2026",
            }
        ),
    ),
    ("icml", frozenset({"icml", "international conference on machine learning"})),
    ("iclr", frozenset({"iclr", "international conference on learning representations"})),
    (
        "cvpr",
        frozenset(
            {
                "cvpr",
                "computer vision and pattern recognition",
                "ieee conference on computer vision and pattern recognition",
                "ieee cvf conference on computer vision and pattern recognition",
                "cvf conference on computer vision and pattern recognition",
            }
        ),
    ),
    (
        "iccv",
        frozenset(
            {
                "iccv",
                "international conference on computer vision",
                "ieee international conference on computer vision",
                "ieee cvf international conference on computer vision",
            }
        ),
    ),
    ("eccv", frozenset({"eccv", "european conference on computer vision"})),
    (
        "acl",
        frozenset(
            {
                "acl",
                "annual meeting of the association for computational linguistics",
                "meeting of the association for computational linguistics",
                "association for computational linguistics",
            }
        ),
    ),
    (
        "emnlp",
        frozenset(
            {
                "emnlp",
                "empirical methods in natural language processing",
                "conference on empirical methods in natural language processing",
                "empirical methods in natural language processing (emnlp)",
            }
        ),
    ),
    (
        "naacl",
        frozenset(
            {
                "naacl",
                "north american chapter of the association for computational linguistics",
                "conference of the north american chapter of the association for "
                "computational linguistics",
                "human language technologies",
                "conference on human language technologies",
            }
        ),
    ),
    ("aaai", frozenset({"aaai", "aaai conference on artificial intelligence"})),
    ("ijcai", frozenset({"ijcai", "international joint conference on artificial intelligence"})),
    (
        "kdd",
        frozenset(
            {
                "kdd",
                "knowledge discovery and data mining",
                "acm sigkdd international conference on knowledge discovery and data mining",
                "acm sigkdd conference on knowledge discovery and data mining",
            }
        ),
    ),
    ("jmlr", frozenset({"jmlr", "journal of machine learning research"})),
    ("tmlr", frozenset({"tmlr", "transactions on machine learning research"})),
    ("icra", frozenset({"icra", "international conference on robotics and automation"})),
    (
        "interspeech",
        frozenset(
            {"interspeech", "conference of the international speech communication association"}
        ),
    ),
)

_VENUE_STOPWORDS = frozenset(
    [
        "conference",
        "international",
        "proceedings",
        "annual",
        "the",
        "of",
        "on",
        "and",
        "for",
        "in",
        "journal",
        "transactions",
        "ieee",
        "cvf",
        "acm",
        "sigkdd",
        "meeting",
        "chapter",
        "north",
        "american",
        "european",
    ]
)


def _venue_tokens(venue: str) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", venue.lower())
    return frozenset(w for w in words if w not in _VENUE_STOPWORDS and len(w) > 1)


def _venue_variants(venue: str) -> list[str]:
    """The normalised venue and a few folded forms an index or bibliography may use.

    Crossref gives "2023 IEEE/CVF Conference on ... (CVPR)" (a year in front, the short
    name in brackets behind); DBLP-style bibliographies write "Proceedings of the ...";
    the same venue appears without them.
    """
    normal = re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", venue.strip().lower())).strip()
    bases = {normal, re.sub(r"^proceedings of the ", "", normal)}
    variants: set[str] = set()
    for base in bases:
        without_year = re.sub(r"^(19|20)\d{2}\s+", "", base)
        variants.add(base)
        variants.add(without_year)
        variants.add(re.sub(r"\b\w+\s*$", "", without_year).strip())
    return [v for v in variants if v]


# Venue strings an index gives that say nothing about where the work was published: an
# arXiv preprint record, or an institutional repository's copy of it (OpenAlex sometimes
# holds the repository record instead of the proceedings one). They never contradict an
# entry's venue.
_NON_CONTRADICTING_VENUES = frozenset(
    {
        "arxiv",
        "arxiv cornell university",
        "corr",
        "arxiv org",
        "ucl discovery",
        "ucl discovery university college london",
        "escholarship",
        "inspire",
        "inspire hep",
        "zenodo",
        "ssrn",
        "dblp",
        "researchgate",
        "core",
        "semanticscholar",
    }
)


def venues_match(a: str | None, b: str | None) -> bool:
    """Whether two venue strings name the same venue.

    Exact after normalisation and folding (a leading year, a trailing short name in
    brackets, a leading "proceedings of the" are dropped), membership in a shared alias
    group (an index gives the proceedings' full name, a bibliography the short one), or
    token containment (each side keeps at least three content words, so "empirical
    methods natural language processing" matches "conference on empirical methods in
    natural language processing" without matching a different conference). A preprint or
    repository placeholder venue says nothing and never contradicts.
    """
    if not a or not b:
        return False
    variants_a, variants_b = _venue_variants(a), _venue_variants(b)
    for variants in (variants_a, variants_b):
        if any(variant in _NON_CONTRADICTING_VENUES for variant in variants):
            return True
    if variants_a and variants_b and set(variants_a) & set(variants_b):
        return True
    groups_a = {
        name
        for name, spellings in _VENUE_GROUPS
        if any(variant in spellings for variant in variants_a)
    }
    groups_b = {
        name
        for name, spellings in _VENUE_GROUPS
        if any(variant in spellings for variant in variants_b)
    }
    if groups_a & groups_b:
        return True
    tokens_a, tokens_b = _venue_tokens(variants_a[0]), _venue_tokens(variants_b[0])
    if len(tokens_a) < 3 or len(tokens_b) < 3:
        return False
    return tokens_a <= tokens_b or tokens_b <= tokens_a


@dataclass(frozen=True)
class Resolution:
    """The outcome of resolving one reference."""

    work: Work | None = None
    reason: str | None = None
    # What was looked up, for LOOKUP evidence: ("crossref doi:10.1234/x", "found"), ...
    queries: tuple[tuple[str, str], ...] = ()
    # The entry gave a DOI and neither Crossref nor OpenAlex knew it: on its face a
    # fabricated identifier, reported even when a title search later found a work.
    doi_unresolved: bool = False

    @property
    def found(self) -> bool:
        return self.work is not None


def _entry_matches(entry: BibEntry, work: Work) -> bool:
    """Whether `work` is plausibly the work `entry` cites."""
    if work.title and entry.title and not titles_match(entry.title, work.title):
        return False
    if entry.year and work.year and abs(entry.year - work.year) > 1:
        return False
    if entry.authors and work.authors:
        entry_surname = first_surname(entry.authors[0])
        work_surnames = {first_surname(a) for a in work.authors}
        if entry_surname and entry_surname not in work_surnames:
            return False
    return True


async def _search_candidates(
    http: HttpClient, query: str, entry: BibEntry, queries: list[tuple[str, str]]
) -> list[Work]:
    """The title-search candidates of `query` that plausibly match `entry`.

    The four indexes are tried in order and the loop stops at the first that yields a
    match; one dead or rate-limited index is skipped, not fatal."""
    candidates: list[Work] = []
    searches: list[tuple[str, Callable[[], Awaitable[list[Work]]]]] = [
        ("openalex", lambda: openalex_search(http, query)),
        ("crossref", lambda: crossref_search(http, query, entry.authors)),
        ("arxiv", lambda: arxiv_search(http, query)),
        ("s2", lambda: s2_search(http, query)),
    ]
    for name, search in searches:
        try:
            found = await search()
        except Exception as exc:  # one dead index must not sink the lookup
            logger.info("%s title search of %r failed: %s", name, query[:40], exc)
            queries.append((f"{name} title search", f"failed ({type(exc).__name__})"))
            continue
        queries.append((f"{name} title search", f"{len(found)} hits"))
        candidates.extend(w for w in found if _entry_matches(entry, w))
        if candidates:
            break
    return candidates


async def resolve_entry(http: HttpClient, entry: BibEntry) -> Resolution:
    """Resolve one bibliography entry to a work, trying identifiers before search."""
    queries: list[tuple[str, str]] = []
    doi_unresolved = False

    arxiv_id = entry.arxiv_id if entry.arxiv_id and is_arxiv_id(entry.arxiv_id) else None

    if entry.doi and is_doi(entry.doi):
        doi = normalize_doi(entry.doi)
        work = await crossref_by_doi(http, doi)
        queries.append((f"crossref doi:{doi}", "found" if work else "not found"))
        if work is None:
            work = await openalex_by_doi(http, doi)
            queries.append((f"openalex doi:{doi}", "found" if work else "not found"))
        if work is not None:
            return Resolution(work=work, queries=tuple(queries))
        # arXiv assigns its DOIs through DataCite, so Crossref and OpenAlex do not know
        # them; such a DOI is real, and the paper is looked up on arXiv instead of being
        # called fabricated.
        if arxiv_id is None:
            arxiv_id = arxiv_id_from_doi(doi)
            if arxiv_id is not None:
                queries.append((f"arxiv id from doi:{doi}", arxiv_id))
        if arxiv_id is None:
            doi_unresolved = True

    if arxiv_id:
        work = await arxiv_by_id(http, arxiv_id)
        queries.append((f"arxiv id:{arxiv_id}", "found" if work else "not found"))
        if work is not None:
            return Resolution(work=work, queries=tuple(queries))

    title = entry.title or entry.raw
    if not title:
        return Resolution(
            reason="the entry has no DOI, arXiv id or title to look up",
            queries=tuple(queries),
        )

    # Queries go out ASCII-folded: bibliographies hold accents and mojibake the search
    # APIs do not match ("DemoirΘing" finds nothing; "Demoiring" does).
    query = ascii_fold(title)
    candidates = await _search_candidates(http, query, entry, queries)
    if not candidates:
        # A long subtitle can push the real title out of the query's useful window;
        # retry once with the first words only.
        words = re.findall(r"[a-z0-9]+", query.lower())
        if len(words) > 8:
            retry = " ".join(words[:8])
            queries.append(("shortened title retry", retry))
            candidates = await _search_candidates(http, retry, entry, queries)

    if candidates:
        # A search can return the arXiv preprint record of the proceedings paper the
        # entry cites (OpenAlex keeps both); prefer the candidate whose venue agrees
        # with the entry, then one whose year does, then the first that matched.
        best = max(
            candidates,
            key=lambda work: (
                venues_match(entry.venue, work.venue),
                entry.year is not None and work.year == entry.year,
            ),
        )
        return Resolution(work=best, queries=tuple(queries), doi_unresolved=doi_unresolved)
    return Resolution(
        reason="no work with this title was found in the scholarly indexes",
        queries=tuple(queries),
        doi_unresolved=doi_unresolved,
    )


@dataclass(frozen=True)
class MarkerName:
    """What an author-year marker says: surnames and a year, when it gives them."""

    surnames: tuple[str, ...] = ()
    year: int | None = None


_YEAR = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
_ET_AL = re.compile(r"([A-Z][A-Za-z'\-]+)(?:\s+et\s+al\.?)?")
_AND = re.compile(r"([A-Z][A-Za-z'\-]+)\s+and\s+([A-Z][A-Za-z'\-]+)")


def parse_marker(marker: str) -> MarkerName:
    """Pull surnames and a year out of an author-year marker such as
    "(Smith et al., 2020; Jones and Lee, 2019a)". Numeric markers give nothing."""
    surnames: list[str] = []
    for match in _ET_AL.finditer(marker):
        surnames.append(match.group(1).lower())
    for match in _AND.finditer(marker):
        surnames.extend(s.lower() for s in match.groups())
    year_match = _YEAR.search(marker)
    year = int(year_match.group(0)[:4]) if year_match else None
    seen: set[str] = set()
    unique: list[str] = []
    for surname in surnames:
        if surname not in seen:
            seen.add(surname)
            unique.append(surname)
    return MarkerName(surnames=tuple(unique), year=year)


def marker_is_author_year(marker: str) -> bool:
    """Whether the marker names authors (so it can be searched), not just a number."""
    return bool(parse_marker(marker).surnames)


def match_marker_to_bibliography(entries: tuple[BibEntry, ...], marker: str) -> BibEntry | None:
    """The bibliography entry an unlinked author-year marker most likely cites.

    The parser could not link the marker, but the entry is usually still there; matching
    surnames and the year against the entries recovers it without any network call.
    """
    parsed = parse_marker(marker)
    if not parsed.surnames:
        return None
    best: BibEntry | None = None
    best_score = 0
    for entry in entries:
        entry_surnames = {first_surname(a) for a in entry.authors if a}
        if not entry_surnames:
            # Fall back to the raw reference text, which holds the surnames too.
            raw = entry.raw or ""
            entry_surnames = {m.group(1).lower() for m in _ET_AL.finditer(raw)}
            entry_surnames |= {m.group(1).lower() for m in _AND.finditer(raw)}
        if not entry_surnames:
            continue
        overlap = len(set(parsed.surnames) & entry_surnames)
        if overlap == 0:
            continue
        score = overlap
        if parsed.year and entry.year == parsed.year:
            score += 2
        elif parsed.year and entry.year and abs(entry.year - parsed.year) > 1:
            continue
        if score > best_score:
            best, best_score = entry, score
    return best


async def resolve_marker(
    http: HttpClient, marker: str, entries: tuple[BibEntry, ...]
) -> Resolution:
    """Resolve an unlinked citation marker: match it to a bibliography entry, then
    resolve that entry. Numeric markers cannot be resolved this way."""
    entry = match_marker_to_bibliography(entries, marker)
    if entry is None:
        return Resolution(
            reason=(
                "the marker could not be linked to a bibliography entry and gives no "
                "author names to search for"
                if not marker_is_author_year(marker)
                else "no bibliography entry matches the marker's authors and year"
            )
        )
    resolution = await resolve_entry(http, entry)
    if resolution.work is None:
        reason = resolution.reason or "the matched entry could not be resolved"
        return Resolution(reason=reason, queries=resolution.queries)
    return resolution
