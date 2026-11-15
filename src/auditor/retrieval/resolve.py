"""Reference resolution (decision P2.2): from a bibliography entry or a marker to a work.

The order follows the identifiers the entry carries: a DOI is authoritative, an arXiv id
is next, and a title with authors is matched against the scholarly indexes. A resolution
that finds nothing says why, so the caller can report it instead of guessing.
"""

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from auditor.infra.http import HttpClient, HttpError
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


def _venue_word_list(venue: str) -> list[str]:
    """The content words of a venue, in the order they appear (abbreviation matching)."""
    words = re.findall(r"[a-z0-9]+", venue.lower())
    return [w for w in words if w not in _VENUE_STOPWORDS and len(w) > 1]


def _venue_raw_words(venue: str) -> list[str]:
    """All the words of a venue, in order, stopwords kept (abbreviation anchoring)."""
    return re.findall(r"[a-z0-9]+", venue.lower())


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
# arXiv preprint record, or a working-paper repository. Institutional repositories are
# matched by pattern (they are named "<X> (<University>)" and are far too many to list).
_NON_CONTRADICTING_VENUES = frozenset(
    {
        "arxiv",
        "arxiv cornell university",
        "corr",
        "arxiv org",
        "repec",
        "repec research papers in economics",
    }
)

_REPOSITORY_VENUE = re.compile(r"\([^()]*universit[a-zè]*[^()]*\)|\brepository\b", re.IGNORECASE)


def _is_repository_venue(venue: str) -> bool:
    """Whether the venue names an institutional repository's copy of a work.

    OpenAlex names them after the repository ("UCL Discovery (University College London)",
    "Digital Access to Scholarship at Harvard (DASH) (Harvard University)"), which says
    nothing about where the work was published."""
    return bool(_REPOSITORY_VENUE.search(venue))


def venues_match(a: str | None, b: str | None) -> bool:
    """Whether two venue strings name the same venue.

    Exact after normalisation and folding (a leading year, a trailing short name in
    brackets, a leading "proceedings of the" are dropped), membership in a shared alias
    group (an index gives the proceedings' full name, a bibliography the short one),
    ISO-4-style abbreviation or acronym ("Mach. Learn." for "Machine Learning", "JMLR"
    for "Journal of Machine Learning Research"), or token containment (each side keeps at
    least three content words, so "empirical methods natural language processing" matches
    "conference on empirical methods in natural language processing" without matching a
    different conference). A preprint or repository placeholder venue says nothing and
    never contradicts.
    """
    if not a or not b:
        return False
    if _is_repository_venue(a) or _is_repository_venue(b):
        return True
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
    if any(
        _venue_abbreviates(
            _venue_raw_words(x), _venue_raw_words(y), _venue_word_list(x), _venue_word_list(y)
        )
        or _venue_abbreviates(
            _venue_raw_words(y), _venue_raw_words(x), _venue_word_list(y), _venue_word_list(x)
        )
        for x in variants_a
        for y in variants_b
    ):
        return True
    tokens_a, tokens_b = _venue_word_list(variants_a[0]), _venue_word_list(variants_b[0])
    if len(tokens_a) < 3 or len(tokens_b) < 3:
        return False
    return tokens_a <= tokens_b or tokens_b <= tokens_a


def _venue_abbreviates(
    short_raw: list[str], long_raw: list[str], short_content: list[str], long_content: list[str]
) -> bool:
    """Whether `short_raw` is an abbreviation of `long_raw`: ISO-4 token prefixes
    ("trans mach learn res" for "transactions on machine learning research"), or a single
    acronym matching the content initials ("jmlr" for "journal machine learning
    research"). The prefix mapping runs on the raw words, so a stopword the long side
    dropped ("transactions") still anchors its abbreviation ("trans")."""
    if not short_raw or not long_raw:
        return False
    if len(short_content) == 1:
        return "".join(token[0] for token in long_content) == short_content[0]
    index = 0
    for token in short_raw:
        stem = token.rstrip(".")
        while index < len(long_raw) and not long_raw[index].startswith(stem):
            index += 1
        if index == len(long_raw):
            return False
        index += 1
    return True


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
    # Nothing was found because nothing could be asked: no index answered the title
    # search (all refused, rate-limited or down), or the entry gives nothing to look up.
    # That is not evidence that the work does not exist.
    unavailable: bool = False

    @property
    def found(self) -> bool:
        return self.work is not None


def _entry_matches(entry: BibEntry, work: Work) -> bool:
    """Whether `work` is plausibly the work `entry` cites."""
    if work.title and entry.title and not titles_match(entry.title, work.title):
        return False
    # One year of drift is normal. A bigger gap is still the same work when the title is
    # exactly the entry's (a bibliography with a wrong year, which the metadata check
    # then reports); only an uncertain title match is rejected on the year.
    year_off = (
        entry.year is not None
        and work.year is not None
        and abs(entry.year - work.year) > 1
        and title_text(entry.title or "") != title_text(work.title or "")
    )
    if year_off:
        return False
    if entry.authors and work.authors:
        entry_surname = first_surname(entry.authors[0])
        work_surnames = {first_surname(a) for a in work.authors}
        if entry_surname and entry_surname not in work_surnames:
            return False
    return True


async def _identifier_lookup(
    name: str, lookup: Callable[[], Awaitable[Work | None]], queries: list[tuple[str, str]]
) -> tuple[Work | None, bool]:
    """One lookup by identifier: the work, and whether the index answered at all."""
    try:
        work = await lookup()
    except HttpError as exc:
        logger.info("%s failed: %s", name, exc)
        queries.append((name, f"failed ({type(exc).__name__})"))
        return None, False
    queries.append((name, "found" if work else "not found"))
    return work, True


async def _search_candidates(
    http: HttpClient, query: str, entry: BibEntry, queries: list[tuple[str, str]]
) -> tuple[list[Work], bool]:
    """The title-search candidates of `query` that plausibly match `entry`, and whether
    any index answered.

    The four indexes are tried in order and the loop stops at the first that yields a
    match; one dead or rate-limited index is skipped, not fatal."""
    candidates: list[Work] = []
    answered = False
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
        answered = True
        queries.append((f"{name} title search", f"{len(found)} hits"))
        candidates.extend(w for w in found if _entry_matches(entry, w))
        if candidates:
            break
    return candidates, answered


async def resolve_entry(http: HttpClient, entry: BibEntry) -> Resolution:
    """Resolve one bibliography entry to a work, trying identifiers before search."""
    queries: list[tuple[str, str]] = []
    doi_unresolved = False

    arxiv_id = entry.arxiv_id if entry.arxiv_id and is_arxiv_id(entry.arxiv_id) else None

    if entry.doi and is_doi(entry.doi):
        doi = normalize_doi(entry.doi)
        work, crossref_answered = await _identifier_lookup(
            f"crossref doi:{doi}", lambda: crossref_by_doi(http, doi), queries
        )
        openalex_answered = True
        if work is None:
            work, openalex_answered = await _identifier_lookup(
                f"openalex doi:{doi}", lambda: openalex_by_doi(http, doi), queries
            )
        if work is not None:
            return Resolution(work=work, queries=tuple(queries))
        # arXiv assigns its DOIs through DataCite, so Crossref and OpenAlex do not know
        # them; such a DOI is real, and the paper is looked up on arXiv instead of being
        # called fabricated.
        if arxiv_id is None:
            arxiv_id = arxiv_id_from_doi(doi)
            if arxiv_id is not None:
                queries.append((f"arxiv id from doi:{doi}", arxiv_id))
        # Only two indexes that both answered "no such DOI" make it look fabricated; one
        # that could not answer says nothing.
        if arxiv_id is None and crossref_answered and openalex_answered:
            doi_unresolved = True

    arxiv_answered = False
    if arxiv_id:
        known_id = arxiv_id
        work, arxiv_answered = await _identifier_lookup(
            f"arxiv id:{known_id}", lambda: arxiv_by_id(http, known_id), queries
        )
        if work is not None:
            return Resolution(work=work, queries=tuple(queries))

    title = entry.title or entry.raw
    if not title:
        if arxiv_answered:
            # arXiv answered that it has no paper with this id: a real miss.
            return Resolution(
                reason=f"arXiv has no paper with the id {arxiv_id}",
                queries=tuple(queries),
                doi_unresolved=doi_unresolved,
            )
        return Resolution(
            reason=(
                "the identifier lookups failed and the entry has no title to search"
                if any(outcome.startswith("failed") for _, outcome in queries)
                else "the entry has no DOI, arXiv id or title to look up"
            ),
            queries=tuple(queries),
            doi_unresolved=doi_unresolved,
            unavailable=True,
        )

    # Queries go out ASCII-folded: bibliographies hold accents and mojibake the search
    # APIs do not match ("DemoirΘing" finds nothing; "Demoiring" does).
    query = ascii_fold(title)
    candidates, answered = await _search_candidates(http, query, entry, queries)
    if not candidates:
        # A long subtitle can push the real title out of the query's useful window;
        # retry once with the first words only.
        words = re.findall(r"[a-z0-9]+", query.lower())
        if len(words) > 8:
            retry = " ".join(words[:8])
            queries.append(("shortened title retry", retry))
            candidates, answered_retry = await _search_candidates(http, retry, entry, queries)
            answered = answered or answered_retry

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
    if not answered:
        return Resolution(
            reason="no scholarly index answered the title search",
            queries=tuple(queries),
            doi_unresolved=doi_unresolved,
            unavailable=True,
        )
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
