"""Finds the paper's own code repository among the links it contains."""

import re

from auditor.schemas import ParsedPaper, RepositoryRef

_REPOSITORY = re.compile(
    r"^https?://(?:www\.)?(?P<host>github\.com|gitlab\.com)/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<name>[A-Za-z0-9_.-]+?)(?:\.git)?/?(?:[#?].*)?$"
)
_CODE_WORDS = re.compile(r"\b(code|implementation|repository|available|release[sd]?)\b", re.I)


def repository_url(url: str) -> str | None:
    """The canonical `https://host/owner/name` form of a repository link, if it is one."""
    match = _REPOSITORY.match(url.strip())
    if match is None:
        return None
    return f"https://{match['host']}/{match['owner']}/{match['name']}"


def find_repository(paper: ParsedPaper) -> RepositoryRef | None:
    """Prefer a repository mentioned next to words like "code is available"."""
    fallback = None
    for link in paper.links:
        url = repository_url(link.url)
        if url is None:
            continue
        if link.sentence_id is not None and _CODE_WORDS.search(
            paper.sentence(link.sentence_id).text
        ):
            return RepositoryRef(url=url)
        fallback = fallback or url
    return RepositoryRef(url=fallback) if fallback else None
