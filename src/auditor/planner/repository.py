"""Finds the paper's own code repository among the links it contains.

Precision matters more than recall here: a wrong repository makes the code agent compare
the paper against someone else's code, while a missed one only skips the code checks
(and can be given with `--repo`). So a repository is taken only from a sentence that
announces code, such as "Our code is available at ..." or "Code: ...", never from a link
that merely appears somewhere in the paper.
"""

import re

from auditor.schemas import ParsedPaper, RepositoryRef

_REPOSITORY = re.compile(
    r"^https?://(?:www\.)?(?P<host>github\.com|gitlab\.com)/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/(?P<name>[A-Za-z0-9_.-]+?)(?:\.git)?(?:[/#?].*)?$"
)
# The part of a link after owner/name that names a snapshot: /tree/<ref>, /commit/<sha>
# or /blob/<ref>/..., with GitLab's extra "/-".
_SNAPSHOT = re.compile(r"^/(?:-/)?(?:tree|commit|blob)/(?P<ref>[^/#?]+)")
_NOT_OWNERS = {"orgs", "features", "topics", "sponsors", "marketplace", "about", "settings"}
_CODE_NOUN = re.compile(r"\b(codes?|implementations?|source|scripts?|software)\b", re.IGNORECASE)
_ANNOUNCEMENT = re.compile(
    r"\b(our|we|available|release[sd]?|open[- ]?sourced?|can be found)\b"
    r"|\b(code|implementation)s?\s*:",
    re.IGNORECASE,
)
# Sentences about reusing someone else's code, such as "based on the open-sourced
# implementation in https://github.com/...".
_REUSE = re.compile(
    r"\b(based on|built on|builds on|adapted from|borrowed from|taken from|forked from)\b",
    re.IGNORECASE,
)


def repository_url(url: str) -> str | None:
    """The canonical `https://host/owner/name` form of a repository link, if it is one."""
    match = _REPOSITORY.match(url.strip())
    if match is None or match["owner"].lower() in _NOT_OWNERS:
        return None
    return f"https://{match['host']}/{match['owner']}/{match['name']}"


def snapshot_ref(url: str) -> str | None:
    """The commit, tag or branch a repository link points at, if it names one."""
    match = _REPOSITORY.match(url.strip())
    if match is None:
        return None
    rest = url.strip()[match.end("name") :].removeprefix(".git")
    found = _SNAPSHOT.match(rest)
    return found["ref"] if found else None


def announces_code(sentence: str) -> bool:
    if _REUSE.search(sentence):
        return False
    return bool(_CODE_NOUN.search(sentence) and _ANNOUNCEMENT.search(sentence))


def find_repository(paper: ParsedPaper) -> RepositoryRef | None:
    for link in paper.links:
        url = repository_url(link.url)
        if url is None or link.sentence_id is None:
            continue
        if announces_code(paper.sentence(link.sentence_id).text):
            return RepositoryRef(url=url)
    return None
