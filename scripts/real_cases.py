"""Collect candidates for real errors in papers from OpenReview and arXiv comments.

A candidate is a paper plus a sentence, written by a reviewer, an area chair or the
authors, that names a concrete error. A person confirms each one by hand before it is used
to evaluate the auditor. Nothing downloaded is treated as instructions, and no model is called.

    uv run python scripts/real_cases.py arxiv --out data/real_cases/arxiv.jsonl
    uv run python scripts/real_cases.py openreview --out data/real_cases/openreview.jsonl

Requests go through the shared HTTP client (host allowlist, response cache, rate limits).
OpenReview answers anonymous API calls with a bot challenge at times; the script stops with
a message instead of working around it.
"""

import argparse
import asyncio
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from auditor.config import Settings
from auditor.infra.http import DEFAULT_MIN_INTERVALS, HttpClient, HttpError

OPENREVIEW_API = "https://api2.openreview.net"
OPENREVIEW_SITE = "https://openreview.net"
ARXIV_API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
QUOTE_LIMIT = 400

# Venues to read, per year. Desk rejections and withdrawals carry the shortest comments.
OPENREVIEW_VENUES = (
    "ICLR.cc/2024/Conference/Desk_Rejected_Submission",
    "ICLR.cc/2024/Conference/Withdrawn_Submission",
    "ICLR.cc/2025/Conference/Desk_Rejected_Submission",
    "ICLR.cc/2025/Conference/Withdrawn_Submission",
    "ICLR.cc/2026/Conference/Desk_Rejected_Submission",
    "ICLR.cc/2026/Conference/Withdrawn_Submission",
)

# Error type -> patterns that, in a comment, suggest it. The first matching type wins, so
# the specific ones come first.
TYPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "unfinished_or_placeholder_content",
        re.compile(
            r"placeholder|simulated results|results? (?:are|were) (?:fabricated|made up)|"
            r"as an ai language model|certainly, here is|todo\b|unfinished|"
            r"unresolved reference|\?\?",
            re.I,
        ),
    ),
    (
        "template_violation",
        re.compile(
            r"page limit|exceeds? the (?:page|length)|anonymi[sz]|de-?anonymi[sz]|"
            r"wrong template|modified (?:the )?template|margins?\b|formatting violation",
            re.I,
        ),
    ),
    (
        "reference_not_found",
        re.compile(
            r"hallucinat\w+ (?:references?|citations?)|fabricated (?:references?|citations?)|"
            r"non-?existent (?:references?|citations?|papers?)|fake (?:references?|citations?)",
            re.I,
        ),
    ),
    (
        "math_error",
        re.compile(
            r"(?:proof|lemma|theorem|proposition|corollary|derivation|equation)\b[^.]{0,80}"
            r"(?:wrong|incorrect|flaw|false|error|invalid|does not hold|not correct)|"
            r"(?:wrong|incorrect|flaw\w*|false|error|mistake) (?:in|with) (?:the )?"
            r"(?:proof|lemma|theorem|proposition|corollary|derivation)|"
            r"mathematical error|counter-?example to",
            re.I,
        ),
    ),
    (
        "statistical_reporting",
        re.compile(r"p-?values?|confidence interval|standard deviation|statistical error", re.I),
    ),
    (
        "code_defect",
        re.compile(r"(?:bug|error|mistake) in (?:the |our )?(?:code|implementation)", re.I),
    ),
    (
        "result_inconsistency",
        re.compile(
            r"(?:table|figure|fig\.|results?|numbers?|values?|accuracy|metric|experiments?)"
            r"\b[^.]{0,80}(?:incorrect|wrong|erroneous|inconsistent|do(?:es)? not match|"
            r"differ|error|mistake|bug)|"
            r"(?:incorrect|wrong|erroneous|inconsistent|mistake|error)\w* (?:in |with )?"
            r"(?:the )?(?:table|figure|results?|numbers?|reported|evaluation|experiments?)",
            re.I,
        ),
    ),
)

WITHDRAWAL = re.compile(
    r"withdr[ae]w\w*|retract\w*|correct(?:ed|ion)|errata|erratum|revis(?:ed|ion)|"
    r"hallucinat\w*|fabricat\w*|placeholder|admin note",
    re.I,
)
# Comments about these describe the version they sit on, not an earlier one.
SAME_VERSION = frozenset({"reference_not_found", "unfinished_or_placeholder_content"})
LOCATION = re.compile(
    r"\b(?:Table|Figure|Fig\.|Lemma|Theorem|Proposition|Corollary|Definition|Algorithm|"
    r"Equation|Eq\.|Section|Sec\.|Appendix|Remark|Claim)\s*\(?[A-Z]?\d+(?:[.\-]\d+)*[a-z]?\)?"
    r"|\bpage limit\b|\banonymity\b|\bplaceholder\b",
    re.I,
)
# Withdrawals for reasons other than an error do not interest us.
NON_ERROR = re.compile(
    r"copyright|publisher|license|accepted (?:at|to|for)|to appear|merged|superseded by|"
    r"replaced by|conference version|journal version|duplicate|submitted by mistake",
    re.I,
)


class Candidate(BaseModel):
    case_id: str
    source: str
    paper_title: str
    paper_url: str
    pdf_url: str
    error_type: str
    location: str
    evidence_quote: str
    evidence_url: str


def classify(text: str) -> str | None:
    """The error type a comment suggests, or None when it names no concrete error."""
    for error_type, pattern in TYPE_PATTERNS:
        if pattern.search(text):
            return error_type
    return None


def find_location(text: str) -> str:
    match = LOCATION.search(text)
    return match.group(0).strip() if match else ""


def clean(text: str, limit: int = QUOTE_LIMIT) -> str:
    """One line, at most `limit` characters, around the first suggestive phrase."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    hits = [m.start() for _, p in TYPE_PATTERNS if (m := p.search(text))]
    start = max(0, min(hits) - limit // 4) if hits else 0
    return text[start : start + limit].strip()


def versioned(arxiv_id: str) -> tuple[str, int]:
    base, _, number = arxiv_id.rpartition("v")
    return (base, int(number)) if number.isdigit() and base else (arxiv_id, 1)


def parse_arxiv(feed: str) -> Iterator[Candidate]:
    """Candidates from an arXiv Atom feed: entries whose comment names an error.

    The comment belongs to the newest version, so the flawed paper is the version before.
    """
    root = ET.fromstring(feed)
    for entry in root.findall(f"{ATOM}entry"):
        comment = (entry.findtext(f"{ARXIV_NS}comment") or "").strip()
        abs_url = (entry.findtext(f"{ATOM}id") or "").strip()
        title = re.sub(r"\s+", " ", entry.findtext(f"{ATOM}title") or "").strip()
        if (
            not comment
            or not abs_url
            or NON_ERROR.search(comment)
            or not WITHDRAWAL.search(comment)
        ):
            continue
        error_type = classify(comment)
        if error_type is None:
            continue
        base, number = versioned(abs_url.rsplit("/abs/", 1)[-1])
        if number < 2 and error_type not in SAME_VERSION:
            continue  # No earlier version to read.
        number = number if error_type in SAME_VERSION else number - 1
        yield Candidate(
            case_id=f"arxiv:{base}v{number}",
            source="arxiv",
            paper_title=title,
            paper_url=f"https://arxiv.org/abs/{base}",
            pdf_url=f"https://arxiv.org/pdf/{base}v{number}",
            error_type=error_type,
            location=find_location(comment),
            evidence_quote=clean(comment),
            evidence_url=f"https://arxiv.org/abs/{abs_url.rsplit('/abs/', 1)[-1]}",
        )


def _value(content: Mapping[str, Any], key: str) -> str:
    field = content.get(key)
    value = field.get("value") if isinstance(field, dict) else field
    return value if isinstance(value, str) else ""


def parse_openreview(
    submission: Mapping[str, Any], replies: Iterable[Mapping[str, Any]]
) -> Iterator[Candidate]:
    """Candidates from one submission and the notes of its forum.

    Only comments by the venue (desk rejection, withdrawal, decision) and official
    reviews are read, because only they come with a claim a person can check.
    """
    forum = str(submission.get("forum") or submission.get("id") or "")
    content = submission.get("content", {})
    title = _value(content, "title")
    if not forum or not title:
        return
    seen: set[str] = set()
    for note in replies:
        if note.get("id") == forum:
            continue
        invitations = " ".join(str(i) for i in note.get("invitations", []))
        if not re.search(
            r"Desk_Rejection|Withdrawal|Decision|Official_Review|Meta_Review", invitations
        ):
            continue
        note_content = note.get("content", {})
        text = " ".join(
            _value(note_content, key)
            for key in (
                "desk_reject_comments",
                "withdrawal_confirmation",
                "comment",
                "metareview",
                "weaknesses",
                "summary",
                "decision",
            )
        )
        error_type = classify(text)
        if error_type is None or str(note.get("id")) in seen:
            continue
        seen.add(str(note.get("id")))
        yield Candidate(
            case_id=f"openreview:{forum}:{note.get('id')}",
            source="openreview",
            paper_title=title,
            paper_url=f"{OPENREVIEW_SITE}/forum?id={forum}",
            pdf_url=f"{OPENREVIEW_SITE}/pdf?id={forum}",
            error_type=error_type,
            location=find_location(text),
            evidence_quote=clean(text),
            evidence_url=f"{OPENREVIEW_SITE}/forum?id={forum}&noteId={note.get('id')}",
        )


ARXIV_QUERIES = (
    'co:withdrawn AND co:"error in"',
    'co:withdrawn AND co:"errors in"',
    "co:withdrawn AND co:erroneous",
    'co:withdrawn AND co:"incorrect"',
    'co:withdrawn AND co:"flaw"',
    'co:withdrawn AND co:"bug"',
    'co:withdrawn AND co:"proof"',
    'co:withdrawn AND co:"table"',
    'co:withdrawn AND co:"lemma"',
    'co:withdrawn AND co:"mistake"',
    'co:"corrected" AND co:"table"',
    'co:"corrected" AND co:"error in"',
    'co:"fixed" AND co:"bug"',
    'co:"placeholder"',
    'co:"results" AND co:"incorrect"',
    'co:"hallucinated references"',
    'co:"hallucinated citations"',
    'co:"fabricated references"',
    'co:"fake references"',
    'co:"nonexistent references"',
    'co:"unfinished"',
    'co:"wrong version"',
    'co:"page limit"',
    'co:"to be replaced"',
)
ARXIV_CATEGORIES = "(cat:cs.* OR cat:stat.ML)"
ARXIV_MATH_QUERIES = ('co:withdrawn AND co:"error in" AND co:"proof"',)


async def collect_arxiv(http: HttpClient, pages: int, page_size: int) -> list[Candidate]:
    found: dict[str, Candidate] = {}
    jobs = [(f"({q}) AND {ARXIV_CATEGORIES}") for q in ARXIV_QUERIES]
    jobs += [f"({q}) AND cat:math.*" for q in ARXIV_MATH_QUERIES]
    for query in jobs:
        for page in range(pages):
            params = {
                "search_query": query,
                "start": page * page_size,
                "max_results": page_size,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            try:
                response = await http.get(ARXIV_API, params=params)
                response.raise_for_status()
                batch = list(parse_arxiv(response.text))
            except (HttpError, ET.ParseError) as error:
                print(f"arxiv: {query!r} page {page}: {error}", file=sys.stderr)
                break
            for candidate in batch:
                found.setdefault(candidate.case_id, candidate)
        print(f"arxiv: {len(found)} candidates after {query!r}", file=sys.stderr)
    return list(found.values())


async def collect_openreview(http: HttpClient, max_per_venue: int) -> list[Candidate]:
    found: list[Candidate] = []
    for venue in OPENREVIEW_VENUES:
        offset = 0
        while offset < max_per_venue:
            params = {"content.venueid": venue, "limit": 100, "offset": offset}
            response = await http.get(f"{OPENREVIEW_API}/notes", params=params)
            if response.status_code == 403 and "Challenge" in response.text:
                raise SystemExit(
                    "OpenReview answered with a bot challenge (HTTP 403 ChallengeRequiredError). "
                    "Not working around it; try again later or from another network."
                )
            response.raise_for_status()
            submissions = response.json().get("notes", [])
            for submission in submissions:
                forum = submission["id"]
                notes = await http.get(f"{OPENREVIEW_API}/notes", params={"forum": forum})
                notes.raise_for_status()
                found.extend(parse_openreview(submission, notes.json().get("notes", [])))
            if len(submissions) < 100:
                break
            offset += 100
        print(f"openreview: {len(found)} candidates after {venue}", file=sys.stderr)
    return found


def write_jsonl(path: Path, candidates: Iterable[Candidate]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for candidate in candidates:
            file.write(json.dumps(candidate.model_dump(), ensure_ascii=False) + "\n")
            count += 1
    return count


async def run(args: argparse.Namespace) -> int:
    settings = Settings(contact_email=args.contact) if args.contact else Settings()
    intervals = {**DEFAULT_MIN_INTERVALS, "api2.openreview.net": 0.5, "openreview.net": 0.5}
    async with HttpClient(settings, min_intervals=intervals) as http:
        if args.source == "arxiv":
            candidates = await collect_arxiv(http, args.pages, args.page_size)
        else:
            candidates = await collect_openreview(http, args.max_per_venue)
    print(f"wrote {write_jsonl(args.out, candidates)} candidates to {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("source", choices=("arxiv", "openreview"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contact", help="email added to the User-Agent")
    parser.add_argument("--pages", type=int, default=3, help="arXiv result pages per query")
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--max-per-venue", type=int, default=2000)
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
