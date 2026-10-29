"""Reference checks (decision P2.5): does the bibliography entry exist and match?

A reference task is checked without a model. The entry is resolved against the scholarly
indexes, and its metadata is compared with the entry's fields field by field. A reference
that resolves to nothing is a `reference_not_found` finding; one that resolves to a
different work is a `reference_metadata_mismatch` finding.
"""

import re

from auditor.infra.context import RunContext
from auditor.retrieval.clients import Work
from auditor.retrieval.resolve import (
    Resolution,
    first_surname,
    title_text,
    title_tokens,
    titles_match,
    venues_match,
)
from auditor.schemas import (
    AuditTask,
    BibEntry,
    BibTarget,
    CandidateFinding,
    ErrorClass,
    Evidence,
    EvidenceKind,
    SourceKind,
    SourceRef,
    TaskKind,
    TaskOutcome,
    TaskResult,
)

# How an entry's author list ends when it knowingly does not name everyone.
_TRUNCATION = re.compile(r"(?:\b(?:et\s+al\.?|and\s+others|others)\s*)+[\.,]?\s*$", re.IGNORECASE)


def _lookup_evidence(entry: BibEntry, resolution: Resolution) -> Evidence:
    """The LOOKUP evidence for a reference check: what was looked up and what came back."""
    work = resolution.work
    identifier = work.id if work else entry.id
    return Evidence(
        kind=EvidenceKind.LOOKUP,
        source=SourceRef(kind=SourceKind.SCHOLARLY_INDEX, id=identifier),
        text=(
            f"Resolved '{entry.raw or entry.title}' "
            f"(queries: {', '.join(q for q, _ in resolution.queries) or 'none'}); "
            f"{'found' if resolution.found else 'not found: ' + (resolution.reason or 'unknown')}"
        ),
    )


def _author_problems(entry: BibEntry, work: Work) -> list[str]:
    """Problems with the entry's author list compared with the work's.

    A name the entry gives that no index author carries (a placeholder or a fabricated
    author), a list that silently stops short of the index's (an author dropped without
    "et al."), and a wrong first author (authors swapped or reordered) are all metadata
    errors. Names compare by surname, so initials and name order do not matter."""
    if not entry.authors or not work.authors:
        return []
    work_surnames = [first_surname(a) for a in work.authors if a and first_surname(a)]
    entry_surnames = [s for s in (first_surname(a) for a in entry.authors) if s]
    if not entry_surnames or not work_surnames:
        return []
    problems: list[str] = []
    missing = [s for s in entry_surnames if s not in work_surnames]
    if missing:
        problems.append(
            f"author: the entry lists {', '.join(missing)}, none of whom is on the work"
        )
    truncated = bool(_TRUNCATION.search(" ".join(entry.authors)))
    if not truncated and len(entry_surnames) < len(work_surnames) and len(work_surnames) >= 3:
        problems.append(
            f"author: the entry lists {len(entry_surnames)} of the work's "
            f"{len(work_surnames)} authors without saying so"
        )
    if (
        entry_surnames[0] != work_surnames[0]
        and entry_surnames[0] in work_surnames
    ):
        problems.append(
            f"author: the entry's first author ({entry_surnames[0]}) is not the work's "
            f"first author ({work_surnames[0]})"
        )
    return problems


def _metadata_findings(task: AuditTask, entry: BibEntry, work: Work) -> list[CandidateFinding]:
    """Findings for the metadata fields that do not match."""
    findings: list[CandidateFinding] = []
    problems: list[str] = []

    if entry.title and work.title:
        if not titles_match(entry.title, work.title):
            problems.append(f"title: the entry says '{entry.title}', the index says '{work.title}'")
        else:
            # A near-miss title ("...You Nedd" for "...You Need") matches the fuzzy
            # threshold but is still a corrupted title. A legitimately shortened record
            # ("NeRF" for the full title) keeps every word of the shorter side; a
            # corruption introduces a word the other side does not have.
            if title_text(entry.title) != title_text(work.title) and not (
                title_tokens(entry.title) <= title_tokens(work.title)
                or title_tokens(work.title) <= title_tokens(entry.title)
            ):
                problems.append(
                    f"title: the entry says '{entry.title}', the index says '{work.title}' "
                    "(near-miss)"
                )
    # A bibliography may cite the proceedings year where the index has the online-first
    # year (or the reverse); one year of drift is normal, two or more is a mismatch.
    if entry.year and work.year and abs(entry.year - work.year) > 1:
        problems.append(f"year: the entry says {entry.year}, the index says {work.year}")
    if entry.venue and work.venue and not venues_match(entry.venue, work.venue):
        problems.append(f"venue: the entry says '{entry.venue}', the index says '{work.venue}'")
    problems.extend(_author_problems(entry, work))

    if not problems:
        return findings

    findings.append(
        CandidateFinding.for_task(
            task,
            0,
            error_class=ErrorClass.REFERENCE_METADATA_MISMATCH,
            summary="The reference's metadata does not match the scholarly indexes: "
            + "; ".join(problems),
            explanation=entry.raw or entry.title or entry.id,
            evidence=(_lookup_evidence(entry, Resolution(work=work)),),
        )
    )
    return findings


async def check_reference(
    task: AuditTask,
    ctx: RunContext,
    resolution: Resolution,
    entry: BibEntry | None = None,
) -> TaskResult:
    """Whether the bibliography entry exists and its metadata matches (decision P2.5).

    `entry` is the bibliography entry the caller already fetched (check() does); when
    not given, it is looked up again, so the check also works on a bare task."""
    if task.kind is not TaskKind.REFERENCE or not isinstance(task.target, BibTarget):
        raise ValueError(f"{task.task_id} is not a reference task")
    if entry is None:
        try:
            entry = ctx.paper.bib_entry(task.target.bib_id)
        except KeyError:
            entry = None
    if entry is None:
        return TaskResult.for_task(
            task,
            TaskOutcome.UNVERIFIABLE,
            reason=f"the bibliography entry {task.target.bib_id} is not in the paper",
        )
    findings: list[CandidateFinding] = []
    if not resolution.found:
        findings.append(
            CandidateFinding.for_task(
                task,
                0,
                error_class=ErrorClass.REFERENCE_NOT_FOUND,
                summary=f"The reference could not be resolved: {resolution.reason}",
                explanation=entry.raw or entry.title or entry.id,
                evidence=(_lookup_evidence(entry, resolution),),
            )
        )
    elif resolution.doi_unresolved:
        # The entry cites a DOI that neither Crossref nor OpenAlex knows; even if a
        # title search then found the real work, the identifier as printed is fabricated.
        findings.append(
            CandidateFinding.for_task(
                task,
                0,
                error_class=ErrorClass.REFERENCE_NOT_FOUND,
                summary=f"The entry's DOI ({entry.doi}) does not resolve in Crossref or OpenAlex",
                explanation=entry.raw or entry.title or entry.id,
                evidence=(_lookup_evidence(entry, resolution),),
            )
        )
    else:
        assert resolution.work is not None
        findings.extend(_metadata_findings(task, entry, resolution.work))

    if not findings:
        return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=[])
    return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=findings)
