"""The audit report for one paper (team plan P4.4, decisions D6 and D20).

The report is advisory. It has three sections:

1. verified flags: findings whose every side is shown by evidence the verifier found again
   in its source;
2. items for human review: suspicions the evidence does not settle, with the verifier's
   reason;
3. items not checked: tasks that ended unverifiable or failed, with their reasons, so a
   missing flag is never read as a clean result (D6).

Dropped findings are listed at the end, so a reader can see what was set aside and why.
Repeats of one problem (D20) are shown once, with the ids of the repeats. Each item shows
the paper's own text at the flagged place and the evidence with its source and location.
The wording describes discrepancies only; it never says why they exist.
"""

from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Literal

from auditor.schemas import (
    AuditTask,
    BibliographyAnchor,
    DocumentAnchor,
    ErrorClass,
    Evidence,
    EvidenceKind,
    PageAnchor,
    ParsedPaper,
    SentenceAnchor,
    SourceKind,
    TaskOutcome,
    TaskResult,
    VerificationState,
    VerifiedFinding,
)
from auditor.schemas.base import Record
from auditor.schemas.tasks import BibTarget, MarkerTarget, RuleTarget, SettingTarget, Target

NOTICE = (
    "This report was produced automatically and is advisory. A verified flag means that "
    "the evidence shown was found again in its source and disagrees with the paper; it "
    "does not say why the difference exists, and it may still be wrong. Items for human "
    "review are open questions, not findings. A human reviewer decides what, if anything, "
    "to raise with the authors."
)

# What each class means, in words that describe the discrepancy and not a cause.
CLASS_TITLES: dict[ErrorClass, str] = {
    ErrorClass.REFERENCE_NOT_FOUND: "Reference not found in scholarly indexes",
    ErrorClass.REFERENCE_METADATA_MISMATCH: "Reference details differ from the indexed record",
    ErrorClass.CLAIM_NOT_SUPPORTED: "Cited work does not appear to support the claim",
    ErrorClass.CLAIM_NUMBER_MISMATCH: "Number attributed to a cited work differs from it",
    ErrorClass.SETTING_MISMATCH: "Setting in the paper differs from the code",
    ErrorClass.DATA_SPLIT_MISMATCH: "Dataset or split in the paper differs from the code",
    ErrorClass.METRIC_MISMATCH: "Metric in the paper differs from the code",
    ErrorClass.DATASET_SIZE_MISMATCH: "Dataset size in the paper differs from the data",
    ErrorClass.LABEL_DISTRIBUTION_MISMATCH: "Classes or class balance differ from the data",
    ErrorClass.DATA_LEAKAGE: "Test or validation examples also appear in training data",
    ErrorClass.METHOD_MISMATCH: "Method in the paper differs from the code",
    ErrorClass.MATH_ERROR: "Possible error in a derivation or proof",
    ErrorClass.UNFINISHED_CONTENT: "Unfinished or placeholder content",
    ErrorClass.CODE_DEFECT: "Released code may not run as published",
    ErrorClass.RESULT_INCONSISTENCY: "Text and results disagree",
    ErrorClass.TEMPLATE_VIOLATION: "Venue formatting rule not met",
}

_SOURCE_NAMES: dict[SourceKind, str] = {
    SourceKind.SUBMISSION: "the paper",
    SourceKind.CITED_WORK: "cited work",
    SourceKind.REPOSITORY: "repository",
    SourceKind.VENUE_RULE: "venue rule",
    SourceKind.SCHOLARLY_INDEX: "scholarly index",
}


class Place(Record):
    """Where in the paper an item is."""

    description: str
    section: str | None = None
    page: int | None = None
    sentence_id: str | None = None
    # The paper's own words at that place: the sentence, or the bibliography entry.
    text: str | None = None


class ReportEvidence(Record):
    kind: EvidenceKind
    source_kind: SourceKind
    source: str
    source_title: str | None = None
    url: str | None = None
    location: str | None = None
    text: str


class ReportItem(Record):
    finding_id: str
    error_class: ErrorClass
    title: str
    severity: str | None = None
    uncertain: bool = False
    place: Place
    about: str
    summary: str
    explanation: str = ""
    evidence: tuple[ReportEvidence, ...] = ()
    verifier_reason: str
    also_reported_as: tuple[str, ...] = ()


class NotChecked(Record):
    task_id: str
    kind: str
    outcome: Literal["unverifiable", "failed"]
    reason: str
    place: Place
    about: str
    claim: str | None = None


class AuditReport(Record):
    paper_id: str
    variant: str
    title: str | None = None
    run_id: str | None = None
    notice: str = NOTICE
    tasks: int
    task_outcomes: dict[str, int]
    verified: tuple[ReportItem, ...] = ()
    review: tuple[ReportItem, ...] = ()
    not_checked: tuple[NotChecked, ...] = ()
    dropped: tuple[ReportItem, ...] = ()


def _place(paper: ParsedPaper, anchor: object) -> Place:
    match anchor:
        case SentenceAnchor(sentence_id=sentence_id):
            try:
                sentence = paper.sentence(sentence_id)
            except KeyError:
                return Place(description=f"sentence {sentence_id}", sentence_id=sentence_id)
            sections = {s.id: s for s in paper.sections}
            section = sections.get(sentence.section_id or "")
            title = None
            if section is not None:
                title = f"{section.number} {section.title}" if section.number else section.title
            page = sentence.boxes[0].page if sentence.boxes else None
            parts = [p for p in (title and f"section {title}", page and f"page {page}") if p]
            return Place(
                description=", ".join(parts) or f"sentence {sentence_id}",
                section=title,
                page=page,
                sentence_id=sentence_id,
                text=sentence.text,
            )
        case BibliographyAnchor(bib_id=bib_id):
            try:
                entry = paper.bib_entry(bib_id)
            except KeyError:
                return Place(description=f"reference {bib_id}")
            text = entry.raw or ", ".join(
                p for p in (entry.title, "; ".join(entry.authors), str(entry.year or "")) if p
            )
            page = entry.boxes[0].page if entry.boxes else None
            return Place(description="bibliography", page=page, text=text or None)
        case PageAnchor(pages=pages):
            label = "page" if len(pages) == 1 else "pages"
            return Place(description=f"{label} {', '.join(map(str, pages))}", page=pages[0])
        case DocumentAnchor():
            return Place(description="the paper as a whole")
    return Place(description="unknown place")


def _about(target: Target, paper: ParsedPaper) -> str:
    match target:
        case BibTarget(bib_id=bib_id):
            try:
                entry = paper.bib_entry(bib_id)
            except KeyError:
                return f"reference {bib_id}"
            return f"reference {entry.title or entry.raw or bib_id}"
        case MarkerTarget(marker=marker):
            return f"citation {marker}"
        case SettingTarget(key=key):
            return key.replace("_", " ")
        case RuleTarget(rule_id=rule_id):
            return f"rule {rule_id}"


def _location(evidence: Evidence) -> str | None:
    if evidence.path is not None:
        lines = ""
        if evidence.line_start is not None:
            end = evidence.line_end
            lines = f":{evidence.line_start}" + (
                f"-{end}" if end is not None and end != evidence.line_start else ""
            )
        return f"{evidence.path}{lines}"
    if evidence.page is not None:
        return f"page {evidence.page}"
    if evidence.char_start is not None:
        return f"characters {evidence.char_start}-{evidence.char_end}"
    return None


def _evidence(evidence: Evidence) -> ReportEvidence:
    return ReportEvidence(
        kind=evidence.kind,
        source_kind=evidence.source.kind,
        source=evidence.source.id,
        source_title=evidence.source.title,
        url=evidence.source.url,
        location=_location(evidence),
        text=evidence.text,
    )


def _item(decision: VerifiedFinding, paper: ParsedPaper, repeats: Sequence[str]) -> ReportItem:
    finding = decision.finding
    # A verified flag shows what the verifier accepted; anything else shows what the
    # specialist gave, since that is what a reviewer has to judge.
    evidence = (
        decision.accepted_evidence
        if decision.state is VerificationState.VERIFIED_FLAG
        else finding.evidence
    )
    return ReportItem(
        finding_id=finding.finding_id,
        error_class=finding.error_class,
        title=CLASS_TITLES[finding.error_class],
        severity=decision.severity.value if decision.severity else None,
        uncertain=finding.uncertain,
        place=_place(paper, finding.anchor),
        about=_about(finding.target, paper),
        summary=finding.summary,
        explanation=finding.explanation,
        evidence=tuple(_evidence(e) for e in evidence),
        verifier_reason=decision.reason,
        also_reported_as=tuple(repeats),
    )


def build_report(
    paper: ParsedPaper,
    tasks: Sequence[AuditTask],
    results: Sequence[TaskResult],
    verified: Sequence[VerifiedFinding],
    *,
    run_id: str | None = None,
) -> AuditReport:
    """The report on one paper from a full-system run. `verified` is in the order the
    synthesis step ranked it (D20): state, then severity, certain before uncertain."""
    repeats: defaultdict[str, list[str]] = defaultdict(list)
    for decision in verified:
        if decision.duplicate_of is not None:
            repeats[decision.duplicate_of].append(decision.finding.finding_id)
    sections: dict[VerificationState, list[ReportItem]] = defaultdict(list)
    for decision in verified:
        if decision.duplicate_of is None:
            item = _item(decision, paper, repeats.get(decision.finding.finding_id, ()))
            sections[decision.state].append(item)

    by_id = {task.task_id: task for task in tasks}
    not_checked = []
    for result in results:
        if result.outcome is TaskOutcome.CHECKED:
            continue
        task = by_id.get(result.task_id)
        not_checked.append(
            NotChecked(
                task_id=result.task_id,
                kind=result.kind.value,
                outcome=result.outcome.value,  # type: ignore[arg-type]
                reason=(result.reason or result.error or "no reason recorded").strip(),
                place=_place(paper, task.anchor) if task else Place(description="unknown"),
                about=_about(task.target, paper) if task else result.task_id,
                claim=task.claim if task else None,
            )
        )
    outcomes = Counter(result.outcome.value for result in results)
    return AuditReport(
        paper_id=paper.paper_id,
        variant=paper.variant.value,
        title=paper.metadata.title,
        run_id=run_id,
        tasks=len(tasks) or len(results),
        task_outcomes={o.value: outcomes[o.value] for o in TaskOutcome},
        verified=tuple(sections[VerificationState.VERIFIED_FLAG]),
        review=tuple(sections[VerificationState.NEEDS_HUMAN_REVIEW]),
        not_checked=tuple(not_checked),
        dropped=tuple(sections[VerificationState.DROPPED]),
    )


# --- Markdown -------------------------------------------------------------------------


def _quote(text: str, limit: int = 600) -> str:
    """A Markdown blockquote; long code or passages are cut at `limit` characters."""
    text = text.strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + " [...]"
    return "\n".join(f"> {line}" if line.strip() else ">" for line in text.splitlines())


def _short_repository(source: str) -> str:
    """`owner/repo@abc1234` for a hosted repository, the folder name for a local copy."""
    base, _, revision = source.rpartition("@")
    if not base:
        base, revision = source, ""
    base = base.removeprefix("file://").rstrip("/")
    for host in ("https://github.com/", "https://gitlab.com/", "https://huggingface.co/"):
        if base.startswith(host):
            base = base.removeprefix(host)
            break
    else:
        if "://" not in base:
            base = base.rsplit("/", 1)[-1]
    revision = revision.removeprefix("content-")[:7]
    return f"{base}@{revision}" if revision else base


def _source_line(evidence: ReportEvidence) -> str:
    name = _SOURCE_NAMES[evidence.source_kind]
    if evidence.source_kind is SourceKind.SUBMISSION:
        label = "The paper"
    elif evidence.source_kind is SourceKind.REPOSITORY:
        label = f"Repository `{_short_repository(evidence.source)}`"
    else:
        label = f"{name.capitalize()} `{evidence.source_title or evidence.source}`"
        if evidence.url and evidence.url != evidence.source:
            label += f" (<{evidence.url}>)"
    if evidence.location:
        label += f", {evidence.location}"
    return label


def _same_text(a: str, b: str) -> bool:
    return " ".join(a.split()) == " ".join(b.split())


def _render_item(number: int, item: ReportItem, *, show_reason: bool) -> list[str]:
    tags = [
        t
        for t in (
            item.severity and f"{item.severity} severity",
            item.uncertain and "marked uncertain by its check",
        )
        if t
    ]
    lines = [
        f"### {number}. {item.title}",
        "",
        f"**Where:** {item.place.description}. **About:** {item.about}."
        + (f" ({'; '.join(tags)})" if tags else ""),
        "",
    ]
    if item.place.text:
        lines += ["What the paper says:", "", _quote(item.place.text), ""]
    lines += [f"**What the check found:** {item.summary}", ""]
    if item.explanation and item.explanation != item.summary:
        lines += [item.explanation, ""]
    shown = [
        e
        for e in item.evidence
        # A quote of the whole flagged sentence is already shown above.
        if not (
            e.source_kind is SourceKind.SUBMISSION
            and item.place.text
            and _same_text(e.text, item.place.text)
        )
    ]
    if shown:
        lines += ["**Evidence:**", ""]
        for evidence in shown:
            if evidence.kind is EvidenceKind.CODE:
                lines += [
                    f"- {_source_line(evidence)}:",
                    "",
                    "  ```",
                    *("  " + line for line in evidence.text.strip().splitlines()[:25]),
                    "  ```",
                    "",
                ]
            else:
                lines += [f"- {_source_line(evidence)}:", "", _quote(evidence.text), ""]
    if show_reason:
        lines += [f"**Why it is here:** {item.verifier_reason}", ""]
    if item.also_reported_as:
        lines += [f"Also reported as {', '.join(item.also_reported_as)}.", ""]
    return lines


def render_markdown(report: AuditReport) -> str:
    title = report.title or report.paper_id
    outcomes = report.task_outcomes
    lines = [
        f"# Audit report: {title}",
        "",
        f"Paper `{report.paper_id}` ({report.variant})"
        + (f", run `{report.run_id}`" if report.run_id else "")
        + ".",
        "",
        f"> {report.notice}",
        "",
        "## Summary",
        "",
        "| | Count |",
        "|---|---|",
        f"| Verified flags | {len(report.verified)} |",
        f"| Items for human review | {len(report.review)} |",
        f"| Checks that could not be done | {len(report.not_checked)} |",
        f"| Suspicions set aside by the verifier | {len(report.dropped)} |",
        f"| Checks run | {report.tasks} ({outcomes.get('checked', 0)} completed, "
        f"{outcomes.get('unverifiable', 0)} unverifiable, {outcomes.get('failed', 0)} failed) |",
        "",
        "## 1. Verified flags",
        "",
    ]
    if report.verified:
        lines += [
            "Each difference below is shown by evidence that was found again in its source.",
            "",
        ]
        for n, item in enumerate(report.verified, start=1):
            lines += _render_item(n, item, show_reason=False)
    else:
        lines += ["No finding was verified.", ""]

    lines += ["## 2. Items for human review", ""]
    if report.review:
        lines += [
            "The evidence for these is incomplete, could not be checked, or rests on a "
            "model's reading. Each says why it was not verified.",
            "",
        ]
        for n, item in enumerate(report.review, start=1):
            lines += _render_item(n, item, show_reason=True)
    else:
        lines += ["Nothing needs review.", ""]

    lines += ["## 3. Items not checked", ""]
    if report.not_checked:
        lines += [
            "These checks could not be completed, so the absence of a flag here says "
            "nothing about the paper.",
            "",
            "| Check | Where | About | Outcome | Reason |",
            "|---|---|---|---|---|",
        ]
        for unchecked in report.not_checked:
            reason = " ".join(unchecked.reason.split()).replace("|", "\\|")
            if len(reason) > 200:
                reason = reason[:200] + " [...]"
            lines.append(
                f"| {unchecked.kind} | {unchecked.place.description} | {unchecked.about} | "
                f"{unchecked.outcome} | {reason} |"
            )
        lines.append("")
    else:
        lines += ["Every check was completed.", ""]

    if report.dropped:
        lines += [
            "## Set aside",
            "",
            "Suspicions the verifier dropped, because their evidence was not found in its "
            "source or they carried none.",
            "",
            "| Finding | Kind | Where | Reason |",
            "|---|---|---|---|",
        ]
        for item in report.dropped:
            reason = " ".join(item.verifier_reason.split()).replace("|", "\\|")
            lines.append(
                f"| {item.finding_id} | {item.title} | {item.place.description} | {reason} |"
            )
        lines.append("")
    return "\n".join(lines)
