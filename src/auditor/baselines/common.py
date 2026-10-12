"""Shared pieces of the B0 and B1 baselines: how the paper is shown to the model, what the
model must answer, and how answers become Predictions (decision D3)."""

import logging
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from auditor.schemas import (
    Anchor,
    BibEntry,
    BibliographyAnchor,
    BibTarget,
    DocumentAnchor,
    ErrorClass,
    MarkerTarget,
    PageAnchor,
    ParsedPaper,
    Prediction,
    RepositoryRef,
    RuleTarget,
    SentenceAnchor,
    SettingTarget,
    SystemName,
    Target,
    VenueProfile,
)

logger = logging.getLogger(__name__)

ERROR_CLASS_GUIDE = """Error classes:
- reference_not_found: a cited work does not seem to exist.
- reference_metadata_mismatch: a cited work exists but its title, authors, year or venue are wrong.
- claim_not_supported: the cited work does not support, or contradicts, what the sentence says.
- claim_number_mismatch: a number attributed to a cited work differs from that work.
- setting_mismatch: a hyperparameter or configuration in the paper differs from the code.
- data_split_mismatch: a dataset or split described in the paper differs from the code.
- metric_mismatch: a metric or evaluation procedure differs from the code.
- template_violation: a venue formatting rule is broken."""

ANSWER_GUIDE = """For every error, give:
- error_class: one of the classes above.
- anchor_kind and anchor_id: where the error is. Use "sentence" with a sentence id such as
  "s12", "bibliography" with an entry id such as "b3", "page" with a page number, or
  "document" with no id.
- target_kind and target: what the error is about. Use "bibliography" with an entry id,
  "marker" with the citation marker text when it has no entry, "setting" with a snake_case
  setting name such as "learning_rate", or "rule" with a venue rule id.
- summary: one sentence describing the problem.
Report only errors you have specific reasons for. An empty list is a valid answer."""


class IncompleteAnswerError(Exception):
    """A baseline stopped without a final answer, for example at its model call limit.

    Raised rather than returning no predictions, which would be indistinguishable from
    a valid "no errors found" answer."""


class ReportedError(BaseModel):
    error_class: ErrorClass
    anchor_kind: Literal["sentence", "bibliography", "page", "document"]
    anchor_id: str | None = None
    target_kind: Literal["bibliography", "marker", "setting", "rule"]
    target: str
    summary: str


class ReportedErrors(BaseModel):
    errors: list[ReportedError] = Field(default_factory=list)


def render_paper(
    paper: ParsedPaper,
    *,
    venue: VenueProfile | None = None,
    repository: RepositoryRef | None = None,
) -> str:
    """The paper as text, with the ids the model must use to point at things."""
    titles = {section.id: section.title for section in paper.sections}
    lines = [f"Title: {paper.metadata.title or paper.paper_id}"]
    if paper.layout is not None:
        lines.append(f"Pages: {paper.layout.page_count}")
    if repository is not None:
        commit = f" (commit {repository.commit})" if repository.commit else ""
        lines.append(f"Code repository: {repository.url}{commit}")
    lines.append("")
    current = None
    for sentence in paper.sentences:
        if sentence.section_id != current and sentence.section_id is not None:
            lines += ["", f"## {titles.get(sentence.section_id, '')}"]
        current = sentence.section_id
        pages = ",".join(str(p) for p in sentence.pages)
        page_note = f" (p. {pages})" if pages else ""
        lines.append(f"[{sentence.id}]{page_note} {sentence.text}")
    if paper.bibliography:
        lines += ["", "## References"]
        lines += [f"[{entry.id}] {_reference(entry)}" for entry in paper.bibliography]
    if venue is not None and venue.rules:
        lines += ["", f"## Venue rules ({venue.name})"]
        lines += [f"[{rule.id}] {rule.description}" for rule in venue.rules]
    return "\n".join(lines)


def _reference(entry: BibEntry) -> str:
    """Every field the specialists see, so a changed author, year or identifier is visible
    to the baselines too. The raw text is the fallback when nothing was parsed."""
    parts = [", ".join(entry.authors), entry.title or "", entry.venue or ""]
    if entry.year:
        parts.append(str(entry.year))
    if entry.doi:
        parts.append(f"DOI {entry.doi}")
    if entry.arxiv_id:
        parts.append(f"arXiv {entry.arxiv_id}")
    if entry.url:
        parts.append(entry.url)
    return ". ".join(p for p in parts if p) or (entry.raw or "")


def _anchor(error: ReportedError) -> Anchor:
    match error.anchor_kind:
        case "sentence":
            return SentenceAnchor(sentence_id=error.anchor_id or "")
        case "bibliography":
            return BibliographyAnchor(bib_id=error.anchor_id or "")
        case "page":
            return PageAnchor(pages=(int(error.anchor_id or ""),))
        case "document":
            return DocumentAnchor()


def _target(error: ReportedError) -> Target:
    match error.target_kind:
        case "bibliography":
            return BibTarget(bib_id=error.target)
        case "marker":
            return MarkerTarget(marker=error.target)
        case "setting":
            key = re.sub(r"[\s-]+", "_", error.target.strip().lower())
            return SettingTarget(key=key)
        case "rule":
            return RuleTarget(rule_id=error.target)


def to_predictions(
    reported: ReportedErrors,
    paper: ParsedPaper,
    *,
    run_id: str,
    system: SystemName,
) -> tuple[list[Prediction], list[str]]:
    """Valid predictions, and a reason for every reported error that had to be dropped."""
    predictions: list[Prediction] = []
    dropped: list[str] = []
    known_entries = {entry.id for entry in paper.bibliography}
    for index, error in enumerate(reported.errors):
        try:
            anchor = _anchor(error)
            paper.validate_anchor(anchor)
            target = _target(error)
            if isinstance(target, BibTarget) and target.bib_id not in known_entries:
                raise ValueError(f"unknown bibliography entry {target.bib_id}")
            predictions.append(
                Prediction(
                    run_id=run_id,
                    system=system,
                    paper_id=paper.paper_id,
                    variant=paper.variant,
                    error_class=error.error_class,
                    anchor=anchor,
                    target=target,
                    summary=error.summary.strip() or error.error_class.value,
                )
            )
        except (ValueError, ValidationError) as exc:
            dropped.append(f"error {index}: {exc}")
    for reason in dropped:
        logger.warning("dropped a reported error: %s", reason)
    return predictions, dropped
