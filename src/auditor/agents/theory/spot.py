"""SPOT as an outside benchmark for the theory checker (decision D21).

SPOT (Son et al., arXiv 2505.11855; data `amphora/SPOT` on Hugging Face, CC BY 4.0) lists
published papers with errors that led to an erratum or a retraction. The subset used here
is the one Google's Paper Assistant Tool reports on: Mathematics and Computer Science
papers whose error is an "Equation / proof" error (26 papers, 29 errors).

SPOT gives each paper as text pages with LaTeX, so a page becomes a section and a paragraph
a sentence. Its labels describe an error in words with a rough location, so a run is scored
by an LLM judge, as in the SPOT and PAT papers: for each annotated error, the judge is shown
the run's reports for that paper and names the ones describing the same error. Every grade
keeps a field for the person who checks it; a grade they reject counts as not found.
"""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from auditor.infra.llm import StructuredLLM
from auditor.schemas import (
    PaperMetadata,
    ParsedPaper,
    ParserInfo,
    Prediction,
    Record,
    Section,
    Sentence,
    SentenceAnchor,
)

SPOT_DATASET = "amphora/SPOT"
SPOT_REVISION = "0641b6aea2495b7262a349856d24ebeea93a8e58"
CATEGORIES = frozenset({"Mathematics", "Computer Science"})
ERROR_CATEGORY = "Equation / proof"

JUDGE_PROMPT = """You grade an automatic reviewer against a known error in a research paper.
You get the annotation of one error that led to an erratum or retraction, and the numbered
issues the reviewer reported for the same paper. Name the issues that describe this error:
the same flaw in the same result or step, even if worded differently or less completely. An
issue that is only about the same section, or about a different flaw, does not count. Answer
with the numbers of the matching issues, or an empty list."""


class SpotError(Record):
    error_id: str
    paper_id: str
    title: str
    paper_category: str
    error_category: str
    location: str
    severity: str
    annotation: str


def spot_subset(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[SpotError], dict[str, list[str]]]:
    """The errors of the subset, and the text pages of each of its papers."""
    errors: list[SpotError] = []
    pages: dict[str, list[str]] = {}
    count: dict[str, int] = {}
    for row in rows:
        if row["paper_category"] not in CATEGORIES or row["error_category"] != ERROR_CATEGORY:
            continue
        paper_id = str(row["doi/arxiv_id"]).replace("/", "_")
        number = count.get(paper_id, 0)
        count[paper_id] = number + 1
        errors.append(
            SpotError(
                error_id=f"{paper_id}-e{number}",
                paper_id=paper_id,
                title=row["title"],
                paper_category=row["paper_category"],
                error_category=row["error_category"],
                location=row["error_location"] or "",
                severity=row["error_severity"] or "",
                annotation=row["error_annotation"],
            )
        )
        if paper_id not in pages:
            pages[paper_id] = [
                entry["text"]
                for entry in row["paper_content"] or []
                if entry.get("type") == "text" and entry.get("text")
            ]
    return errors, pages


def paper_from_pages(paper_id: str, title: str, pages: Sequence[str], revision: str) -> ParsedPaper:
    sections = []
    sentences = []
    for number, page in enumerate(pages, start=1):
        section_id = f"page{number}"
        paragraphs = [p.strip() for p in page.split("\n\n")]
        paragraphs = [p for p in paragraphs if p and p.strip("-* ")]
        if not paragraphs:
            continue
        sections.append(Section(id=section_id, title=f"Page {number}"))
        sentences += [
            Sentence(id=f"p{number}-{i}", section_id=section_id, text=text)
            for i, text in enumerate(paragraphs)
        ]
    return ParsedPaper(
        paper_id=paper_id,
        parser=ParserInfo(name="spot-text", version=revision),
        metadata=PaperMetadata(title=title, arxiv_id=paper_id if "." in paper_id else None),
        sections=tuple(sections),
        sentences=tuple(sentences),
    )


class MatchVerdict(BaseModel):
    matches: list[int]
    reason: str = ""


class Grade(Record):
    error_id: str
    paper_id: str
    # Finding ids (or report numbers, for systems without ids) the judge matched.
    matched: tuple[str, ...]
    reason: str = ""
    # Set by the person who checks the grade. False rejects the judge's match; to record a
    # match the judge missed, put the finding id in `matched` and set True.
    human_agrees: bool | None = None


def _report(number: int, prediction: Prediction, texts: Mapping[str, str]) -> str:
    line = f"{number}. {prediction.summary}"
    if isinstance(prediction.anchor, SentenceAnchor):
        text = texts.get(prediction.anchor.sentence_id, "")
        if text:
            line += f'\n   (at: "{text[:300]}")'
    return line


async def judge_paper(
    llm: StructuredLLM,
    errors: Sequence[SpotError],
    reports: Sequence[Prediction],
    texts: Mapping[str, str],
) -> list[Grade]:
    """One grade per annotated error of a paper. `texts` maps sentence ids to their text,
    so the judge sees where each report points."""
    if not reports:
        return [Grade(error_id=e.error_id, paper_id=e.paper_id, matched=()) for e in errors]
    listing = "\n".join(_report(i, p, texts) for i, p in enumerate(reports, start=1))
    grades = []
    for error in errors:
        prompt = (
            f"Paper: {error.title}\n\nAnnotated error (location: {error.location}):\n"
            f"{error.annotation}\n\nReported issues:\n{listing}"
        )
        verdict = (
            await llm.generate(MatchVerdict, [SystemMessage(JUDGE_PROMPT), HumanMessage(prompt)])
        ).value
        matched = tuple(
            reports[n - 1].finding_id or str(n)
            for n in dict.fromkeys(verdict.matches)
            if 1 <= n <= len(reports)
        )
        grades.append(
            Grade(
                error_id=error.error_id,
                paper_id=error.paper_id,
                matched=matched,
                reason=verdict.reason,
            )
        )
    return grades


@dataclass(frozen=True)
class SpotScores:
    errors: int
    found: int
    predictions: int
    matched: int
    # Grades nobody has checked yet; numbers are reported only when this is 0.
    unchecked: int
    # Papers with an annotated error, and those with at least one of them found.
    papers: int = 0
    papers_found: int = 0

    @property
    def recall(self) -> float:
        return self.found / self.errors if self.errors else 0.0

    @property
    def precision(self) -> float:
        return self.matched / self.predictions if self.predictions else 0.0

    @property
    def paper_recall(self) -> float:
        """The screening view: a paper counts as caught when any of its errors is found."""
        return self.papers_found / self.papers if self.papers else 0.0


def split_of(paper_id: str, paper_ids: Sequence[str]) -> Literal["dev", "test"]:
    """A fixed half of the papers for tuning ("dev") and the other half for reporting
    ("test"): the papers are ordered by a hash of their id, and the first half is dev. The
    split never depends on the order of the list or on any result."""

    def key(pid: str) -> str:
        return hashlib.sha256(pid.encode("utf-8")).hexdigest()

    ranked = sorted(set(paper_ids), key=key)
    if paper_id not in ranked:
        raise ValueError(f"{paper_id} is not one of the papers")
    return "dev" if ranked.index(paper_id) < (len(ranked) + 1) // 2 else "test"


def score_grades(grades: Sequence[Grade], *, predictions: int) -> SpotScores:
    """Recall over annotated errors and precision over reports. A report matched to an
    annotated error counts once; a grade the checker rejected counts as not found."""
    accepted = [g for g in grades if g.matched and g.human_agrees is not False]
    matched = {(g.paper_id, finding) for g in accepted for finding in g.matched}
    return SpotScores(
        errors=len(grades),
        found=len(accepted),
        predictions=predictions,
        matched=len(matched),
        unchecked=sum(1 for g in grades if g.human_agrees is None),
        papers=len({g.paper_id for g in grades}),
        papers_found=len({g.paper_id for g in accepted}),
    )
