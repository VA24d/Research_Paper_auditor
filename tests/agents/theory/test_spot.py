from typing import Any

import pytest

from auditor.agents.theory.spot import (
    Grade,
    MatchVerdict,
    SpotError,
    judge_paper,
    paper_from_pages,
    score_grades,
    split_of,
    spot_subset,
)
from auditor.infra.llm import StructuredLLM
from auditor.schemas import (
    ErrorClass,
    PaperVariant,
    Prediction,
    SentenceAnchor,
    SettingTarget,
    SystemName,
    VerificationState,
)
from auditor.testing import ScriptedChatModel, tool_reply


def row(**fields: Any) -> dict[str, Any]:
    base = {
        "doi/arxiv_id": "2405.01133v3",
        "title": "A missing theorem on dual spaces",
        "paper_category": "Mathematics",
        "error_category": "Equation / proof",
        "error_location": "Lemma 3,4",
        "error_severity": "retract",
        "error_annotation": "Proposition 2.3 is false, so Lemmas 3 and 4 fail.",
        "paper_content": [
            {"type": "text", "text": "ABSTRACT. We answer a question.\n\nSecond paragraph."},
            {"type": "image_url", "text": None, "image_url": {"url": "x"}},
            {"type": "text", "text": "---\n\nLemma 3. Every map is bounded.\n\n"},
        ],
    }
    return {**base, **fields}


def test_the_subset_keeps_math_and_cs_proof_errors() -> None:
    rows = [
        row(),
        row(**{"doi/arxiv_id": "2405.01133v3", "error_location": "Theorem 4.1"}),
        row(**{"doi/arxiv_id": "x1", "paper_category": "Computer Science"}),
        row(**{"doi/arxiv_id": "x2", "paper_category": "Biology"}),
        row(**{"doi/arxiv_id": "x3", "error_category": "Figure duplication"}),
    ]
    errors, pages = spot_subset(rows)
    assert [e.paper_id for e in errors] == ["2405.01133v3", "2405.01133v3", "x1"]
    assert [e.error_id for e in errors] == ["2405.01133v3-e0", "2405.01133v3-e1", "x1-e0"]
    assert errors[0].annotation.startswith("Proposition 2.3")
    assert set(pages) == {"2405.01133v3", "x1"}


def test_pages_become_sections_and_paragraphs_become_sentences() -> None:
    _, pages = spot_subset([row()])
    paper = paper_from_pages("2405.01133v3", "A missing theorem", pages["2405.01133v3"], "rev")
    # Text entries are pages; images sit between them.
    assert [s.title for s in paper.sections] == ["Page 1", "Page 2"]
    assert [(s.id, s.section_id, s.text) for s in paper.sentences] == [
        ("p1-0", "page1", "ABSTRACT. We answer a question."),
        ("p1-1", "page1", "Second paragraph."),
        ("p2-0", "page2", "Lemma 3. Every map is bounded."),
    ]
    assert paper.metadata.arxiv_id == "2405.01133v3"
    assert paper.parser.version == "rev"


def prediction(summary: str, number: int) -> Prediction:
    return Prediction(
        run_id="r",
        system=SystemName.FULL,
        paper_id="2405.01133v3",
        variant=PaperVariant.ORIGINAL,
        error_class=ErrorClass.MATH_ERROR,
        anchor=SentenceAnchor(sentence_id="p2-0"),
        target=SettingTarget(key="math"),
        state=VerificationState.NEEDS_HUMAN_REVIEW,
        summary=summary,
        finding_id=f"f{number}",
    )


def error(number: int) -> SpotError:
    errors, _ = spot_subset([row(), row(error_location="Theorem 4.1")])
    return errors[number]


async def test_the_judge_matches_reports_to_annotated_errors() -> None:
    model = ScriptedChatModel(
        replies=[
            tool_reply(MatchVerdict, {"matches": [2], "reason": "same flaw"}),
            tool_reply(MatchVerdict, {"matches": [], "reason": "not reported"}),
        ]
    )
    llm = StructuredLLM(model, name="test:s")
    reports = [prediction("Eq. 1: a typo.", 0), prediction("Lemma 3: Prop 2.3 is false.", 1)]
    grades = await judge_paper(llm, [error(0), error(1)], reports, {"p2-0": "Lemma 3. ..."})
    assert [(g.error_id, g.matched) for g in grades] == [
        ("2405.01133v3-e0", ("f1",)),
        ("2405.01133v3-e1", ()),
    ]
    prompt = " ".join(str(m.content) for m in model.received[0])
    assert "Lemma 3: Prop 2.3 is false." in prompt
    assert "Proposition 2.3 is false" in prompt


async def test_a_paper_without_reports_needs_no_judge() -> None:
    model = ScriptedChatModel(replies=[])
    grades = await judge_paper(StructuredLLM(model, name="t"), [error(0)], [], {})
    assert grades == [Grade(error_id="2405.01133v3-e0", paper_id="2405.01133v3", matched=())]
    assert model.received == []


async def test_numbers_the_judge_invents_are_ignored() -> None:
    model = ScriptedChatModel(replies=[tool_reply(MatchVerdict, {"matches": [9], "reason": ""})])
    grades = await judge_paper(StructuredLLM(model, name="t"), [error(0)], [prediction("x", 0)], {})
    assert grades[0].matched == ()


def test_scores_use_the_human_check_when_it_disagrees() -> None:
    grades = [
        Grade(error_id="a-e0", paper_id="a", matched=("f1",)),
        Grade(error_id="a-e1", paper_id="a", matched=("f2",), human_agrees=False),
        Grade(error_id="b-e0", paper_id="b", matched=()),
    ]
    scores = score_grades(grades, predictions=4)
    assert (scores.errors, scores.found, scores.predictions, scores.matched) == (3, 1, 4, 1)
    assert scores.recall == pytest.approx(1 / 3)
    assert scores.precision == pytest.approx(1 / 4)
    assert scores.unchecked == 2


def test_papers_found_counts_a_paper_once_whatever_its_errors() -> None:
    grades = [
        Grade(error_id="a-e0", paper_id="a", matched=("f1",)),
        Grade(error_id="a-e1", paper_id="a", matched=()),
        Grade(error_id="b-e0", paper_id="b", matched=()),
    ]
    scores = score_grades(grades, predictions=2)
    assert (scores.papers, scores.papers_found) == (2, 1)
    assert scores.paper_recall == pytest.approx(0.5)


def test_the_split_is_fixed_and_half_and_half() -> None:
    ids = [f"2401.{n:05d}v1" for n in range(26)]
    dev = [i for i in ids if split_of(i, ids) == "dev"]
    assert len(dev) == 13
    assert dev == [i for i in reversed(ids) if split_of(i, list(reversed(ids))) == "dev"][::-1]
