import dataclasses
from pathlib import Path

from auditor.planner.tasks import plan_paper
from auditor.planner.theory import formal_density, theory_tasks
from auditor.schemas import (
    ParsedPaper,
    ParserInfo,
    Section,
    Sentence,
    SentenceAnchor,
    SettingTarget,
    TaskKind,
)
from auditor.testing import offline_context

PROSE = "Large language models changed how researchers write and read scientific papers today."
FORMAL = "By Lemma 2 we have f(x) = sum_i a_i x^i <= C n^2 for all x in [0, 1]."


def paper(sections: dict[str, list[str]]) -> ParsedPaper:
    return ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sections=tuple(Section(id=key, title=key) for key in sections),
        sentences=tuple(
            Sentence(id=f"{key}-{i}", section_id=key, text=text)
            for key, texts in sections.items()
            for i, text in enumerate(texts)
        ),
    )


def test_formal_text_is_denser_than_prose() -> None:
    assert formal_density([PROSE]) == 1.0
    assert formal_density([FORMAL]) > 2.0


def test_only_formal_sections_become_theory_tasks() -> None:
    tasks = theory_tasks(
        paper({"intro": [PROSE] * 3, "proofs": [FORMAL] * 3, "outlook": [PROSE]}), 3
    )
    (task,) = tasks
    assert task.kind is TaskKind.THEORY
    assert task.sections == ("proofs",)
    assert task.anchor == SentenceAnchor(sentence_id="proofs-0")
    assert task.target == SettingTarget(key="math")


def test_theorem_statements_count_even_in_words() -> None:
    statement = "Theorem 3. Every bounded monotone sequence of real numbers converges to a limit."
    (task,) = theory_tasks(paper({"intro": [PROSE], "main": [statement]}), 3)
    assert task.sections == ("main",)


def test_neighbouring_formal_sections_form_one_segment() -> None:
    (task,) = theory_tasks(paper({"a": [FORMAL], "b": [FORMAL], "c": [PROSE]}), 3)
    assert task.sections == ("a", "b")


def test_the_densest_segments_are_kept_within_the_limit() -> None:
    light = "Theorem 1 holds for every graph that has at least one vertex in it."
    sections = {
        "s1": [light],
        "gap1": [PROSE],
        "s2": [FORMAL] * 4,
        "gap2": [PROSE],
        "s3": [FORMAL] * 2,
    }
    tasks = theory_tasks(paper(sections), 2)
    assert [t.sections for t in tasks] == [("s2",), ("s3",)]


def test_long_formal_runs_are_split_into_segments() -> None:
    sections = {f"s{i}": [FORMAL] * 60 for i in range(3)}
    tasks = theory_tasks(paper(sections), 5)
    assert len(tasks) >= 2
    assert sorted(key for t in tasks for key in t.sections or ()) == ["s0", "s1", "s2"]


async def test_the_planner_adds_theory_tasks_only_when_enabled(tmp_path: Path) -> None:
    formal = paper({"intro": [PROSE], "proofs": [FORMAL] * 2})
    ctx = offline_context(tmp_path, paper=formal)
    assert not [t for t in await plan_paper(formal, ctx) if t.kind is TaskKind.THEORY]
    settings = ctx.settings.model_copy(update={"theory_check_enabled": True})
    tasks = await plan_paper(formal, dataclasses.replace(ctx, settings=settings))
    assert [t.sections for t in tasks if t.kind is TaskKind.THEORY] == [("proofs",)]
