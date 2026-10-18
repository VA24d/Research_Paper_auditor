from pathlib import Path

from auditor.agents.integrity.set_d import apply_plants, label_for, plan_paper, plant_code
from auditor.schemas import (
    ErrorClass,
    PaperVariant,
    ParsedPaper,
    ParserInfo,
    Section,
    Sentence,
    SentenceAnchor,
    SentenceKind,
    Table,
)

ROWS = ["Method | Acc | F1", "Baseline | 71.4 | 60.2", "Ours | 75.9 | 64.8"]


def paper() -> ParsedPaper:
    table_rows = [
        Sentence(id=f"t{i}", section_id="exp", kind=SentenceKind.TABLE, text=row)
        for i, row in enumerate(ROWS)
    ]
    return ParsedPaper(
        paper_id="p1",
        parser=ParserInfo(name="test", version="1"),
        sections=(Section(id="exp", title="Experiments"),),
        sentences=(
            Sentence(id="s0", section_id="exp", text="We evaluate on two benchmarks."),
            Sentence(id="s1", section_id="exp", text="Table 1 shows that Ours reaches 75.9."),
            *table_rows,
            Sentence(
                id="s2",
                section_id="exp",
                text="Ours also gains clearly on the F1 score, 64.8 against 60.2 for Baseline.",
            ),
            Sentence(id="s3", section_id="exp", text="The method is simple to train."),
        ),
        tables=(Table(id="tab1", label="Table 1", row_ids=tuple(s.id for s in table_rows)),),
    )


def test_plants_use_the_papers_own_tables_and_are_deterministic() -> None:
    plants = plan_paper(paper())
    assert plants == plan_paper(paper())
    kinds = {p.kind for p in plants}
    assert kinds == {"placeholder", "number", "comparison", "gain", "grim"}
    injected = apply_plants(paper(), plants)
    assert injected.variant is PaperVariant.INJECTED
    texts = {s.id: s.text for s in injected.sentences}
    (number,) = [p for p in plants if p.kind == "number"]
    assert number.original in {"64.8", "60.2"}
    assert number.text in texts["s2"]
    (comparison,) = [p for p in plants if p.kind == "comparison"]
    assert comparison.sentence_id is not None
    assert "Baseline reaches 71.4, higher than Ours (75.9)" in texts[comparison.sentence_id]
    # The original is left as it was.
    assert number.original in paper().sentence("s2").text


def test_labels_name_the_planted_sentence_and_class() -> None:
    plants = plan_paper(paper())
    labels = {p.kind: label_for(p) for p in plants}
    assert labels["placeholder"].error_class is ErrorClass.UNFINISHED_CONTENT
    assert labels["gain"].error_class is ErrorClass.RESULT_INCONSISTENCY
    number = labels["number"].anchor
    assert isinstance(number, SentenceAnchor)
    assert number.sentence_id == "s2"


def test_a_code_defect_is_planted_in_a_file_the_experiment_runs(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text(
        "def main():\n    model = 1\n    print(model)\n\n\nif __name__ == '__main__':\n    main()\n"
    )
    (repo / "old.py").write_text("def unused():\n    value = 2\n    return value\n")
    plants = [plant_code(pid, repo, tmp_path / f"out-{pid}") for pid in ("a", "b", "c", "d")]
    assert all(p is not None and p.path == "train.py" for p in plants)
    kinds = {p.kind for p in plants if p is not None}
    assert kinds <= {"undefined_name", "syntax"}
    for pid, plant in zip(("a", "b", "c", "d"), plants, strict=True):
        assert plant is not None
        changed = (tmp_path / f"out-{pid}" / "train.py").read_text()
        assert changed != (repo / "train.py").read_text()
        label = label_for(plant, "https://github.com/x/y")
        assert label.error_class is ErrorClass.CODE_DEFECT
    # The original repository is untouched.
    assert "print(model)" in (repo / "train.py").read_text()
