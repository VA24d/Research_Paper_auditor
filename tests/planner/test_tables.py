from pathlib import Path

import pytest

from auditor.parser.tei import tei_to_paper
from auditor.planner.tables import canonical_key, split_cells, table_settings
from auditor.schemas import (
    CELL_SEPARATOR,
    PaperSource,
    ParsedPaper,
    ParserInfo,
    Sentence,
    SentenceKind,
    Table,
)

SAMPLE = Path(__file__).parents[1] / "fixtures" / "tei" / "sample.tei.xml"


def paper_with(*tables: list[list[str]]) -> ParsedPaper:
    sentences = [Sentence(id="s0", text="Body text.")]
    built = []
    for t_index, rows in enumerate(tables):
        row_ids = []
        for row in rows:
            sentence_id = f"s{len(sentences)}"
            sentences.append(
                Sentence(id=sentence_id, kind=SentenceKind.TABLE, text=CELL_SEPARATOR.join(row))
            )
            row_ids.append(sentence_id)
        built.append(Table(id=f"t{t_index}", row_ids=tuple(row_ids)))
    return ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sentences=tuple(sentences),
        tables=tuple(built),
    )


def found(paper: ParsedPaper) -> list[tuple[str, str, str | None]]:
    return [(s.key, s.raw_value, s.context) for s in table_settings(paper)]


def test_key_value_table() -> None:
    paper = paper_with([["Parameter", "Value"], ["dim", "4096"], ["n_layers", "32"]])
    assert found(paper) == [("hidden_size", "4096", None), ("num_layers", "32", None)]


def test_table_without_header_row() -> None:
    assert found(paper_with([["Batch size", "64"], ["Optimizer", "AdamW"]])) == [
        ("batch_size", "64", None),
        ("optimizer", "AdamW", None),
    ]


def test_one_column_per_configuration_uses_the_header_as_context() -> None:
    paper = paper_with([["", "SST-2", "MRPC"], ["Batch Size", "16", "32"], ["# Epochs", "3", "3"]])
    assert found(paper) == [
        ("batch_size", "16", "SST-2"),
        ("batch_size", "32", "MRPC"),
        ("epochs", "3", None),
    ]


def test_one_row_per_configuration_uses_the_row_label_as_context() -> None:
    paper = paper_with([["Model", "Epochs", "Base LR"], ["ViT-B/16", "7", "8e-4"]])
    assert found(paper) == [("epochs", "7", "ViT-B/16"), ("learning_rate", "8e-4", "ViT-B/16")]


@pytest.mark.parametrize(
    "rows",
    [
        # A numeric first row is data, so its numbers cannot label the columns.
        [["Batch Size", "32", "16", "1"], ["Sequence Length", "512", "256", "128"]],
        # Several values merged into one cell.
        [["", "A", "B"], ["Learning Rate", "5e-4 1e-4", "2e-4"]],
        # Different values with no header to say which is which.
        [["Learning rate", "1e-4", "2e-5"]],
        # A results table.
        [["System", "MNLI", "QQP"], ["BERT", "86.7", "72.1"]],
    ],
)
def test_ambiguous_tables_give_nothing(rows: list[list[str]]) -> None:
    assert found(paper_with(rows)) == []


def test_spans_point_at_the_cell() -> None:
    paper = paper_with([["Weight decay", "0.01"]])
    (setting,) = table_settings(paper)
    assert isinstance(setting.source, PaperSource)
    anchor = setting.source.anchor
    assert anchor.start is not None
    assert anchor.end is not None
    assert paper.sentence(anchor.sentence_id).text[anchor.start : anchor.end] == "0.01"


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("Learning Rate (LR)", "learning_rate"),
        ("# Epochs", "epochs"),
        ("n_kv_heads", "num_kv_heads"),
        ("Max Seq. Len.", "max_sequence_length"),
        ("Accuracy", None),
    ],
)
def test_canonical_key(name: str, key: str | None) -> None:
    assert canonical_key(name) == key


def test_split_cells_keeps_offsets() -> None:
    row = CELL_SEPARATOR.join(["", "Batch", " 32 "])
    cells = split_cells(row)
    assert [c.text for c in cells] == ["", "Batch", "32"]
    assert all(row[c.start : c.end] == c.text for c in cells)


def test_grobid_table_rows_become_table_sentences() -> None:
    table = (
        '<figure type="table" xml:id="tab_0"><head>Table 3:</head>'
        "<figDesc>Hyperparameters.</figDesc>"
        '<table coords="2,100,100,300,60"><row><cell>Parameter</cell><cell>Value</cell></row>'
        "<row><cell>batch size</cell><cell>256</cell></row>"
        "<row><cell></cell><cell></cell></row></table></figure>"
    )
    tei = SAMPLE.read_text(encoding="utf-8").replace("</body>", table + "</body>")
    paper = tei_to_paper(tei, paper_id="x")
    (parsed,) = paper.tables
    assert parsed.label == "Table 3"
    assert [paper.sentence(r).text for r in parsed.row_ids] == [
        "Parameter | Value",
        "batch size | 256",
    ]
    assert all(paper.sentence(r).kind is SentenceKind.TABLE for r in parsed.row_ids)
    assert found(paper) == [("batch_size", "256", None)]
