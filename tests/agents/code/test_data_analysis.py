import json
from pathlib import Path

import pytest

from auditor.agents.code.data.analysis import AnalysisLimits, analyse
from auditor.agents.code.data.readers import UnsupportedDataError, read_rows
from auditor.agents.code.data.splits import map_splits

# --- mapping files to splits -----------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "split"),
    [
        ("data/train.csv", "train"),
        ("data/training_set.jsonl", "train"),
        ("dev/part-0.csv", "validation"),
        ("splits/valid.tsv", "validation"),
        ("test/data.parquet", "test"),
        ("plain_text/test-00000-of-00001.parquet", "test"),
        ("latest.csv", None),
        ("contest/results.csv", None),
        ("config.json", None),
    ],
)
def test_files_map_to_splits_by_name(path: str, split: str | None) -> None:
    mapping, skipped = map_splits([path])
    found = [name for name, paths in mapping.items() if path in paths]
    assert found == ([split] if split else [])
    assert skipped == {}


def test_a_name_with_two_splits_is_ambiguous() -> None:
    mapping, skipped = map_splits(["data/train_test.csv"])
    assert mapping == {}
    assert skipped == {"data/train_test.csv": "names more than one split"}


def test_the_file_name_wins_over_its_folder() -> None:
    mapping, _ = map_splits(["train/test.csv"])
    assert mapping == {"test": ["train/test.csv"]}


def test_shards_are_kept_together_in_order() -> None:
    paths = ["d/train-00001-of-00002.parquet", "d/train-00000-of-00002.parquet"]
    mapping, _ = map_splits(paths)
    assert mapping == {"train": sorted(paths)}


def test_card_data_files_take_precedence() -> None:
    card = {"train": ("raw/a-*",), "test": ("raw/b-*",)}
    mapping, _ = map_splits(["raw/a-0.csv", "raw/b-0.csv", "raw/test.csv"], card)
    assert mapping == {"train": ["raw/a-0.csv"], "test": ["raw/b-0.csv"]}


# --- readers ---------------------------------------------------------------------------


def test_csv_with_a_bom_and_a_long_field(tmp_path: Path) -> None:
    path = tmp_path / "train.csv"
    long = "x" * 1_000_000
    path.write_bytes(f"﻿text,label\n{long},1\nshort,0\n".encode())
    table = read_rows(path)
    assert table.columns == ("text", "label")
    assert [row["label"] for row in table.rows] == ["1", "0"]


def test_tsv_uses_tabs(tmp_path: Path) -> None:
    path = tmp_path / "dev.tsv"
    path.write_text("a\tb\n1\t2\n", encoding="utf-8")
    table = read_rows(path)
    assert list(table.rows) == [{"a": "1", "b": "2"}]


def test_jsonl_counts_bad_lines(tmp_path: Path) -> None:
    path = tmp_path / "test.jsonl"
    path.write_text(
        '{"text": "a", "label": 1}\nnot json\n{"text": "b", "label": 0}\n\n', encoding="utf-8"
    )
    table = read_rows(path)
    assert [row["text"] for row in table.rows] == ["a", "b"]
    assert table.bad_lines == 1


def test_json_array_of_records(tmp_path: Path) -> None:
    path = tmp_path / "test.json"
    path.write_text(json.dumps([{"q": "x", "a": 1}, {"q": "y", "a": 2}]), encoding="utf-8")
    assert [row["a"] for row in read_rows(path).rows] == ["1", "2"]


def test_json_config_object_is_not_data(tmp_path: Path) -> None:
    path = tmp_path / "test.json"
    path.write_text('{"lr": 0.1}', encoding="utf-8")
    with pytest.raises(UnsupportedDataError, match="not a list of records"):
        list(read_rows(path).rows)


def test_unsafe_formats_are_never_read(tmp_path: Path) -> None:
    path = tmp_path / "train.pkl"
    path.write_bytes(b"\x80\x04")
    with pytest.raises(UnsupportedDataError):
        read_rows(path)


# --- analysis --------------------------------------------------------------------------


def write_csv(path: Path, header: str, rows: list[str]) -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_rows_labels_columns_and_leakage(tmp_path: Path) -> None:
    train = write_csv(
        tmp_path / "train.csv",
        "id,text,label",
        ["1,good film,pos", "2,bad film,neg", "3,fine  film,pos", "4,  okay film ,neg"],
    )
    # Row ids differ and whitespace differs, but "fine film" and "okay film" are the same
    # input as in train: both leak.
    test = write_csv(
        tmp_path / "test.csv",
        "id,text,label",
        ["10,fine film,pos", "11,new film,neg", "12,okay film,neg"],
    )
    report = analyse({"train": [train], "test": [test]}, AnalysisLimits())
    assert report.splits["train"].rows == 4
    assert report.splits["test"].rows == 3
    assert report.splits["train"].columns == ("id", "text", "label")
    assert report.label_column == "label"
    assert report.splits["train"].label_counts == {"neg": 2, "pos": 2}
    (overlap,) = report.leakage
    assert (overlap.split, overlap.against) == ("test", "train")
    assert overlap.overlapping_rows == 2
    assert overlap.share == pytest.approx(66.6667)
    assert overlap.examples == (0, 2)
    assert not overlap.lower_bound
    assert report.input_columns == ("text",)


def test_separators_inside_cells_do_not_collide(tmp_path: Path) -> None:
    train = write_csv(tmp_path / "train.csv", "a,b,label", ['"x|y",z,0'])
    test = write_csv(tmp_path / "test.csv", "a,b,label", ['x,"y|z",0'])
    report = analyse({"train": [train], "test": [test]}, AnalysisLimits())
    assert report.leakage[0].overlapping_rows == 0


def test_hashing_cap_gives_a_lower_bound(tmp_path: Path) -> None:
    train = write_csv(tmp_path / "train.csv", "text", ["a", "b", "c"])
    test = write_csv(tmp_path / "test.csv", "text", ["a", "b", "c"])
    report = analyse({"train": [train], "test": [test]}, AnalysisLimits(max_hashed_rows=2))
    overlap = report.leakage[0]
    assert overlap.overlapping_rows == 2
    assert overlap.lower_bound


def test_label_hint_and_missing_label(tmp_path: Path) -> None:
    train = write_csv(tmp_path / "train.csv", "sentence,y_true", ["a,1", "b,0"])
    hinted = analyse({"train": [train]}, AnalysisLimits(), label_hint="y_true")
    assert hinted.label_column == "y_true"
    unknown = analyse({"train": [write_csv(tmp_path / "t2.csv", "a,b", ["1,2"])]}, AnalysisLimits())
    assert unknown.label_column is None
    assert unknown.label_reason is not None


def test_reports_are_identical_across_runs(tmp_path: Path) -> None:
    train = write_csv(tmp_path / "train.csv", "text,label", ["a,0", "b,1"])
    test = write_csv(tmp_path / "test.csv", "text,label", ["a,0"])
    first = analyse({"train": [train], "test": [test]}, AnalysisLimits())
    second = analyse({"train": [train], "test": [test]}, AnalysisLimits())
    assert first.model_dump_json() == second.model_dump_json()


def test_bad_json_lines_are_counted_in_the_report(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text('{"text": "a", "label": 1}\nbroken\n', encoding="utf-8")
    report = analyse({"train": [train]}, AnalysisLimits())
    assert report.splits["train"].rows == 1
    assert report.splits["train"].bad_lines == 1


def test_parquet_files_are_read_in_batches(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    path = tmp_path / "test-00000-of-00001.parquet"
    pq.write_table(pa.table({"text": ["a", "b"], "label": [0, 1]}), path)
    table = read_rows(path)
    assert table.columns == ("text", "label")
    assert list(table.rows) == [{"text": "a", "label": "0"}, {"text": "b", "label": "1"}]
