import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from auditor.agents.integrity.results import ImageTable, TableImages
from auditor.schemas import ParsedPaper, ParserInfo, Sentence, SentenceKind, Table

ROOT = Path(__file__).parent.parent


def _load(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bench = _load("table_bench", ROOT / "scripts" / "table_bench.py")
pdfs = _load("table_crops_pdfs", ROOT / "tests" / "parser" / "test_table_crops.py")

TRUTH = {
    "2401.00001": {
        "Table 1": [
            ["Method", "Acc", "F1"],
            ["Base", "70.1", "0.61"],
            ["Ours", r"\textbf{85.3}$_{\pm 0.2}$", "0.75"],
        ],
        "Table 2": [["Setting", "Time"], ["small", "1,234"], ["large", "$-$0.5"]],
    }
}


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        ("85.3%", [85.3]),
        ("**85.3**", [85.3]),
        ("1,234", [1234.0]),
        ("\u22120.5", [-0.5]),
        ("60.5 \u00b1 0.39", [60.5, 0.39]),
        ("60.5+-0.39", [60.5, 0.39]),
        ("74.5 (+1.0)", [74.5, 1.0]),
        ("0.754 / 0.602", [0.754, 0.602]),
        ("6.58M", [6.58]),
        ("ResNet-50", []),
        ("F1", []),
        ("1e-5", [1e-05]),
    ],
)
def test_numbers_are_read_whatever_their_format(cell: str, expected: list[float]) -> None:
    assert bench.numbers(cell) == expected


def test_spreads_can_be_left_out() -> None:
    assert bench.numbers("60.5 \u00b1 0.39", spreads=False) == [60.5]
    assert bench.numbers("3.2 (\u00b10.1)", spreads=False) == [3.2]


@pytest.mark.parametrize(
    ("cell", "expected"),
    [
        (r"\textbf{85.3}$_{\pm 0.2}$", "85.3 \u00b1 0.2"),
        (r"\multirow{2}{*}{Method}", "Method"),
        (r"Tran \cite{tran2016", "Tran"),
        (r"\citet{a} x", "x"),
        (r"$-$0.3", "-0.3"),
        (r"10^{-3}", "10-3"),
        (r"\# Nodes", "# Nodes"),
        (r"[0.2em] Adam", "Adam"),
        (r"\cellcolor{gray!20}91.2\%", "91.2%"),
    ],
)
def test_latex_cells_are_cleaned(cell: str, expected: str) -> None:
    assert bench.clean_latex_cell(cell) == expected


def test_a_perfect_reading_scores_full_marks_despite_formatting() -> None:
    predicted = {
        "2401.00001v2": {
            "Table 1": [
                ["Method", "Acc", "F1"],
                ["Base", "70.1", "0.61"],
                ["Ours", "**85.3** \u00b1 0.2", "0.75"],
            ],
            "Table 2": [["Setting", "Time"], ["small", "1234"], ["large", "\u22120.5"]],
        }
    }
    result = bench.score(bench_truth(), predicted)
    overall = result.overall()
    assert overall["recall"] == overall["precision"] == overall["label_accuracy"] == 1.0
    assert overall["matched_tables"] == 2


def bench_truth() -> dict[str, dict[str, list[list[str]]]]:
    return {
        paper: {
            label: [[bench.clean_latex_cell(c) for c in row] for row in rows]
            for label, rows in tables.items()
        }
        for paper, tables in TRUTH.items()
    }


def test_tables_are_matched_by_their_numbers_when_labels_differ() -> None:
    predicted = {"2401.00001": {"Table 7": [["small", "1234"], ["large", "-0.5"]]}}
    result = bench.score(bench_truth(), predicted)
    by_label = {t.truth_label: t for t in result.tables}
    assert by_label["Table 2"].predicted_label == "Table 7"
    assert by_label["Table 2"].recall == 1.0
    assert by_label["Table 1"].predicted_label is None
    assert by_label["Table 1"].recall == 0.0


def test_numbers_in_the_wrong_row_are_not_recalled() -> None:
    # The two body rows run together into one: the numbers are all in the table, but the
    # row is aligned with "Base", so the three numbers of "Ours" are not recalled.
    predicted = {
        "2401.00001": {
            "Table 1": [["Method", "Acc", "F1"], ["Base", "70.1 85.3", "0.61 0.75 0.2"]],
            "Table 2": [["Setting", "Time"], ["large", "-0.5"], ["small", "1234"]],
        }
    }
    result = bench.score(bench_truth(), predicted)
    table1, table2 = result.tables
    assert table1.bag_matched == table1.truth_numbers == 5
    assert table1.matched_numbers == 2
    assert table1.label_accuracy == pytest.approx(2 / 3)
    assert table2.recall == 1.0  # row order does not matter


def test_unmatched_predictions_lower_precision() -> None:
    predicted = {
        "2401.00001": {
            "Table 2": [["small", "1234"], ["large", "-0.5"]],
            "Table 9": [["x", "11.1", "22.2", "33.3"]],
        }
    }
    overall = bench.score(bench_truth(), predicted).overall()
    assert overall["precision"] == pytest.approx(2 / 5)
    assert overall["predicted_tables"] == 2


def test_papers_without_predictions_are_listed_not_scored() -> None:
    result = bench.score(bench_truth(), {})
    assert result.papers == 0
    assert result.papers_without_prediction == ["2401.00001"]


def test_truth_comes_from_the_latex_tables(tmp_path: Path) -> None:
    folder = tmp_path / "2401.00002"
    folder.mkdir()
    (folder / "main.tex").write_text(
        "\\documentclass{article}\n\\begin{document}\nText.\n"
        "\\begin{table}\\caption{Scores.}\n\\begin{tabular}{lc}\n\\toprule\n"
        "Model & Acc \\\\\n\\midrule\nBERT~\\cite{devlin} & 84.5\\% \\\\\n\\bottomrule\n"
        "\\end{tabular}\n\\end{table}\n\\end{document}\n",
        encoding="utf-8",
    )
    truth = bench.truth_from_sources(tmp_path)
    assert truth == {"2401.00002": {"Table 1": [["Model", "Acc"], ["BERT", "84.5%"]]}}


def test_grobid_strategy_reads_only_grobids_tables() -> None:
    paper = ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sentences=(
            Sentence(id="r1", kind=SentenceKind.TABLE, text="Base | 70.1"),
            Sentence(id="r2", kind=SentenceKind.TABLE, text="Ours | 71.5"),
        ),
        tables=(
            Table(id="t0", label="Table 1", row_ids=("r1",)),
            Table(id="layout-p3-0", label="Table 1", row_ids=("r2",), page=3),
        ),
    )
    assert bench.predict_grobid(paper) == {"Table 1": [["Base", "70.1"]]}
    layout = {"Table 1": [["Ours", "71.5"]]}
    assert bench.predict_parsed(layout, {"Table 1": [["x"]], "Table 2": [["y"]]}) == {
        "Table 1": [["Ours", "71.5"]],
        "Table 2": [["y"]],
    }


def crop_request(label: str | None = "Table 1") -> object:
    return bench.ModelRequest(
        id="p/p1/Table1", paper_id="p", page=1, label=label, prompt="P", text="T", png=b"x"
    )


def test_a_crops_single_table_takes_the_crops_label() -> None:
    reply = (
        '```json\n{"tables": [{"label": "", "rows": [["A", "B", ""], ["", ""], ["1", "2"]]}]}\n```'
    )
    assert bench.parse_reply(crop_request(), reply) == {"Table 1": [["A", "B"], ["1", "2"]]}


def test_a_page_reply_keeps_the_models_labels() -> None:
    reply = TableImages(
        tables=[
            ImageTable(label="Table 2", rows=[["a", "1"]]),
            ImageTable(label="", rows=[["b", "2"]]),
        ]
    )
    assert bench.parse_reply(crop_request(None), reply) == {
        "Table 2": [["a", "1"]],
        "page 1 table 2": [["b", "2"]],
    }
    assert bench.parse_reply(crop_request(None), reply.model_dump()) == bench.parse_reply(
        crop_request(None), reply
    )


def test_crop_requests_carry_the_image_and_the_pdfs_text(tmp_path: Path) -> None:
    pdf = pdfs.write_pdf(tmp_path / "paper.pdf", [pdfs.ruled_table_page()])
    (request,) = bench.model_requests("crop-image+words", pdf, dpi=72)
    assert (request.id, request.page, request.label) == ("paper/p1/Table1", 1, "Table 1")
    assert "Base | 70.1 | 60.5 | 10M" in request.text
    assert request.prompt == bench.WORDS_PROMPT
    system, human = request.messages()
    assert system.content == bench.WORDS_PROMPT
    assert human.content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    (plain,) = bench.model_requests("crop-image", pdf, dpi=72)
    assert "Base |" not in plain.text
    (page,) = bench.model_requests("page-image", pdf)
    assert (page.id, page.label) == ("paper/p1", None)


def test_dumped_requests_turn_replies_into_predictions(tmp_path: Path) -> None:
    pdf = pdfs.write_pdf(tmp_path / "paper.pdf", [pdfs.ruled_table_page()])
    requests = bench.model_requests("crop-image", pdf, dpi=72)
    bench.dump_requests(requests, tmp_path / "requests")
    loaded = bench.load_requests(tmp_path / "requests")
    assert [r.id for r in loaded] == [r.id for r in requests]
    assert loaded[0].png == requests[0].png
    replies = {"paper/p1/Table1": {"tables": [{"label": "Table 1", "rows": [["Base", "70.1"]]}]}}
    assert bench.predictions_from_replies(loaded, replies) == {
        "paper": {"Table 1": [["Base", "70.1"]]}
    }
    assert bench.predictions_from_replies(loaded, {}) == {"paper": {}}


async def test_the_model_hook_runs_each_request_through_the_call(tmp_path: Path) -> None:
    pdf = pdfs.write_pdf(tmp_path / "paper.pdf", [pdfs.ruled_table_page()])
    seen = []

    async def call(request: object) -> str:
        seen.append(request)
        return json.dumps({"tables": [{"label": "Table 1", "rows": [["Ours", "72.1"]]}]})

    predictions = await bench.predict_with_model("crop-image", [pdf], call, dpi=72)
    assert predictions == {"paper": {"Table 1": [["Ours", "72.1"]]}}
    assert len(seen) == 1


def test_score_command_prints_a_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    truth = tmp_path / "truth.json"
    predicted = tmp_path / "predicted.json"
    truth.write_text(json.dumps(bench_truth()), encoding="utf-8")
    predicted.write_text(json.dumps({"2401.00001": {}}), encoding="utf-8")
    assert bench.main(["score", str(truth), str(predicted), "--out", str(tmp_path / "s.json")]) == 0
    assert "recall   0.0%" in capsys.readouterr().out
    saved = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert saved["overall"]["truth_tables"] == 2


def test_crop_text_reads_the_pdfs_text_inside_each_box(tmp_path: Path) -> None:
    pdf = pdfs.write_pdf(tmp_path / "paper.pdf", [pdfs.ruled_table_page()])
    (rows,) = bench.predict_crop_text(pdf).values()
    assert rows[0][0].startswith("Table 1: Accuracy")
    assert ["Base", "70.1", "60.5", "10M"] in rows
