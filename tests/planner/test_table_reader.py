from pathlib import Path
from typing import Any

from auditor.infra.llm import StructuredLLM
from auditor.planner.table_reader import TableReading, read_tables, table_units
from auditor.planner.tasks import PlannerOptions, plan_paper
from auditor.schemas import (
    CELL_SEPARATOR,
    ParsedPaper,
    ParserInfo,
    RepositoryRef,
    Sentence,
    SentenceKind,
    Table,
    TaskKind,
)
from auditor.testing import ScriptedChatModel, offline_context, tool_reply

GARBLED = [
    "Hyperparameters | Fine-Tune | LoRA",
    "Learning Rate | 5.00E-06 | 5.00E-04 1.00E-04",
    "Batch Size | 16 32",
]
FLATTENED = "SST-2 MRPC CoLA Batch Size 16 16 16 Learning Rate 5E-04 4E-04 4E-04"


def garbled_paper() -> ParsedPaper:
    sentences = [Sentence(id="s0", text="We fine-tune with LoRA.")]
    row_ids = []
    for row in GARBLED:
        sentence_id = f"s{len(sentences)}"
        sentences.append(Sentence(id=sentence_id, kind=SentenceKind.TABLE, text=row))
        row_ids.append(sentence_id)
    sentences.append(Sentence(id=f"s{len(sentences)}", text=FLATTENED))
    sentences.append(Sentence(id=f"s{len(sentences)}", text="Results improve."))
    return ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sentences=tuple(sentences),
        tables=(Table(id="t0", row_ids=tuple(row_ids)),),
    )


def reading(**fields: Any) -> dict[str, Any]:
    return {"key": "learning_rate", "row": 1, "value": "5.00E-04", "context": "LoRA", **fields}


def scripted(*answers: list[dict[str, Any]]) -> StructuredLLM:
    model = ScriptedChatModel(replies=[tool_reply(TableReading, {"settings": a}) for a in answers])
    return StructuredLLM(model, name="test:scripted")


def test_units_are_garbled_tables_and_flattened_rows() -> None:
    units = table_units(garbled_paper(), already=[])
    assert [[s.id for s in unit] for unit in units] == [["s1", "s2", "s3"], ["s4"]]


async def test_only_readings_that_quote_the_table_are_kept() -> None:
    paper = garbled_paper()
    llm = scripted(
        [
            reading(),
            reading(value="3e-4"),
            reading(context="RoBERTa"),
            reading(key="momentum"),
            reading(row=9),
            reading(key="batch_size", row=2, value="16", context=None),
        ],
        [],
    )
    found = await read_tables(paper, llm, already=[])
    assert [(s.key, s.raw_value, s.context) for s in found] == [
        ("learning_rate", "5.00E-04", "LoRA"),
        ("batch_size", "16", None),
    ]
    anchor = found[0].source.anchor  # type: ignore[union-attr]
    text = paper.sentence(anchor.sentence_id).text
    assert text[anchor.start : anchor.end] == "5.00E-04"


async def test_planner_reads_tables_only_when_there_is_a_repository(tmp_path: Path) -> None:
    paper = garbled_paper()
    no_repo = await plan_paper(
        paper,
        offline_context(tmp_path, paper=paper, llm=scripted()),
        PlannerOptions(detect_repository=False),
    )
    assert not [t for t in no_repo if t.kind is TaskKind.CODE]

    llm = scripted([reading()], [])
    tasks = await plan_paper(
        paper,
        offline_context(tmp_path, paper=paper, llm=llm),
        PlannerOptions(repository=RepositoryRef(url="https://github.com/a/b")),
    )
    code = [t for t in tasks if t.kind is TaskKind.CODE]
    assert [(t.setting.key, t.setting.context) for t in code if t.setting] == [
        ("learning_rate", "LoRA")
    ]
    assert CELL_SEPARATOR in paper.sentence(code[0].anchor.sentence_id).text  # type: ignore[union-attr]
