"""LLM reading of hyperparameter tables the deterministic reader could not use.

GROBID often mangles tables: merged cells, headers split from their columns, or rows
flattened into body text. The model is shown the rows with their numbers and returns
settings with values copied from a row; a value that is not literally in that row, or a
context that is not in the table, is discarded.
"""

import logging
import re
from collections.abc import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from auditor.infra.llm import StructuredLLM
from auditor.planner.settings import looks_like_table_row
from auditor.planner.tables import ALIASES, canonical_key, table_value
from auditor.schemas import (
    ExtractedSetting,
    PaperSource,
    ParsedPaper,
    Sentence,
    SentenceAnchor,
    SentenceKind,
)

logger = logging.getLogger(__name__)


class TableSettingReading(BaseModel):
    key: str = Field(description="Which setting this is, as one of the listed keys")
    row: int = Field(description="Number of the row the value is in")
    value: str = Field(description="The value, copied exactly as it appears in that row")
    context: str | None = Field(
        default=None,
        description="The model, dataset or task the value applies to, copied exactly from "
        "the table, or null if it applies to everything",
    )


class TableReading(BaseModel):
    settings: list[TableSettingReading] = Field(default_factory=list)


SYSTEM_PROMPT = """You read hyperparameter tables from research papers.

You get the rows of one table, numbered, with cells separated by " | ". Some rows are
garbled by PDF extraction, so use the row and column labels to work out which value
belongs to which setting and configuration.

Report only training or architecture settings of the paper's own experiments, using these
keys: {keys}. Copy each value exactly as it appears in its row. If the table gives different
values for different models, datasets or tasks, give one entry per value with the label it
applies to as "context", copied exactly. Skip anything you are unsure about; an empty list
is a valid answer."""

_HINT_WORDS = ("hyperparam", "learning rate", "batch", "epoch", "optimizer", "warmup", "decay")


def table_units(paper: ParsedPaper, already: Sequence[ExtractedSetting]) -> list[list[Sentence]]:
    """Tables, and runs of table-like body sentences, that look like hyperparameters."""
    units: list[list[Sentence]] = [
        [paper.sentence(row_id) for row_id in table.row_ids] for table in paper.tables
    ]
    run: list[Sentence] = []
    for sentence in paper.sentences:
        if sentence.kind is not SentenceKind.TABLE and looks_like_table_row(sentence.text):
            run.append(sentence)
            continue
        if run:
            units.append(run)
            run = []
    if run:
        units.append(run)

    def relevant(unit: list[Sentence]) -> bool:
        text = " ".join(s.text for s in unit).lower()
        names = (s.text.split("|", 1)[0] for s in unit)
        return any(w in text for w in _HINT_WORDS) or any(canonical_key(n) for n in names)

    return [unit for unit in units if relevant(unit)]


def _accept(
    reading: TableSettingReading, unit: list[Sentence], taken: set[tuple[str, int]]
) -> ExtractedSetting | None:
    """The reading as a setting, or None if it does not quote the table. A value that
    occurs several times in its row ("16 | 16" for two models) takes the first occurrence
    no earlier reading has taken, so each configuration keeps its own anchor."""
    if reading.key not in ALIASES or not 0 <= reading.row < len(unit):
        return None
    row = unit[reading.row]
    value = reading.value.strip()
    starts = [m.start() for m in re.finditer(re.escape(value), row.text)] if value else []
    if not starts:
        return None
    start = next((s for s in starts if (row.id, s) not in taken), starts[0])
    context = reading.context.strip() if reading.context else None
    if context and not any(context in s.text for s in unit):
        return None
    parsed = table_value(reading.key, value)
    if parsed is None:
        return None
    return ExtractedSetting(
        key=reading.key,
        raw_name=reading.key.replace("_", " "),
        raw_value=value,
        value=parsed,
        unit="%" if value.endswith("%") else None,
        context=context,
        source=PaperSource(
            anchor=SentenceAnchor(sentence_id=row.id, start=start, end=start + len(value))
        ),
    )


def _identity(setting: ExtractedSetting) -> tuple[object, ...]:
    """One value per setting, configuration and row."""
    anchor = setting.source.anchor if isinstance(setting.source, PaperSource) else None
    return (setting.key, setting.context, anchor.sentence_id if anchor else None)


def _position(setting: ExtractedSetting) -> tuple[str, int] | None:
    if not isinstance(setting.source, PaperSource) or setting.source.anchor.start is None:
        return None
    return (setting.source.anchor.sentence_id, setting.source.anchor.start)


async def read_tables(
    paper: ParsedPaper, llm: StructuredLLM, already: Sequence[ExtractedSetting]
) -> list[ExtractedSetting]:
    found: list[ExtractedSetting] = []
    seen = {_identity(s) for s in already}
    taken = {p for p in map(_position, already) if p is not None}
    system = SystemMessage(SYSTEM_PROMPT.format(keys=", ".join(ALIASES)))
    for unit in table_units(paper, already):
        rows = "\n".join(f"Row {index}: {sentence.text}" for index, sentence in enumerate(unit))
        result = await llm.generate(
            TableReading, [system, HumanMessage(rows)], purpose="table-read"
        )
        for reading in result.value.settings:
            setting = _accept(reading, unit, taken)
            if setting is None:
                logger.warning("discarded a table reading that does not quote the table")
            elif _identity(setting) not in seen:
                seen.add(_identity(setting))
                position = _position(setting)
                if position is not None:
                    taken.add(position)
                found.append(setting)
    return found
