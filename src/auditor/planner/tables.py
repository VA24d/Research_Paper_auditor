"""Settings from hyperparameter tables.

Two table shapes are read without an LLM:
- a setting per row: "batch size | 32" or, with one column per configuration,
  "batch size | 16 | 32" under a header "| SST-2 | MRPC";
- a setting per column: a header "Model | Epochs | Base LR" and one row per configuration.

A value is only taken when it is clear which configuration it belongs to; anything else
is left for the optional LLM reader, which must quote the table exactly.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass

from auditor.planner.evaluation_setup import held_out_setting
from auditor.planner.settings import extract_settings, parse_number
from auditor.schemas import (
    CELL_SEPARATOR,
    ExtractedSetting,
    PaperSource,
    ParsedPaper,
    SentenceAnchor,
    SettingValue,
)

# Canonical keys and the names papers use for them in tables.
ALIASES: dict[str, tuple[str, ...]] = {
    "learning_rate": (
        "learning rate",
        "lr",
        "base lr",
        "peak lr",
        "peak learning rate",
        "initial learning rate",
        "max learning rate",
    ),
    "batch_size": ("batch size", "batch", "train batch size", "global batch size"),
    "epochs": ("epochs", "epoch", "num epochs", "number of epochs", "training epochs"),
    "training_steps": ("steps", "training steps", "train steps", "max steps", "iterations"),
    "warmup_steps": ("warmup steps", "warm-up steps", "warmup iterations"),
    "warmup_ratio": ("warmup ratio", "warm-up ratio", "warmup proportion"),
    "weight_decay": ("weight decay", "wd"),
    "dropout": ("dropout", "dropout rate", "dropout prob", "dropout probability"),
    "label_smoothing": ("label smoothing",),
    "optimizer": ("optimizer", "optimiser"),
    "lr_schedule": ("lr schedule", "lr scheduler", "learning rate schedule", "lr decay"),
    "seed": ("seed", "random seed"),
    "max_sequence_length": (
        "max seq len",
        "max seq length",
        "max sequence length",
        "maximum sequence length",
        "sequence length",
        "context length",
        "context len",
        "max length",
    ),
    "hidden_size": ("dim", "d model", "hidden size", "model dimension", "embedding dim"),
    "intermediate_size": ("hidden dim", "ffn dim", "d ff", "intermediate size", "mlp size"),
    "num_layers": ("n layers", "layers", "num layers", "number of layers", "depth"),
    "num_attention_heads": ("n heads", "heads", "num heads", "attention heads"),
    "num_kv_heads": ("n kv heads", "kv heads"),
    "head_dim": ("head dim",),
    "vocab_size": ("vocab size", "vocabulary size"),
    "sliding_window": ("window size", "sliding window"),
}
TEXT_VALUED = {"optimizer", "lr_schedule"}
_NAME_TO_KEY = {name: key for key, names in ALIASES.items() for name in names}
_TEXT_VALUE = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]{1,30}$")
_LABEL = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class Cell:
    text: str
    start: int
    end: int


def split_cells(row: str) -> list[Cell]:
    cells = []
    offset = 0
    for part in row.split(CELL_SEPARATOR):
        stripped = part.strip()
        lead = len(part) - len(part.lstrip())
        cells.append(Cell(stripped, offset + lead, offset + lead + len(stripped)))
        offset += len(part) + len(CELL_SEPARATOR)
    return cells


def canonical_key(name: str) -> str | None:
    """`# Epochs`, `Learning Rate (LR)` and `n_layers` to their canonical keys."""
    cleaned = re.sub(r"\([^)]*\)", " ", name.lower())
    cleaned = re.sub(r"[_\-#.:]+", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    return _NAME_TO_KEY.get(cleaned)


def table_value(key: str, raw: str) -> SettingValue | None:
    if key in TEXT_VALUED:
        return raw.lower() if _TEXT_VALUE.match(raw) else None
    return parse_number(raw)


def _setting(
    key: str, name: str, row_id: str, cell: Cell, value: SettingValue, context: str | None
) -> ExtractedSetting:
    return ExtractedSetting(
        key=key,
        raw_name=name,
        raw_value=cell.text,
        value=value,
        unit="%" if cell.text.endswith("%") else None,
        context=context,
        source=PaperSource(
            anchor=SentenceAnchor(sentence_id=row_id, start=cell.start, end=cell.end)
        ),
    )


def _row_settings(row_id: str, cells: list[Cell], header: list[Cell]) -> Iterator[ExtractedSetting]:
    key = canonical_key(cells[0].text) if cells else None
    if key is None:
        return
    values = [
        (index, cell, table_value(key, cell.text))
        for index, cell in enumerate(cells[1:], start=1)
        if cell.text
    ]
    parsed = [(index, cell, value) for index, cell, value in values if value is not None]
    if not parsed or len(parsed) != len(values):
        return
    if len({repr(value) for _, _, value in parsed}) == 1:
        _, cell, value = parsed[0]
        yield _setting(key, cells[0].text, row_id, cell, value, None)
    elif len(header) == len(cells):
        labels = [header[index].text for index, _, _ in parsed]
        # Column labels name configurations ("SST-2"); a row of numbers is data, not a header.
        if not all(_LABEL.search(label) for label in labels):
            return
        for (_, cell, value), label in zip(parsed, labels, strict=True):
            yield _setting(key, cells[0].text, row_id, cell, value, label)


def _column_settings(
    rows: list[tuple[str, list[Cell]]], header: list[Cell]
) -> Iterator[ExtractedSetting]:
    keys = [canonical_key(cell.text) for cell in header]
    if not any(keys[1:]):
        return
    for row_id, cells in rows:
        if len(cells) != len(header) or not _LABEL.search(cells[0].text):
            continue
        for index, key in enumerate(keys):
            if index == 0 or key is None or not cells[index].text:
                continue
            value = table_value(key, cells[index].text)
            if value is not None:
                yield _setting(key, header[index].text, row_id, cells[index], value, cells[0].text)


def table_settings(paper: ParsedPaper) -> list[ExtractedSetting]:
    found: list[ExtractedSetting] = []
    for table in paper.tables:
        # Layout tables repeat GROBID's tables page by page; reading both would make every
        # setting twice.
        if table.page is not None:
            continue
        rows = [(row_id, split_cells(paper.sentence(row_id).text)) for row_id in table.row_ids]
        if not rows:
            continue
        header = rows[0][1]
        # The first row is either a header or, in a table without one, a setting row.
        found.extend(_row_settings(rows[0][0], header, []))
        for row_id, cells in rows[1:]:
            found.extend(_row_settings(row_id, cells, header))
        found.extend(_column_settings(rows[1:], header))
    return found


def paper_settings(paper: ParsedPaper) -> list[ExtractedSetting]:
    """Settings stated in sentences and in tables, plus the held-out check implied by a
    stated data setup."""
    settings = [*extract_settings(paper), *table_settings(paper)]
    held_out = held_out_setting(settings)
    return [*settings, held_out] if held_out is not None else settings
