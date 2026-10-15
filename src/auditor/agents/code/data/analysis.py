"""What the data files of a dataset actually hold: rows per split, columns, the label
distribution, and test or validation rows that also appear in the training data.

Rows are counted in constant memory. For leakage, the inputs of the held-out splits are
hashed (up to a cap) and the training split is streamed past them. Inputs are the columns
every split shares, minus the label and id-like columns, normalised (Unicode NFC, outer
spaces stripped, inner runs of whitespace collapsed; case kept) and encoded as a JSON list
of [column, value] pairs, so a separator inside a cell cannot make two rows collide.
"""

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from auditor.agents.code.data.readers import UnsupportedDataError, read_rows

ANALYSIS_VERSION = 1
LABEL_NAMES = ("label", "labels", "target", "class", "category", "sentiment", "y")
_ID_COLUMN = re.compile(r"^(?:id|idx|guid|uid|index|unnamed: ?0|.*_id)$", re.IGNORECASE)
_HELD_OUT = ("test", "validation")
_EXAMPLES = 5
_EMPTY = "\x00"


@dataclass(frozen=True)
class AnalysisLimits:
    max_hashed_rows: int = 2_000_000


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SplitStats(_Model):
    rows: int
    files: tuple[str, ...]
    columns: tuple[str, ...]
    bad_lines: int = 0
    label_counts: dict[str, int] | None = None
    unreadable: dict[str, str] = {}


class Overlap(_Model):
    """Rows of `split` whose inputs also occur in `against`."""

    split: str
    against: str
    rows_checked: int
    overlapping_rows: int
    share: float
    # 0-based row numbers in `split` of the first overlapping rows.
    examples: tuple[int, ...]
    # True when only the first `rows_checked` rows could be hashed.
    lower_bound: bool


class DataReport(_Model):
    version: int = ANALYSIS_VERSION
    splits: dict[str, SplitStats]
    label_column: str | None
    label_reason: str | None = None
    input_columns: tuple[str, ...]
    leakage: tuple[Overlap, ...]
    # Data files of a split that were not downloaded, with the reason.
    not_read: dict[str, str] = {}


def analyse(
    files: Mapping[str, Sequence[Path]],
    limits: AnalysisLimits,
    *,
    label_hint: str | None = None,
    names: Mapping[Path, str] | None = None,
) -> DataReport:
    """Analyse local data files grouped by split. `names` gives the repository path to
    report for each local file (defaults to the file name)."""
    names = names or {}
    columns = {split: _columns(paths) for split, paths in files.items()}
    shared = _shared(list(columns.values()))
    label, label_reason = _label_column(shared, label_hint)
    inputs = tuple(c for c in shared if c != label and not _ID_COLUMN.match(c))

    # For each held-out split: the first row of every distinct input, and how many rows
    # share it (duplicates inside a held-out split leak as well).
    first: dict[str, dict[int, int]] = {}
    counts: dict[str, Counter[int]] = {}
    capped: set[str] = set()
    stats: dict[str, SplitStats] = {}
    for split in sorted(files):
        hashing = bool(split in _HELD_OUT and inputs and "train" in files)
        rows, bad, labels, unreadable = 0, [0], Counter[str](), {}
        seen: dict[int, int] = {}
        repeats: Counter[int] = Counter()
        for index, row, problem, path in _rows(files[split], bad):
            if problem is not None:
                unreadable[names.get(path, path.name)] = problem
                continue
            assert row is not None
            rows = index + 1
            if label is not None and label in row:
                labels[row[label]] += 1
            key = _input_hash(row, inputs) if hashing else None
            if key is None:
                continue
            if key in seen:
                repeats[key] += 1
            elif len(seen) < limits.max_hashed_rows:
                seen[key] = index
                repeats[key] += 1
            else:
                capped.add(split)
        if hashing:
            first[split], counts[split] = seen, repeats
        stats[split] = SplitStats(
            rows=rows,
            files=tuple(names.get(p, p.name) for p in files[split]),
            columns=columns[split],
            bad_lines=bad[0],
            label_counts=dict(sorted(labels.items())) if label is not None else None,
            unreadable=unreadable,
        )

    matched: dict[str, set[int]] = {split: set() for split in first}
    if first:
        for _, row, _, _ in _rows(files["train"]):
            key = _input_hash(row, inputs) if row is not None else None
            if key is None:
                continue
            for split, table in first.items():
                if key in table:
                    matched[split].add(key)
    leakage = []
    for split in sorted(first):
        overlapping = sum(counts[split][key] for key in matched[split])
        checked = stats[split].rows
        leakage.append(
            Overlap(
                split=split,
                against="train",
                rows_checked=checked,
                overlapping_rows=overlapping,
                share=round(100 * overlapping / checked, 4) if checked else 0.0,
                examples=tuple(sorted(first[split][key] for key in matched[split])[:_EXAMPLES]),
                lower_bound=split in capped,
            )
        )
    return DataReport(
        splits=stats,
        label_column=label,
        label_reason=label_reason,
        input_columns=inputs,
        leakage=tuple(leakage),
    )


def _rows(
    paths: Sequence[Path], bad_lines: list[int] | None = None
) -> Iterator[tuple[int, dict[str, str] | None, str | None, Path]]:
    """(row number across the split's files, row, problem with the file, file). Lines
    that did not parse are added to `bad_lines[0]`."""
    index = 0
    for path in paths:
        try:
            table = read_rows(path)
            for row in table.rows:
                yield index, row, None, path
                index += 1
        except (UnsupportedDataError, OSError, ValueError) as exc:
            yield index, None, str(exc), path
            continue
        if bad_lines is not None:
            bad_lines[0] += table.bad_lines


def _columns(paths: Sequence[Path]) -> tuple[str, ...]:
    for path in paths:
        try:
            return read_rows(path).columns
        except (UnsupportedDataError, OSError, ValueError):
            continue
    return ()


def _shared(all_columns: Sequence[tuple[str, ...]]) -> tuple[str, ...]:
    """The columns every split has, in the order of the first split that has any."""
    lists = [columns for columns in all_columns if columns]
    if not lists:
        return ()
    common = set(lists[0]).intersection(*map(set, lists[1:]))
    return tuple(c for c in lists[0] if c in common)


def _label_column(columns: Sequence[str], hint: str | None) -> tuple[str | None, str | None]:
    if hint is not None and hint in columns:
        return hint, None
    lowered = {c.lower(): c for c in columns}
    for name in LABEL_NAMES:
        if name in lowered:
            return lowered[name], None
    return None, "no label column found (looked for a dataset card label or a column named " + (
        ", ".join(LABEL_NAMES) + ")"
    )


def _normalise(value: str) -> str:
    text = " ".join(unicodedata.normalize("NFC", value).split())
    return text if text and text != "null" else _EMPTY


def _input_hash(row: Mapping[str, str], inputs: Sequence[str]) -> int | None:
    pairs = [[column, _normalise(row.get(column, ""))] for column in sorted(inputs)]
    if all(value == _EMPTY for _, value in pairs):
        return None
    encoded = json.dumps(pairs, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), "big")
