"""Streams rows out of data files without loading them whole.

CSV, TSV, JSON Lines, JSON arrays (up to 50 MB) and Parquet (with the optional `data`
extra) are read. Every value comes back as a string. Formats that can run code when
loaded, such as pickle, are refused, as is anything else.
"""

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# A cell can be large (whole documents), but sys.maxsize overflows on Windows.
csv.field_size_limit(2**27)
MAX_JSON_ARRAY_BYTES = 50_000_000
DATA_SUFFIXES = frozenset({".csv", ".tsv", ".jsonl", ".json", ".parquet"})
_PARQUET_BATCH_ROWS = 10_000


class UnsupportedDataError(Exception):
    """The file cannot be read as rows."""


class Table:
    """A file's columns and a single-use iterator over its rows. `bad_lines` counts JSON
    Lines that did not parse, as the rows are read."""

    def __init__(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix not in DATA_SUFFIXES:
            raise UnsupportedDataError(f"{path.name}: {suffix or 'no extension'} is not read")
        self.path = path
        self.bad_lines = 0
        self.columns: tuple[str, ...] = ()
        self.rows: Iterator[dict[str, str]]
        match suffix:
            case ".csv" | ".tsv":
                self.rows = self._delimited("\t" if suffix == ".tsv" else ",")
            case ".jsonl":
                self.rows = self._json_lines()
            case ".json":
                self.rows = self._json_array()
            case _:
                self.rows = self._parquet()

    def _delimited(self, delimiter: str) -> Iterator[dict[str, str]]:
        handle = self.path.open(encoding="utf-8-sig", errors="replace", newline="")
        reader = csv.reader(handle, delimiter=delimiter)
        header = next(reader, None)
        self.columns = tuple(header or ())

        def rows() -> Iterator[dict[str, str]]:
            with handle:
                for row in reader:
                    if row:
                        yield dict(zip(self.columns, row, strict=False))

        return rows()

    def _json_lines(self) -> Iterator[dict[str, str]]:
        handle = self.path.open(encoding="utf-8-sig", errors="replace")
        records = self._records(handle)
        first = next(records, None)
        if first is not None:
            self.columns = tuple(first)

        def rows() -> Iterator[dict[str, str]]:
            with handle:
                if first is not None:
                    yield first
                yield from records

        return rows()

    def _records(self, handle: Any) -> Iterator[dict[str, str]]:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                self.bad_lines += 1
                continue
            if isinstance(record, dict):
                yield _strings(record)
            else:
                self.bad_lines += 1

    def _json_array(self) -> Iterator[dict[str, str]]:
        if self.path.stat().st_size > MAX_JSON_ARRAY_BYTES:
            raise UnsupportedDataError(f"{self.path.name} is a JSON file larger than 50 MB")
        data = json.loads(self.path.read_text(encoding="utf-8-sig", errors="replace"))
        if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
            raise UnsupportedDataError(f"{self.path.name} is not a list of records")
        if data:
            self.columns = tuple(data[0])
        return (_strings(record) for record in data)

    def _parquet(self) -> Iterator[dict[str, str]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise UnsupportedDataError(
                "parquet needs the optional dependency: uv sync --extra data"
            ) from exc
        # Opened from a file handle, not memory-mapped, so Windows can delete it afterwards.
        handle = self.path.open("rb")
        parquet = pq.ParquetFile(handle)
        self.columns = tuple(parquet.schema_arrow.names)

        def rows() -> Iterator[dict[str, str]]:
            with handle:
                for batch in parquet.iter_batches(batch_size=_PARQUET_BATCH_ROWS):
                    for record in batch.to_pylist():
                        yield _strings(record)

        return rows()


def _strings(record: dict[str, Any]) -> dict[str, str]:
    return {
        str(key): value
        if isinstance(value, str)
        else json.dumps(value, sort_keys=True, default=str)
        for key, value in record.items()
    }


def read_rows(path: Path) -> Table:
    return Table(path)
