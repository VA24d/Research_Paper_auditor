"""Which data files belong to which split.

A Hugging Face card's `data_files` decide when there is one. Otherwise each file is named
by the deepest part of its path that names a split ("dev/part-0.csv" is validation,
"train/test.csv" is test). A part that names two splits ("train_test.csv") is ambiguous
and skipped with a reason, and words that merely contain a split name ("latest",
"contest") do not count.
"""

import fnmatch
import re
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath

_TOKEN = re.compile(
    r"(?<![a-z0-9])(train|training|valid|validation|val|dev|test|testing)(?![a-z0-9])"
)
_SPLIT = {
    "train": "train",
    "training": "train",
    "valid": "validation",
    "validation": "validation",
    "val": "validation",
    "dev": "validation",
    "test": "test",
    "testing": "test",
}
AMBIGUOUS = "names more than one split"


class AmbiguousSplitError(ValueError):
    """A path names more than one split, such as "train_test.csv"."""


def split_of(path: str) -> str | None:
    """The split a path names, or None if it names none."""
    for part in reversed(PurePosixPath(path).parts):
        stem = part.lower().rsplit(".", 1)[0]
        found = {_SPLIT[m.group(1)] for m in _TOKEN.finditer(stem)}
        if len(found) > 1:
            raise AmbiguousSplitError(path)
        if found:
            return found.pop()
    return None


def map_splits(
    paths: Iterable[str], card_files: Mapping[str, tuple[str, ...]] | None = None
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Paths grouped by split (each list sorted, so shards keep their order), and the
    paths left out with the reason."""
    mapping: dict[str, list[str]] = {}
    skipped: dict[str, str] = {}
    for path in sorted(set(paths)):
        if card_files:
            for name, patterns in card_files.items():
                if any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns):
                    mapping.setdefault(_SPLIT.get(name, name), []).append(path)
                    break
            continue
        try:
            split = split_of(path)
        except AmbiguousSplitError:
            skipped[path] = AMBIGUOUS
            continue
        if split is not None:
            mapping.setdefault(split, []).append(path)
    return mapping, skipped
