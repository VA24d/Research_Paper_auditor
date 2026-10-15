"""Splits defined by list files, such as `splits/train.txt` and `splits/test.txt` with one
example per line."""

import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import PurePosixPath

from auditor.agents.code.extract.normalise import split_from_parts
from auditor.schemas import ExtractedSetting, RepositoryRef, RepositorySource

LFS_POINTER = "version https://git-lfs.github.com/spec/v1"
_SPLITS = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "dev": "validation",
    "test": "test",
    "testing": "test",
}
# The whole file name must name a split: "train", "cifar_train", "train_list". Names such as
# "test_results" or "requirements" are not split lists.
_LIST_NAME = re.compile(
    r"^(?:[a-z0-9]+[_-])?(?P<split>" + "|".join(_SPLITS) + r")"
    r"(?:[_-](?:list|ids|files|split|set|images|idx|index))?$"
)
_ORDER = ("train", "validation", "test")


def scan_split_lists(files: Mapping[str, str], repository: RepositoryRef) -> list[ExtractedSetting]:
    """`files` maps paths of `.txt` files to their text. Each folder holding at least a
    train and a test list gives one `data_split` from the line counts."""
    folders: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(files):
        pure = PurePosixPath(path)
        match = _LIST_NAME.match(pure.stem.lower())
        if pure.suffix.lower() == ".txt" and match:
            folders[str(pure.parent)][_SPLITS[match["split"]]].append(path)

    found = []
    for lists in folders.values():
        # Two lists for one split (train.txt and train_list.txt) leave the split unclear.
        if any(len(paths) > 1 for paths in lists.values()):
            continue
        counts = {}
        for split, (path,) in lists.items():
            text = files[path]
            if text.startswith(LFS_POINTER):
                break
            counts[split] = (path, sum(1 for line in text.splitlines() if line.strip()))
        else:
            if "train" not in counts or "test" not in counts:
                continue
            ordered = [(split, *counts[split]) for split in _ORDER if split in counts]
            value = split_from_parts([float(count) for _, _, count in ordered])
            if value is None:
                continue
            test_path = counts["test"][0]
            found.append(
                ExtractedSetting(
                    key="data_split",
                    raw_name="split list files",
                    raw_value=" / ".join(
                        f"{PurePosixPath(path).name} {count}" for _, path, count in ordered
                    ),
                    value=value,
                    source=RepositorySource(repository=repository, path=test_path, line=1),
                )
            )
            # Each list's line count is the number of examples in that split.
            found += [
                ExtractedSetting(
                    key=f"{split}_examples",
                    raw_name="split list file",
                    raw_value=f"{PurePosixPath(path).name} {count} lines",
                    value=count,
                    source=RepositorySource(repository=repository, path=path, line=1),
                )
                for split, path, count in ordered
            ]
    return found
