"""Split settings in Python code, read with `ast` (the code is never imported or run).

Recognised: argparse options and function defaults named like `test_size` or
`val_split`, the same names as keyword arguments or assignments, `train_test_split`,
`random_split`, Keras `fit(validation_split=...)` and `load_dataset(name, config, split=...)`.
"""

import ast
import re
from dataclasses import dataclass

from auditor.agents.code.extract.normalise import (
    DATASET_NAMES,
    as_percent,
    setting_name,
    split_from_parts,
    split_key,
)
from auditor.schemas import ExtractedSetting, RepositoryRef, RepositorySource, SettingValue

# `load_dataset` names that load local files rather than a dataset on the Hub.
_BUILDERS = frozenset(
    {
        "csv",
        "json",
        "parquet",
        "text",
        "arrow",
        "pandas",
        "generator",
        "sql",
        "imagefolder",
        "audiofolder",
        "videofolder",
        "webdataset",
    }
)
# "train[90%:]", "train[:10%]", "train[80%:90%]".
_SLICE = re.compile(r"^(?P<split>\w+)\[(?:(?P<start>\d+)%)?:(?:(?P<end>\d+)%)?\]$")


@dataclass
class _Found:
    line: int
    column: int
    setting: ExtractedSetting


class _Scanner:
    def __init__(self, path: str, text: str, repository: RepositoryRef) -> None:
        self.path = path
        self.text = text
        self.repository = repository
        self.found: list[_Found] = []

    def add(
        self,
        node: ast.AST,
        key: str,
        raw_name: str,
        value: SettingValue,
        context: str | None = None,
    ) -> None:
        raw_value = ast.get_source_segment(self.text, node) or str(value)
        line = getattr(node, "lineno", 1)
        self.found.append(
            _Found(
                line,
                getattr(node, "col_offset", 0),
                ExtractedSetting(
                    key=key,
                    raw_name=raw_name,
                    raw_value=raw_value,
                    value=value,
                    context=context,
                    source=RepositorySource(repository=self.repository, path=self.path, line=line),
                ),
            )
        )

    def named_value(self, raw_name: str, node: ast.expr | None, context: str | None = None) -> None:
        """A value under a name such as `test_size`, if the name is about a split."""
        if node is None:
            return
        name = setting_name(raw_name)
        value = _constant(node)
        key = split_key(name)
        if key is not None:
            percent = as_percent(value, name)
            if percent is not None:
                self.add(node, key, raw_name, percent, context)
        elif name in DATASET_NAMES and isinstance(value, str) and value:
            self.add(node, "dataset", raw_name, value)

    def call(self, node: ast.Call, target: str | None) -> None:
        function = _function_name(node.func)
        if function == "add_argument":
            options = [_constant(arg) for arg in node.args]
            names = [o for o in options if isinstance(o, str)]
            default = _keyword(node, "default")
            if names and default is not None:
                for option in names:
                    if split_key(setting_name(option)) or setting_name(option) in DATASET_NAMES:
                        self.named_value(option, default)
                        break
            return
        if function == "load_dataset":
            self.load_dataset(node, target)
            return
        if function == "random_split":
            lengths = node.args[1] if len(node.args) > 1 else _keyword(node, "lengths")
            parts = _numbers(lengths)
            split = split_from_parts(parts) if parts else None
            if lengths is not None and split is not None:
                self.add(lengths, "data_split", "random_split", split)
            return
        if function == "train_test_split" and _keyword(node, "test_size") is None:
            train = _keyword(node, "train_size")
            share = as_percent(_constant(train)) if train is not None else None
            if train is not None and share is not None:
                self.add(train, "test_fraction", "train_size", round(100 - share, 4))
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            context = None
            if function == "fit" and setting_name(keyword.arg) == "validation_split":
                context = "of the training data"
            if split_key(setting_name(keyword.arg)) is not None:
                self.named_value(keyword.arg, keyword.value, context)

    def load_dataset(self, node: ast.Call, target: str | None) -> None:
        name = _constant(node.args[0]) if node.args else _constant(_keyword(node, "path"))
        if not isinstance(name, str) or not name or name.lower() in _BUILDERS:
            return
        config = (
            _constant(node.args[1]) if len(node.args) > 1 else _constant(_keyword(node, "name"))
        )
        first = node.args[0] if node.args else _keyword(node, "path")
        assert first is not None
        self.add(
            first, "dataset", "load_dataset", name, config if isinstance(config, str) else None
        )

        split_node = _keyword(node, "split")
        split = _constant(split_node)
        match = _SLICE.match(split) if isinstance(split, str) else None
        if split_node is None or match is None or target is None:
            return
        # A slice of one split, such as validation carved out of train, is a share of that
        # split, not of the whole dataset.
        share = int(match["end"] or 100) - int(match["start"] or 0)
        held_out = setting_name(target)
        key = None
        if re.search(r"val|dev", held_out):
            key = "validation_fraction"
        elif "test" in held_out:
            key = "test_fraction"
        if key is not None and 0 < share < 100:
            self.add(
                split_node, key, f"split={split}", float(share), f"of the {match['split']} split"
            )


def _function_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _keyword(node: ast.Call, name: str) -> ast.expr | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _constant(node: ast.expr | None) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _numbers(node: ast.expr | None) -> list[float]:
    if not isinstance(node, ast.List | ast.Tuple):
        return []
    values = [_constant(item) for item in node.elts]
    if not all(isinstance(v, int | float) and not isinstance(v, bool) for v in values):
        return []
    return [float(v) for v in values if isinstance(v, int | float)]


def _target_name(node: ast.Assign | ast.AnnAssign) -> str | None:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    if len(targets) == 1 and isinstance(targets[0], ast.Name):
        return targets[0].id
    return None


def scan_python(path: str, text: str, repository: RepositoryRef) -> list[ExtractedSetting]:
    """Split settings in one Python file, in source order. A file that does not parse
    gives nothing."""
    try:
        tree = ast.parse(text, filename=path)
    except (SyntaxError, ValueError, RecursionError):
        return []
    scanner = _Scanner(path, text, repository)
    targets: dict[int, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None:
            name = _target_name(node)
            if name is not None and isinstance(node.value, ast.Call):
                targets[id(node.value)] = name
            elif name is not None:
                scanner.named_value(name, node.value)
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            arguments = node.args
            positional = [*arguments.posonlyargs, *arguments.args]
            for arg, default in zip(
                positional[len(positional) - len(arguments.defaults) :],
                arguments.defaults,
                strict=True,
            ):
                scanner.named_value(arg.arg, default)
            for arg, kw_default in zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True):
                scanner.named_value(arg.arg, kw_default)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            scanner.call(node, targets.get(id(node)))
    scanner.found.sort(key=lambda f: (f.line, f.column))
    return [f.setting for f in scanner.found]
