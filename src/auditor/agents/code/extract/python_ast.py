"""Settings in Python code, read with `ast` (the code is never imported or run).

Recognised: argparse options, function defaults, keyword arguments and assignments named
like a setting (`lr`, `num_train_epochs`, `test_size`, see `names.CODE_NAMES`);
optimizer constructors (`torch.optim.AdamW`), seeding calls, `nn.Dropout`, metric
functions and `evaluate.load`; `train_test_split`, `random_split`, Keras
`fit(validation_split=...)` and `load_dataset(name, config, split=...)`.
"""

import ast
import re
import warnings
from dataclasses import dataclass

from auditor.agents.code.extract.names import classify, metric_key
from auditor.agents.code.extract.normalise import (
    as_percent,
    setting_name,
    split_from_parts,
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
        """A value under a name such as `test_size` or `lr`, if the name is a setting."""
        if node is None:
            return
        found = classify(setting_name(raw_name), _constant(node))
        if found is not None:
            self.add(node, found.key, raw_name, found.value, context or found.context)

    def call(self, node: ast.Call, target: str | None) -> None:
        function = _function_name(node.func)
        if function == "add_argument":
            options = [_constant(arg) for arg in node.args]
            names = [o for o in options if isinstance(o, str)]
            default = _keyword(node, "default")
            if names and default is not None:
                for option in names:
                    if classify(setting_name(option), _constant(default)) is not None:
                        self.named_value(option, default)
                        break
            return
        if function in _OPTIMIZERS and isinstance(node.func, ast.Attribute | ast.Name):
            self.add(node.func, "optimizer", function, function.lower())
        if function in _SEED_FUNCTIONS and node.args:
            self.named_value("seed", node.args[0])
        if function == "Dropout":
            self.named_value("dropout", node.args[0] if node.args else _keyword(node, "p"))
        metric = _metric_call(function, node)
        if metric is not None:
            self.add(node, "eval_metric", function or "metric", metric)
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
            if function == "Dropout" and keyword.arg == "p":
                continue
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


_OPTIMIZERS = frozenset(
    {"Adam", "AdamW", "SGD", "Adafactor", "Adagrad", "Adadelta", "RMSprop", "LAMB", "Lamb", "Lion"}
)
_SEED_FUNCTIONS = frozenset({"manual_seed", "set_seed", "seed", "seed_everything"})
_METRIC_FUNCTIONS = {
    "accuracy_score": "accuracy",
    "matthews_corrcoef": "matthews_correlation",
    "roc_auc_score": "auc",
    "pearsonr": "pearson_correlation",
    "spearmanr": "spearman_correlation",
    "corpus_bleu": "bleu",
}
_F1_AVERAGES = {"macro": "macro_f1", "micro": "micro_f1"}


def _metric_call(function: str | None, node: ast.Call) -> str | None:
    """The metric a call computes: sklearn-style functions, or `evaluate.load("bleu")` and
    `load_metric("bleu")`."""
    if function in _METRIC_FUNCTIONS:
        return _METRIC_FUNCTIONS[function]
    if function == "f1_score":
        average = _constant(_keyword(node, "average"))
        return _F1_AVERAGES.get(average, "f1") if isinstance(average, str) else "f1"
    if function in {"load", "load_metric"} and node.args:
        name = _constant(node.args[0])
        owner = node.func.value if isinstance(node.func, ast.Attribute) else None
        is_metric = function == "load_metric" or (
            isinstance(owner, ast.Name) and owner.id == "evaluate"
        )
        if is_metric and isinstance(name, str):
            return metric_key(name)
    return None


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
    # Negative numbers are a unary minus applied to a constant.
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int | float)
    ):
        return -node.operand.value
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
        # Repository code often has invalid escape sequences; that is not ours to report.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
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
