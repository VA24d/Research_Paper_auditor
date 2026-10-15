"""Split sizes, labels and data files from a Hugging Face dataset card's YAML front matter.

The card is read from the snapshot pinned to a commit, so its `num_examples` describe that
revision, unlike the Hub's `/size` service, which describes the latest one.
"""

from dataclasses import dataclass, field
from typing import Any, Self

import yaml

from auditor.agents.code.extract.normalise import split_from_parts
from auditor.schemas import ExtractedSetting, RepositoryRef, RepositorySource, SettingValue

_VALIDATION = ("validation", "valid", "dev")
_UNLABELLED = frozenset({"unsupervised", "unlabeled", "unlabelled"})


@dataclass(frozen=True)
class CardConfig:
    name: str
    splits: dict[str, int] = field(default_factory=dict)
    # 1-based line in the card of each split's `num_examples`.
    split_lines: dict[str, int] = field(default_factory=dict)
    label_column: str | None = None
    label_names: tuple[str, ...] | None = None
    # 1-based line of the label feature's `class_label`.
    label_line: int | None = None
    data_files: dict[str, tuple[str, ...]] = field(default_factory=dict)
    default: bool = False


@dataclass(frozen=True)
class CardInfo:
    configs: tuple[CardConfig, ...]

    def config(self, name: str | None) -> CardConfig | None:
        """The named config; without a name, the only one or the one marked default."""
        if name is not None:
            return next((c for c in self.configs if c.name == name), None)
        if len(self.configs) == 1:
            return self.configs[0]
        defaults = [c for c in self.configs if c.default]
        return defaults[0] if len(defaults) == 1 else None

    def settings(
        self, path: str, repository: RepositoryRef, config_name: str | None
    ) -> list[ExtractedSetting]:
        """The card's split sizes as `data_split` and `<split>_examples` settings, their sum
        as `total_examples`, and the label classes as `num_classes`."""
        config = self.config(config_name)
        if config is None:
            return []

        def setting(
            key: str, name: str, raw: str, value: SettingValue, line: int
        ) -> ExtractedSetting:
            return ExtractedSetting(
                key=key,
                raw_name=name,
                raw_value=raw,
                value=value,
                context=config.name,
                source=RepositorySource(repository=repository, path=path, line=line),
            )

        splits, lines = config.splits, config.split_lines
        ordered = ["train", *[s for s in _VALIDATION if s in splits][:1], "test"]
        found = []
        value = split_from_parts([float(splits[s]) for s in ordered if s in splits])
        if "train" in splits and "test" in splits and value is not None:
            raw = " / ".join(f"{s} {splits[s]}" for s in ordered)
            found.append(setting("data_split", "dataset_info.splits", raw, value, lines["test"]))
        for split in ordered:
            if split in splits:
                key = f"{'validation' if split in _VALIDATION else split}_examples"
                name = f"dataset_info.splits[{split}].num_examples"
                found.append(setting(key, name, str(splits[split]), splits[split], lines[split]))
        # Unlabelled extras such as IMDB's "unsupervised" split are not part of the dataset
        # a paper evaluates on.
        labelled = {s: n for s, n in splits.items() if s not in _UNLABELLED}
        if labelled:
            raw = " + ".join(f"{s} {n}" for s, n in labelled.items())
            first = min(lines[s] for s in labelled)
            found.append(
                setting("total_examples", "dataset_info.splits", raw, sum(labelled.values()), first)
            )
        if config.label_names:
            names = config.label_names
            raw = f"{len(names)} classes: {', '.join(names)}"
            found.append(
                setting("num_classes", "class_label.names", raw, len(names), config.label_line or 1)
            )
        return found

    @classmethod
    def from_front_matter(cls, root: yaml.MappingNode, offset: int) -> Self:
        data_files = {
            str(_value(_get(entry, "config_name")) or "default"): (
                _data_files(_get(entry, "data_files")),
                _value(_get(entry, "default")) is True,
            )
            for entry in _items(_get(root, "configs"))
        }
        infos = _get(root, "dataset_info")
        entries = _items(infos) if isinstance(infos, yaml.SequenceNode) else [infos]
        configs = []
        for entry in entries:
            if not isinstance(entry, yaml.MappingNode):
                continue
            name = str(_value(_get(entry, "config_name")) or "default")
            splits: dict[str, int] = {}
            lines: dict[str, int] = {}
            for split in _items(_get(entry, "splits")):
                split_name = _value(_get(split, "name"))
                count_node = _get(split, "num_examples")
                count = _value(count_node)
                if isinstance(split_name, str) and isinstance(count, int) and count_node:
                    splits[split_name] = count
                    lines[split_name] = count_node.start_mark.line + 1 + offset
            label_column, label_names, label_node = _class_label(_get(entry, "features"))
            label_line = label_node.start_mark.line + 1 + offset if label_node else None
            files, default = data_files.get(name, ({}, False))
            configs.append(
                CardConfig(
                    name, splits, lines, label_column, label_names, label_line, files, default
                )
            )
        return cls(tuple(configs))


def parse_card(text: str) -> CardInfo | None:
    """The card's dataset information, or None without readable YAML front matter."""
    lines = text.lstrip("﻿").splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    try:
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return None
    try:
        root = yaml.compose("\n".join(lines[1:end]), Loader=yaml.SafeLoader)
    except yaml.YAMLError:
        return None
    if not isinstance(root, yaml.MappingNode):
        return None
    # Node lines count from the line after the opening "---".
    return CardInfo.from_front_matter(root, offset=1)


def _get(node: yaml.Node | None, key: str) -> yaml.Node | None:
    if isinstance(node, yaml.MappingNode):
        for key_node, value_node in node.value:
            if isinstance(key_node, yaml.ScalarNode) and key_node.value == key:
                found: yaml.Node = value_node
                return found
    return None


def _items(node: yaml.Node | None) -> list[yaml.Node]:
    return list(node.value) if isinstance(node, yaml.SequenceNode) else []


def _value(node: yaml.Node | None) -> Any:
    """A scalar's value, typed by the safe loader's rules; None for anything else."""
    if not isinstance(node, yaml.ScalarNode):
        return None
    try:
        return yaml.SafeLoader("").construct_object(node)
    except yaml.YAMLError:
        return None


def _data_files(node: yaml.Node | None) -> dict[str, tuple[str, ...]]:
    """`data_files` as split -> path patterns. A bare string or list means one train split."""
    if isinstance(node, yaml.ScalarNode):
        return {"train": (str(node.value),)}
    files: dict[str, tuple[str, ...]] = {}
    for entry in _items(node):
        if isinstance(entry, yaml.ScalarNode):
            files["train"] = (*files.get("train", ()), str(entry.value))
            continue
        split = _value(_get(entry, "split"))
        paths = _get(entry, "path")
        if not isinstance(split, str):
            continue
        if isinstance(paths, yaml.ScalarNode):
            files[split] = (str(paths.value),)
        else:
            files[split] = tuple(
                str(p.value) for p in _items(paths) if isinstance(p, yaml.ScalarNode)
            )
    return files


def _class_label(
    features: yaml.Node | None,
) -> tuple[str | None, tuple[str, ...] | None, yaml.Node | None]:
    """The first feature with a `class_label` dtype, its class names in id order, and the
    `class_label` node."""
    for feature in _items(features):
        class_label = _get(_get(feature, "dtype"), "class_label")
        names = _get(class_label, "names")
        if names is None:
            continue
        column = _value(_get(feature, "name"))
        if isinstance(names, yaml.SequenceNode):
            labels = tuple(str(_value(n)) for n in names.value)
        elif isinstance(names, yaml.MappingNode):
            pairs = sorted(
                ((int(str(k.value)), str(_value(v))) for k, v in names.value),
                key=lambda pair: pair[0],
            )
            labels = tuple(label for _, label in pairs)
        else:
            continue
        return (
            (str(column) if column is not None else None),
            labels,
            _key_node(_get(feature, "dtype"), "class_label"),
        )
    return None, None, None


def _key_node(node: yaml.Node | None, key: str) -> yaml.Node | None:
    if isinstance(node, yaml.MappingNode):
        for key_node, _ in node.value:
            if isinstance(key_node, yaml.ScalarNode) and key_node.value == key:
                found: yaml.Node = key_node
                return found
    return None
