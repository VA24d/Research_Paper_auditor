"""Settings in YAML, JSON and TOML configuration files, with the line of each key.

YAML is only composed into nodes with the safe loader, never constructed, so tags such as
`!!python/object` are inert.
"""

import json
import re
import tomllib
from collections.abc import Iterator
from typing import Any

import yaml

from auditor.agents.code.extract.names import classify
from auditor.agents.code.extract.normalise import setting_name
from auditor.schemas import ExtractedSetting, RepositoryRef, RepositorySource, SettingValue

_YAML_NUMBER_TAGS = {"tag:yaml.org,2002:int", "tag:yaml.org,2002:float"}
_JSON_KEY = re.compile(r'"((?:[^"\\]|\\.)*)"\s*:')


def _setting(
    key: str,
    raw_name: str,
    raw_value: str,
    value: SettingValue,
    path: str,
    line: int,
    repository: RepositoryRef,
) -> ExtractedSetting:
    return ExtractedSetting(
        key=key,
        raw_name=raw_name,
        raw_value=raw_value,
        value=value,
        source=RepositorySource(repository=repository, path=path, line=line),
    )


def _named(
    raw_name: str, value: object, raw_value: str, path: str, line: int, repository: RepositoryRef
) -> ExtractedSetting | None:
    found = classify(setting_name(raw_name), value)
    if found is None:
        return None
    return ExtractedSetting(
        key=found.key,
        raw_name=raw_name,
        raw_value=raw_value,
        value=found.value,
        context=found.context,
        source=RepositorySource(repository=repository, path=path, line=line),
    )


def scan_yaml(path: str, text: str, repository: RepositoryRef) -> list[ExtractedSetting]:
    try:
        documents = list(yaml.compose_all(text, Loader=yaml.SafeLoader))
    except yaml.YAMLError:
        return []
    found = []
    for document in documents:
        for key_node, value_node in _yaml_pairs(document):
            if not isinstance(value_node, yaml.ScalarNode):
                continue
            raw = str(value_node.value)
            value: object = raw
            if value_node.tag in _YAML_NUMBER_TAGS:
                try:
                    value = yaml.SafeLoader("").construct_object(value_node)
                except yaml.YAMLError:
                    continue
            line = key_node.start_mark.line + 1
            setting = _named(str(key_node.value), value, raw, path, line, repository)
            if setting is not None:
                found.append(setting)
    return found


def _yaml_pairs(node: yaml.Node | None) -> Iterator[tuple[yaml.ScalarNode, yaml.Node]]:
    if isinstance(node, yaml.MappingNode):
        for key, value in node.value:
            if isinstance(key, yaml.ScalarNode):
                yield key, value
            yield from _yaml_pairs(value)
    elif isinstance(node, yaml.SequenceNode):
        for item in node.value:
            yield from _yaml_pairs(item)


def scan_json(path: str, text: str, repository: RepositoryRef) -> list[ExtractedSetting]:
    """Split settings in a JSON object. A file whose top level is a list holds data
    records, not configuration, and is skipped."""
    try:
        data = json.loads(text)
    except (ValueError, RecursionError):
        return []
    if not isinstance(data, dict):
        return []
    # Keys appear in the text in the order a depth-first walk visits them, so each key is
    # matched to the next occurrence of its name.
    keys = [
        (json.loads(f'"{m.group(1)}"'), text.count("\n", 0, m.start()) + 1)
        for m in _JSON_KEY.finditer(text)
    ]
    position = 0
    found = []
    for name, value in _json_pairs(data):
        line = 1
        for index in range(position, len(keys)):
            if keys[index][0] == name:
                line = keys[index][1]
                position = index + 1
                break
        if isinstance(value, dict | list):
            continue
        setting = _named(name, value, json.dumps(value), path, line, repository)
        if setting is not None:
            found.append(setting)
    return found


def _json_pairs(value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key), item
            yield from _json_pairs(item)
    elif isinstance(value, list):
        for item in value:
            yield from _json_pairs(item)


def scan_toml(path: str, text: str, repository: RepositoryRef) -> list[ExtractedSetting]:
    """Settings in a TOML file; each key's line is the next line, in order, that assigns
    a key of that name."""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return []
    assigned = [
        (match.group(1), text.count("\n", 0, match.start()) + 1)
        for match in re.finditer(r'(?m)(?:^|[{,])\s*"?([A-Za-z0-9_.-]+)"?\s*=', text)
    ]
    position = 0
    found = []
    for name, value in _json_pairs(data):
        line = 1
        for index in range(position, len(assigned)):
            if assigned[index][0] == name:
                line = assigned[index][1]
                position = index + 1
                break
        if isinstance(value, dict | list):
            continue
        setting = _named(name, value, json.dumps(value), path, line, repository)
        if setting is not None:
            found.append(setting)
    return found
