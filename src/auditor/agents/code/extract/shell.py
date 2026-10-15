"""Settings passed on the command line in shell scripts, such as
`python train.py --lr 3e-4 --seed=7`. Scripts are only read, never run."""

import re
import shlex

from auditor.agents.code.extract.names import classify
from auditor.agents.code.extract.normalise import setting_name
from auditor.schemas import ExtractedSetting, RepositoryRef, RepositorySource

_OPTION = re.compile(r"^--?(?P<name>[A-Za-z][\w.-]*)(?:=(?P<value>.+))?$")


def scan_shell(path: str, text: str, repository: RepositoryRef) -> list[ExtractedSetting]:
    found = []
    lines = text.splitlines()
    for number, line in enumerate(lines, start=1):
        stripped = line.split("#", 1)[0] if not line.lstrip().startswith("#!") else ""
        try:
            tokens = shlex.split(stripped.rstrip().removesuffix("\\"))
        except ValueError:
            continue
        for index, token in enumerate(tokens):
            match = _OPTION.match(token)
            if match is None:
                continue
            raw = match.group("value")
            if raw is None:
                following = tokens[index + 1] if index + 1 < len(tokens) else None
                if following is None or (following.startswith("-") and not _is_number(following)):
                    continue
                raw = following
            name = match.group("name")
            setting = classify(setting_name(name), raw)
            if setting is None:
                continue
            found.append(
                ExtractedSetting(
                    key=setting.key,
                    raw_name=f"--{name}",
                    raw_value=raw,
                    value=setting.value,
                    context=setting.context,
                    source=RepositorySource(repository=repository, path=path, line=number),
                )
            )
    return found


def _is_number(token: str) -> bool:
    return re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", token) is not None
