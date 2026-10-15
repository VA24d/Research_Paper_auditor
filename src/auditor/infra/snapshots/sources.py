"""Where a paper's code or data lives, read from a link, a DOI or a local folder."""

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname


class SourceKind(StrEnum):
    GITHUB = "github"
    GITLAB = "gitlab"
    HUGGINGFACE = "huggingface"
    KAGGLE = "kaggle"
    ANONYMOUS_GITHUB = "anonymous_github"
    ZENODO = "zenodo"
    LOCAL = "local"


class SourceError(ValueError):
    """The value does not name a source the auditor can fetch."""


@dataclass(frozen=True)
class SourceSpec:
    """A parsed source. `url` is the canonical link (a file URI for a local folder),
    `identity` the host's name for it, and `ref` the requested revision, if any."""

    kind: SourceKind
    url: str
    identity: str
    ref: str | None = None
    # Hugging Face only: "model", "dataset" or "space".
    repo_type: str | None = None
    local_path: Path | None = None


_NAME = r"[A-Za-z0-9_.-]+"
_NOT_GITHUB_OWNERS = {"orgs", "features", "topics", "sponsors", "marketplace", "about", "settings"}
_SNAPSHOT_WORDS = {"tree", "commit", "blob", "resolve"}
_ZENODO_DOI = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/)?10\.5281/zenodo\.(?P<id>\d+)$")
# Hosts whose snapshots are fixed by the host itself, so a requested revision means nothing.
_UNPINNABLE = {
    SourceKind.ANONYMOUS_GITHUB: "Anonymous GitHub snapshots",
    SourceKind.ZENODO: "Zenodo records (each version has its own record)",
    SourceKind.LOCAL: "Local folders",
}


def parse_source(value: str, ref: str | None = None) -> SourceSpec:
    """Read `value` as a supported source, with `ref` as the revision to pin, if given.

    Raises SourceError for anything else, including a `ref` that disagrees with a revision
    the link already names.
    """
    value = value.strip()
    spec = _parse(value)
    if ref is not None:
        if spec.kind in _UNPINNABLE:
            raise SourceError(f"{_UNPINNABLE[spec.kind]} cannot be pinned to a revision ({ref})")
        if spec.ref is not None and spec.ref != ref:
            raise SourceError(f"{value} names {spec.ref}, but the revision given is {ref}")
        if spec.kind is SourceKind.KAGGLE and not ref.isdigit():
            raise SourceError(f"a Kaggle dataset revision is a version number, not {ref!r}")
        spec = SourceSpec(spec.kind, spec.url, spec.identity, ref, spec.repo_type, spec.local_path)
    return spec


def _parse(value: str) -> SourceSpec:
    doi = _ZENODO_DOI.match(value)
    if doi:
        return _zenodo(doi["id"])
    if value.startswith("file:"):
        return _local(Path(url2pathname(urlsplit(value).path)))
    if "://" not in value:
        return _local(Path(value))

    parts = urlsplit(value)
    host = (parts.hostname or "").lower().removeprefix("www.")
    segments = [s for s in parts.path.split("/") if s]
    if parts.scheme in {"http", "https"}:
        match host:
            case "github.com":
                return _github(value, segments)
            case "gitlab.com":
                return _gitlab(value, segments)
            case "huggingface.co" | "hf.co":
                return _huggingface(value, segments)
            case "kaggle.com":
                return _kaggle(value, segments)
            case "anonymous.4open.science":
                if len(segments) >= 2 and segments[0] == "r":
                    return SourceSpec(
                        SourceKind.ANONYMOUS_GITHUB,
                        f"https://anonymous.4open.science/r/{segments[1]}",
                        segments[1],
                    )
            case "zenodo.org":
                if (
                    len(segments) >= 2
                    and segments[0] in {"record", "records"}
                    and segments[1].isdigit()
                ):
                    return _zenodo(segments[1])
    raise _unsupported(value)


def _unsupported(value: str) -> SourceError:
    return SourceError(
        f"{value} is not a supported code or data source (GitHub, GitLab, Hugging Face, "
        "Kaggle datasets, Anonymous GitHub, Zenodo or a local folder)"
    )


def _is_name(segment: str) -> bool:
    return re.fullmatch(_NAME, segment) is not None


def _github(value: str, segments: list[str]) -> SourceSpec:
    if len(segments) < 2 or segments[0].lower() in _NOT_GITHUB_OWNERS:
        raise _unsupported(value)
    owner, name = segments[0], segments[1].removesuffix(".git")
    if not (_is_name(owner) and _is_name(name)):
        raise _unsupported(value)
    rest = segments[2:]
    ref = rest[1] if len(rest) >= 2 and rest[0] in _SNAPSHOT_WORDS else None
    return SourceSpec(
        SourceKind.GITHUB, f"https://github.com/{owner}/{name}", f"{owner}/{name}", ref
    )


def _gitlab(value: str, segments: list[str]) -> SourceSpec:
    # Projects can sit in nested groups; "/-/" separates the project from the page in it.
    if "-" in segments:
        split = segments.index("-")
        project, rest = segments[:split], segments[split + 1 :]
    else:
        project, rest = segments, []
    if project:
        project[-1] = project[-1].removesuffix(".git")
    if len(project) < 2 or not all(_is_name(s) for s in project):
        raise _unsupported(value)
    ref = rest[1] if len(rest) >= 2 and rest[0] in _SNAPSHOT_WORDS else None
    path = "/".join(project)
    return SourceSpec(SourceKind.GITLAB, f"https://gitlab.com/{path}", path, ref)


def _huggingface(value: str, segments: list[str]) -> SourceSpec:
    repo_type, prefix = "model", ""
    if segments and segments[0] in {"datasets", "spaces"}:
        repo_type = segments[0].removesuffix("s")
        prefix = f"{segments[0]}/"
        segments = segments[1:]
    # Old models and datasets have a single-part name ("gpt2", "imdb"); the rest are
    # owner/name.
    size = 1 if len(segments) == 1 or (len(segments) > 1 and segments[1] in _SNAPSHOT_WORDS) else 2
    name, rest = segments[:size], segments[size:]
    if len(name) != size or not all(_is_name(s) for s in name):
        raise _unsupported(value)
    identity = "/".join(name)
    ref = rest[1] if len(rest) >= 2 and rest[0] in _SNAPSHOT_WORDS else None
    return SourceSpec(
        SourceKind.HUGGINGFACE,
        f"https://huggingface.co/{prefix}{identity}",
        identity,
        ref,
        repo_type=repo_type,
    )


def _kaggle(value: str, segments: list[str]) -> SourceSpec:
    if not segments or segments[0] != "datasets":
        raise SourceError(
            f"{value}: only Kaggle datasets can be fetched; notebooks, models and "
            "competitions need a Kaggle login"
        )
    if len(segments) < 3 or not (_is_name(segments[1]) and _is_name(segments[2])):
        raise _unsupported(value)
    identity = f"{segments[1]}/{segments[2]}"
    rest = segments[3:]
    ref = rest[1] if len(rest) >= 2 and rest[0] == "versions" and rest[1].isdigit() else None
    return SourceSpec(
        SourceKind.KAGGLE, f"https://www.kaggle.com/datasets/{identity}", identity, ref
    )


def _zenodo(record: str) -> SourceSpec:
    return SourceSpec(SourceKind.ZENODO, f"https://zenodo.org/records/{record}", record)


def _local(path: Path) -> SourceSpec:
    if not path.is_dir():
        raise SourceError(f"no such directory: {path}")
    resolved = path.resolve()
    return SourceSpec(SourceKind.LOCAL, resolved.as_uri(), str(resolved), local_path=resolved)
