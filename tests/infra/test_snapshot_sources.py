from pathlib import Path

import pytest

from auditor.infra.snapshots import SourceError, SourceKind, SourceSpec, parse_source


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            "https://github.com/microsoft/LoRA",
            SourceSpec(SourceKind.GITHUB, "https://github.com/microsoft/LoRA", "microsoft/LoRA"),
        ),
        (
            "https://www.github.com/microsoft/LoRA.git",
            SourceSpec(SourceKind.GITHUB, "https://github.com/microsoft/LoRA", "microsoft/LoRA"),
        ),
        (
            "https://github.com/microsoft/LoRA/tree/0123abc/examples",
            SourceSpec(
                SourceKind.GITHUB,
                "https://github.com/microsoft/LoRA",
                "microsoft/LoRA",
                ref="0123abc",
            ),
        ),
        (
            "https://gitlab.com/group/sub/project/-/tree/v1.0",
            SourceSpec(
                SourceKind.GITLAB,
                "https://gitlab.com/group/sub/project",
                "group/sub/project",
                ref="v1.0",
            ),
        ),
        (
            "https://huggingface.co/google-bert/bert-base-uncased",
            SourceSpec(
                SourceKind.HUGGINGFACE,
                "https://huggingface.co/google-bert/bert-base-uncased",
                "google-bert/bert-base-uncased",
                repo_type="model",
            ),
        ),
        (
            "https://huggingface.co/gpt2/tree/main",
            SourceSpec(
                SourceKind.HUGGINGFACE,
                "https://huggingface.co/gpt2",
                "gpt2",
                ref="main",
                repo_type="model",
            ),
        ),
        (
            "https://hf.co/datasets/stanfordnlp/imdb/blob/abc123/README.md",
            SourceSpec(
                SourceKind.HUGGINGFACE,
                "https://huggingface.co/datasets/stanfordnlp/imdb",
                "stanfordnlp/imdb",
                ref="abc123",
                repo_type="dataset",
            ),
        ),
        (
            "https://www.kaggle.com/datasets/uciml/iris/versions/2",
            SourceSpec(
                SourceKind.KAGGLE,
                "https://www.kaggle.com/datasets/uciml/iris",
                "uciml/iris",
                ref="2",
            ),
        ),
        (
            "https://anonymous.4open.science/r/840c8c57-3c32/README.md",
            SourceSpec(
                SourceKind.ANONYMOUS_GITHUB,
                "https://anonymous.4open.science/r/840c8c57-3c32",
                "840c8c57-3c32",
            ),
        ),
        (
            "https://zenodo.org/record/10796477",
            SourceSpec(SourceKind.ZENODO, "https://zenodo.org/records/10796477", "10796477"),
        ),
        (
            "https://doi.org/10.5281/zenodo.10796477",
            SourceSpec(SourceKind.ZENODO, "https://zenodo.org/records/10796477", "10796477"),
        ),
        (
            "10.5281/zenodo.10796477",
            SourceSpec(SourceKind.ZENODO, "https://zenodo.org/records/10796477", "10796477"),
        ),
    ],
)
def test_supported_links_are_read(value: str, expected: SourceSpec) -> None:
    assert parse_source(value) == expected


def test_a_given_ref_pins_a_link_without_one() -> None:
    spec = parse_source("https://github.com/a/b", "v2.0")
    assert spec.ref == "v2.0"


def test_a_given_ref_must_agree_with_the_link() -> None:
    with pytest.raises(SourceError, match="names 0123abc"):
        parse_source("https://github.com/a/b/commit/0123abc", "fff0000")


@pytest.mark.parametrize(
    "value",
    [
        "https://anonymous.4open.science/r/abc",
        "https://zenodo.org/records/1",
    ],
)
def test_sources_without_revisions_refuse_a_ref(value: str) -> None:
    with pytest.raises(SourceError, match="cannot be pinned"):
        parse_source(value, "main")


def test_kaggle_versions_are_numbers() -> None:
    with pytest.raises(SourceError, match="version number"):
        parse_source("https://www.kaggle.com/datasets/uciml/iris", "main")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("https://example.com/a/b", "not a supported"),
        ("https://www.kaggle.com/code/someone/notebook", "only Kaggle datasets"),
        ("https://github.com/orgs/microsoft", "not a supported"),
        ("https://github.com/microsoft", "not a supported"),
        ("ftp://github.com/a/b", "not a supported"),
    ],
)
def test_unsupported_links_are_refused(value: str, message: str) -> None:
    with pytest.raises(SourceError, match=message):
        parse_source(value)


def test_a_local_directory_is_a_source(tmp_path: Path) -> None:
    folder = tmp_path / "patched repo"
    folder.mkdir()
    spec = parse_source(str(folder))
    assert spec.kind is SourceKind.LOCAL
    assert spec.url == folder.resolve().as_uri()
    assert spec.local_path == folder.resolve()
    assert parse_source(spec.url) == spec


def test_a_missing_local_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="no such directory"):
        parse_source(str(tmp_path / "missing"))


def test_a_local_directory_cannot_be_pinned(tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="cannot be pinned"):
        parse_source(str(tmp_path), "main")
