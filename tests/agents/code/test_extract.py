import pytest

from auditor.agents.code.extract import (
    parse_card,
    scan_json,
    scan_python,
    scan_split_lists,
    scan_yaml,
)
from auditor.agents.code.extract.normalise import split_parts, splits_match
from auditor.schemas import ExtractedSetting, RepositoryRef, RepositorySource

REPO = RepositoryRef(url="https://github.com/o/r", commit="a" * 40)


def found(settings: list[ExtractedSetting]) -> list[tuple[str, object, int, str | None]]:
    rows: list[tuple[str, object, int, str | None]] = []
    for s in settings:
        assert isinstance(s.source, RepositorySource)
        assert s.source.repository == REPO
        rows.append((s.key, s.value, s.source.line, s.context))
    return rows


# --- normalise -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "parts"),
    [
        ("80/10/10", (80.0, 10.0, 10.0)),
        ("8/1/1", (80.0, 10.0, 10.0)),
        ("70/30", (70.0, 30.0)),
    ],
)
def test_split_parts_are_percentages(value: str, parts: tuple[float, ...]) -> None:
    assert split_parts(value) == parts


def test_splits_match_within_one_point() -> None:
    assert splits_match("80/10/10", "8/1/1")
    assert splits_match("80/10/10", "79.6/10.2/10.2")
    assert not splits_match("80/10/10", "70/15/15")
    assert not splits_match("80/20", "80/10/10")


# --- python ----------------------------------------------------------------------------

PYTHON = """\
import argparse
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--test-size", type=float, default=0.2)
parser.add_argument("--val_ratio", default=0.1)
parser.add_argument("--dataset", default="cifar10")

X_train, X_test = train_test_split(X, y, test_size=0.25, random_state=0)
rows = train_test_split(X, test_size=500)
train, val, test = random_split(ds, [0.8, 0.1, 0.1])
parts = random_split(ds, [8000, 1000, 1000])
data = load_dataset("glue", "sst2")
val_ds = load_dataset("imdb", split="train[90%:]")
model.fit(x, y, validation_split=0.15)
TEST_FRACTION = 0.3

def loaders(batch_size=32, val_split=0.05):
    pass
"""


def test_python_split_settings_keep_their_line() -> None:
    assert found(scan_python("train.py", PYTHON, REPO)) == [
        ("test_fraction", 20.0, 7, None),
        ("validation_fraction", 10.0, 8, None),
        ("dataset", "cifar10", 9, None),
        ("test_fraction", 25.0, 11, None),
        ("data_split", "80/10/10", 13, None),
        ("data_split", "80/10/10", 14, None),
        ("dataset", "glue", 15, "sst2"),
        ("dataset", "imdb", 16, None),
        ("validation_fraction", 10.0, 16, "of the train split"),
        ("validation_fraction", 15.0, 17, "of the training data"),
        ("test_fraction", 30.0, 18, None),
        ("validation_fraction", 5.0, 20, None),
    ]


def test_python_setting_keeps_the_code_as_written() -> None:
    first = scan_python("train.py", PYTHON, REPO)[0]
    assert first.raw_name == "--test-size"
    assert first.raw_value == "0.2"


def test_python_that_does_not_parse_gives_nothing() -> None:
    assert scan_python("broken.py", "def f(:\n", REPO) == []


def test_slices_of_two_splits_and_builders_are_not_datasets() -> None:
    text = 'd = load_dataset("csv", data_files="x.csv")\ne = load_dataset("imdb", split="test")\n'
    assert found(scan_python("a.py", text, REPO)) == [("dataset", "imdb", 2, None)]


# --- yaml and json ---------------------------------------------------------------------

YAML = """\
data:
  name: imdb
  splits:
    test_size: 0.2
    val_split: 10%
model:
  lr: 3e-4
"""


def test_yaml_keys_keep_their_line() -> None:
    assert found(scan_yaml("configs/base.yaml", YAML, REPO)) == [
        ("test_fraction", 20.0, 4, None),
        ("validation_fraction", 10.0, 5, None),
    ]


def test_yaml_with_python_tags_is_read_safely() -> None:
    text = "a: !!python/object/apply:os.system ['echo hi']\ntest_size: 0.3\n"
    assert found(scan_yaml("x.yaml", text, REPO)) == [("test_fraction", 30.0, 2, None)]


def test_broken_yaml_gives_nothing() -> None:
    assert scan_yaml("x.yaml", "a: [1, 2\n", REPO) == []


def test_json_keys_keep_their_line() -> None:
    text = (
        '{\n  "train": {\n    "lr": 0.1,\n    "validation_split": 0.2\n  },\n'
        '  "test_ratio": 0.1\n}\n'
    )
    assert found(scan_json("config.json", text, REPO)) == [
        ("validation_fraction", 20.0, 4, None),
        ("test_fraction", 10.0, 6, None),
    ]


def test_json_records_are_data_not_config() -> None:
    assert scan_json("data/test.json", '[{"test_size": 0.5}]', REPO) == []


# --- split lists -----------------------------------------------------------------------


def test_split_lists_in_one_folder_give_the_split() -> None:
    files = {
        "splits/train.txt": "\n".join(f"img{i}.png" for i in range(80)) + "\n",
        "splits/val.txt": "\n".join(f"v{i}" for i in range(10)) + "\n\n",
        "splits/test.txt": "\n".join(f"t{i}" for i in range(10)),
        "requirements.txt": "torch\n",
        "other/train.txt": "a\n",
    }
    settings = scan_split_lists(files, REPO)
    assert found(settings) == [
        ("data_split", "80/10/10", 1, None),
        ("train_examples", 80, 1, None),
        ("validation_examples", 10, 1, None),
        ("test_examples", 10, 1, None),
    ]
    assert settings[0].raw_value == "train.txt 80 / val.txt 10 / test.txt 10"
    paths = [s.source.path for s in settings if isinstance(s.source, RepositorySource)]
    assert paths == ["splits/test.txt", "splits/train.txt", "splits/val.txt", "splits/test.txt"]


def test_split_lists_stored_in_git_lfs_are_ignored() -> None:
    pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n"
    files = {"train.txt": pointer, "test.txt": pointer}
    assert scan_split_lists(files, REPO) == []


# --- Hugging Face dataset cards --------------------------------------------------------

IMDB_CARD = """\
---
pretty_name: IMDB
dataset_info:
  config_name: plain_text
  features:
  - name: text
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': neg
          '1': pos
  splits:
  - name: train
    num_bytes: 33432823
    num_examples: 25000
  - name: test
    num_bytes: 32650685
    num_examples: 25000
  - name: unsupervised
    num_examples: 50000
configs:
- config_name: plain_text
  data_files:
  - split: train
    path: plain_text/train-*
  - split: test
    path: plain_text/test-*
  default: true
---
# IMDB
"""


def test_card_gives_split_sizes_labels_and_files() -> None:
    card = parse_card(IMDB_CARD)
    assert card is not None
    config = card.config(None)
    assert config is not None
    assert config.name == "plain_text"
    assert config.splits == {"train": 25000, "test": 25000, "unsupervised": 50000}
    assert config.split_lines["test"] == 20
    assert config.label_column == "label"
    assert config.label_names == ("neg", "pos")
    assert config.data_files == {"train": ("plain_text/train-*",), "test": ("plain_text/test-*",)}


def test_card_split_becomes_a_setting() -> None:
    card = parse_card(IMDB_CARD)
    assert card is not None
    settings = card.settings("README.md", REPO, None)
    assert found(settings) == [
        ("data_split", "50/50", 20, "plain_text"),
        ("train_examples", 25000, 17, "plain_text"),
        ("test_examples", 25000, 20, "plain_text"),
        ("total_examples", 50000, 17, "plain_text"),
        ("num_classes", 2, 10, "plain_text"),
    ]
    assert settings[0].raw_value == "train 25000 / test 25000"
    assert settings[-1].raw_value == "2 classes: neg, pos"


def test_card_with_several_configs_picks_the_named_one() -> None:
    text = """\
---
dataset_info:
- config_name: a
  splits:
  - name: train
    num_examples: 90
  - name: test
    num_examples: 10
- config_name: b
  splits:
  - name: train
    num_examples: 70
  - name: validation
    num_examples: 10
  - name: test
    num_examples: 20
---
"""
    card = parse_card(text)
    assert card is not None
    config = card.config("b")
    assert config is not None
    assert config.splits == {"train": 70, "validation": 10, "test": 20}
    assert card.config("missing") is None
    # With several configs and no name, there is no single answer.
    assert card.config(None) is None


@pytest.mark.parametrize("text", ["# no front matter\n", "---\na: [1\n---\n", ""])
def test_card_without_readable_front_matter_is_none(text: str) -> None:
    assert parse_card(text) is None
