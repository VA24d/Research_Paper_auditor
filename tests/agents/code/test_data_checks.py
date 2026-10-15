import dataclasses
import random
from pathlib import Path
from typing import Any

import httpx
import pytest

from auditor.agents.code.data_checks import check_data_setting
from auditor.config import Settings
from auditor.infra.context import RunContext
from auditor.infra.http import HttpClient
from auditor.planner.evaluation_setup import evaluation_settings, held_out_setting
from auditor.schemas import (
    AuditTask,
    ErrorClass,
    EvidenceKind,
    PaperSource,
    ParsedPaper,
    RepositoryRef,
    SettingTarget,
    SourceKind,
    TaskKind,
    TaskOutcome,
)
from auditor.schemas.examples import example_paper
from auditor.testing import offline_context

SHA = "b" * 40


def paper_saying(text: str) -> ParsedPaper:
    data = example_paper().model_dump()
    data["sentences"][4]["text"] = text
    return ParsedPaper.model_validate(data)


def task_for(paper: ParsedPaper, repository: str, key: str | None = None) -> AuditTask:
    found = evaluation_settings(paper.sentence("s4"))
    if key == "test_overlap":
        held_out = held_out_setting(found)
        assert held_out is not None
        found = [held_out]
    (setting,) = [s for s in found if key is None or s.key == key]
    assert isinstance(setting.source, PaperSource)
    return AuditTask.create(
        paper_id=paper.paper_id,
        variant=paper.variant,
        kind=TaskKind.CODE,
        anchor=setting.source.anchor,
        target=SettingTarget(key=setting.key),
        setting=setting,
        repository=RepositoryRef(url=repository),
    )


def repo(tmp_path: Path, files: dict[str, str]) -> str:
    root = tmp_path / "repo"
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")
    root.mkdir(exist_ok=True)
    return root.resolve().as_uri()


SPLIT_8_1_1 = "We randomly split the data into 80/10/10 train/validation/test sets."


async def check(
    tmp_path: Path, text: str, files: dict[str, str], ctx: RunContext | None = None
) -> Any:
    paper = paper_saying(text)
    ctx = ctx or offline_context(tmp_path, paper=paper)
    return await check_data_setting(task_for(paper, repo(tmp_path, files)), ctx)


async def test_matching_code_split_is_checked_without_findings(tmp_path: Path) -> None:
    result = await check(
        tmp_path, SPLIT_8_1_1, {"train.py": "a, b, c = random_split(ds, [0.8, 0.1, 0.1])\n"}
    )
    assert result.outcome is TaskOutcome.CHECKED
    assert result.findings == ()


async def test_counts_within_one_point_match(tmp_path: Path) -> None:
    code = "parts = random_split(ds, [8530, 1066, 1066])\n"
    result = await check(tmp_path, SPLIT_8_1_1, {"data.py": code})
    assert result.findings == ()


async def test_different_code_split_is_a_finding_with_code_evidence(tmp_path: Path) -> None:
    code = "import torch\n\nparts = random_split(ds, [0.7, 0.15, 0.15])\n"
    result = await check(tmp_path, SPLIT_8_1_1, {"src/data.py": code})
    assert result.outcome is TaskOutcome.CHECKED
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.DATA_SPLIT_MISMATCH
    assert not finding.uncertain
    assert "80/10/10" in finding.summary
    assert "70/15/15" in finding.summary
    quote, code_evidence = finding.evidence
    assert quote.kind is EvidenceKind.QUOTE
    assert quote.source.kind is SourceKind.SUBMISSION
    assert quote.text == SPLIT_8_1_1
    assert code_evidence.kind is EvidenceKind.CODE
    assert code_evidence.path == "src/data.py"
    assert code_evidence.line_start == 3
    assert code_evidence.text == "parts = random_split(ds, [0.7, 0.15, 0.15])"
    assert code_evidence.source.kind is SourceKind.REPOSITORY
    assert code_evidence.source.id.startswith("file://")
    assert "@content-" in code_evidence.source.id


async def test_one_matching_value_among_several_is_enough(tmp_path: Path) -> None:
    files = {
        "a.py": "x = random_split(ds, [0.7, 0.15, 0.15])\n",
        "b.py": "y = random_split(ds, [0.8, 0.1, 0.1])\n",
    }
    result = await check(tmp_path, SPLIT_8_1_1, files)
    assert result.findings == ()


async def test_several_values_none_matching_is_an_uncertain_finding(tmp_path: Path) -> None:
    files = {
        "a.py": "x = random_split(ds, [0.7, 0.15, 0.15])\n",
        "b.py": "y = random_split(ds, [0.6, 0.2, 0.2])\n",
    }
    result = await check(tmp_path, SPLIT_8_1_1, files)
    (finding,) = result.findings
    assert finding.uncertain
    assert len(finding.evidence) == 3


async def test_paper_split_compared_with_code_test_size(tmp_path: Path) -> None:
    files = {"train.py": "X_tr, X_te = train_test_split(X, test_size=0.2)\n"}
    agree = await check(tmp_path, "We split the dataset into 80/20.", files)
    assert agree.findings == ()
    files = {"train.py": "X_tr, X_te = train_test_split(X, test_size=0.3)\n"}
    disagree = await check(tmp_path / "2", "We use 20% of the data for testing.", files)
    (finding,) = disagree.findings
    assert finding.error_class is ErrorClass.DATA_SPLIT_MISMATCH


async def test_validation_share_of_train_is_only_compared_with_the_same_base(
    tmp_path: Path,
) -> None:
    code = 'val = load_dataset("imdb", split="train[85%:]")\n'
    of_train = await check(
        tmp_path, "We hold out 10% of the training data for validation.", {"a.py": code}
    )
    (finding,) = of_train.findings
    assert "15" in finding.summary


async def test_split_lists_are_used_when_code_says_nothing(tmp_path: Path) -> None:
    files = {
        "splits/train.txt": "a\n" * 70,
        "splits/val.txt": "b\n" * 15,
        "splits/test.txt": "c\n" * 15,
    }
    result = await check(tmp_path, SPLIT_8_1_1, files)
    (finding,) = result.findings
    (_, lookup) = finding.evidence
    assert lookup.kind is EvidenceKind.LOOKUP
    assert lookup.path == "splits/test.txt"
    assert "train.txt 70" in lookup.text


async def test_code_evidence_wins_over_split_lists(tmp_path: Path) -> None:
    files = {
        "data.py": "x = random_split(ds, [0.8, 0.1, 0.1])\n",
        "splits/train.txt": "a\n" * 70,
        "splits/test.txt": "c\n" * 30,
    }
    result = await check(tmp_path, SPLIT_8_1_1, files)
    assert result.findings == ()


async def test_no_evidence_with_data_files_needs_the_data(tmp_path: Path) -> None:
    files = {"train.py": "print('hi')\n", "data/train.csv": "a,b\n1,2\n"}
    result = await check(tmp_path, SPLIT_8_1_1, files)
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "row counts need the data files" in result.reason


async def test_no_evidence_at_all_is_unverifiable(tmp_path: Path) -> None:
    result = await check(tmp_path, SPLIT_8_1_1, {"train.py": "print('hi')\n"})
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "no split settings" in result.reason


async def test_unreachable_repository_is_unverifiable(tmp_path: Path) -> None:
    paper = paper_saying(SPLIT_8_1_1)
    task = task_for(paper, (tmp_path / "missing").as_uri())
    result = await check_data_setting(task, offline_context(tmp_path, paper=paper))
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "repository" in result.reason


async def test_dataset_name_matches_the_config_it_loads(tmp_path: Path) -> None:
    code = 'data = load_dataset("glue", "sst2")\n'
    result = await check(tmp_path, "We fine-tune on the SST-2 dataset.", {"a.py": code})
    assert result.findings == ()


async def test_other_dataset_is_an_uncertain_finding(tmp_path: Path) -> None:
    code = 'data = load_dataset("imdb")\n'
    result = await check(tmp_path, "We fine-tune on the SST-2 dataset.", {"a.py": code})
    (finding,) = result.findings
    assert finding.uncertain
    assert finding.error_class is ErrorClass.DATA_SPLIT_MISMATCH


# --- Hugging Face cards, followed from load_dataset ------------------------------------


def card(splits: dict[str, int]) -> str:
    rows = "".join(
        f"  - name: {name}\n    num_examples: {count}\n" for name, count in splits.items()
    )
    return f"---\ndataset_info:\n  splits:\n{rows}---\n# Data\n"


def hub(readme: str, size: dict[str, Any] | None = None) -> tuple[httpx.MockTransport, list[str]]:
    requested: list[str] = []
    api = "https://huggingface.co/api/datasets/org/data"

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        requested.append(url)
        if url == f"{api}/revision/main":
            return httpx.Response(200, json={"sha": SHA})
        if url == f"{api}/tree/{SHA}?recursive=true":
            return httpx.Response(200, json=[{"type": "file", "path": "README.md", "size": 99}])
        if url == f"https://huggingface.co/datasets/org/data/resolve/{SHA}/README.md":
            return httpx.Response(200, text=readme)
        if url.startswith("https://datasets-server.huggingface.co/size") and size is not None:
            return httpx.Response(200, json=size)
        return httpx.Response(404)

    return httpx.MockTransport(handler), requested


def hub_context(tmp_path: Path, paper: ParsedPaper, transport: httpx.MockTransport) -> RunContext:
    settings = Settings(cache_dir=tmp_path / "cache", scratch_dir=tmp_path / "scratch")
    base = offline_context(tmp_path, paper=paper)
    http = HttpClient(settings, transport=transport, use_default_cache=False, rng=random.Random(0))
    return RunContext(settings=settings, paper=paper, http=http, llm=None, logger=base.logger)


async def test_card_of_the_loaded_dataset_is_used(tmp_path: Path) -> None:
    transport, requested = hub(card({"train": 700, "validation": 150, "test": 150}))
    assert requested == []
    paper = paper_saying(SPLIT_8_1_1)
    ctx = hub_context(tmp_path, paper, transport)
    task = task_for(paper, repo(tmp_path, {"a.py": 'd = load_dataset("org/data")\n'}))
    result = await check_data_setting(task, ctx)
    (finding,) = result.findings
    (_, lookup) = finding.evidence
    assert lookup.kind is EvidenceKind.LOOKUP
    assert lookup.path == "README.md"
    assert lookup.source.id == f"https://huggingface.co/datasets/org/data@{SHA}"
    assert "train 700" in lookup.text


async def test_latest_revision_sizes_never_make_a_mismatch(tmp_path: Path) -> None:
    size = {
        "size": {
            "splits": [
                {"config": "default", "split": "train", "num_rows": 50},
                {"config": "default", "split": "test", "num_rows": 50},
            ]
        }
    }
    transport, _ = hub("# no card data\n", size)
    paper = paper_saying("We split the dataset into 80/20.")
    ctx = hub_context(tmp_path, paper, transport)
    task = task_for(paper, "https://huggingface.co/datasets/org/data")
    result = await check_data_setting(task, ctx)
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "latest revision" in result.reason
    (finding,) = result.findings
    assert finding.uncertain


def test_the_check_covers_the_evaluation_data_keys() -> None:
    import auditor.agents.code.data_checks as module

    assert {
        "data_split",
        "test_fraction",
        "validation_fraction",
        "dataset",
        "train_examples",
        "validation_examples",
        "test_examples",
        "total_examples",
        "num_classes",
        "class_balance",
        "test_overlap",
    } == module.DATA_KEYS


# --- dataset size, classes, balance and leakage ----------------------------------------

IMDB_LIKE = (
    "---\ndataset_info:\n  features:\n  - name: text\n    dtype: string\n"
    "  - name: label\n    dtype:\n      class_label:\n        names:\n"
    "          '0': neg\n          '1': pos\n  splits:\n"
    "  - name: train\n    num_examples: 25000\n  - name: test\n    num_examples: 25000\n"
    "---\n"
)


async def hub_check(tmp_path: Path, text: str, key: str) -> Any:
    transport, _ = hub(IMDB_LIKE)
    paper = paper_saying(text)
    ctx = hub_context(tmp_path, paper, transport)
    task = task_for(paper, "https://huggingface.co/datasets/org/data", key)
    return await check_data_setting(task, ctx)


async def test_stated_split_sizes_are_checked_against_the_card(tmp_path: Path) -> None:
    text = "We use 25,000 training examples and 2,000 test examples."
    train = await hub_check(tmp_path, text, "train_examples")
    assert train.outcome is TaskOutcome.CHECKED
    assert train.findings == ()
    test = await hub_check(tmp_path, text, "test_examples")
    (finding,) = test.findings
    assert finding.error_class is ErrorClass.DATASET_SIZE_MISMATCH
    assert "2,000" in finding.summary or "2000" in finding.summary
    assert "25000" in finding.summary


async def test_rounded_sizes_allow_for_their_rounding(tmp_path: Path) -> None:
    rounded = await hub_check(tmp_path, "The test set contains 25k images.", "test_examples")
    assert rounded.findings == ()


async def test_total_size_is_the_sum_of_the_splits(tmp_path: Path) -> None:
    result = await hub_check(tmp_path, "Our dataset consists of 50,000 samples.", "total_examples")
    assert result.findings == ()


async def test_number_of_classes_comes_from_the_card_labels(tmp_path: Path) -> None:
    result = await hub_check(tmp_path, "We train a classifier over 10 classes.", "num_classes")
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.LABEL_DISTRIBUTION_MISMATCH
    (_, lookup) = finding.evidence
    assert "neg" in lookup.text


async def test_split_list_line_counts_are_example_counts(tmp_path: Path) -> None:
    files = {"splits/train.txt": "a\n" * 70, "splits/test.txt": "c\n" * 30}
    agree = await check(tmp_path, "The test set contains 30 images.", files)
    assert agree.findings == ()


@pytest.mark.parametrize(
    ("text", "key"),
    [
        ("We construct a class-balanced test set.", "class_balance"),
        (SPLIT_8_1_1, "test_overlap"),
    ],
)
async def test_balance_and_leakage_need_the_data(tmp_path: Path, text: str, key: str) -> None:
    paper = paper_saying(text)
    files = {"train.py": "x = random_split(ds, [0.8, 0.1, 0.1])\n"}
    task = task_for(paper, repo(tmp_path, files), key)
    result = await check_data_setting(task, offline_context(tmp_path, paper=paper))
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "data analysis mode" in result.reason


# --- data analysis mode ----------------------------------------------------------------


def data_mode(ctx: RunContext) -> RunContext:
    return dataclasses.replace(
        ctx, settings=ctx.settings.model_copy(update={"data_analysis_enabled": True})
    )


async def data_check(tmp_path: Path, text: str, files: dict[str, str], key: str | None) -> Any:
    paper = paper_saying(text)
    task = task_for(paper, repo(tmp_path, files), key)
    return await check_data_setting(task, data_mode(offline_context(tmp_path, paper=paper)))


TRAIN_CSV = "id,text,label\n1,good,pos\n2,bad,neg\n3,fine,pos\n4,poor,neg\n"


async def test_test_rows_found_in_training_data_are_leakage(tmp_path: Path) -> None:
    files = {
        "data/train.csv": TRAIN_CSV,
        "data/test.csv": "id,text,label\n9,fine,pos\n10,new,neg\n",
    }
    result = await data_check(tmp_path, SPLIT_8_1_1, files, "test_overlap")
    assert result.outcome is TaskOutcome.CHECKED
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.DATA_LEAKAGE
    assert "1 of 2 test rows" in finding.summary
    (_, lookup) = finding.evidence
    assert lookup.kind is EvidenceKind.LOOKUP
    assert lookup.path == "data/test.csv"
    assert list((tmp_path / "scratch").rglob("data_report-*.json"))


async def test_disjoint_splits_are_checked_without_findings(tmp_path: Path) -> None:
    files = {"data/train.csv": TRAIN_CSV, "data/test.csv": "id,text,label\n9,new,pos\n"}
    result = await data_check(tmp_path, SPLIT_8_1_1, files, "test_overlap")
    assert result.outcome is TaskOutcome.CHECKED
    assert result.findings == ()


async def test_an_unbalanced_test_set_contradicts_a_balance_claim(tmp_path: Path) -> None:
    test = "text,label\n" + "".join(f"t{i},pos\n" for i in range(9)) + "u,neg\n"
    files = {"data/train.csv": TRAIN_CSV, "data/test.csv": test}
    result = await data_check(
        tmp_path, "We construct a class-balanced test set.", files, "class_balance"
    )
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.LABEL_DISTRIBUTION_MISMATCH
    assert "neg 1" in finding.summary
    assert "pos 9" in finding.summary


async def test_row_counts_settle_a_split_the_code_does_not_state(tmp_path: Path) -> None:
    train = "text\n" + "".join(f"a{i}\n" for i in range(70))
    test = "text\n" + "".join(f"b{i}\n" for i in range(30))
    files = {"data/train.csv": train, "data/test.csv": test}
    result = await data_check(tmp_path, "We split the dataset into 80/20.", files, None)
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.DATA_SPLIT_MISMATCH
    assert "70/30" in finding.summary
    (_, lookup) = finding.evidence
    assert "train 70" in lookup.text


async def test_data_mode_without_split_files_is_unverifiable(tmp_path: Path) -> None:
    files = {"train.py": "print(1)\n", "data/all.csv": TRAIN_CSV}
    result = await data_check(tmp_path, SPLIT_8_1_1, files, "test_overlap")
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "train" in result.reason


async def test_dataset_named_anywhere_in_the_repository_is_not_flagged(tmp_path: Path) -> None:
    files = {
        "a.py": 'data = load_dataset("imdb")\n',
        "README.md": "We also report results on SST-2 with scripts/sst2.sh.\n",
    }
    result = await check(tmp_path, "We fine-tune on the SST-2 dataset.", files)
    assert result.outcome is TaskOutcome.CHECKED
    assert result.findings == ()


async def test_several_other_datasets_are_unverifiable(tmp_path: Path) -> None:
    files = {"a.py": 'a = load_dataset("imdb")\nb = load_dataset("ag_news")\n'}
    result = await check(tmp_path, "We fine-tune on the SST-2 dataset.", files)
    assert result.outcome is TaskOutcome.UNVERIFIABLE
