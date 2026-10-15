from pathlib import Path

import pytest

from auditor.agents.code import check
from auditor.planner.tables import paper_settings
from auditor.planner.tasks import code_tasks
from auditor.schemas import (
    AuditTask,
    ErrorClass,
    EvidenceKind,
    ParsedPaper,
    RepositoryRef,
    TaskOutcome,
    TaskResult,
)
from auditor.schemas.examples import example_paper
from auditor.testing import offline_context

DEFAULT = "We train with a learning rate of 3e-4 and a batch size of 256."


def paper_saying(text: str) -> ParsedPaper:
    data = example_paper().model_dump()
    data["sentences"][4]["text"] = text
    return ParsedPaper.model_validate(data)


def repo(tmp_path: Path, files: dict[str, str]) -> str:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")
    return root.resolve().as_uri()


async def run(tmp_path: Path, text: str, files: dict[str, str], key: str) -> TaskResult:
    paper = paper_saying(text)
    tasks = code_tasks(paper, RepositoryRef(url=repo(tmp_path, files)), paper_settings(paper))
    task: AuditTask = next(t for t in tasks if t.setting is not None and t.setting.key == key)
    return await check(task, offline_context(tmp_path, paper=paper))


async def test_matching_settings_are_checked(tmp_path: Path) -> None:
    files = {
        "train.py": (
            'parser.add_argument("--lr", type=float, default=3e-4)\n'
            "loader = DataLoader(ds, batch_size=256)\n"
        )
    }
    for key in ("learning_rate", "batch_size"):
        result = await run(tmp_path, DEFAULT, files, key)
        assert result.outcome is TaskOutcome.CHECKED
        assert result.findings == ()


async def test_different_value_is_a_setting_mismatch(tmp_path: Path) -> None:
    files = {"train.py": "import torch\nopt = torch.optim.Adam(params, lr=1e-3)\n"}
    result = await run(tmp_path, DEFAULT, files, "learning_rate")
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.SETTING_MISMATCH
    assert not finding.uncertain
    assert "3e-4" in finding.summary
    assert "0.001" in finding.summary
    quote, code = finding.evidence
    assert quote.kind is EvidenceKind.QUOTE
    assert code.kind is EvidenceKind.CODE
    assert (code.path, code.line_start) == ("train.py", 2)
    assert code.text == "opt = torch.optim.Adam(params, lr=1e-3)"


async def test_one_matching_value_among_several_is_enough(tmp_path: Path) -> None:
    files = {
        "train.py": 'parser.add_argument("--lr", default=1e-3)\n',
        "scripts/run.sh": "python train.py --lr 3e-4\n",
    }
    result = await run(tmp_path, DEFAULT, files, "learning_rate")
    assert result.findings == ()


async def test_several_other_values_make_an_uncertain_finding(tmp_path: Path) -> None:
    files = {
        "train.py": 'parser.add_argument("--lr", default=1e-3)\n',
        "scripts/run.sh": "python train.py --lr 5e-4\n",
    }
    result = await run(tmp_path, DEFAULT, files, "learning_rate")
    (finding,) = result.findings
    assert finding.uncertain
    assert len(finding.evidence) == 3


async def test_per_device_batch_size_is_only_a_suspicion(tmp_path: Path) -> None:
    files = {"args.py": "args = TrainingArguments(per_device_train_batch_size=32)\n"}
    result = await run(tmp_path, DEFAULT, files, "batch_size")
    (finding,) = result.findings
    assert finding.uncertain
    assert "per device" in finding.summary


async def test_setting_missing_from_the_code_is_unverifiable(tmp_path: Path) -> None:
    result = await run(tmp_path, DEFAULT, {"train.py": "print('hi')\n"}, "learning_rate")
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "learning_rate" in result.reason


async def test_percent_in_the_paper_matches_a_fraction_in_code(tmp_path: Path) -> None:
    files = {"model.py": "drop = nn.Dropout(0.1)\n"}
    result = await run(tmp_path, "We use a dropout rate of 10%.", files, "dropout")
    assert result.outcome is TaskOutcome.CHECKED
    assert result.findings == ()


@pytest.mark.parametrize(("code", "mismatch"), [("AdamW", False), ("Adam", True)])
async def test_optimizer_names_are_compared_exactly(
    tmp_path: Path, code: str, mismatch: bool
) -> None:
    files = {"train.py": f"opt = torch.optim.{code}(model.parameters())\n"}
    result = await run(tmp_path, "We train the model with AdamW.", files, "optimizer")
    assert bool(result.findings) is mismatch


async def test_reported_metric_must_be_computed(tmp_path: Path) -> None:
    text = "We report macro-F1 on the test set."
    computed = {"eval.py": 'score = f1_score(y, p, average="macro")\n'}
    assert (await run(tmp_path, text, computed, "eval_metric")).findings == ()
    other = {"eval.py": "score = accuracy_score(y, p)\n"}
    (finding,) = (await run(tmp_path / "b", text, other, "eval_metric")).findings
    assert finding.error_class is ErrorClass.METRIC_MISMATCH


async def test_generic_metric_names_cover_their_variants(tmp_path: Path) -> None:
    files = {"eval.py": 'score = f1_score(y, p, average="macro")\n'}
    result = await run(tmp_path, "We report F1 scores.", files, "eval_metric")
    assert result.findings == ()


async def test_data_keys_go_to_the_data_checks(tmp_path: Path) -> None:
    text = "We randomly split the data into 80/10/10 train/validation/test sets."
    files = {"data.py": "a, b, c = random_split(ds, [0.7, 0.15, 0.15])\n"}
    result = await run(tmp_path, text, files, "data_split")
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.DATA_SPLIT_MISMATCH


async def test_unreachable_repository_is_unverifiable(tmp_path: Path) -> None:
    paper = paper_saying(DEFAULT)
    missing = RepositoryRef(url=(tmp_path / "missing").as_uri())
    (task, *_) = code_tasks(paper, missing, paper_settings(paper))
    result = await check(task, offline_context(tmp_path, paper=paper))
    assert result.outcome is TaskOutcome.UNVERIFIABLE
