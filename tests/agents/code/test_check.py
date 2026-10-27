import dataclasses
from pathlib import Path

import pytest

from auditor.agents.code import check
from auditor.agents.code.second_look import Applies
from auditor.infra.context import RunContext
from auditor.infra.llm import StructuredLLM
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
from auditor.testing import ScriptedChatModel, offline_context, tool_reply

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


async def test_per_device_batch_size_alone_is_unverifiable(tmp_path: Path) -> None:
    files = {"args.py": "args = TrainingArguments(per_device_train_batch_size=32)\n"}
    result = await run(tmp_path, DEFAULT, files, "batch_size")
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "per device" in result.reason


async def test_eval_loader_batch_sizes_are_ignored(tmp_path: Path) -> None:
    files = {
        "train.py": (
            "test_loader = DataLoader(test_set, batch_size=1)\n"
            "val = DataLoader(val_set, batch_size=2)\n"
        )
    }
    result = await run(tmp_path, DEFAULT, files, "batch_size")
    assert result.outcome is TaskOutcome.UNVERIFIABLE


async def test_library_function_defaults_are_not_the_experiment(tmp_path: Path) -> None:
    files = {
        "optim/radam.py": (
            "class RAdam(Optimizer):\n    def __init__(self, params, lr=1e-3):\n        pass\n"
        ),
        "utils/helpers.py": "def make_optimizer(model, lr=5e-4):\n    pass\n",
    }
    result = await run(tmp_path, DEFAULT, files, "learning_rate")
    assert result.outcome is TaskOutcome.UNVERIFIABLE


async def test_defaults_of_an_entry_point_count(tmp_path: Path) -> None:
    files = {
        "main.py": ("def main(lr=1e-3):\n    pass\n\n\nif __name__ == '__main__':\n    main()\n")
    }
    (finding,) = (await run(tmp_path, DEFAULT, files, "learning_rate")).findings
    assert finding.error_class is ErrorClass.SETTING_MISMATCH


async def test_vendored_and_checkpoint_files_are_not_scanned(tmp_path: Path) -> None:
    files = {
        "third_party/lib/opt.py": "lr = 1e-3\n",
        "checkpoints/run1/trainer_state.json": '{"learning_rate": 0.3}',
        "outputs/args.yaml": "lr: 0.1\n",
    }
    result = await run(tmp_path, DEFAULT, files, "learning_rate")
    assert result.outcome is TaskOutcome.UNVERIFIABLE


async def test_many_configs_unrelated_to_the_sentence_are_unverifiable(tmp_path: Path) -> None:
    files = {
        "configs/cifar.yaml": "lr: 0.1\n",
        "configs/imagenet.yaml": "lr: 0.01\n",
        "configs/mnist.yaml": "lr: 0.05\n",
    }
    result = await run(tmp_path, DEFAULT, files, "learning_rate")
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "configurations" in result.reason


async def test_a_config_named_in_the_sentence_is_still_compared(tmp_path: Path) -> None:
    text = "On CIFAR we train with a learning rate of 3e-4."
    files = {
        "configs/cifar.yaml": "lr: 0.1\n",
        "configs/imagenet.yaml": "lr: 0.01\n",
        "configs/mnist.yaml": "lr: 0.05\n",
    }
    (finding,) = (await run(tmp_path, text, files, "learning_rate")).findings
    assert "cifar" in finding.summary


async def test_specific_metric_matches_the_generic_one_in_code(tmp_path: Path) -> None:
    files = {"eval.py": 'metric = evaluate.load("rouge")\n'}
    result = await run(tmp_path, "We report ROUGE-L on the test set.", files, "eval_metric")
    assert result.outcome is TaskOutcome.CHECKED
    assert result.findings == ()


async def test_metric_dispatch_tables_are_unverifiable(tmp_path: Path) -> None:
    files = {
        "metrics.py": (
            "a = accuracy_score(y, p)\nb = matthews_corrcoef(y, p)\nc = roc_auc_score(y, p)\n"
        )
    }
    result = await run(tmp_path, "We report macro-F1 on the test set.", files, "eval_metric")
    assert result.outcome is TaskOutcome.UNVERIFIABLE


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


async def test_model_constructor_defaults_count(tmp_path: Path) -> None:
    files = {
        "model.py": (
            "class Net(nn.Module):\n"
            "    def __init__(self, dropout=0.3):\n"
            "        super().__init__()\n"
        )
    }
    (finding,) = (await run(tmp_path, "We use a dropout rate of 10%.", files, "dropout")).findings
    assert finding.error_class is ErrorClass.SETTING_MISMATCH


async def test_the_config_holding_the_papers_other_settings_is_used(tmp_path: Path) -> None:
    files = {
        "configs/a.yaml": "lr: 0.1\nbatch_size: 64\n",
        "configs/b.yaml": "lr: 0.01\nbatch_size: 256\n",
        "configs/c.yaml": "lr: 0.05\nbatch_size: 16\n",
    }
    (finding,) = (await run(tmp_path, DEFAULT, files, "learning_rate")).findings
    assert "configs/b.yaml" in finding.summary
    assert "configs/a.yaml" not in finding.summary


# --- second look by a small model on uncertain findings --------------------------------


def with_llm(
    ctx: RunContext, replies: list[dict[str, str]]
) -> tuple[RunContext, ScriptedChatModel]:
    model = ScriptedChatModel(replies=[tool_reply(Applies, reply) for reply in replies])
    return dataclasses.replace(ctx, llm=StructuredLLM(model, name="test:scripted")), model


TWO_VALUES = {
    "train.py": 'parser.add_argument("--lr", default=1e-3)\n',
    "scripts/finetune.sh": "python train.py --lr 5e-4\n",
}


async def second_look(tmp_path: Path, answers: list[str]) -> tuple[TaskResult, ScriptedChatModel]:
    paper = paper_saying(DEFAULT)
    tasks = code_tasks(paper, RepositoryRef(url=repo(tmp_path, TWO_VALUES)), paper_settings(paper))
    task = next(t for t in tasks if t.setting is not None and t.setting.key == "learning_rate")
    ctx, model = with_llm(
        offline_context(tmp_path, paper=paper), [{"answer": a, "reason": "r"} for a in answers]
    )
    return await check(task, ctx), model


async def test_values_confirmed_by_the_second_look_make_a_firm_finding(tmp_path: Path) -> None:
    result, model = await second_look(tmp_path, ["yes", "no"])
    (finding,) = result.findings
    assert not finding.uncertain
    first = " ".join(str(m.content) for m in model.received[0])
    assert "learning rate of 3e-4" in first  # the paper sentence
    confirmed, rejected = (
        ("scripts/finetune.sh", "train.py") if "finetune.sh" in first else ("train.py", "finetune")
    )
    assert confirmed in finding.summary
    assert rejected not in finding.summary


async def test_values_rejected_by_the_second_look_are_unverifiable(tmp_path: Path) -> None:
    result, _ = await second_look(tmp_path, ["no", "no"])
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.findings == ()


async def test_an_unsure_second_look_keeps_the_suspicion(tmp_path: Path) -> None:
    result, _ = await second_look(tmp_path, ["unsure", "unsure"])
    (finding,) = result.findings
    assert finding.uncertain


async def test_multiple_distinct_values_confirmed_keeps_uncertain(tmp_path: Path) -> None:
    result, _ = await second_look(tmp_path, ["yes", "yes"])
    (finding,) = result.findings
    assert finding.uncertain


async def test_same_values_confirmed_by_second_look_not_uncertain(tmp_path: Path) -> None:
    same_values = {
        "train.py": 'parser.add_argument("--lr", default=1e-3)\n',
        "scripts/finetune.sh": "python train.py --lr 1e-3\n",
    }
    paper = paper_saying(DEFAULT)
    tasks = code_tasks(paper, RepositoryRef(url=repo(tmp_path, same_values)), paper_settings(paper))
    task = next(t for t in tasks if t.setting is not None and t.setting.key == "learning_rate")
    answers = [{"answer": a, "reason": "r"} for a in ["yes", "yes"]]
    ctx, _ = with_llm(offline_context(tmp_path, paper=paper), answers)
    result = await check(task, ctx)
    (finding,) = result.findings
    assert not finding.uncertain
