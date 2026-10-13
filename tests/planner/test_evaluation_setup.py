from pathlib import Path

import pytest

from auditor.planner.evaluation_setup import evaluation_settings
from auditor.planner.tasks import PlannerOptions, plan_paper
from auditor.schemas import PaperSource, RepositoryRef, Sentence, SettingValue, TaskKind
from auditor.schemas.examples import example_paper
from auditor.testing import offline_context


def found(text: str) -> list[tuple[str, str, SettingValue | None]]:
    sentence = Sentence(id="s", text=text)
    return [(s.key, s.raw_value, s.value) for s in evaluation_settings(sentence)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "We use SQuAD with an 80/20 train/test split and report macro-F1.",
            [("data_split", "80/20", "80/20"), ("eval_metric", "macro-F1", "macro_f1")],
        ),
        (
            "We split the data into 80/10/10 for training, validation and testing.",
            [("data_split", "80/10/10", "80/10/10")],
        ),
        ("We use an 8:1:1 split.", [("data_split", "8:1:1", "8/1/1")]),
        (
            "We hold out 10% of the training data for validation.",
            [("validation_fraction", "10%", 10)],
        ),
        ("We keep 20% of the examples as test data.", [("test_fraction", "20%", 20)]),
        (
            "We report top-1 accuracy and ROUGE-L.",
            [
                ("eval_metric", "top-1 accuracy", "top1_accuracy"),
                ("eval_metric", "ROUGE-L", "rouge_l"),
            ],
        ),
        ("We report Exact Match (EM) scores.", [("eval_metric", "Exact Match", "exact_match")]),
        (
            "We train on CIFAR-10 and evaluate on the SQuAD v1.1 dataset.",
            [("dataset", "CIFAR-10", "CIFAR-10"), ("dataset", "SQuAD v1.1", "SQuAD v1.1")],
        ),
        ("We fine-tune on the PASCAL VOC sets.", [("dataset", "PASCAL VOC", "PASCAL VOC")]),
        # Not settings: a date-like ratio, a metric without an evaluation verb, hardware, a
        # model with its size, a task family, another paper's dataset, a plain word.
        ("Accuracy rose from 80/20 in 2019.", []),
        ("Our model improves accuracy by 2 points.", []),
        ("We train on NVIDIA V100 GPUs.", []),
        ("We train on GPT-3 175B.", []),
        ("Our method is an alternative to fine-tuning on NLU.", []),
        ("They train on the ImageNet dataset.", []),
        ("We train on Wikipedia text.", []),
        ("Prior work reports 92.1% exact match on SQuAD [2, 3].", []),
        ("The accuracy is evaluated by mAP.", [("eval_metric", "mAP", "map")]),
    ],
)
def test_evaluation_setup_in_sentences(
    text: str, expected: list[tuple[str, str, SettingValue | None]]
) -> None:
    assert found(text) == expected


def test_spans_quote_the_sentence() -> None:
    sentence = Sentence(id="s", text="We use an 80/20 train/test split and report macro-F1.")
    for setting in evaluation_settings(sentence):
        assert isinstance(setting.source, PaperSource)
        anchor = setting.source.anchor
        assert sentence.text[anchor.start : anchor.end] == setting.raw_value


async def test_split_and_metric_become_code_tasks(tmp_path: Path) -> None:
    paper = example_paper()
    text = "We use SQuAD with an 80/20 train/test split and report macro-F1."
    changed = paper.sentences[1].model_copy(update={"text": text})
    paper = paper.model_copy(
        update={"sentences": (paper.sentences[0], changed, *paper.sentences[2:])}
    )
    tasks = await plan_paper(
        paper,
        offline_context(tmp_path, paper=paper),
        PlannerOptions(repository=RepositoryRef(url="https://github.com/a/b"), use_llm=False),
    )
    keys = {t.setting.key for t in tasks if t.kind is TaskKind.CODE and t.setting}
    assert {"data_split", "eval_metric"} <= keys
