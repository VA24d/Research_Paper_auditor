from pathlib import Path

import pytest

from auditor.planner.settings import extract_settings, parse_number, settings_in_sentence
from auditor.planner.tasks import plan_paper
from auditor.schemas import PaperSource, ParsedPaper, Sentence
from auditor.schemas.examples import example_paper
from auditor.testing import offline_context

MINUS = chr(0x2212)
TIMES = chr(0x00D7)
BULLET = chr(0x2022)


def found(text: str) -> list[tuple[str, str, object]]:
    return [
        (s.key, s.raw_value, s.value) for s in settings_in_sentence(Sentence(id="s", text=text))
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "We train with a learning rate of 3e-4 and a batch size of 256.",
            [("learning_rate", "3e-4", 3e-4), ("batch_size", "256", 256)],
        ),
        (
            "We train for 10 epochs with AdamW, weight decay 0.01 and dropout 0.1.",
            [
                ("epochs", "10", 10),
                ("optimizer", "AdamW", "adamw"),
                ("weight_decay", "0.01", 0.01),
                ("dropout", "0.1", 0.1),
            ],
        ),
        (
            "We use 2,000 warmup steps and 100k training steps.",
            [("warmup_steps", "2,000", 2000), ("training_steps", "100k", 100_000)],
        ),
        (
            f"A peak learning rate of 1 {TIMES} 10^{MINUS}4 is used.",
            [("learning_rate", f"1 {TIMES} 10^{MINUS}4", 1e-4)],
        ),
        (
            "Results use random seed 42 and a maximum sequence length of 512.",
            [("seed", "42", 42), ("max_sequence_length", "512", 512)],
        ),
        (
            "We optimise with SGD at lr = 0.1.",
            [("optimizer", "SGD", "sgd"), ("learning_rate", "0.1", 0.1)],
        ),
        ("Dropout is set to 10%.", [("dropout", "10%", 10)]),
        (
            f"We train with a learning rate of 8 {BULLET} 10 -4.",
            [("learning_rate", f"8 {BULLET} 10 -4", 8e-4)],
        ),
        ("We use a learning rate of 10 -3.", [("learning_rate", "10 -3", 1e-3)]),
        ("We use a learning rate of 10^-3.", [("learning_rate", "10^-3", 1e-3)]),
        ("We use a learning rate of 10^{-3}.", [("learning_rate", "10^{-3}", 1e-3)]),
        ("We use a batch size of 10 - 20.", [("batch_size", "10", 10)]),
        ("Following Adam [12], we report accuracy.", []),
        ("Compared to GPT-3 175B fine-tuned with Adam, LoRA uses less memory.", []),
        ("The model is trained with AdamW.", []),
        ("Batch Size 8 8 32 4 6 8 4 4 Learning Rate 1E- 45", []),
        ("We set 7 and 5 steps for HotpotQA and FEVER.", []),
        (
            "We finetune ReAct for 4, 000 steps.",
            [("training_steps", "4, 000", 4000)],
        ),
        (
            "We use learning rate warmup over the first 10,000 steps.",
            [("warmup_steps", "10,000", 10_000)],
        ),
        ("Accuracy improves by 3 points.", []),
    ],
)
def test_settings_in_sentence(text: str, expected: list[tuple[str, str, object]]) -> None:
    assert found(text) == expected


def test_spans_point_at_the_value() -> None:
    sentence = Sentence(id="s4", text="We train with a learning rate of 3e-4.")
    (setting,) = settings_in_sentence(sentence)
    assert isinstance(setting.source, PaperSource)
    anchor = setting.source.anchor
    assert anchor.start is not None
    assert anchor.end is not None
    assert sentence.text[anchor.start : anchor.end] == "3e-4"


def test_percent_values_keep_their_unit() -> None:
    (setting,) = settings_in_sentence(Sentence(id="s", text="Dropout is set to 10%."))
    assert setting.unit == "%"


def test_extract_settings_reads_the_whole_paper() -> None:
    keys = [s.key for s in extract_settings(example_paper())]
    assert keys == ["learning_rate", "batch_size"]


@pytest.mark.parametrize(
    ("raw", "value"),
    [
        ("3e-4", 3e-4),
        (f"3e{MINUS}4", 3e-4),
        ("1 x 10^-4", 1e-4),
        (f"2 {TIMES} 10^{{{MINUS}5}}", 2e-5),
        ("2,000", 2000),
        ("100k", 100_000),
        ("1.5M", 1_500_000.0),
        ("0.1", 0.1),
        ("256", 256),
        ("10%", 10),
        ("abc", None),
        ("16 32", None),
        ("20 000", 20_000),
        ("", None),
        ("%", None),
        (f"8 {BULLET} 10 -4", 8e-4),
        ("10 -3", 1e-3),
        ("10^-4", 1e-4),
        ("10-20", None),
        ("10 - 20", None),
        ("10 3", None),
        ("100", 100),
    ],
)
def test_parse_number(raw: str, value: float | None) -> None:
    result = parse_number(raw)
    assert result == pytest.approx(value) if value is not None else result is None


async def test_two_values_of_one_setting_become_two_code_tasks(tmp_path: Path) -> None:
    data = example_paper().model_dump()
    data["sentences"][4]["text"] = (
        "We use a learning rate of 1e-4 for BERT and a learning rate of 2e-5 for RoBERTa."
    )
    paper = ParsedPaper.model_validate(data)
    tasks = await plan_paper(paper, offline_context(tmp_path, paper=paper))
    values = sorted(
        t.setting.raw_value for t in tasks if t.setting is not None and t.setting.key != "method"
    )
    assert values == ["1e-4", "2e-5"]


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        ("Grathwohl et al. use a batch size of 900.", False),
        ("EWC [39] uses a batch size of 32.", False),
        ("We follow [3] and use a batch size of 32.", True),
        ("The model is trained with a batch size of 32.", True),
        ("The model is trained following [12] with a batch size of 32.", True),
        ("SGD [61] is used as the optimizer with a batch size of 32.", True),
        ("OGD [13] uses a small batch size of 20.", False),
        ("Models are trained as in Smith et al. (2020) with a batch size of 32.", True),
    ],
)
def test_settings_of_cited_work_are_not_the_papers(text: str, kept: bool) -> None:
    found = [s.key for s in settings_in_sentence(Sentence(id="s", text=text))]
    assert ("batch_size" in found) is kept


@pytest.mark.parametrize(
    ("text", "value"),
    [
        (
            "Categorical Cross Entropy has been used as the loss function, while Adam "
            "(Adaptive Moment Estimation) optimizer [11] has been used for weight update.",
            "adam",
        ),
        ("The SGD optimizer is used for all runs.", "sgd"),
        ("GPT-3 was fine-tuned with Adam.", None),
        ("The Adam optimizer proposed by Kingma and Ba is popular.", None),
    ],
)
def test_passive_optimizer_sentences_state_the_papers_setup(text: str, value: str | None) -> None:
    found = [
        s.value for s in settings_in_sentence(Sentence(id="s", text=text)) if s.key == "optimizer"
    ]
    assert found == ([value] if value else [])
