import pytest

from auditor.planner.settings import extract_settings, parse_number, settings_in_sentence
from auditor.schemas import PaperSource, Sentence
from auditor.schemas.examples import example_paper

MINUS = chr(0x2212)
TIMES = chr(0x00D7)


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
            "The model is trained for 10 epochs with AdamW, weight decay 0.01 and dropout 0.1.",
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
        ("Following Adam [12], we report accuracy.", []),
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
    ],
)
def test_parse_number(raw: str, value: float | None) -> None:
    result = parse_number(raw)
    assert result == pytest.approx(value) if value is not None else result is None
