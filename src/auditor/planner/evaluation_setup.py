"""Finds the evaluation setup a paper states: datasets, data splits and metrics.

These are what `data_split_mismatch` and `metric_mismatch` errors are about (decision D4).
Like `auditor.planner.settings`, the patterns are narrow and every match keeps its exact
span. Keys: `dataset`, `data_split` ("80/10/10"), `test_fraction` and
`validation_fraction` (percent), and `eval_metric` (a normalised name such as `macro_f1`);
for the data itself `train_examples`, `validation_examples`, `test_examples`,
`total_examples`, `num_classes` and `class_balance`, plus the derived `test_overlap`.
"""

import re

from auditor.schemas import ExtractedSetting, PaperSource, Sentence, SentenceAnchor, SettingValue

_FIRST_PERSON = re.compile(r"\b(we|our)\b", re.IGNORECASE)
# A passive statement of the paper's own setup ("accuracy is evaluated by mAP"), as long
# as the sentence does not cite or describe other work.
_PASSIVE = re.compile(
    r"\b(?:is|are|was|were)\s+(?:\w+\s+)?(?:evaluated|measured|reported|computed)\b"
    r"|\bmetrics?\s+(?:is|are)\b",
    re.IGNORECASE,
)
_OTHER_WORK = re.compile(
    r"\bet al\.|\[\d|\b(?:prior|previous\w*|existing|they|their)\b", re.IGNORECASE
)

# Metric names, longest first so "macro-F1" wins over "F1", with their normalised keys.
_METRICS = {
    "top-1 accuracy": "top1_accuracy",
    "top-5 accuracy": "top5_accuracy",
    "macro-f1": "macro_f1",
    "micro-f1": "micro_f1",
    "macro f1": "macro_f1",
    "micro f1": "micro_f1",
    "f1 score": "f1",
    "f1": "f1",
    "exact match": "exact_match",
    "em": "exact_match",
    "accuracy": "accuracy",
    "sacrebleu": "sacrebleu",
    "bleu": "bleu",
    "rouge-1": "rouge_1",
    "rouge-2": "rouge_2",
    "rouge-l": "rouge_l",
    "meteor": "meteor",
    "perplexity": "perplexity",
    "auroc": "auc",
    "roc-auc": "auc",
    "auc": "auc",
    "mean average precision": "map",
    "map": "map",
    "mrr": "mrr",
    "ndcg": "ndcg",
    "matthews correlations": "matthews_correlation",
    "matthews correlation": "matthews_correlation",
    "mcc": "matthews_correlation",
    "pearson correlations": "pearson_correlation",
    "pearson correlation": "pearson_correlation",
    "spearman correlations": "spearman_correlation",
    "spearman correlation": "spearman_correlation",
    "word error rate": "wer",
    "wer": "wer",
    "cider": "cider",
    "fid": "fid",
}
_METRIC = re.compile(
    r"(?<![\w-])(?P<value>"
    + "|".join(re.escape(name).replace(r"\ ", r"[\s-]+") for name in _METRICS)
    + r")(?![\w-])",
    re.IGNORECASE,
)
_METRIC_KEYS = {re.sub(r"[\s-]+", " ", name): key for name, key in _METRICS.items()}
# A metric counts only after a verb saying it is what the paper measures, in the same clause.
_EVALUATES = re.compile(
    r"\b(?:report\w*|evaluat\w*|measur\w*|comput\w*|in\s+terms\s+of|metrics?\s+(?:is|are|of))\b",
    re.IGNORECASE,
)
_CLAUSE_END = re.compile(r"[.;]\s|$")

_SPLIT_PART = r"(?:train(?:ing)?|dev(?:elopment)?|val(?:idation)?|test(?:ing)?)"
_SPLIT_RATIO = r"\d{1,2}(?:\s*%?\s*[/:]\s*\d{1,2}){1,2}\s*%?"
_SPLITS = (
    re.compile(
        rf"(?P<value>{_SPLIT_RATIO})\s+(?:{_SPLIT_PART}(?:\s*[/:-]\s*{_SPLIT_PART}){{1,2}}\s+)?"
        r"(?:split|partition)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bsplit\w*\b[^.;]{{0,40}}?\binto\s+(?P<value>{_SPLIT_RATIO})",
        re.IGNORECASE,
    ),
)
_FRACTION = re.compile(
    r"(?P<value>\d{1,2}(?:\.\d+)?\s*%)\s+of\s+(?:the\s+)?(?:[\w-]+\s+){0,3}?"
    r"(?:data|examples|samples|images|instances|set)\s+(?:\w+\s+){0,2}?(?:for|as)\s+"
    r"(?:the\s+|a\s+)?(?P<name>test\w*|validation|val|dev\w*|held-?out)",
    re.IGNORECASE,
)

# After the first word, a dataset name may go on with an all-caps word ("PASCAL VOC",
# "MS COCO") and a version or year ("SQuAD v1.1", "ImageNet 2012").
_NAME_TAIL = r"(?:\s+[A-Z]{2,}\d*)?(?:\s+v?\d+(?:\.\d+)?\b)?"
_DATASETS = (
    # "the SQuAD dataset", "the GLUE benchmark"
    re.compile(
        rf"\bthe\s+(?P<value>[A-Z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*{_NAME_TAIL})"
        r"\s+(?:dataset|benchmark|corpus)\b"
    ),
    # "train on CIFAR-10", "evaluate on SQuAD v1.1": the name must look like a dataset name,
    # with an inner capital or a digit, so "train on Wikipedia text" is left alone.
    re.compile(
        r"\b(?:[Tt]rain\w*|[Ff]ine-?tun\w*|[Ee]valuat\w*|[Tt]est\w*|[Pp]re-?train\w*)\s+"
        r"(?:\w+\s+){0,2}?on\s+(?:the\s+)?"
        rf"(?P<value>[A-Z][A-Za-z]*(?:[A-Z]|\d)[A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*{_NAME_TAIL})"
    ),
)
# Hardware, task families and model families are named the same way as datasets.
_NOT_DATASETS = re.compile(
    r"^(?:NVIDIA|GPUs?|TPU\w*|[AHV]100|CPUs?|NLU|NLP|NLG|NLI|QA|MT|ASR"
    r"|GPT|BERT|RoBERTa|DeBERTa|ALBERT|T5|BART|LLaMA|Llama|PaLM|LaMDA|OPT|BLOOM|ViT|ResNet)\b"
)
# A parameter count right after the name marks a model: "GPT-3 175B".
_MODEL_SIZE = re.compile(r"\s*\d+(?:\.\d+)?\s*[BMK]\b")


# Sizes of the data, its classes and their balance (decision D4: dataset_size_mismatch and
# label_distribution_mismatch). "25,000", "10k" and "1.2M" are read by parse_number.
_COUNT = r"\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s?[kKM]\b|\d+"
_EXAMPLE_NOUN = (
    r"(?:examples|samples|instances|images|sentences|documents|pairs|questions|utterances"
    r"|records|items|data\s+points)\b"
)
_SPLIT_NAME = r"train(?:ing)?|val(?:idation)?|dev(?:elopment)?|test(?:ing)?"
_HAS = r"(?:has|have|had|contains?|contained|consists\s+of|comprises|includes|of|with)"
_SPLIT_SIZES = (
    # "25,000 training examples"
    re.compile(rf"(?P<value>{_COUNT})\s+(?P<name>{_SPLIT_NAME})\s+{_EXAMPLE_NOUN}", re.IGNORECASE),
    # "the test set contains 10k images"
    re.compile(
        rf"\b(?P<name>{_SPLIT_NAME})\s+(?:set|split|data)\s+{_HAS}\s+(?P<value>{_COUNT})\s+"
        rf"{_EXAMPLE_NOUN}",
        re.IGNORECASE,
    ),
)
_TOTAL_SIZE = re.compile(
    rf"\b(?:dataset|corpus|benchmark)\s+{_HAS}\s+(?P<value>{_COUNT})\s+{_EXAMPLE_NOUN}",
    re.IGNORECASE,
)
_SPLIT_KEYS = {"tr": "train_examples", "va": "validation_examples", "de": "validation_examples"}
_NUMBER_WORDS = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
_CLASSES = (
    re.compile(
        r"\b(?P<value>\d+|"
        + "|".join(_NUMBER_WORDS)
        + r")[\s-]+(?:classes|categories|class\s+labels)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?P<value>\d+)-way\s+classification\b", re.IGNORECASE),
    re.compile(r"\b(?P<value>binary)\s+classification\b", re.IGNORECASE),
)
# "class-balanced" and "balanced", but not "imbalanced" or "balanced accuracy", and only
# about the data.
_BALANCED = re.compile(r"\b(?P<value>(?:class[- ])?balanced)\b(?!\s+accuracy)", re.IGNORECASE)
_ABOUT_DATA = re.compile(
    r"\b(?:sets?|splits?|data(?:set)?|classes|labels|samples)\b", re.IGNORECASE
)
# Keys whose sentence describes the paper's data, in order of preference for anchoring the
# held-out check.
_DATA_SETUP_KEYS = ("data_split", "test_fraction", "test_examples", "dataset")


def _setting(
    sentence: Sentence, key: str, name: str, raw: str, start: int, value: SettingValue
) -> ExtractedSetting:
    return ExtractedSetting(
        key=key,
        raw_name=name,
        raw_value=raw,
        value=value,
        unit="%" if raw.endswith("%") else None,
        source=PaperSource(
            anchor=SentenceAnchor(sentence_id=sentence.id, start=start, end=start + len(raw))
        ),
    )


def _split_value(raw: str) -> str | None:
    """ "80 / 10 / 10" as "80/10/10", if the parts add up to 100 (or 10, as in 8:1:1)."""
    parts = [int(p) for p in re.findall(r"\d+", raw)]
    return "/".join(map(str, parts)) if sum(parts) in (10, 100) else None


def evaluation_settings(sentence: Sentence) -> list[ExtractedSetting]:
    text = sentence.text
    found: list[ExtractedSetting] = []
    taken: list[tuple[int, int]] = []

    def add(key: str, name: str, match: re.Match[str], value: SettingValue | None) -> None:
        raw = match.group("value").strip()
        start = match.start("value")
        if value is None or any(start < e and s < start + len(raw) for s, e in taken):
            return
        if any(setting.key == key and setting.value == value for setting in found):
            return
        taken.append((start, start + len(raw)))
        found.append(_setting(sentence, key, name, raw, start, value))

    # Sizes and classes describe the data itself, so they count unless the sentence is
    # about other work.
    if not _OTHER_WORK.search(text):
        for pattern in _SPLIT_SIZES:
            for match in pattern.finditer(text):
                key = _SPLIT_KEYS.get(match.group("name")[:2].lower(), "test_examples")
                add(key, match.group("name"), match, _count(match.group("value")))
        for match in _TOTAL_SIZE.finditer(text):
            add("total_examples", "dataset size", match, _count(match.group("value")))
        for pattern in _CLASSES:
            for match in pattern.finditer(text):
                word = match.group("value").lower()
                count = 2 if word == "binary" else _NUMBER_WORDS.get(word) or int(word)
                add("num_classes", "classes", match, count)
        if _ABOUT_DATA.search(text):
            for match in _BALANCED.finditer(text):
                add("class_balance", "class balance", match, "balanced")

    # Splits and metrics count when the paper states its own setup, not when it reports
    # someone else's ("Prior work reports 92.1% exact match [2]").
    own = bool(_FIRST_PERSON.search(text)) or bool(
        _PASSIVE.search(text) and not _OTHER_WORK.search(text)
    )
    if not own:
        return sorted(found, key=_start)
    for pattern in _SPLITS:
        for match in pattern.finditer(text):
            add("data_split", "split", match, _split_value(match.group("value")))
    for match in _FRACTION.finditer(text):
        is_test = match.group("name").lower().startswith("test")
        number = float(match.group("value").rstrip("% "))
        key = "test_fraction" if is_test else "validation_fraction"
        add(key, match.group("name"), match, int(number) if number.is_integer() else number)

    for verb in _EVALUATES.finditer(text):
        end = _CLAUSE_END.search(text, verb.end())
        clause_end = end.start() if end else len(text)
        for match in _METRIC.finditer(text, verb.end(), clause_end):
            name = re.sub(r"[\s-]+", " ", match.group("value").lower())
            add("eval_metric", "metric", match, _METRIC_KEYS.get(name))

    # Dataset names are only taken from first-person sentences: a passive sentence such as
    # "the model is evaluated on GLUE" is too often about someone else's model.
    if _FIRST_PERSON.search(text):
        for pattern in _DATASETS:
            for match in pattern.finditer(text):
                raw = match.group("value").strip()
                is_model = _MODEL_SIZE.match(text, match.start("value") + len(raw))
                if not _NOT_DATASETS.match(raw) and not is_model:
                    add("dataset", "dataset", match, raw)
    return sorted(found, key=_start)


def _start(setting: ExtractedSetting) -> int:
    source = setting.source
    return (source.anchor.start or 0) if isinstance(source, PaperSource) else 0


def _count(raw: str) -> SettingValue | None:
    # Imported here: auditor.planner.settings imports this module.
    from auditor.planner.settings import parse_number

    return parse_number(raw)


def held_out_setting(settings: list[ExtractedSetting]) -> ExtractedSetting | None:
    """The implied claim that the test data is held out (`test_overlap` 0, decision D4
    `data_leakage`), anchored where the paper first describes its data split or dataset."""
    stated = [s for s in settings if isinstance(s.source, PaperSource)]
    for key in _DATA_SETUP_KEYS:
        for setting in stated:
            if setting.key == key:
                return ExtractedSetting(
                    key="test_overlap",
                    raw_name="held-out test data",
                    raw_value=setting.raw_value,
                    value=0,
                    source=setting.source,
                )
    return None
