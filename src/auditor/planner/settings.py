"""Finds experimental settings stated in a paper's sentences, such as "a learning rate of 3e-4".

Pattern-based on purpose: every match keeps its exact span, so the code agent and the
verifier can quote the paper. Missing a setting costs a check; inventing one would cost a
false accusation, so the patterns are narrow.
"""

import re
from dataclasses import dataclass

from auditor.planner.evaluation_setup import evaluation_settings
from auditor.schemas import (
    ExtractedSetting,
    PaperSource,
    ParsedPaper,
    Sentence,
    SentenceAnchor,
    SentenceKind,
    SettingValue,
)

_MINUS = "-\u2212\u2013"
# "4, 000" is how GROBID sometimes renders 4,000, so a comma may be followed by a space
# when exactly three digits come next.
# Powers of ten use any multiplication sign ("8 x 10^-4"), may stand alone ("10^-3",
# "10^{-3}"), and a bare "10 -3" is a flattened superscript. parse_number reads the same
# forms.
_NUMBER = (
    rf"(?:10\s*\^\s*\{{?\s*[{_MINUS}]?\s*\d+\s*\}}?|"
    rf"10\s+[{_MINUS}]\d+(?!\d|\.\d)|"
    rf"\d+(?:[.,]\d+|,\s\d{{3}}(?!\d))*(?:\s*[eE]\s*[{_MINUS}+]?\s*\d+)?"
    rf"(?:\s*(?:[x\u00d7\u00b7\u2022\u22c5]|\\times)\s*10\s*\^?\s*\{{?\s*[{_MINUS}]?\s*\d+\s*\}}?)?"
    r"\s*[kKM]?)"
)
_LINK = r"(?:\s+(?:of|is|was|were|=|:|set\s+to|to|at|as))*\s*:?\s*"


_NUMERIC_TOKEN = re.compile(r"[\d.,%+-]*\d[\d.,%+-]*[eE]?[\d+-]*")
_FIRST_PERSON = re.compile(r"\b(we|our)\b", re.IGNORECASE)
_STEP_QUALIFIER = re.compile(r"training|gradient|optimi[sz]ation", re.IGNORECASE)
_TRAINING_CONTEXT = re.compile(
    r"\b(train\w*|pre-?train\w*|fine-?tun\w*|finetun\w*|optimi[sz]\w*)\b", re.IGNORECASE
)


@dataclass(frozen=True)
class _Pattern:
    key: str
    regex: re.Pattern[str]
    numeric: bool = True


def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


PATTERNS = (
    _Pattern(
        "learning_rate",
        _compile(
            rf"(?P<name>(?:peak\s+|initial\s+|base\s+)?learning\s+rate|\blr\b){_LINK}(?P<value>{_NUMBER})"
        ),
    ),
    _Pattern("batch_size", _compile(rf"(?P<name>batch\s+size){_LINK}(?P<value>{_NUMBER})")),
    _Pattern(
        "epochs",
        _compile(r"(?P<value>\d+)\s+(?P<name>(?:training\s+)?epochs)\b"),
    ),
    _Pattern(
        "warmup_steps",
        _compile(rf"(?P<value>{_NUMBER})\s+(?P<name>warm-?up\s+(?:steps|iterations))"),
    ),
    _Pattern(
        "warmup_steps",
        _compile(
            rf"(?P<name>warm-?up)\s+(?:over|for|of|during)\s+(?:the\s+first\s+)?"
            rf"(?P<value>{_NUMBER})\s+(?:steps|iterations)"
        ),
    ),
    _Pattern(
        "training_steps",
        _compile(
            rf"(?<!warmup )(?<!warm-up )(?P<value>{_NUMBER})\s+"
            r"(?P<name>(?:training\s+|gradient\s+|optimi[sz]ation\s+)?(?:steps|iterations))\b"
        ),
    ),
    _Pattern("weight_decay", _compile(rf"(?P<name>weight\s+decay){_LINK}(?P<value>{_NUMBER})")),
    _Pattern(
        "dropout",
        _compile(rf"(?P<name>dropout(?:\s+(?:rate|probability))?){_LINK}(?P<value>{_NUMBER}\s*%?)"),
    ),
    _Pattern("seed", _compile(rf"(?P<name>(?:random\s+)?seed){_LINK}(?P<value>\d+)\b")),
    _Pattern(
        "max_sequence_length",
        _compile(
            rf"(?P<name>(?:maximum|max\.?)\s+(?:sequence|input|context)\s+length){_LINK}"
            rf"(?P<value>{_NUMBER})"
        ),
    ),
    _Pattern(
        "optimizer",
        _compile(
            r"(?:\b(?:use|used|using|with|train\w*|optimi[sz]\w*)\b[^.]{0,40}?)"
            r"\b(?P<value>AdamW|Adam|SGD|Adafactor|LAMB|RMSprop|Lion)\b(?P<name>)"
        ),
        numeric=False,
    ),
)


def parse_number(raw: str) -> int | float | None:
    """`3e-4`, `3 x 10^-4`, `1,000`, `10k` and `5%` to numbers (percent stays as written)."""
    text = raw.strip().rstrip("%").strip()
    for minus in "\u2212\u2013":
        text = text.replace(minus, "-")
    if not text:
        return None
    multiplier = 1
    if text[-1] in "kKM":
        multiplier = 1_000_000 if text[-1] == "M" else 1_000
        text = text[:-1].strip()
    # A power of ten: "3 x 10^-4" (any multiplication sign), or a bare "10 -3" whose
    # superscript was flattened. A bare power needs a caret, or a minus with a space before it
    # and none after; "100", "10 3", "10-20" and "10 - 20" are not read as powers.
    power = re.match(
        r"^(?:(?P<base>[\d.]+)\s*(?:[x\u00d7\u00b7\u2022\u22c5]|\\times)\s*)?10\s*(?P<caret>\^)?\s*\{?\s*(?P<exp>-?\s*\d+)\s*\}?$",
        text,
    )
    if power and not (power["base"] or power["caret"] or re.match(r"10\s+-\d", text)):
        power = None
    try:
        if power:
            base = float(power["base"]) if power["base"] else 1.0
            value: float = base * 10 ** int(power["exp"].replace(" ", ""))
        else:
            # Spaces and commas count only as thousands separators ("20 000", "4, 000");
            # "16 32" is two numbers from merged table cells, not 1632.
            compact = re.sub(r"(?<=\d),? (?=\d{3}(?!\d))", "", text)
            if re.fullmatch(r"\d{1,3}(?:,\d{3})+", compact):
                compact = compact.replace(",", "")
            value = float(compact)
    except ValueError:
        return None
    value *= multiplier
    return (
        int(value) if value.is_integer() and "e" not in text.lower() and "." not in text else value
    )


def _value(pattern: _Pattern, raw: str) -> SettingValue | None:
    if not pattern.numeric:
        return raw.lower()
    return parse_number(raw)


def looks_like_table_row(text: str) -> bool:
    """Table cells flattened into text ("Batch Size 8 8 32 4 6") give no single value."""
    tokens = text.split()
    numeric = sum(1 for token in tokens if _NUMERIC_TOKEN.fullmatch(token))
    return numeric >= 5 and numeric / len(tokens) >= 0.3


def settings_in_sentence(sentence: Sentence) -> list[ExtractedSetting]:
    if looks_like_table_row(sentence.text):
        return []
    first_person = bool(_FIRST_PERSON.search(sentence.text))
    found: list[ExtractedSetting] = []
    taken: list[tuple[int, int]] = []
    for pattern in PATTERNS:
        # Optimizer names also appear when describing other people's systems
        # ("GPT-3 fine-tuned with Adam"), so only first-person sentences count.
        if pattern.key == "optimizer" and not first_person:
            continue
        for match in pattern.regex.finditer(sentence.text):
            # A bare "7 steps" can be reasoning or environment steps; it only counts as
            # training when qualified or when the sentence is about training.
            if (
                pattern.key == "training_steps"
                and not _STEP_QUALIFIER.search(match.group("name"))
                and not _TRAINING_CONTEXT.search(sentence.text)
            ):
                continue
            raw_value = match.group("value").strip()
            start = match.start("value")
            end = start + len(raw_value)
            if any(start < t_end and t_start < end for t_start, t_end in taken):
                continue
            value = _value(pattern, raw_value)
            if value is None:
                continue
            taken.append((start, end))
            name = match.group("name").strip() or pattern.key.replace("_", " ")
            found.append(
                ExtractedSetting(
                    key=pattern.key,
                    raw_name=name,
                    raw_value=raw_value,
                    value=value,
                    unit="%" if raw_value.endswith("%") else None,
                    source=PaperSource(
                        anchor=SentenceAnchor(sentence_id=sentence.id, start=start, end=end)
                    ),
                )
            )
    return sorted(found, key=_start)


def _start(setting: ExtractedSetting) -> int:
    source = setting.source
    return (source.anchor.start or 0) if isinstance(source, PaperSource) else 0


def extract_settings(paper: ParsedPaper) -> list[ExtractedSetting]:
    """Settings and the evaluation setup stated in sentences; table rows are read by
    `auditor.planner.tables`."""
    return [
        setting
        for sentence in paper.sentences
        if sentence.kind is not SentenceKind.TABLE and not looks_like_table_row(sentence.text)
        for setting in (*settings_in_sentence(sentence), *evaluation_settings(sentence))
    ]
