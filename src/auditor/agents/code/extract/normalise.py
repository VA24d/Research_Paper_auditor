"""Split values in the forms the paper side uses (decision D3 keys from
`auditor.planner.evaluation_setup`): percentages for `test_fraction` and
`validation_fraction`, and "80/10/10" strings for `data_split`."""

import re
from collections.abc import Sequence

from auditor.planner.settings import parse_number

# Names code and configs use for the share of data held out, after `setting_name`.
TEST_NAMES = frozenset(
    {
        "test_size",
        "test_ratio",
        "test_split",
        "test_frac",
        "test_fraction",
        "test_pct",
        "test_percent",
        "test_percentage",
        "test_portion",
        "test_proportion",
        "holdout_ratio",
        "holdout_size",
    }
)
VALIDATION_NAMES = frozenset(
    {
        "val_size",
        "val_ratio",
        "val_split",
        "val_frac",
        "val_fraction",
        "val_pct",
        "val_percent",
        "valid_size",
        "valid_ratio",
        "valid_split",
        "validation_size",
        "validation_ratio",
        "validation_split",
        "validation_fraction",
        "dev_size",
        "dev_ratio",
        "dev_split",
    }
)
DATASET_NAMES = frozenset({"dataset", "dataset_name"})

# How close two percentages must be to count as the same split: real counts rarely
# divide exactly (8530/1066/1066 is "80/10/10").
TOLERANCE_POINTS = 1.0


def setting_name(raw: str) -> str:
    """`--Test-Size` and `data.test_size` style names as `test_size`."""
    return re.sub(r"[-.\s]+", "_", raw.strip().lstrip("-")).lower()


def split_key(name: str) -> str | None:
    """The paper-side key a normalised name stands for, if it is about a held-out share."""
    if name in TEST_NAMES:
        return "test_fraction"
    if name in VALIDATION_NAMES:
        return "validation_fraction"
    return None


def as_percent(value: object, name: str = "") -> float | None:
    """A held-out share as a percentage: 0.2 and "20%" are 20. Whole numbers of 1 or more
    are row counts, not shares, unless the name says percent."""
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        text = value.strip()
        number = parse_number(text)
        if number is None:
            return None
        if text.endswith("%"):
            return _percent_if_valid(float(number))
        value = number
    if not isinstance(value, int | float):
        return None
    number = float(value)
    if 0 < number < 1:
        return round(number * 100, 4)
    if any(word in name for word in ("pct", "percent")):
        return _percent_if_valid(number)
    return None


def _percent_if_valid(number: float) -> float | None:
    return round(number, 4) if 0 < number < 100 else None


def split_from_parts(parts: Sequence[float]) -> str | None:
    """Fractions, percentages or row counts of each split as "80/10/10"."""
    if len(parts) < 2 or any(part <= 0 for part in parts):
        return None
    total = sum(parts)
    return "/".join(_format(part / total * 100) for part in parts)


def _format(percent: float) -> str:
    nearest = round(percent)
    return str(nearest) if abs(percent - nearest) < 0.05 else f"{percent:.1f}"


def split_parts(value: str) -> tuple[float, ...]:
    """ "80/10/10" and "8/1/1" as percentages (80.0, 10.0, 10.0)."""
    numbers = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", value)]
    total = sum(numbers)
    if total <= 0:
        return ()
    return tuple(round(n / total * 100, 4) for n in numbers)


def splits_match(a: str, b: str, tolerance: float = TOLERANCE_POINTS) -> bool:
    pa, pb = split_parts(a), split_parts(b)
    return len(pa) == len(pb) and all(
        abs(x - y) <= tolerance + 1e-9 for x, y in zip(pa, pb, strict=True)
    )


def percents_match(a: float, b: float, tolerance: float = TOLERANCE_POINTS) -> bool:
    return abs(a - b) <= tolerance + 1e-9
