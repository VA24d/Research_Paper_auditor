"""The names code and configs give settings, mapped to the paper-side keys.

`classify(name, value)` decides whether a named value is a setting the paper can state
and normalises it: shares become percentages, numbers stay numbers, optimizer and
schedule names are lower-cased, metric names become the planner's metric keys.
"""

import re
from dataclasses import dataclass

from auditor.agents.code.extract.normalise import DATASET_NAMES, as_percent, split_key
from auditor.planner.evaluation_setup import METRIC_KEYS
from auditor.planner.settings import parse_number
from auditor.schemas import SettingValue

# Hyperparameter and architecture keys (as in auditor.planner.tables.ALIASES) and the
# names code uses for them, after `normalise.setting_name`.
CODE_NAMES: dict[str, tuple[str, ...]] = {
    "learning_rate": (
        "lr",
        "learning_rate",
        "base_lr",
        "peak_lr",
        "init_lr",
        "initial_lr",
        "max_lr",
    ),
    "batch_size": (
        "batch_size",
        "bs",
        "train_batch_size",
        "per_device_train_batch_size",
        "per_gpu_train_batch_size",
        "global_batch_size",
        "train_bs",
    ),
    "epochs": (
        "epochs",
        "num_epochs",
        "n_epochs",
        "max_epochs",
        "num_train_epochs",
        "train_epochs",
    ),
    "training_steps": (
        "max_steps",
        "num_steps",
        "train_steps",
        "training_steps",
        "total_steps",
        "num_training_steps",
        "max_iters",
    ),
    "warmup_steps": ("warmup_steps", "num_warmup_steps", "warmup_iters"),
    "warmup_ratio": ("warmup_ratio", "warmup_proportion"),
    "weight_decay": ("weight_decay", "wd"),
    "dropout": ("dropout", "dropout_rate", "dropout_prob", "hidden_dropout_prob", "drop_rate"),
    "label_smoothing": ("label_smoothing", "label_smoothing_factor"),
    "optimizer": ("optimizer", "optim", "optimizer_name", "optimizer_type"),
    "lr_schedule": ("lr_scheduler_type", "lr_scheduler", "lr_schedule", "scheduler"),
    "seed": ("seed", "random_seed", "manual_seed"),
    "max_sequence_length": (
        "max_seq_length",
        "max_seq_len",
        "max_length",
        "max_len",
        "seq_len",
        "max_source_length",
        "block_size",
        "context_length",
    ),
    "hidden_size": ("hidden_size", "d_model", "n_embd", "embed_dim", "emb_dim"),
    "intermediate_size": ("intermediate_size", "d_ff", "ffn_dim", "dim_feedforward", "mlp_dim"),
    "num_layers": ("num_layers", "n_layers", "num_hidden_layers", "n_layer", "depth"),
    "num_attention_heads": ("num_attention_heads", "n_heads", "num_heads", "n_head", "nhead"),
    "num_kv_heads": ("num_key_value_heads", "n_kv_heads", "num_kv_heads"),
    "head_dim": ("head_dim",),
    "vocab_size": ("vocab_size", "n_vocab"),
    "sliding_window": ("sliding_window", "window_size"),
}
TEXT_KEYS = frozenset({"optimizer", "lr_schedule"})
METRIC_SETTING_NAMES = frozenset({"metric_for_best_model", "eval_metric", "metric", "metric_name"})
_KEY_OF = {name: key for key, names in CODE_NAMES.items() for name in names}
# Names that hold a value per device, which a paper usually states for the whole run.
_PER_DEVICE = re.compile(r"per_(?:device|gpu)")
_TEXT_VALUE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,40}$")


@dataclass(frozen=True)
class Classified:
    key: str
    value: SettingValue
    context: str | None = None


def metric_key(name: str) -> str | None:
    """A metric name as the planner's key ("Macro-F1" -> "macro_f1")."""
    text = re.sub(r"[\s_-]+", " ", name.strip().lower())
    return METRIC_KEYS.get(text) or (text.replace(" ", "_") if text in _CODE_METRICS else None)


# Metric names libraries use that are not written that way in papers.
_CODE_METRICS = {"sacrebleu", "exact match", "squad", "rouge"}


def classify(name: str, value: object) -> Classified | None:
    """The setting a normalised `name` with a constant `value` stands for, if any."""
    split = split_key(name)
    if split is not None:
        percent = as_percent(value, name)
        return Classified(split, percent) if percent is not None else None
    if name in DATASET_NAMES:
        return (
            Classified("dataset", value.strip())
            if isinstance(value, str) and value.strip()
            else None
        )
    if name in METRIC_SETTING_NAMES:
        found = metric_key(value) if isinstance(value, str) else None
        return Classified("eval_metric", found) if found else None
    key = _KEY_OF.get(name)
    if key is None:
        return None
    context = "per device" if _PER_DEVICE.search(name) else None
    if key in TEXT_KEYS:
        if isinstance(value, str) and _TEXT_VALUE.match(value.strip()):
            return Classified(key, value.strip().lower(), context)
        return None
    number = _number(value)
    return Classified(key, number, context) if number is not None else None


def _number(value: object) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        return parse_number(value)
    return None
