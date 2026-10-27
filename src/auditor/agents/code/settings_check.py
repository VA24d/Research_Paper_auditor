"""Checks a hyperparameter, architecture size, optimizer, seed or metric the paper states
against the values its repository uses.

Every value the static scan finds for the key counts (argparse defaults, configs, keyword
arguments, command lines in scripts): the paper agrees with the repository if any of them
matches. Otherwise the finding lists them; it is uncertain when the repository holds
several different values, a value is per device, or the paper's value belongs to one of
several configurations, since the experiment may use another one.
"""

import math
import re
from collections import Counter

from auditor.agents.code.data_checks import (
    Candidate,
    Tier,
    paper_quote,
    paper_sentence,
    repository_evidence,
    static_candidates,
)
from auditor.agents.code.second_look import second_look
from auditor.infra.context import RunContext
from auditor.infra.snapshots import SnapshotUnavailableError, SourceError, fetch_snapshot
from auditor.planner.tables import paper_settings
from auditor.schemas import (
    AuditTask,
    CandidateFinding,
    ErrorClass,
    ExtractedSetting,
    RepositorySource,
    TaskOutcome,
    TaskResult,
)

# Numbers count as equal within this relative tolerance ("3e-4" and "0.0003").
RELATIVE_TOLERANCE = 1e-6
# A generic metric name in the paper covers the code's more specific ones.
_METRIC_FAMILIES = {
    "f1": ("macro_f1", "micro_f1"),
    "rouge": ("rouge_1", "rouge_2", "rouge_l"),
    "bleu": ("sacrebleu",),
    "exact_match": ("squad",),
}


async def check_setting(task: AuditTask, ctx: RunContext) -> TaskResult:
    setting, repository = task.setting, task.repository
    if setting is None or repository is None:
        raise ValueError(f"{task.task_id} is not a code task")
    try:
        snapshot = await fetch_snapshot(repository, ctx.http, ctx.settings)
    except (SnapshotUnavailableError, SourceError) as exc:
        return TaskResult.for_task(
            task, TaskOutcome.UNVERIFIABLE, reason=f"the repository could not be fetched: {exc}"
        )
    candidates = [
        c
        for c in await static_candidates(snapshot)
        if c.tier is Tier.CODE and c.setting.key == setting.key
    ]
    if not candidates:
        reason = f"no {setting.key} found in the repository's code, configs or scripts"
        if snapshot.manifest.incomplete:
            reason += f" (the snapshot is incomplete: {'; '.join(snapshot.manifest.incomplete)})"
        return TaskResult.for_task(task, TaskOutcome.UNVERIFIABLE, reason=reason)
    if any(_agrees(setting, c.setting) for c in candidates):
        return TaskResult.for_task(task, TaskOutcome.CHECKED)

    sentence = paper_sentence(task, ctx)
    undecided = _undecided(
        setting,
        sentence,
        candidates,
        _experiment_files(ctx, setting, await static_candidates(snapshot)),
    )
    if isinstance(undecided, str):
        return TaskResult.for_task(task, TaskOutcome.UNVERIFIABLE, reason=undecided)
    candidates = undecided

    values = {_shown(c.setting) for c in candidates}
    uncertain = (
        len(values) > 1 or setting.context is not None or any(c.setting.context for c in candidates)
    )
    if uncertain and ctx.llm is not None and ctx.settings.code_second_look:
        confirmed, kept, rejected_all = await second_look(ctx, setting, sentence, candidates)
        if rejected_all:
            return TaskResult.for_task(
                task,
                TaskOutcome.UNVERIFIABLE,
                reason=(
                    f"every repository value of {setting.key} was judged to belong to something "
                    "other than the experiment the sentence describes"
                ),
            )
        if confirmed:
            candidates = confirmed
            uncertain = (
                len({_shown(c.setting) for c in confirmed}) > 1 or setting.context is not None
            )
        else:
            candidates = kept
    error_class = (
        ErrorClass.METRIC_MISMATCH if setting.key == "eval_metric" else ErrorClass.SETTING_MISMATCH
    )
    finding = CandidateFinding.for_task(
        task,
        0,
        error_class=error_class,
        summary=(
            f"The paper states {setting.key} {setting.raw_value}; the repository has "
            + ", ".join(_where(c) for c in candidates[:4])
            + (f" and {len(candidates) - 4} more" if len(candidates) > 4 else "")
        ),
        explanation=_explanation(setting, candidates, snapshot_url=snapshot.source_id),
        evidence=[paper_quote(task, sentence), *repository_evidence(candidates)],
        uncertain=uncertain,
    )
    return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=[finding])


_PATH_TOKEN = re.compile(r"[a-z0-9]{3,}")
_GENERIC_PATH_TOKENS = frozenset(
    {
        "configs",
        "config",
        "conf",
        "yaml",
        "yml",
        "json",
        "toml",
        "train",
        "base",
        "default",
        "scripts",
        "script",
        "src",
        "run",
        "main",
        "experiments",
        "exp",
        "args",
    }
)
# A repository with this many different values from different files is a set of
# configurations; the paper's value is compared only with those it names.
MANY_CONFIGURATIONS = 3


def _experiment_files(
    ctx: RunContext, setting: ExtractedSetting, found: list[Candidate]
) -> set[str]:
    """The repository files that hold the most of the paper's other stated settings: the
    configuration of the experiment the paper describes, when the repository has several."""
    counts: Counter[str] = Counter()
    for other in paper_settings(ctx.paper):
        if other.key == setting.key:
            continue
        for candidate in found:
            if (
                candidate.tier is Tier.CODE
                and candidate.setting.key == other.key
                and _agrees(other, candidate.setting)
            ):
                counts[_path(candidate)] += 1
    if not counts:
        return set()
    best = max(counts.values())
    return {path for path, count in counts.items() if count == best}


def _undecided(
    paper: ExtractedSetting,
    sentence: str,
    candidates: list[Candidate],
    experiment_files: frozenset[str] | set[str] = frozenset(),
) -> str | list[Candidate]:
    """Why no repository value can be compared with the paper's, or the values that can.

    Per-device batch sizes need the device count and accumulation, which the code does not
    state; a metric module defining several metrics does not say which one the paper's table
    used; and many configurations only count when the sentence names one of them."""
    if any(c.setting.context == "per device" for c in candidates):
        candidates = [c for c in candidates if c.setting.context != "per device"]
        if not candidates:
            return (
                f"the repository gives {paper.key} only per device; the global value needs the "
                "device count and gradient accumulation, which the code does not state"
            )
    distinct = {str(c.setting.value) for c in candidates}
    if paper.key == "eval_metric" and len(distinct) > 1:
        return (
            "the repository computes several metrics ("
            + ", ".join(sorted(distinct))
            + "), and the code does not say which one the paper reports"
        )
    files = {_path(c) for c in candidates}
    if len(distinct) >= MANY_CONFIGURATIONS and len(files) > 1:
        words = set(_PATH_TOKEN.findall(sentence.lower()))
        named = [
            c
            for c in candidates
            if _path(c) in experiment_files
            or (set(_PATH_TOKEN.findall(_path(c).lower())) - _GENERIC_PATH_TOKENS) & words
        ]
        if not named:
            return (
                f"the repository holds {len(distinct)} values of {paper.key} in "
                f"{len(files)} files, configurations for different experiments, and the sentence "
                "names none of them"
            )
        return named
    return candidates


def _path(candidate: Candidate) -> str:
    source = candidate.setting.source
    assert isinstance(source, RepositorySource)
    return source.path


def _agrees(paper: ExtractedSetting, repo: ExtractedSetting) -> bool:
    if paper.key == "eval_metric":
        wanted, offered = str(paper.value), str(repo.value)
        # Generic and specific names cover each other: "rouge" and "rouge_l", "f1" and
        # "macro_f1".
        return (
            offered == wanted
            or offered in _METRIC_FAMILIES.get(wanted, ())
            or wanted in _METRIC_FAMILIES.get(offered, ())
        )
    if isinstance(paper.value, str) or isinstance(repo.value, str):
        return _text(paper.value) == _text(repo.value)
    if paper.value is None or repo.value is None:
        return False
    ours, theirs = float(paper.value), float(repo.value)
    if math.isclose(ours, theirs, rel_tol=RELATIVE_TOLERANCE):
        return True
    # "10%" in the paper is 0.1 in code.
    return paper.unit == "%" and math.isclose(ours / 100, theirs, rel_tol=RELATIVE_TOLERANCE)


def _text(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _shown(setting: ExtractedSetting) -> str:
    return f"{setting.value}" + (f" ({setting.context})" if setting.context else "")


def _where(candidate: Candidate) -> str:
    source = candidate.setting.source
    assert isinstance(source, RepositorySource)
    return f"{_shown(candidate.setting)} ({source.path}:{source.line})"


def _explanation(setting: ExtractedSetting, candidates: list[Candidate], snapshot_url: str) -> str:
    text = f"Static scan of {snapshot_url}; no value found for {setting.key} matches the paper."
    if setting.context:
        text += (
            f" The paper's value is for {setting.context}; the experiment may use another "
            "configuration."
        )
    if any(c.setting.context == "per device" for c in candidates):
        text += (
            " Per-device values multiply by the number of devices, which the code does not state."
        )
    return text


def values_agree(paper: ExtractedSetting, repo: ExtractedSetting) -> bool:
    """Whether a repository value matches the paper's under the settings check's rules."""
    return _agrees(paper, repo)
