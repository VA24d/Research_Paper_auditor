from typing import Any

import pytest
from pydantic import ValidationError

from auditor.schemas import (
    AuditTask,
    BibliographyAnchor,
    BibTarget,
    DocumentAnchor,
    ExtractedSetting,
    MarkerTarget,
    PaperSource,
    PaperVariant,
    RuleTarget,
    SentenceAnchor,
    SettingTarget,
    TaskKind,
    compute_task_id,
)
from auditor.schemas.examples import REPOSITORY, example_tasks, example_venue


def test_task_ids_are_pinned() -> None:
    # Changing how ids are computed invalidates cached results and gold-label links,
    # so a change here must be deliberate.
    assert [t.task_id for t in example_tasks()] == [
        "reference-716e993b187e7804",
        "citation-ff90ef73025cd80d",
        "citation-a7af1d358c8d24c7",
        "code-35a0d3269a95fa69",
        "template-c90a0151950ba034",
    ]


def test_task_id_depends_on_variant() -> None:
    anchor = BibliographyAnchor(bib_id="b0")
    target = BibTarget(bib_id="b0")
    original = compute_task_id("p1", PaperVariant.ORIGINAL, TaskKind.REFERENCE, anchor, target)
    injected = compute_task_id("p1", PaperVariant.INJECTED, TaskKind.REFERENCE, anchor, target)
    assert original != injected


def test_tampered_task_id_is_rejected() -> None:
    data = example_tasks()[0].model_dump()
    data["target"]["bib_id"] = "b1"
    with pytest.raises(ValidationError, match="task_id should be"):
        AuditTask.model_validate(data)


def test_tasks_round_trip_through_json() -> None:
    for task in example_tasks():
        assert AuditTask.model_validate_json(task.model_dump_json()) == task


def make(**fields: Any) -> AuditTask:
    return AuditTask.create(paper_id="p1", variant=PaperVariant.ORIGINAL, **fields)


def setting(key: str = "learning_rate") -> ExtractedSetting:
    return ExtractedSetting(
        key=key,
        raw_name="lr",
        raw_value="3e-4",
        value=3e-4,
        source=PaperSource(anchor=SentenceAnchor(sentence_id="s4")),
    )


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            {
                "kind": TaskKind.REFERENCE,
                "anchor": SentenceAnchor(sentence_id="s1"),
                "target": BibTarget(bib_id="b0"),
            },
            "BibliographyAnchor",
        ),
        (
            {
                "kind": TaskKind.REFERENCE,
                "anchor": BibliographyAnchor(bib_id="b0"),
                "target": BibTarget(bib_id="b0"),
                "claim": "extra",
            },
            "do not take 'claim'",
        ),
        (
            {
                "kind": TaskKind.CITATION,
                "anchor": SentenceAnchor(sentence_id="s1"),
                "target": BibTarget(bib_id="b0"),
            },
            "need 'claim'",
        ),
        (
            {
                "kind": TaskKind.CITATION,
                "anchor": SentenceAnchor(sentence_id="s1"),
                "target": RuleTarget(rule_id="r1"),
                "claim": "x",
            },
            "BibTarget or MarkerTarget",
        ),
        (
            {
                "kind": TaskKind.CODE,
                "anchor": SentenceAnchor(sentence_id="s4"),
                "target": SettingTarget(key="learning_rate"),
                "setting": setting(),
            },
            "need 'repository'",
        ),
        (
            {
                "kind": TaskKind.CODE,
                "anchor": SentenceAnchor(sentence_id="s4"),
                "target": SettingTarget(key="batch_size"),
                "setting": setting(),
                "repository": REPOSITORY,
            },
            "must match its target",
        ),
        (
            {
                "kind": TaskKind.TEMPLATE,
                "anchor": DocumentAnchor(),
                "target": RuleTarget(rule_id="body-font-size"),
                "rule": example_venue().rule("page-limit"),
            },
            "must match its target",
        ),
    ],
)
def test_kind_specific_rules(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        make(**fields)


def test_a_theory_task_names_the_sections_it_checks() -> None:
    fields: dict[str, Any] = {
        "kind": TaskKind.THEORY,
        "anchor": SentenceAnchor(sentence_id="s4"),
        "target": SettingTarget(key="math"),
    }
    with pytest.raises(ValidationError, match="need 'sections'"):
        make(**fields)
    task = make(**fields, sections=("sec2", "sec3"))
    assert task.sections == ("sec2", "sec3")
    with pytest.raises(ValidationError, match="do not take 'sections'"):
        make(
            kind=TaskKind.REFERENCE,
            anchor=BibliographyAnchor(bib_id="b0"),
            target=BibTarget(bib_id="b0"),
            sections=("sec2",),
        )


def test_citation_task_for_unlinked_marker_is_valid() -> None:
    task = make(
        kind=TaskKind.CITATION,
        anchor=SentenceAnchor(sentence_id="s3"),
        target=MarkerTarget(marker="[7]"),
        claim="Retrieval also improves factual accuracy [7].",
    )
    assert task.kind is TaskKind.CITATION


def test_setting_key_must_be_snake_case() -> None:
    with pytest.raises(ValidationError):
        setting(key="Learning Rate")
    with pytest.raises(ValidationError):
        SettingTarget(key="Learning Rate")


@pytest.mark.parametrize(
    ("raw", "expected_type"),
    [(3e-4, float), (256, int), (True, bool), ("adamw", str)],
)
def test_setting_values_keep_their_type(raw: object, expected_type: type) -> None:
    value = ExtractedSetting.model_validate({**setting().model_dump(), "value": raw}).value
    assert type(value) is expected_type
