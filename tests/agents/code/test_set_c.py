from pathlib import Path

import pytest

from auditor.agents.code.set_c import (
    EvidenceType,
    InjectionRecord,
    candidates,
    clean_label,
    inject_code,
    inject_paper,
    mutate,
    propose,
    score_code_module,
    validate,
)
from auditor.planner.tables import paper_settings
from auditor.schemas import (
    ErrorClass,
    Evidence,
    EvidenceKind,
    GoldLabel,
    PaperSource,
    PaperVariant,
    ParsedPaper,
    Prediction,
    RepositoryRef,
    SentenceAnchor,
    SettingTarget,
    SourceKind,
    SourceRef,
    SystemName,
)
from auditor.schemas.examples import example_paper
from auditor.testing import offline_context

TEXT = "We train with a learning rate of 3e-4 and a batch size of 256 [1]."


def paper() -> ParsedPaper:
    data = example_paper().model_dump()
    data["sentences"][4]["text"] = TEXT
    data["citations"] = [c for c in data["citations"] if c["sentence_id"] != "s4"]
    data["citations"].append(
        {
            "id": "c-s4",
            "sentence_id": "s4",
            "start": TEXT.index("[1]"),
            "end": TEXT.index("[1]") + 3,
            "marker": "[1]",
            "bib_ids": [data["bibliography"][0]["id"]],
            "link_method": "parser",
        }
    )
    return ParsedPaper.model_validate(data)


def lr_anchor(p: ParsedPaper) -> SentenceAnchor:
    (setting,) = [s for s in paper_settings(p) if s.key == "learning_rate"]
    assert isinstance(setting.source, PaperSource)
    return setting.source.anchor


def test_paper_injection_replaces_the_value_and_keeps_markers_aligned() -> None:
    original = paper()
    injected, label, record = inject_paper(
        original,
        lr_anchor(original),
        "1e-3",
        key="learning_rate",
        rationale="code uses 3e-4 (train.py:2); the paper now says 1e-3",
        evidence_type=EvidenceType.CODE,
        evidence=[_code_evidence()],
    )
    assert injected.variant is PaperVariant.INJECTED
    assert injected.sentence("s4").text == TEXT.replace("3e-4", "1e-3")
    (marker,) = injected.mentions_in("s4")
    assert injected.sentence("s4").text[marker.start : marker.end] == "[1]"
    assert label.error_class == ErrorClass.SETTING_MISMATCH
    assert label.target == SettingTarget(key="learning_rate")
    assert isinstance(label.anchor, SentenceAnchor)
    assert injected.sentence("s4").text[label.anchor.start : label.anchor.end] == "1e-3"
    assert record.side == "paper"
    assert (record.original, record.injected) == ("3e-4", "1e-3")
    # The planner finds the injected value at the label's anchor.
    assert lr_anchor(injected) == label.anchor


def test_injection_that_changes_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="same"):
        inject_paper(
            paper(),
            lr_anchor(paper()),
            "3e-4",
            key="learning_rate",
            rationale="r",
            evidence_type=EvidenceType.CODE,
            evidence=[_code_evidence()],
        )


def test_code_injection_patches_a_copy(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    (source / ".git").mkdir(parents=True)
    (source / "train.py").write_text("lr = 3e-4\nbs = 256\n", encoding="utf-8")
    patched = inject_code(source, tmp_path / "patched", "train.py", 1, "3e-4", "1e-3")
    assert (patched / "train.py").read_text(encoding="utf-8") == "lr = 1e-3\nbs = 256\n"
    assert (source / "train.py").read_text(encoding="utf-8") == "lr = 3e-4\nbs = 256\n"
    assert not (patched / ".git").exists()
    with pytest.raises(ValueError, match="line 2"):
        inject_code(source, tmp_path / "again", "train.py", 2, "3e-4", "1e-3")


def _code_evidence() -> Evidence:
    return Evidence(
        kind=EvidenceKind.CODE,
        source=SourceRef(kind=SourceKind.REPOSITORY, id="https://github.com/o/r@abc"),
        text="lr = 3e-4",
        path="train.py",
        line_start=1,
    )


async def test_candidates_list_paper_settings_with_repository_values(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("lr = 3e-4\nbatch_size = 128\n", encoding="utf-8")
    p = paper()
    rows = await candidates(
        p, RepositoryRef(url=repo.resolve().as_uri()), offline_context(tmp_path, paper=p)
    )
    by_key = {row.key: row for row in rows}
    assert by_key["learning_rate"].outcome == "checked"
    assert by_key["learning_rate"].agrees is True
    assert by_key["learning_rate"].evidence_types == (EvidenceType.CODE,)
    assert by_key["batch_size"].agrees is False
    assert "train.py:2" in by_key["batch_size"].repository_values[0]


def label(
    label_id: str, error_class: ErrorClass | str, key: str, variant: PaperVariant
) -> GoldLabel:
    return GoldLabel(
        label_id=label_id,
        paper_id="example-paper",
        variant=variant,
        error_class=error_class,  # type: ignore[arg-type]
        anchor=SentenceAnchor(sentence_id="s4"),
        target=SettingTarget(key=key),
        rationale="r",
        evidence=(_code_evidence(),),
        reviewed_by=("anurag",),
    )


def record(label_id: str, key: str, evidence_type: EvidenceType) -> InjectionRecord:
    return InjectionRecord(
        label_id=label_id,
        paper_id="example-paper",
        key=key,
        side="paper",
        evidence_type=evidence_type,
        original="3e-4",
        injected="1e-3",
    )


def test_validation_requires_data_mode_for_data_file_labels() -> None:
    labels = [label("l1", ErrorClass.DATA_LEAKAGE, "test_overlap", PaperVariant.INJECTED)]
    records = [record("l1", "test_overlap", EvidenceType.DATA_ANALYSIS)]
    problems = validate(labels, records, data_mode=False)
    assert any("--analyse-data" in p for p in problems)
    assert validate(labels, records, data_mode=True) == []


def test_validation_finds_missing_records_and_wrong_classes() -> None:
    labels = [
        label("l1", ErrorClass.METRIC_MISMATCH, "learning_rate", PaperVariant.INJECTED),
        label("l2", ErrorClass.SETTING_MISMATCH, "batch_size", PaperVariant.INJECTED),
    ]
    problems = validate(labels, [record("l1", "learning_rate", EvidenceType.CODE)], data_mode=False)
    assert any("l1" in p and "setting_mismatch" in p for p in problems)
    assert any("l2" in p and "no injection record" in p for p in problems)


def test_validation_wants_a_reviewed_sample() -> None:
    unreviewed = label("l1", ErrorClass.SETTING_MISMATCH, "learning_rate", PaperVariant.INJECTED)
    unreviewed = unreviewed.model_copy(update={"reviewed_by": ()})
    problems = validate(
        [unreviewed], [record("l1", "learning_rate", EvidenceType.CODE)], data_mode=False
    )
    assert any("20%" in p for p in problems)


def test_clean_label_has_no_error() -> None:
    control = clean_label(paper(), lr_anchor(paper()), "learning_rate", [_code_evidence()], "ok")
    assert control.error_class == "none"
    assert control.variant is PaperVariant.ORIGINAL


def prediction(error_class: ErrorClass, key: str, variant: PaperVariant) -> Prediction:
    return Prediction(
        run_id="r",
        system=SystemName.B1,
        paper_id="example-paper",
        variant=variant,
        error_class=error_class,
        anchor=SentenceAnchor(sentence_id="s4"),
        target=SettingTarget(key=key),
        summary="s",
    )


def test_code_module_scores() -> None:
    labels = [
        label("hit", ErrorClass.SETTING_MISMATCH, "learning_rate", PaperVariant.INJECTED),
        label("miss", ErrorClass.SETTING_MISMATCH, "batch_size", PaperVariant.INJECTED),
        label("clean", "none", "epochs", PaperVariant.ORIGINAL),
    ]
    predictions = [
        prediction(ErrorClass.SETTING_MISMATCH, "learning_rate", PaperVariant.INJECTED),
        prediction(ErrorClass.SETTING_MISMATCH, "epochs", PaperVariant.ORIGINAL),
        prediction(ErrorClass.SETTING_MISMATCH, "seed", PaperVariant.INJECTED),
    ]
    scores = score_code_module(labels, predictions)
    setting = scores.per_class[ErrorClass.SETTING_MISMATCH]
    assert (setting.true_positives, setting.false_positives, setting.false_negatives) == (1, 1, 1)
    assert setting.precision == pytest.approx(0.5)
    assert setting.recall == pytest.approx(0.5)
    assert scores.false_positive_rate == pytest.approx(1.0)
    assert scores.unlabelled == 1
    assert scores.macro_f1 == pytest.approx(0.5)


# --- proposing instances ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "raw", "expected"),
    [
        ("learning_rate", "3e-4", "3e-3"),
        ("learning_rate", "0.001", "0.01"),
        ("learning_rate", "1 x 10^-4", None),
        ("weight_decay", "0.01", "0.1"),
        ("batch_size", "256", "512"),
        ("epochs", "3", "6"),
        ("seed", "42", "43"),
        ("dropout", "0.1", "0.3"),
        ("dropout", "10%", "30%"),
        ("data_split", "80/10/10", "70/15/15"),
        ("data_split", "80/20", "70/30"),
        ("data_split", "8:1:1", None),
        ("test_fraction", "20%", "30%"),
        ("train_examples", "25,000", "50,000"),
        ("test_examples", "10k", "20k"),
        ("num_classes", "10", "11"),
        ("num_classes", "binary", None),
        ("eval_metric", "macro-F1", "accuracy"),
        ("eval_metric", "accuracy", "macro-F1"),
        ("optimizer", "AdamW", "SGD"),
        ("optimizer", "SGD", "Adam"),
        ("dataset", "SQuAD", None),
    ],
)
def test_mutations_keep_the_paper_style(key: str, raw: str, expected: str | None) -> None:
    assert mutate(key, raw) == expected


async def test_propose_injects_agreeing_settings_and_keeps_controls(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("lr = 3e-4\nbatch_size = 256\nseed = 42\n", encoding="utf-8")
    text = "We train with a learning rate of 3e-4, a batch size of 256 and random seed 42."
    data = example_paper().model_dump()
    data["sentences"][4]["text"] = text
    p = ParsedPaper.model_validate(data)
    ref = RepositoryRef(url=repo.resolve().as_uri())
    rows = await candidates(p, ref, offline_context(tmp_path, paper=p))
    proposal = propose(p, rows, errors=2, controls=1, seed=7)
    assert len(proposal.labels) == 2 + 2  # two errors, one control in both variants
    errors = [label for label in proposal.labels if label.is_error]
    assert {label.variant for label in errors} == {PaperVariant.INJECTED}
    assert len({label.target for label in errors}) == 2
    controls = [label for label in proposal.labels if not label.is_error]
    assert {label.variant for label in controls} == {PaperVariant.ORIGINAL, PaperVariant.INJECTED}
    assert all(label.evidence[0].kind is EvidenceKind.CODE for label in proposal.labels)
    # Every injected value is found again by the planner, at the label's anchor.
    injected_settings = paper_settings(proposal.injected)
    for label in errors:
        assert any(
            s.key == label.target.key  # type: ignore[union-attr]
            and isinstance(s.source, PaperSource)
            and s.source.anchor == label.anchor
            for s in injected_settings
        )
    assert {r.label_id for r in proposal.records} == {label.label_id for label in errors}
    # The same seed gives the same proposal.
    again = propose(p, rows, errors=2, controls=1, seed=7)
    assert again.labels == proposal.labels


async def test_controls_after_a_shorter_injection_move_with_the_text(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "train.py").write_text("lr = 0.001\nbatch_size = 256\n", encoding="utf-8")
    text = "We use a learning rate of 0.001 and a batch size of 256."
    data = example_paper().model_dump()
    data["sentences"][4]["text"] = text
    p = ParsedPaper.model_validate(data)
    rows = await candidates(
        p, RepositoryRef(url=repo.resolve().as_uri()), offline_context(tmp_path, paper=p)
    )
    lr_first = [r for r in rows if r.key == "learning_rate"] + [
        r for r in rows if r.key != "learning_rate"
    ]
    proposal = propose(p, lr_first, errors=1, controls=1, seed=0)
    labels = {(label.variant, label.error_class): label for label in proposal.labels}
    if (PaperVariant.INJECTED, "none") not in labels:
        pytest.skip("this seed injected the batch size")
    control = labels[(PaperVariant.INJECTED, "none")]
    assert isinstance(control.anchor, SentenceAnchor)
    sentence = proposal.injected.sentence("s4").text
    assert sentence[control.anchor.start : control.anchor.end] in {"256", "0.001"}
    ids = [label.label_id for label in proposal.labels]
    assert len(ids) == len(set(ids))
