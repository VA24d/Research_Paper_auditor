"""Offline tests for set B's injection machinery, labels, validation and scoring (P2.8)."""

from pathlib import Path

import pytest

from auditor.agents.citation.set_b import (
    ClassScore,
    InjectionKind,
    InjectionRecord,
    append_jsonl,
    clean_label,
    gold_passage_for,
    inject_claim,
    inject_metadata,
    inject_not_found,
    inject_number,
    read_jsonl,
    score_citation_module,
    validate,
    write_paper,
)
from auditor.schemas import (
    BibTarget,
    ErrorClass,
    Evidence,
    EvidenceKind,
    GoldLabel,
    PaperVariant,
    ParsedPaper,
    Prediction,
    SentenceAnchor,
    SourceKind,
    SourceRef,
    SystemName,
)
from auditor.schemas.examples import example_paper

SOURCE_TEXT = (
    "Prior work reports 92.1% exact match on the SQuAD benchmark. " * 40
    + "The dataset holds 100,000 questions. " * 40
)
GOLD_PASSAGE = gold_passage_for(SOURCE_TEXT, "Prior work reports 92.1% exact match on SQuAD")

EVIDENCE = (
    Evidence(
        kind=EvidenceKind.QUOTE,
        source=SourceRef(kind=SourceKind.CITED_WORK, id="arxiv:1606.05250"),
        text="Prior work reports 92.1% exact match on the SQuAD benchmark.",
        char_start=0,
        char_end=59,
    ),
)


def paper() -> ParsedPaper:
    return example_paper()


def span(p: ParsedPaper, sentence_id: str, needle: str) -> SentenceAnchor:
    text = p.sentence(sentence_id).text
    start = text.index(needle)
    return SentenceAnchor(sentence_id=sentence_id, start=start, end=start + len(needle))


def test_inject_number_edits_the_sentence_and_shifts_the_markers() -> None:
    p = paper()
    anchor = span(p, "s2", "92.1%")
    injected, label, record = inject_number(
        p,
        anchor,
        "88.4%",
        bib_id="b1",
        rationale="the cited work reports 88.4, not 92.1",
        evidence=EVIDENCE,
        gold_passage=GOLD_PASSAGE,
        source_text=SOURCE_TEXT,
    )
    assert injected.variant is PaperVariant.INJECTED
    assert "88.4%" in injected.sentence("s2").text
    assert "92.1%" not in injected.sentence("s2").text
    # The citation marker after the number moved with the text.
    mention = next(c for c in injected.citations if c.id == "c1")
    assert injected.sentence("s2").text[mention.start : mention.end] == "[2, 3]"
    assert label.error_class is ErrorClass.CLAIM_NUMBER_MISMATCH
    assert label.variant is PaperVariant.INJECTED
    assert isinstance(label.target, BibTarget)
    assert label.target.bib_id == "b1"
    assert isinstance(label.anchor, SentenceAnchor)
    assert injected.sentence("s2").text[label.anchor.start : label.anchor.end] == "88.4%"
    assert record.original == "92.1%"
    assert record.injected == "88.4%"
    assert record.gold_passage == GOLD_PASSAGE.text
    assert record.gold_passage_start == GOLD_PASSAGE.char_start


def test_inject_number_rejects_the_same_value() -> None:
    p = paper()
    anchor = span(p, "s2", "92.1%")
    with pytest.raises(ValueError, match="same as the old one"):
        inject_number(p, anchor, "92.1%", bib_id="b1", rationale="r", evidence=EVIDENCE)


def test_inject_claim_rewrites_the_claim_sentence() -> None:
    p = paper()
    anchor = span(p, "s3", "improves factual accuracy")
    injected, label, _record = inject_claim(
        p,
        anchor,
        "fails to improve factual accuracy",
        bib_id="b1",
        rationale="the cited work shows the opposite direction",
        evidence=EVIDENCE,
        gold_passage=GOLD_PASSAGE,
        source_text=SOURCE_TEXT,
    )
    assert injected.variant is PaperVariant.INJECTED
    assert "fails to improve factual accuracy" in injected.sentence("s3").text
    assert label.error_class is ErrorClass.CLAIM_NOT_SUPPORTED
    mention = next(c for c in injected.citations if c.id == "c2")
    assert injected.sentence("s3").text[mention.start : mention.end] == "[7]"


def test_inject_claim_relocates_later_markers_when_the_rewrite_changes_the_length() -> None:
    """A whole-sentence rewrite that changes the length BEFORE a later in-sentence
    marker must still leave every marker's span on its own marker text (the session-8
    set B growth hit exactly this: reduces->increases with a [Sohn, 2016] later in the
    sentence)."""
    p = paper()
    sentence = p.sentence("s3")
    # both markers of s3 sit before AND after the phrase; rewrite the whole sentence
    anchor = SentenceAnchor(sentence_id="s3", start=0, end=len(sentence.text))
    swapped = sentence.text.replace("improves factual accuracy", "harms factual accuracy")
    assert len(swapped) != len(sentence.text)
    injected, _label, _record = inject_claim(
        p,
        anchor,
        swapped,
        bib_id="b1",
        rationale="the claim's direction was reversed",
        evidence=EVIDENCE,
        gold_passage=GOLD_PASSAGE,
        source_text=SOURCE_TEXT,
    )
    new_sentence = injected.sentence("s3")
    assert new_sentence.text == swapped
    for mention in injected.citations:
        if mention.sentence_id == "s3":
            assert new_sentence.text[mention.start : mention.end] == mention.marker


def test_inject_claim_fails_when_the_marker_is_gone() -> None:
    p = paper()
    sentence = p.sentence("s3")
    anchor = SentenceAnchor(sentence_id="s3", start=0, end=len(sentence.text))
    with pytest.raises(ValueError, match="is not in the edited sentence"):
        inject_claim(
            p,
            anchor,
            "a rewrite that drops every citation marker entirely",
            bib_id="b1",
            rationale="r",
            evidence=EVIDENCE,
            gold_passage=GOLD_PASSAGE,
            source_text=SOURCE_TEXT,
        )


def test_inject_metadata_edits_the_field_and_the_raw_text() -> None:
    p = paper()
    injected, label, record = inject_metadata(
        p,
        "b0",
        field_name="year",
        new_value="2019",
        rationale="the entry's year is wrong",
        evidence=EVIDENCE,
    )
    entry = injected.bib_entry("b0")
    assert entry.year == 2019
    # D1: the raw text was edited with the field, so the two never disagree.
    assert entry.raw is not None
    assert "2019" in entry.raw
    assert "2017" not in entry.raw
    assert label.error_class is ErrorClass.REFERENCE_METADATA_MISMATCH
    assert record.original == "2017"
    assert record.injected == "2019"


def test_inject_metadata_edits_only_the_field_when_the_entry_has_no_raw() -> None:
    p = paper()  # b1 carries no raw text, so there is nothing to disagree with
    injected, _, record = inject_metadata(
        p, "b1", field_name="year", new_value="2019", rationale="r", evidence=EVIDENCE
    )
    entry = injected.bib_entry("b1")
    assert entry.year == 2019
    assert entry.raw is None
    assert record.original == "2016"


def test_inject_metadata_rejects_a_field_the_raw_text_lacks() -> None:
    p = paper()
    data = p.model_dump()
    data["bibliography"][0]["raw"] = "Vaswani 2017."  # no title in the raw text
    stripped = ParsedPaper.model_validate(data)
    with pytest.raises(ValueError, match="raw text does not contain"):
        inject_metadata(
            stripped,
            "b0",
            field_name="title",
            new_value="A Different Title",
            rationale="r",
            evidence=EVIDENCE,
        )


def test_inject_not_found_clears_the_identifiers() -> None:
    p = paper()
    injected, label, record = inject_not_found(
        p,
        "b0",
        new_title="A Fabricated Study of Nothing Real",
        rationale="the title is fabricated",
        evidence=EVIDENCE,
    )
    entry = injected.bib_entry("b0")
    assert entry.title == "A Fabricated Study of Nothing Real"
    assert entry.doi is None
    assert entry.arxiv_id is None
    assert entry.raw is not None
    assert entry.title in entry.raw
    assert label.error_class is ErrorClass.REFERENCE_NOT_FOUND
    assert record.original == "Attention Is All You Need"


def test_clean_label_is_a_none_error_on_the_original_variant() -> None:
    p = paper()
    label = clean_label(p, span(p, "s2", "92.1%"), "b1", EVIDENCE, "the source reports this")
    assert label.error_class == "none"
    assert label.variant is p.variant
    assert label.is_error is False


def test_validate_passes_a_consistent_set() -> None:
    p = paper()
    anchor = span(p, "s2", "92.1%")
    injected, label, record = inject_number(
        p,
        anchor,
        "88.4%",
        bib_id="b1",
        rationale="r",
        evidence=EVIDENCE,
        gold_passage=GOLD_PASSAGE,
        source_text=SOURCE_TEXT,
    )
    label = label.model_copy(update={"reviewed_by": ("second",)})
    problems = validate(
        [label],
        [record],
        papers={label.paper_id: injected},
        sources={(label.paper_id, record.bib_id): SOURCE_TEXT},
    )
    assert problems == []


def test_validate_reports_broken_labels() -> None:
    p = paper()
    anchor = span(p, "s2", "92.1%")
    _, label, record = inject_number(
        p,
        anchor,
        "88.4%",
        bib_id="b1",
        rationale="r",
        evidence=EVIDENCE,
        gold_passage=GOLD_PASSAGE,
    )
    clean = clean_label(p, anchor, "b1", EVIDENCE, "r")
    problems = validate([label, clean, label], [record])
    text = "\n".join(problems)
    assert "duplicate label id" in text
    assert "no injection record" not in text
    assert "a clean control sits on a sentence an injection edits" in text
    assert "second review" in text  # D18's 20% review share

    # A record whose kind disagrees with its label.
    broken = record.model_copy(update={"kind": InjectionKind.CLAIM_NOT_SUPPORTED})
    problems = validate([label], [broken])
    assert any("the record plants" in problem for problem in problems)

    # A gold passage that is not in the recorded source text.
    problems = validate(
        [label], [record], sources={(label.paper_id, record.bib_id): "unrelated text"}
    )
    assert any("gold passage is not in" in problem for problem in problems)


def test_validate_against_papers_checks_the_injected_copy() -> None:
    p = paper()
    anchor = span(p, "s2", "92.1%")
    injected, label, record = inject_number(
        p, anchor, "88.4%", bib_id="b1", rationale="r", evidence=EVIDENCE
    )
    label = label.model_copy(update={"reviewed_by": ("second",)})
    problems = validate([label], [record], papers={label.paper_id: injected})
    assert problems == []

    # A label whose injected text is not in the paper.
    other = p.model_copy(update={"variant": PaperVariant.INJECTED})
    problems = validate([label], [record], papers={label.paper_id: other})
    assert any("injected text is not in sentence" in problem for problem in problems)

    # A label whose paper is missing.
    problems = validate([label], [record], papers={})
    assert any("no injected paper" in problem for problem in problems)


def test_gold_passage_for_needs_passages() -> None:
    with pytest.raises(ValueError, match="no passages"):
        gold_passage_for("", "claim")


def test_swap_direction_reverses_the_first_direction_word() -> None:
    from auditor.agents.citation.set_b import swap_direction

    assert swap_direction("Method A improves over the baseline [1].") == (
        "Method A degrades over the baseline [1]."
    )
    assert swap_direction("Accuracy is higher on SQuAD [2].") == ("Accuracy is lower on SQuAD [2].")
    assert swap_direction("The dataset holds 100,000 questions [3].") is None


def test_perturb_number_moves_the_value_and_keeps_its_shape() -> None:
    import random

    from auditor.agents.citation.set_b import perturb_number

    rng = random.Random(0)
    assert perturb_number("92.1", rng) is not None
    value = perturb_number("92.1", rng)
    assert value is not None
    assert "." in value
    assert value != "92.1"
    assert perturb_number("0", rng) is None
    assert perturb_number("not-a-number", rng) is None


def test_propose_picks_pairs_and_controls_deterministically() -> None:
    import random

    from auditor.agents.citation.set_b import InjectionKind, propose

    data = paper().model_dump()
    data["sentences"][1]["text"] = (
        "Transformers [1] improves over recurrent networks for most sequence tasks."
    )
    p = ParsedPaper.model_validate(data)
    sources = {"b1": SOURCE_TEXT, "b0": "The Transformer improves over recurrence. " * 20}
    rng = random.Random(0)
    proposals, clean = propose(p, sources, pairs=6, controls=3, rng=rng)
    kinds = [proposal.kind for proposal in proposals]
    assert InjectionKind.CLAIM_NUMBER_MISMATCH in kinds
    assert InjectionKind.CLAIM_NOT_SUPPORTED in kinds
    assert InjectionKind.REFERENCE_METADATA_MISMATCH in kinds  # b0/b1 have a year in raw
    assert InjectionKind.REFERENCE_NOT_FOUND in kinds
    assert len(proposals) <= 6
    assert len(clean) <= 3
    # Deterministic: the same seed gives the same pick.
    again, clean_again = propose(p, sources, pairs=6, controls=3, rng=random.Random(0))
    assert [(pr.kind, pr.bib_id, pr.new_value) for pr in proposals] == [
        (pr.kind, pr.bib_id, pr.new_value) for pr in again
    ]
    assert clean == clean_again


def test_propose_only_uses_pairs_with_a_source(tmp_path: Path) -> None:
    import random

    from auditor.agents.citation.set_b import propose

    p = paper()
    proposals, _ = propose(p, {"b1": SOURCE_TEXT}, pairs=6, controls=2, rng=random.Random(0))
    assert all(proposal.bib_id in {"b1", "b0", "b2"} for proposal in proposals)
    # b2 has no source text: never proposed, never a clean pair with a b2 target.
    assert all(
        proposal.bib_id != "b2" or proposal.kind.value.startswith("reference")
        for proposal in proposals
    )


def test_metadata_year_proposals_are_beyond_the_year_tolerance() -> None:
    """One year of drift is normal in real bibliographies; the injected year must be at
    least two years off, or the tolerant checker would never flag it."""
    import random

    from auditor.agents.citation.set_b import InjectionKind, propose

    p = paper()
    years = {entry.id: entry.year for entry in p.bibliography if entry.year}
    proposals, _ = propose(p, {}, pairs=100, controls=0, rng=random.Random(1))
    year_props = [
        proposal
        for proposal in proposals
        if proposal.kind is InjectionKind.REFERENCE_METADATA_MISMATCH
        and proposal.field_name == "year"
    ]
    assert year_props  # the paper's bibliography has years to propose on
    for proposal in year_props:
        assert abs(int(proposal.new_value) - years[proposal.bib_id]) >= 2


def test_score_citation_module_counts_matches_one_to_one() -> None:
    p = paper()
    anchor = span(p, "s2", "92.1%")
    _, number_label, _ = inject_number(
        p, anchor, "88.4%", bib_id="b1", rationale="r", evidence=EVIDENCE
    )
    _, claim_label, _ = inject_claim(
        p,
        span(p, "s3", "improves factual accuracy"),
        "fails to improve",
        bib_id="b1",
        rationale="r",
        evidence=EVIDENCE,
    )
    clean = clean_label(p, span(p, "s1", "replaced recurrent"), "b0", EVIDENCE, "r")
    labels = [number_label, claim_label, clean]

    def prediction(label: GoldLabel, error_class: ErrorClass) -> Prediction:
        return Prediction(
            run_id="r1",
            system=SystemName.B1,
            paper_id=label.paper_id,
            variant=label.variant,
            error_class=error_class,
            anchor=label.anchor,
            target=label.target,
            summary="found it",
        )

    scores = score_citation_module(
        labels,
        [
            prediction(number_label, ErrorClass.CLAIM_NUMBER_MISMATCH),
            prediction(claim_label, ErrorClass.CLAIM_NOT_SUPPORTED),
            prediction(clean, ErrorClass.CLAIM_NUMBER_MISMATCH),  # false positive on clean
        ],
    )
    assert scores.per_class[ErrorClass.CLAIM_NUMBER_MISMATCH] == ClassScore(
        true_positives=1, false_positives=1
    )
    assert scores.per_class[ErrorClass.CLAIM_NOT_SUPPORTED].true_positives == 1
    assert scores.false_positive_rate == pytest.approx(1.0)
    assert scores.unlabelled == 0

    # A prediction matching no label is unlabelled, not a false positive.
    stray = clean_label(p, span(p, "s5", "Code is available"), "b0", EVIDENCE, "r")
    scores = score_citation_module(labels, [prediction(stray, ErrorClass.CLAIM_NOT_SUPPORTED)])
    assert scores.unlabelled == 1
    assert scores.false_positive_rate == 0.0

    # A wrong class is a false positive for the label's class and a miss for its own.
    scores = score_citation_module(
        labels, [prediction(number_label, ErrorClass.CLAIM_NOT_SUPPORTED)]
    )
    assert scores.per_class[ErrorClass.CLAIM_NUMBER_MISMATCH].false_negatives == 1
    assert scores.per_class[ErrorClass.CLAIM_NOT_SUPPORTED].false_positives == 1


def test_read_jsonl_and_write_paper_round_trip(tmp_path: Path) -> None:
    p = paper()
    anchor = span(p, "s2", "92.1%")
    injected, label, record = inject_number(
        p, anchor, "88.4%", bib_id="b1", rationale="r", evidence=EVIDENCE
    )
    labels_path = tmp_path / "labels.jsonl"
    records_path = tmp_path / "injections.jsonl"
    append_jsonl(labels_path, label)
    append_jsonl(records_path, record)
    assert read_jsonl(labels_path, GoldLabel) == [label]
    assert read_jsonl(records_path, InjectionRecord) == [record]

    paper_path = write_paper(injected, tmp_path)
    loaded = ParsedPaper.model_validate_json(paper_path.read_bytes())
    assert loaded == injected


def test_validate_against_papers_checks_bibliography_edits() -> None:
    """A bibliography injection has no sentence; its edit is checked in the entry."""
    p = paper()
    injected, label, record = inject_metadata(
        p, "b0", field_name="year", new_value="2019", rationale="r", evidence=EVIDENCE
    )
    label = label.model_copy(update={"reviewed_by": ("second",)})
    assert record.sentence_id is None
    assert validate([label], [record], papers={label.paper_id: injected}) == []

    # The original paper, marked injected, still says 2017: the edit was never applied.
    unedited = p.model_copy(update={"variant": PaperVariant.INJECTED})
    problems = validate([label], [record], papers={label.paper_id: unedited})
    assert any("injected value is not in the entry" in problem for problem in problems)


def test_score_citation_module_agrees_with_the_shared_evaluation() -> None:
    """A task can flag a wrong class before the right one on the same claim; the label
    goes to the matching class, as `auditor evaluate` assigns it."""
    from auditor.agents.citation.set_b import CITATION_CLASSES
    from auditor.evaluation.metrics import score

    p = paper()
    _, number_label, _ = inject_number(
        p, span(p, "s2", "92.1%"), "88.4%", bib_id="b1", rationale="r", evidence=EVIDENCE
    )

    def prediction(error_class: ErrorClass) -> Prediction:
        return Prediction(
            run_id="r1",
            system=SystemName.B1,
            paper_id=number_label.paper_id,
            variant=number_label.variant,
            error_class=error_class,
            anchor=number_label.anchor,
            target=number_label.target,
            summary="found it",
        )

    for predictions in (
        [prediction(ErrorClass.CLAIM_NOT_SUPPORTED), prediction(ErrorClass.CLAIM_NUMBER_MISMATCH)],
        [prediction(ErrorClass.CLAIM_NUMBER_MISMATCH), prediction(ErrorClass.CLAIM_NOT_SUPPORTED)],
    ):
        ours = score_citation_module([number_label], predictions)
        shared = score([number_label], predictions, classes=CITATION_CLASSES)
        assert ours.per_class[ErrorClass.CLAIM_NUMBER_MISMATCH].true_positives == 1
        assert ours.macro_f1 == pytest.approx(shared.macro_f1)
        assert ours.unlabelled == shared.unlabelled == 1


def test_validate_against_papers_needs_the_structured_field_edited() -> None:
    """An edit that reached the raw text but not the entry's field is not applied."""
    p = paper()
    injected, label, record = inject_metadata(
        p, "b0", field_name="year", new_value="2019", rationale="r", evidence=EVIDENCE
    )
    label = label.model_copy(update={"reviewed_by": ("second",)})
    data = injected.model_dump()
    data["bibliography"][0]["year"] = 2017  # the raw text still says 2019
    half_done = ParsedPaper.model_validate(data)
    problems = validate([label], [record], papers={label.paper_id: half_done})
    assert any("injected value is not in the entry" in problem for problem in problems)


def test_an_injected_value_elsewhere_in_the_entry_does_not_hide_a_missing_edit() -> None:
    """The year edit is missing, though "2019" happens to be inside the DOI."""
    p = paper()
    injected, label, record = inject_metadata(
        p, "b0", field_name="year", new_value="2019", rationale="r", evidence=EVIDENCE
    )
    label = label.model_copy(update={"reviewed_by": ("second",)})
    data = injected.model_dump()
    data["bibliography"][0]["year"] = 2017
    data["bibliography"][0]["doi"] = "10.1234/2019.forecast"
    half_done = ParsedPaper.model_validate(data)
    problems = validate([label], [record], papers={label.paper_id: half_done})
    assert any("injected value is not in the entry" in problem for problem in problems)
