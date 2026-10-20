"""Building set B, the claim-source support set (P2.8, decision D18).

Set B is the citation agent's main test: about 600 claim-source pairs from 60 recent arXiv
papers. Errors are injected by editing a copy of the parsed paper (decision D1), the
unchanged original is kept, and every injected error and every sampled clean pair gets a
`GoldLabel` whose evidence holds the gold passage of the cited work's text (needed for
Recall@5). The four citation error classes (D4) are injected:

- `claim_number_mismatch`: a number the claim attributes to the cited work is changed.
- `claim_not_supported`: the claim sentence is rewritten so the cited work no longer
  supports it (the new text is chosen by the caller, at build time).
- `reference_metadata_mismatch`: a bibliography field (title, authors, year, venue) is
  changed; the entry's `raw` text is edited with it, so the two never disagree (D1).
- `reference_not_found`: the entry's title and identifiers are replaced with a fabricated
  work that resolves to nothing.

`validate` runs the D18 checks before the labels are frozen, and `score_citation_module`
scores a run's predictions against the labels.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from auditor.retrieval.passages import Passage, rank_passages
from auditor.schemas import (
    BibliographyAnchor,
    BibTarget,
    ErrorClass,
    Evidence,
    GoldLabel,
    PaperVariant,
    ParsedPaper,
    Prediction,
    SentenceAnchor,
)


def _replace_in_raw(raw: str, original: str, new_value: str) -> str:
    """Replace `original` in the entry's raw text, case-insensitively (a parser's raw text
    often differs from the fields in case), so the two never disagree (decision D1)."""
    pattern = re.compile(re.escape(original), re.IGNORECASE)
    return pattern.sub(new_value.replace("\\", "\\\\"), raw, count=1)


# The share of labels a second person must review before freezing (decision D18).
REVIEW_SHARE = 0.2

CITATION_CLASSES = frozenset(
    {
        ErrorClass.REFERENCE_NOT_FOUND,
        ErrorClass.REFERENCE_METADATA_MISMATCH,
        ErrorClass.CLAIM_NOT_SUPPORTED,
        ErrorClass.CLAIM_NUMBER_MISMATCH,
    }
)


class InjectionKind(StrEnum):
    """Which of the citation error classes an injection plants."""

    CLAIM_NUMBER_MISMATCH = "claim_number_mismatch"
    CLAIM_NOT_SUPPORTED = "claim_not_supported"
    REFERENCE_METADATA_MISMATCH = "reference_metadata_mismatch"
    REFERENCE_NOT_FOUND = "reference_not_found"


CLASS_FOR_KIND: dict[InjectionKind, ErrorClass] = {
    InjectionKind.CLAIM_NUMBER_MISMATCH: ErrorClass.CLAIM_NUMBER_MISMATCH,
    InjectionKind.CLAIM_NOT_SUPPORTED: ErrorClass.CLAIM_NOT_SUPPORTED,
    InjectionKind.REFERENCE_METADATA_MISMATCH: ErrorClass.REFERENCE_METADATA_MISMATCH,
    InjectionKind.REFERENCE_NOT_FOUND: ErrorClass.REFERENCE_NOT_FOUND,
}


class InjectionRecord(BaseModel):
    """One line of the injection log, kept next to the gold labels."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label_id: str
    paper_id: str
    kind: InjectionKind
    bib_id: str
    # A sentence injection's place (None for a bibliography injection).
    sentence_id: str | None = None
    start: int | None = None
    end: int | None = None
    original: str
    injected: str
    # The gold evidence passage: the place in the cited work's text the claim was checked
    # against, with its exact offsets in that text (Recall@5).
    gold_passage: str | None = None
    gold_passage_start: int | None = None
    gold_passage_end: int | None = None
    # Which text the passage came from, so the passage can be re-checked later.
    source_text_sha256: str | None = None


def _label_id(paper_id: str, kind: InjectionKind, bib_id: str, detail: str) -> str:
    digest = sha256(f"{paper_id}|{kind.value}|{bib_id}|{detail}".encode()).hexdigest()
    return f"b-{digest[:12]}"


def gold_passage_for(source_text: str, claim: str) -> Passage:
    """The passage of the cited work's text the claim rests on: the best-ranked passage
    for the original (uninjected) claim. Stored with the label for Recall@5."""
    passages = rank_passages(source_text, claim, top=1)
    if not passages:
        raise ValueError("the source text has no passages to rank")
    return passages[0]


def _sentence_edited_copy(paper: ParsedPaper, anchor: SentenceAnchor, new_text: str) -> ParsedPaper:
    """A copy of the paper with the span `anchor` of one sentence replaced by `new_text`.

    Citation markers that start at or after the edited span move with the text, and the
    copy is marked as injected (decision D1)."""
    if anchor.start is None or anchor.end is None:
        raise ValueError("a sentence injection needs an anchor with a span")
    sentence = paper.sentence(anchor.sentence_id)
    original = sentence.text[anchor.start : anchor.end]
    if original == new_text:
        raise ValueError(f"the new text is the same as the old one ({original})")
    text = sentence.text[: anchor.start] + new_text + sentence.text[anchor.end :]
    delta = len(new_text) - len(original)
    data = paper.model_dump()
    data["variant"] = PaperVariant.INJECTED.value
    for row in data["sentences"]:
        if row["id"] == anchor.sentence_id:
            row["text"] = text
    for mention in data["citations"]:
        if mention["sentence_id"] == anchor.sentence_id and mention["start"] >= anchor.end:
            mention["start"] += delta
            mention["end"] += delta
    return ParsedPaper.model_validate(data)


def inject_number(
    paper: ParsedPaper,
    anchor: SentenceAnchor,
    new_value: str,
    *,
    bib_id: str,
    rationale: str,
    evidence: tuple[Evidence, ...],
    gold_passage: Passage | None = None,
    source_text: str | None = None,
) -> tuple[ParsedPaper, GoldLabel, InjectionRecord]:
    """Change a number the claim attributes to the cited work: the source says something
    else, so the claim is a `claim_number_mismatch`."""
    return _inject_sentence(
        paper,
        anchor,
        new_value,
        kind=InjectionKind.CLAIM_NUMBER_MISMATCH,
        bib_id=bib_id,
        rationale=rationale,
        evidence=evidence,
        gold_passage=gold_passage,
        source_text=source_text,
    )


def inject_claim(
    paper: ParsedPaper,
    anchor: SentenceAnchor,
    new_text: str,
    *,
    bib_id: str,
    rationale: str,
    evidence: tuple[Evidence, ...],
    gold_passage: Passage | None = None,
    source_text: str | None = None,
) -> tuple[ParsedPaper, GoldLabel, InjectionRecord]:
    """Rewrite the claim so the cited work no longer supports it: a
    `claim_not_supported`. The new text is chosen at build time (the machinery does not
    write prose)."""
    return _inject_sentence(
        paper,
        anchor,
        new_text,
        kind=InjectionKind.CLAIM_NOT_SUPPORTED,
        bib_id=bib_id,
        rationale=rationale,
        evidence=evidence,
        gold_passage=gold_passage,
        source_text=source_text,
    )


def _inject_sentence(
    paper: ParsedPaper,
    anchor: SentenceAnchor,
    new_text: str,
    *,
    kind: InjectionKind,
    bib_id: str,
    rationale: str,
    evidence: tuple[Evidence, ...],
    gold_passage: Passage | None,
    source_text: str | None,
) -> tuple[ParsedPaper, GoldLabel, InjectionRecord]:
    sentence = paper.sentence(anchor.sentence_id)
    original = sentence.text[anchor.start : anchor.end]
    if anchor.start is None or anchor.end is None:
        raise ValueError("a sentence injection needs an anchor with a span")
    injected = _sentence_edited_copy(paper, anchor, new_text)
    new_anchor = SentenceAnchor(
        sentence_id=anchor.sentence_id, start=anchor.start, end=anchor.start + len(new_text)
    )
    label_id = _label_id(paper.paper_id, kind, bib_id, f"{anchor.model_dump_json()}|{new_text}")
    label = GoldLabel(
        label_id=label_id,
        paper_id=paper.paper_id,
        variant=PaperVariant.INJECTED,
        error_class=CLASS_FOR_KIND[kind],
        anchor=new_anchor,
        target=BibTarget(bib_id=bib_id),
        rationale=rationale,
        evidence=evidence,
    )
    record = InjectionRecord(
        label_id=label_id,
        paper_id=paper.paper_id,
        kind=kind,
        bib_id=bib_id,
        sentence_id=anchor.sentence_id,
        start=anchor.start,
        end=anchor.end,
        original=original,
        injected=new_text,
        gold_passage=gold_passage.text if gold_passage else None,
        gold_passage_start=gold_passage.char_start if gold_passage else None,
        gold_passage_end=gold_passage.char_end if gold_passage else None,
        source_text_sha256=(
            sha256(source_text.encode("utf-8")).hexdigest() if source_text else None
        ),
    )
    return injected, label, record


# Bibliography fields a metadata injection can change, and how the new value is stored.
_BIB_FIELDS = ("title", "venue", "year", "doi")


def _field_str(entry: object, field_name: str) -> str:
    value = getattr(entry, field_name)
    return "" if value is None else str(value)


def inject_metadata(
    paper: ParsedPaper,
    bib_id: str,
    *,
    field_name: Literal["title", "venue", "year", "doi"],
    new_value: str,
    rationale: str,
    evidence: tuple[Evidence, ...],
) -> tuple[ParsedPaper, GoldLabel, InjectionRecord]:
    """Change one bibliography field so it disagrees with the scholarly indexes: a
    `reference_metadata_mismatch`. The entry's `raw` text is edited with it (D1)."""
    kind = InjectionKind.REFERENCE_METADATA_MISMATCH
    entry = paper.bib_entry(bib_id)
    original = _field_str(entry, field_name)
    if original == new_value:
        raise ValueError(f"the new {field_name} is the same as the old one ({original})")
    if entry.raw is not None and original and original not in entry.raw:
        raise ValueError(
            f"the entry's raw text does not contain the {field_name} '{original}'; "
            "edit the entry by hand instead"
        )
    data = paper.model_dump()
    data["variant"] = PaperVariant.INJECTED.value
    for row in data["bibliography"]:
        if row["id"] != bib_id:
            continue
        row[field_name] = int(new_value) if field_name == "year" else new_value
        if row["raw"] is not None and original:
            row["raw"] = _replace_in_raw(str(row["raw"]), original, new_value)
    injected = ParsedPaper.model_validate(data)
    label_id = _label_id(paper.paper_id, kind, bib_id, f"{field_name}|{new_value}")
    label = GoldLabel(
        label_id=label_id,
        paper_id=paper.paper_id,
        variant=PaperVariant.INJECTED,
        error_class=CLASS_FOR_KIND[kind],
        anchor=BibliographyAnchor(bib_id=bib_id),
        target=BibTarget(bib_id=bib_id),
        rationale=rationale,
        evidence=evidence,
    )
    record = InjectionRecord(
        label_id=label_id,
        paper_id=paper.paper_id,
        kind=kind,
        bib_id=bib_id,
        original=original,
        injected=new_value,
    )
    return injected, label, record


def inject_not_found(
    paper: ParsedPaper,
    bib_id: str,
    *,
    new_title: str,
    rationale: str,
    evidence: tuple[Evidence, ...],
) -> tuple[ParsedPaper, GoldLabel, InjectionRecord]:
    """Turn one entry into a fabricated work: its title is replaced and its identifiers
    are cleared, so no lookup can resolve it (`reference_not_found`). The `raw` text is
    edited with the title (D1)."""
    kind = InjectionKind.REFERENCE_NOT_FOUND
    entry = paper.bib_entry(bib_id)
    original = entry.title or ""
    if not original:
        raise ValueError(f"the entry {bib_id} has no title to replace; edit it by hand instead")
    if entry.raw is not None and original.lower() not in entry.raw.lower():
        raise ValueError(
            f"the entry's raw text does not contain the title '{original}'; "
            "edit the entry by hand instead"
        )
    if new_title == original:
        raise ValueError("the new title is the same as the old one")
    data = paper.model_dump()
    data["variant"] = PaperVariant.INJECTED.value
    for row in data["bibliography"]:
        if row["id"] != bib_id:
            continue
        row["title"] = new_title
        row["doi"] = None
        row["arxiv_id"] = None
        if row["raw"] is not None:
            row["raw"] = _replace_in_raw(str(row["raw"]), original, new_title)
    injected = ParsedPaper.model_validate(data)
    label_id = _label_id(paper.paper_id, kind, bib_id, new_title)
    label = GoldLabel(
        label_id=label_id,
        paper_id=paper.paper_id,
        variant=PaperVariant.INJECTED,
        error_class=CLASS_FOR_KIND[kind],
        anchor=BibliographyAnchor(bib_id=bib_id),
        target=BibTarget(bib_id=bib_id),
        rationale=rationale,
        evidence=evidence,
    )
    record = InjectionRecord(
        label_id=label_id,
        paper_id=paper.paper_id,
        kind=kind,
        bib_id=bib_id,
        original=original,
        injected=new_title,
    )
    return injected, label, record


def clean_label(
    paper: ParsedPaper,
    anchor: SentenceAnchor,
    bib_id: str,
    evidence: tuple[Evidence, ...],
    rationale: str,
) -> GoldLabel:
    """A clean control: the claim is one the cited work genuinely supports."""
    label_id = _label_id(
        paper.paper_id,
        InjectionKind.CLAIM_NOT_SUPPORTED,
        bib_id,
        f"none|{anchor.model_dump_json()}",
    )
    return GoldLabel(
        label_id=label_id,
        paper_id=paper.paper_id,
        variant=paper.variant,
        error_class="none",
        anchor=anchor,
        target=BibTarget(bib_id=bib_id),
        rationale=rationale,
        evidence=evidence,
    )


def validate(
    labels: list[GoldLabel],
    records: list[InjectionRecord],
    *,
    papers: dict[str, ParsedPaper] | None = None,
    sources: dict[str, str] | None = None,
) -> list[str]:
    """Problems that stop set B from being frozen (decision D18).

    `papers` maps a paper id to the injected copy the labels were built from; `sources`
    maps a paper id to the cited work's text a label's gold passage came from. Both are
    optional; the checks that need them run only when they are given.
    """
    problems: list[str] = []
    by_id = {record.label_id: record for record in records}
    ids = Counter(label.label_id for label in labels)
    problems += [f"{i}: duplicate label id" for i, count in ids.items() if count > 1]
    for label in labels:
        if not isinstance(label.target, BibTarget):
            problems.append(f"{label.label_id}: set B labels target a bibliography entry")
            continue
        if label.is_error:
            record = by_id.get(label.label_id)
            if record is None:
                problems.append(f"{label.label_id}: no injection record")
            elif CLASS_FOR_KIND[record.kind] != label.error_class:
                problems.append(
                    f"{label.label_id}: the record plants {record.kind.value} but the label "
                    f"says {label.error_class}"
                )
            elif (
                record.gold_passage
                and sources
                and record.gold_passage not in sources.get(label.paper_id, "")
            ):
                problems.append(
                    f"{label.label_id}: the gold passage is not in the recorded source text"
                )
        elif label.label_id in by_id:
            problems.append(f"{label.label_id}: a clean control has an injection record")
        elif isinstance(label.anchor, SentenceAnchor) and any(
            record.paper_id == label.paper_id and record.sentence_id == label.anchor.sentence_id
            for record in records
        ):
            problems.append(
                f"{label.label_id}: a clean control sits on a sentence an injection edits"
            )
    for record in records:
        if record.label_id not in ids:
            problems.append(f"{record.label_id}: injection record without a label")
    if papers is not None:
        problems += _validate_against_papers(labels, by_id, papers)
    reviewed = sum(1 for label in labels if label.reviewed_by)
    if labels and reviewed < REVIEW_SHARE * len(labels):
        problems.append(
            f"only {reviewed} of {len(labels)} labels have a second review; D18 asks for 20%"
        )
    return problems


def _validate_against_papers(
    labels: list[GoldLabel],
    records: dict[str, InjectionRecord],
    papers: dict[str, ParsedPaper],
) -> list[str]:
    problems: list[str] = []
    for label in labels:
        paper = papers.get(label.paper_id)
        if paper is None:
            problems.append(f"{label.label_id}: no injected paper {label.paper_id}")
            continue
        if paper.variant is not PaperVariant.INJECTED:
            problems.append(f"{label.label_id}: {label.paper_id} is not marked as injected")
        if not isinstance(label.target, BibTarget):
            continue
        entry_ids = {entry.id for entry in paper.bibliography}
        if label.target.bib_id not in entry_ids:
            problems.append(f"{label.label_id}: entry {label.target.bib_id} is not in the paper")
        record = records.get(label.label_id)
        if record is None or record.sentence_id is None:
            continue
        sentence = next((s for s in paper.sentences if s.id == record.sentence_id), None)
        if sentence is None:
            problems.append(f"{label.label_id}: sentence {record.sentence_id} is not in the paper")
            continue
        if record.injected not in sentence.text:
            problems.append(
                f"{label.label_id}: the injected text is not in sentence {record.sentence_id}"
            )
        if record.kind in (
            InjectionKind.REFERENCE_METADATA_MISMATCH,
            InjectionKind.REFERENCE_NOT_FOUND,
        ):
            entry = paper.bib_entry(label.target.bib_id)
            if entry.raw is not None and record.injected not in entry.raw:
                problems.append(
                    f"{label.label_id}: the entry's raw text was not edited with the field (D1)"
                )
    return problems


@dataclass
class ClassScore:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        found = self.true_positives + self.false_positives
        return self.true_positives / found if found else 0.0

    @property
    def recall(self) -> float:
        wanted = self.true_positives + self.false_negatives
        return self.true_positives / wanted if wanted else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


@dataclass
class CitationScores:
    per_class: dict[ErrorClass, ClassScore] = field(default_factory=dict)
    false_positive_rate: float = 0.0
    unlabelled: int = 0

    @property
    def macro_f1(self) -> float:
        """Over the classes the labels contain (decision D4)."""
        scores = [s for s in self.per_class.values() if s.true_positives + s.false_negatives]
        return sum(s.f1 for s in scores) / len(scores) if scores else 0.0


def score_citation_module(
    labels: list[GoldLabel],
    predictions: list[Prediction],
    *,
    before_verifier: bool = False,
) -> CitationScores:
    """Scores of the citation classes (decision D3 matching, one to one; only positive
    predictions count). A positive prediction matching a clean control is a false
    positive; one matching no label is unlabelled, not a false positive."""
    from auditor.schemas.evaluation import matches
    from auditor.schemas.findings import VerificationState

    scores = CitationScores()
    in_scope = [
        label for label in labels if not label.is_error or label.error_class in CITATION_CLASSES
    ]
    for label in in_scope:
        if label.is_error:
            scores.per_class.setdefault(ErrorClass(label.error_class), ClassScore())
    unmatched = list(in_scope)
    clean_hit = 0

    def positive(prediction: Prediction) -> bool:
        if before_verifier:
            return prediction.state is not VerificationState.DROPPED
        return prediction.is_positive

    for prediction in (p for p in predictions if positive(p) and p.error_class in CITATION_CLASSES):
        matched = next((lbl for lbl in unmatched if matches(prediction, lbl)), None)
        if matched is None:
            scores.unlabelled += 1
            continue
        score = scores.per_class.setdefault(prediction.error_class, ClassScore())
        unmatched.remove(matched)
        if not matched.is_error:
            clean_hit += 1
            score.false_positives += 1
        elif matched.error_class == prediction.error_class:
            score.true_positives += 1
        else:
            score.false_positives += 1
            scores.per_class[ErrorClass(matched.error_class)].false_negatives += 1
    for label in unmatched:
        if label.is_error:
            scores.per_class[ErrorClass(label.error_class)].false_negatives += 1
    clean = sum(1 for label in in_scope if not label.is_error)
    scores.false_positive_rate = clean_hit / clean if clean else 0.0
    return scores


def read_jsonl[T: BaseModel](path: Path, schema: type[T]) -> list[T]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [schema.model_validate_json(line) for line in lines if line.strip()]


def append_jsonl(path: Path, *models: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for model in models:
            handle.write(model.model_dump_json() + "\n")


def write_paper(injected: ParsedPaper, out_dir: Path) -> Path:
    path = out_dir / "papers" / f"{injected.paper_id}-{injected.variant.value}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(injected.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    return path
