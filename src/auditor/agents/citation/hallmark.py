"""HALLMARK (set A, P2.7): the citation-hallucination benchmark as a sanity check.

HALLMARK (Kim, Brendel and Reizinger, 2026; arXiv 2607.18360) is a benchmark of BibTeX
entries labelled VALID or HALLUCINATED across 14 hallucination types, released at
https://github.com/rpatrik96/hallmark (data/v1.2, MIT licence). Our use is the sanity
check of the citation lookup (decision D15): the reference check should resolve the valid
entries and flag the hallucinated ones. Each entry is turned into a one-entry paper and
run through the same reference check the agent uses, so the benchmark measures the
existence and metadata checks end to end.
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from auditor.infra.context import RunContext
from auditor.schemas import BibEntry, ErrorClass, PaperMetadata, ParsedPaper, ParserInfo

# The benchmark buries canary entries in its splits to detect training-data contamination;
# they are not benchmark items and are dropped on load, as the original loader does.
CANARY_PREFIX = "__canary__"

# The data release this loader reads.
DATA_URL = "https://raw.githubusercontent.com/rpatrik96/hallmark/main/data/v1.2/{split}.jsonl"
SPLITS = ("dev_public", "test_public", "stress_test")


class HallmarkEntry(BaseModel):
    """One record of a HALLMARK split file (JSON Lines)."""

    model_config = ConfigDict(frozen=True, extra="allow")

    bibtex_key: str
    bibtex_type: str = "misc"
    fields: dict[str, JsonValue] = Field(default_factory=dict)
    label: Literal["VALID", "HALLUCINATED"]
    hallucination_type: str | None = None
    difficulty_tier: int | None = None
    explanation: str = ""
    generation_method: str = ""
    source_conference: str | None = None
    subtests: dict[str, bool | None] = Field(default_factory=dict)
    raw_bibtex: str | None = None

    @property
    def is_hallucinated(self) -> bool:
        return self.label == "HALLUCINATED"

    def field(self, name: str) -> str | None:
        """One BibTeX field as a string, or None when the entry does not give it."""
        value = self.fields.get(name)
        if value is None or isinstance(value, bool):
            return None
        text = str(value).strip()
        return text or None


def load_split(path: Path) -> list[HallmarkEntry]:
    """The entries of one HALLMARK split file, canaries dropped, duplicate keys an error."""
    entries: list[HallmarkEntry] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entries.append(HallmarkEntry.model_validate_json(line))
    entries = [e for e in entries if not e.bibtex_key.startswith(CANARY_PREFIX)]
    duplicates = [key for key, count in Counter(e.bibtex_key for e in entries).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate bibtex_key(s) in {path}: {duplicates[:10]}")
    return entries


def bib_entry_for(entry: HallmarkEntry) -> BibEntry:
    """The bibliography entry a HALLMARK record describes.

    The `raw` text is rendered from the same fields, so raw and fields never disagree
    (decision D1); the benchmark's own raw_bibtex is used when it is given.
    """
    authors = tuple(
        part.strip() for part in (entry.field("author") or "").split(" and ") if part.strip()
    )
    year: int | None = None
    raw_year = entry.field("year")
    if raw_year and raw_year[:4].isdigit():
        year = int(raw_year[:4])
    raw = entry.raw_bibtex or (
        f"@{entry.bibtex_type}{{{entry.bibtex_key}, "
        + ", ".join(f"{name}={value}" for name, value in sorted(entry.fields.items()))
        + "}"
    )
    return BibEntry(
        id=entry.bibtex_key,
        raw=raw,
        title=entry.field("title"),
        authors=authors,
        year=year,
        venue=entry.field("booktitle") or entry.field("journal"),
        doi=entry.field("doi"),
        url=entry.field("url"),
    )


def paper_for(entry: HallmarkEntry) -> ParsedPaper:
    """A one-entry paper, so the entry can go through the agent's reference check."""
    from auditor.schemas import Sentence, SentenceKind

    bib = bib_entry_for(entry)
    return ParsedPaper(
        paper_id=f"hallmark-{entry.bibtex_key}",
        parser=ParserInfo(name="hallmark", version="1.2"),
        metadata=PaperMetadata(title=bib.title, authors=bib.authors, year=bib.year),
        sentences=(Sentence(id="s0", kind=SentenceKind.BODY, text=bib.raw or bib.title or bib.id),),
        bibliography=(bib,),
    )


class Flag(StrEnum):
    """What the reference check made of one entry."""

    CLEAN = "clean"  # resolved, metadata matches
    METADATA_MISMATCH = "reference_metadata_mismatch"
    NOT_FOUND = "reference_not_found"
    UNVERIFIABLE = "unverifiable"  # the check could not run (no identifiers, no network)


@dataclass(frozen=True)
class ExistenceResult:
    """The outcome of one entry's existence check."""

    bibtex_key: str
    label: str  # VALID | HALLUCINATED
    hallucination_type: str | None
    difficulty_tier: int | None
    flag: Flag
    reason: str | None = None

    @property
    def flagged(self) -> bool:
        return self.flag in (Flag.METADATA_MISMATCH, Flag.NOT_FOUND)

    @property
    def is_hallucinated(self) -> bool:
        return self.label == "HALLUCINATED"


async def check_entry(ctx: RunContext, entry: HallmarkEntry) -> ExistenceResult:
    """Run the agent's reference check on one HALLMARK entry."""
    import dataclasses

    from auditor.agents.citation import check as agent_check
    from auditor.schemas import BibliographyAnchor, BibTarget, TaskKind
    from auditor.schemas.tasks import AuditTask

    paper = paper_for(entry)
    task = AuditTask.create(
        paper_id=paper.paper_id,
        variant=paper.variant,
        kind=TaskKind.REFERENCE,
        anchor=BibliographyAnchor(bib_id=entry.bibtex_key),
        target=BibTarget(bib_id=entry.bibtex_key),
    )
    result = await agent_check(task, dataclasses.replace(ctx, paper=paper))
    if result.outcome.value == "unverifiable":
        flag = Flag.UNVERIFIABLE
    elif any(f.error_class is ErrorClass.REFERENCE_NOT_FOUND for f in result.findings):
        flag = Flag.NOT_FOUND
    elif any(f.error_class is ErrorClass.REFERENCE_METADATA_MISMATCH for f in result.findings):
        flag = Flag.METADATA_MISMATCH
    else:
        flag = Flag.CLEAN
    return ExistenceResult(
        bibtex_key=entry.bibtex_key,
        label=entry.label,
        hallucination_type=entry.hallucination_type,
        difficulty_tier=entry.difficulty_tier,
        flag=flag,
        reason=result.reason,
    )


@dataclass
class TypeScore:
    """Precision, recall and F1 of flagging one hallucination type (or all of them)."""

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
class ExistenceScores:
    """The sanity-check numbers over one run's results (decision D15)."""

    overall: TypeScore = field(default_factory=TypeScore)
    per_type: dict[str, TypeScore] = field(default_factory=dict)
    valid_entries: int = 0
    valid_flagged: int = 0
    unverifiable: int = 0

    @property
    def false_positive_rate(self) -> float:
        return self.valid_flagged / self.valid_entries if self.valid_entries else 0.0

    @property
    def accuracy(self) -> float:
        decided = self.overall.true_positives + self.overall.false_positives
        decided += self.overall.false_negatives + self.valid_entries - self.valid_flagged
        correct = self.overall.true_positives + self.valid_entries - self.valid_flagged
        return correct / decided if decided else 0.0


def score(results: list[ExistenceResult]) -> ExistenceScores:
    """Flagging scores against the benchmark's labels.

    A hallucinated entry the check flags is a true positive; a valid entry it flags is a
    false positive (the benchmark's false-positive rate is what limits deployment, since
    every flag needs human review). Unverifiable outcomes are reported, not scored.
    """
    scores = ExistenceScores()
    for result in results:
        if result.flag is Flag.UNVERIFIABLE:
            scores.unverifiable += 1
            continue
        if result.is_hallucinated:
            type_score = scores.per_type.setdefault(result.hallucination_type or "?", TypeScore())
            if result.flagged:
                scores.overall.true_positives += 1
                type_score.true_positives += 1
            else:
                scores.overall.false_negatives += 1
                type_score.false_negatives += 1
        else:
            scores.valid_entries += 1
            if result.flagged:
                scores.valid_flagged += 1
                scores.overall.false_positives += 1
    return scores


def write_results(results: list[ExistenceResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    {
                        "bibtex_key": result.bibtex_key,
                        "label": result.label,
                        "hallucination_type": result.hallucination_type,
                        "difficulty_tier": result.difficulty_tier,
                        "flag": result.flag.value,
                        "reason": result.reason,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def read_results(path: Path) -> list[ExistenceResult]:
    results = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        results.append(
            ExistenceResult(
                bibtex_key=row["bibtex_key"],
                label=row["label"],
                hallucination_type=row.get("hallucination_type"),
                difficulty_tier=row.get("difficulty_tier"),
                flag=Flag(row["flag"]),
                reason=row.get("reason"),
            )
        )
    return results
