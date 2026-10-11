"""Small, valid examples of every contract, for tests and for building against the schemas.

The JSON files in `tests/fixtures/schemas/` are generated from this module:

    uv run python -m auditor.schemas.examples tests/fixtures/schemas
"""

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from auditor.schemas.anchors import BibliographyAnchor, DocumentAnchor, SentenceAnchor
from auditor.schemas.evaluation import NO_ERROR, GoldLabel, Prediction, RunManifest, SystemName
from auditor.schemas.experiment import ExtractedSetting, PaperSource, RepositoryRef
from auditor.schemas.findings import (
    CandidateFinding,
    ErrorClass,
    Evidence,
    EvidenceKind,
    SourceKind,
    SourceRef,
    TaskOutcome,
    TaskResult,
    VerificationState,
    VerifiedFinding,
)
from auditor.schemas.paper import (
    BibEntry,
    Box,
    CitationMention,
    FontUsage,
    Layout,
    Link,
    LinkMethod,
    LinkSource,
    PageLayout,
    PaperMetadata,
    PaperVariant,
    ParsedPaper,
    ParserInfo,
    Section,
    Sentence,
    SentenceKind,
)
from auditor.schemas.tasks import (
    AuditTask,
    BibTarget,
    MarkerTarget,
    RuleTarget,
    SettingTarget,
    TaskKind,
)
from auditor.schemas.venue import VenueProfile, VenueRule

PAPER_ID = "example-paper"
REPOSITORY = RepositoryRef(url="https://github.com/example/auditor-demo", commit="0123abcd")

_SENTENCES = {
    "s0": "We study how to audit citations and code in research papers.",
    "s1": "Transformers [1] replaced recurrent networks for most sequence tasks.",
    "s2": "Prior work reports 92.1% exact match on SQuAD [2, 3].",
    "s3": "Retrieval also improves factual accuracy [7].",
    "s4": "We train with a learning rate of 3e-4 and a batch size of 256.",
    "s5": "Code is available at https://github.com/example/auditor-demo.",
}


def _mention(mention_id: str, sentence_id: str, marker: str, *bib_ids: str) -> CitationMention:
    start = _SENTENCES[sentence_id].index(marker)
    return CitationMention(
        id=mention_id,
        sentence_id=sentence_id,
        start=start,
        end=start + len(marker),
        marker=marker,
        bib_ids=bib_ids,
        link_method=LinkMethod.PARSER if bib_ids else LinkMethod.NONE,
    )


def _span(sentence_id: str, text: str) -> SentenceAnchor:
    start = _SENTENCES[sentence_id].index(text)
    return SentenceAnchor(sentence_id=sentence_id, start=start, end=start + len(text))


def _box(page: int, line: int) -> Box:
    return Box(page=page, x=72.0, y=90.0 + 12.0 * line, width=468.0, height=10.0)


def example_paper() -> ParsedPaper:
    sentence_meta = {
        "s0": (None, SentenceKind.ABSTRACT, 1, 0),
        "s1": ("sec1", SentenceKind.BODY, 1, 4),
        "s2": ("sec1", SentenceKind.BODY, 1, 5),
        "s3": ("sec1", SentenceKind.BODY, 1, 6),
        "s4": ("sec2", SentenceKind.BODY, 2, 0),
        "s5": ("sec2", SentenceKind.BODY, 2, 1),
    }
    sentences = tuple(
        Sentence(id=sid, section_id=section, kind=kind, text=text, boxes=(_box(page, line),))
        for sid, text in _SENTENCES.items()
        for section, kind, page, line in [sentence_meta[sid]]
    )
    return ParsedPaper(
        paper_id=PAPER_ID,
        parser=ParserInfo(name="grobid", version="0.9.1", pdf_sha256="0" * 64),
        metadata=PaperMetadata(
            title="An Example Paper for Auditing",
            authors=("Ada Example", "Alan Sample"),
            year=2026,
        ),
        sections=(
            Section(id="sec1", title="Introduction", number="1"),
            Section(id="sec2", title="Experiments", number="4"),
        ),
        sentences=sentences,
        bibliography=(
            BibEntry(
                id="b0",
                raw="Vaswani et al. Attention is all you need. NeurIPS 2017.",
                title="Attention Is All You Need",
                authors=("Ashish Vaswani", "Noam Shazeer"),
                year=2017,
                venue="NeurIPS",
                arxiv_id="1706.03762",
            ),
            BibEntry(
                id="b1",
                title="SQuAD: 100,000+ Questions for Machine Comprehension of Text",
                authors=("Pranav Rajpurkar", "Jian Zhang"),
                year=2016,
                venue="EMNLP",
                doi="10.18653/v1/D16-1264",
            ),
            BibEntry(
                id="b2",
                title="Deep Residual Learning for Image Recognition",
                authors=("Kaiming He", "Xiangyu Zhang"),
                year=2016,
                venue="CVPR",
            ),
        ),
        citations=(
            _mention("c0", "s1", "[1]", "b0"),
            _mention("c1", "s2", "[2, 3]", "b1", "b2"),
            _mention("c2", "s3", "[7]"),
        ),
        links=(
            Link(
                url="https://github.com/example/auditor-demo",
                page=2,
                sentence_id="s5",
                source=LinkSource.ANNOTATION,
            ),
        ),
        layout=Layout(
            pages=(
                PageLayout(
                    number=1,
                    width=612.0,
                    height=792.0,
                    text_left=72.0,
                    text_top=72.0,
                    text_right=540.0,
                    text_bottom=720.0,
                ),
                PageLayout(
                    number=2,
                    width=612.0,
                    height=792.0,
                    text_left=72.0,
                    text_top=72.0,
                    text_right=540.0,
                    text_bottom=400.0,
                ),
            ),
            fonts=(FontUsage(name="Times-Roman", size=10.0, chars=2400),),
        ),
    )


def example_venue() -> VenueProfile:
    return VenueProfile(
        id="example-venue-2026",
        name="Example Venue 2026",
        rules=(
            VenueRule(
                id="page-limit",
                description="The main text is at most 9 pages.",
                check="max_pages",
                params={"max_pages": 9},
            ),
            VenueRule(
                id="body-font-size",
                description="Body text is at least 10 pt.",
                check="min_body_font_size",
                params={"min_size": 10.0},
            ),
            VenueRule(
                id="anonymous",
                description="The submission does not identify its authors.",
                check="anonymity",
                needs_visual_inspection=True,
            ),
        ),
    )


def _learning_rate() -> ExtractedSetting:
    return ExtractedSetting(
        key="learning_rate",
        raw_name="learning rate",
        raw_value="3e-4",
        value=3e-4,
        source=PaperSource(anchor=_span("s4", "3e-4")),
    )


def example_tasks() -> list[AuditTask]:
    paper = example_paper()
    venue = example_venue()
    variant = PaperVariant.ORIGINAL
    setting = _learning_rate()
    return [
        AuditTask.create(
            paper_id=PAPER_ID,
            variant=variant,
            kind=TaskKind.REFERENCE,
            anchor=BibliographyAnchor(bib_id="b0"),
            target=BibTarget(bib_id="b0"),
        ),
        AuditTask.create(
            paper_id=PAPER_ID,
            variant=variant,
            kind=TaskKind.CITATION,
            anchor=SentenceAnchor(sentence_id="s2"),
            target=BibTarget(bib_id="b1"),
            claim=paper.sentence("s2").text,
        ),
        AuditTask.create(
            paper_id=PAPER_ID,
            variant=variant,
            kind=TaskKind.CITATION,
            anchor=SentenceAnchor(sentence_id="s3"),
            target=MarkerTarget(marker="[7]"),
            claim=paper.sentence("s3").text,
        ),
        AuditTask.create(
            paper_id=PAPER_ID,
            variant=variant,
            kind=TaskKind.CODE,
            anchor=SentenceAnchor(sentence_id="s4"),
            target=SettingTarget(key=setting.key),
            setting=setting,
            repository=REPOSITORY,
        ),
        AuditTask.create(
            paper_id=PAPER_ID,
            variant=variant,
            kind=TaskKind.TEMPLATE,
            anchor=DocumentAnchor(),
            target=RuleTarget(rule_id="page-limit"),
            rule=venue.rule("page-limit"),
        ),
    ]


def _squad_evidence() -> Evidence:
    return Evidence(
        kind=EvidenceKind.QUOTE,
        source=SourceRef(
            kind=SourceKind.CITED_WORK,
            id="doi:10.18653/v1/D16-1264",
            title="SQuAD: 100,000+ Questions for Machine Comprehension of Text",
        ),
        text="which achieves an F1 score of 51.0%",
        page=1,
    )


def example_finding() -> CandidateFinding:
    task = example_tasks()[1]
    return CandidateFinding(
        finding_id=f"{task.task_id}-f0",
        task_id=task.task_id,
        paper_id=task.paper_id,
        variant=task.variant,
        error_class=ErrorClass.CLAIM_NUMBER_MISMATCH,
        anchor=_span("s2", "92.1%"),
        target=task.target,
        summary="The cited work does not report 92.1% exact match on SQuAD.",
        explanation="The cited paper reports a 51.0% F1 baseline and no 92.1% figure.",
        evidence=(_squad_evidence(),),
    )


def example_task_results() -> list[TaskResult]:
    tasks = example_tasks()
    finding = example_finding()
    return [
        TaskResult(
            task_id=tasks[0].task_id,
            paper_id=PAPER_ID,
            variant=tasks[0].variant,
            kind=tasks[0].kind,
            outcome=TaskOutcome.CHECKED,
        ),
        TaskResult(
            task_id=tasks[1].task_id,
            paper_id=PAPER_ID,
            variant=tasks[1].variant,
            kind=tasks[1].kind,
            outcome=TaskOutcome.CHECKED,
            findings=(finding,),
        ),
        TaskResult(
            task_id=tasks[2].task_id,
            paper_id=PAPER_ID,
            variant=tasks[2].variant,
            kind=tasks[2].kind,
            outcome=TaskOutcome.UNVERIFIABLE,
            reason="Citation marker [7] is not linked to any bibliography entry.",
        ),
        TaskResult(
            task_id=tasks[3].task_id,
            paper_id=PAPER_ID,
            variant=tasks[3].variant,
            kind=tasks[3].kind,
            outcome=TaskOutcome.FAILED,
            error="TimeoutError: repository clone did not finish in 120 s",
            attempts=3,
        ),
    ]


def example_verified_findings() -> list[VerifiedFinding]:
    finding = example_finding()
    return [
        VerifiedFinding(
            finding=finding,
            state=VerificationState.VERIFIED_FLAG,
            reason="The quoted figure is verbatim in the cited source and contradicts the claim.",
            accepted_evidence=finding.evidence,
        ),
        VerifiedFinding(
            finding=finding.model_copy(update={"finding_id": f"{finding.task_id}-f1"}),
            state=VerificationState.NEEDS_HUMAN_REVIEW,
            reason="Only the abstract of the cited work was available.",
        ),
    ]


def example_predictions() -> list[Prediction]:
    finding = example_finding()
    return [
        Prediction(
            run_id="run-example",
            system=SystemName.FULL,
            paper_id=PAPER_ID,
            variant=PaperVariant.ORIGINAL,
            error_class=finding.error_class,
            anchor=finding.anchor,
            target=finding.target,
            state=VerificationState.VERIFIED_FLAG,
            summary=finding.summary,
            evidence=finding.evidence,
            finding_id=finding.finding_id,
        ),
        Prediction(
            run_id="run-example",
            system=SystemName.B0,
            paper_id=PAPER_ID,
            variant=PaperVariant.ORIGINAL,
            error_class=ErrorClass.SETTING_MISMATCH,
            anchor=SentenceAnchor(sentence_id="s4"),
            target=SettingTarget(key="learning_rate"),
            summary="The learning rate may differ from the released code.",
        ),
    ]


def example_gold_labels() -> list[GoldLabel]:
    return [
        GoldLabel(
            label_id="example-paper-l0",
            paper_id=PAPER_ID,
            variant=PaperVariant.ORIGINAL,
            error_class=ErrorClass.CLAIM_NUMBER_MISMATCH,
            anchor=_span("s2", "92.1%"),
            target=BibTarget(bib_id="b1"),
            rationale="The cited work never reports 92.1%; the number was injected.",
            evidence=(_squad_evidence(),),
            reviewed_by=("second-reviewer",),
        ),
        GoldLabel(
            label_id="example-paper-l1",
            paper_id=PAPER_ID,
            variant=PaperVariant.ORIGINAL,
            error_class=NO_ERROR,
            anchor=SentenceAnchor(sentence_id="s1"),
            target=BibTarget(bib_id="b0"),
            rationale="The cited work introduces the Transformer, as the sentence says.",
            evidence=(
                Evidence(
                    kind=EvidenceKind.QUOTE,
                    source=SourceRef(kind=SourceKind.CITED_WORK, id="arxiv:1706.03762"),
                    text="We propose a new simple network architecture, the Transformer",
                    page=1,
                ),
            ),
        ),
    ]


def example_run_manifest() -> RunManifest:
    return RunManifest(
        run_id="run-example",
        system=SystemName.FULL,
        started_at=datetime(2026, 10, 10, 12, 0, tzinfo=UTC),
        finished_at=datetime(2026, 10, 10, 12, 5, tzinfo=UTC),
        git_commit="0" * 40,
        git_dirty=False,
        model="example-model",
        python_version="3.13.3",
        platform="Linux-x86_64",
        packages={"langgraph": "1.2.14"},
        config={"max_concurrency": 8},
    )


def all_examples() -> dict[str, BaseModel | Sequence[BaseModel]]:
    """Every example, keyed by the file name it is written to."""
    return {
        "parsed_paper": example_paper(),
        "venue_profile": example_venue(),
        "audit_tasks": example_tasks(),
        "task_results": example_task_results(),
        "verified_findings": example_verified_findings(),
        "predictions": example_predictions(),
        "gold_labels": example_gold_labels(),
        "run_manifest": example_run_manifest(),
    }


def render(value: BaseModel | Sequence[BaseModel]) -> str:
    if isinstance(value, BaseModel):
        data: object = value.model_dump(mode="json")
    else:
        data = [item.model_dump(mode="json") for item in value]
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def write_examples(directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, value in all_examples().items():
        path = directory / f"{name}.json"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(render(value))
        written.append(path)
    return written


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Write example JSON for every contract.")
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    for path in write_examples(args.directory):
        print(path)


if __name__ == "__main__":
    main()
