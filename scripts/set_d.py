"""Build, run and score set D, the planted integrity and code errors (plan W5).

    uv run python scripts/set_d.py build --set-c data/set_c --out datasets/set_d --data data/set_d
    uv run python scripts/set_d.py run --data data/set_d --set-c data/set_c \
        --out data/set_d/run.jsonl
    uv run python scripts/set_d.py score datasets/set_d/labels.jsonl data/set_d/run.jsonl

`build` plants errors in copies of the set C originals and their repositories
(`auditor.agents.integrity.set_d`): the plant records and gold labels go to `--out` (to
commit), the injected papers, patched repositories and the dataset file to `--data`.
`run` checks every original and injected copy with the integrity screen, the results
check and the code-defect check only, and writes one JSON line per paper and variant
with its findings. `score` reports, per kind of plant, how many were found, and the
findings on the originals (clean controls) and the extra findings on the copies.
"""

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from auditor.agents.code.defects import check_defects
from auditor.agents.integrity import check as check_integrity
from auditor.agents.integrity.set_d import Plant, apply_plants, label_for, plan_paper, plant_code
from auditor.config import Settings
from auditor.infra.context import RunContext
from auditor.infra.http import HttpClient
from auditor.infra.llm import StructuredLLM, closing_llm
from auditor.infra.snapshots import fetch_snapshot
from auditor.parser.layout_tables import add_layout_tables, layout_pages
from auditor.planner.integrity import integrity_tasks
from auditor.planner.tasks import defect_tasks
from auditor.schemas import (
    CandidateFinding,
    DatasetEntry,
    ErrorClass,
    GoldLabel,
    PaperVariant,
    ParsedPaper,
    RepositoryRef,
    SentenceAnchor,
)

PLANTS = "plants.jsonl"
LABELS = "labels.jsonl"
DATASET = "dataset.jsonl"


def _load(path: str | Path) -> ParsedPaper:
    return ParsedPaper.model_validate_json(Path(path).read_bytes())


def _done(out: Path) -> set[tuple[str, str]]:
    """The papers and variants a resumed run already has."""
    if not out.exists():
        return set()
    lines = out.read_text(encoding="utf-8").splitlines()
    return {(r["paper_id"], r["variant"]) for r in map(json.loads, lines)}


def _originals(set_c: Path) -> list[DatasetEntry]:
    entries = [
        DatasetEntry.model_validate_json(line)
        for line in (set_c / DATASET).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [e for e in entries if "/papers/" in e.paper]


async def _build(args: argparse.Namespace) -> int:
    settings = Settings()
    out: Path = args.out
    data: Path = args.data
    for folder in (out, data / "injected", data / "repos"):
        folder.mkdir(parents=True, exist_ok=True)
    plants: list[Plant] = []
    labels: list[GoldLabel] = []
    dataset: list[str] = []
    async with HttpClient(settings) as http:
        for entry in _originals(args.set_c):
            paper = _load(entry.paper)
            # Planted against the tables the checks will see: the layout tables of the PDF
            # when there is one, and the tables read from its boxes, as in `run`.
            pdf = args.set_c / "pdfs" / f"{paper.paper_id}.pdf"
            pages = layout_pages(pdf) if pdf.exists() else None
            paper_plants = plan_paper(
                add_layout_tables(paper, pages) if pages else paper,
                pdf if pdf.exists() else None,
            )
            injected = apply_plants(paper, paper_plants)
            (data / "injected" / f"{paper.paper_id}.json").write_text(
                injected.model_dump_json(), encoding="utf-8"
            )
            repository = entry.repository
            patched = None
            if repository:
                ref = RepositoryRef(url=repository, commit=entry.repository_commit)
                snapshot = await fetch_snapshot(ref, http, settings)
                destination = data / "repos" / paper.paper_id
                code = plant_code(paper.paper_id, snapshot.root, destination)
                if code is not None:
                    paper_plants.append(code)
                    patched = destination
            plants += paper_plants
            labels += [label_for(p, repository) for p in paper_plants]
            dataset.append(entry.model_dump_json())
            dataset.append(
                DatasetEntry(
                    paper=str((data / "injected" / f"{paper.paper_id}.json").resolve()),
                    repository=str(patched.resolve()) if patched else repository,
                    repository_commit=None if patched else entry.repository_commit,
                ).model_dump_json()
            )
    (out / PLANTS).write_text("".join(p.model_dump_json() + "\n" for p in plants), encoding="utf-8")
    (out / LABELS).write_text("".join(g.model_dump_json() + "\n" for g in labels), encoding="utf-8")
    (data / DATASET).write_text("".join(line + "\n" for line in dataset), encoding="utf-8")
    kinds = Counter(p.kind for p in plants)
    print(f"{len(dataset) // 2} papers, {len(plants)} plants: {dict(sorted(kinds.items()))}")
    return 0


async def _run(args: argparse.Namespace) -> int:
    settings = Settings()
    entries = [
        DatasetEntry.model_validate_json(line)
        for line in (args.data / DATASET).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out: Path = args.out
    done = _done(out)
    llm = StructuredLLM.from_settings(settings) if not args.no_llm else None
    gate = asyncio.Semaphore(args.concurrency)
    async with HttpClient(settings) as http, closing_llm(llm):

        async def one(entry: DatasetEntry) -> None:
            paper = _load(entry.paper)
            if (paper.paper_id, paper.variant.value) in done:
                return
            pdf = args.set_c / "pdfs" / f"{paper.paper_id}.pdf"
            repository = (
                RepositoryRef(url=entry.repository, commit=entry.repository_commit)
                if entry.repository
                else None
            )
            ctx = RunContext(
                settings=settings,
                paper=paper,
                http=http,
                llm=llm,
                logger=logging.getLogger(),
                repository=repository,
                source_pdf=pdf if pdf.exists() else None,
            )
            async with gate:
                findings: list[CandidateFinding] = []
                errors = []
                for task in [*integrity_tasks(paper), *defect_tasks(paper, repository)]:
                    checker = check_defects if task.kind.value == "code" else check_integrity
                    try:
                        result = await checker(task, ctx)
                    except Exception as exc:
                        errors.append(f"{task.task_id}: {type(exc).__name__}: {exc}"[:300])
                        continue
                    findings += result.findings
            record = {
                "paper_id": paper.paper_id,
                "variant": paper.variant.value,
                "findings": [json.loads(f.model_dump_json()) for f in findings],
                "errors": errors,
            }
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(paper.paper_id, paper.variant.value, len(findings), errors[:1], flush=True)

        await asyncio.gather(*(one(e) for e in entries))
    if llm is not None:
        print(f"cost ${llm.spend.spent_usd:.3f}, {llm.calls} calls")
    return 0


def _matches(label: GoldLabel, finding: CandidateFinding) -> bool:
    if finding.error_class != label.error_class:
        return False
    if label.error_class is ErrorClass.CODE_DEFECT:
        planted = label.evidence[0]
        return any(
            e.path == planted.path and e.line_start == planted.line_start for e in finding.evidence
        )
    return (
        isinstance(label.anchor, SentenceAnchor)
        and isinstance(finding.anchor, SentenceAnchor)
        and finding.anchor.sentence_id == label.anchor.sentence_id
    )


def _score(args: argparse.Namespace) -> int:
    labels = [
        GoldLabel.model_validate_json(line)
        for line in args.labels.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    plants = {
        p.label_id: p
        for p in map(
            Plant.model_validate_json,
            (args.labels.parent / PLANTS).read_text(encoding="utf-8").splitlines(),
        )
    }
    runs = [json.loads(line) for line in args.run.read_text(encoding="utf-8").splitlines()]
    findings = {
        (r["paper_id"], r["variant"]): [CandidateFinding.model_validate(f) for f in r["findings"]]
        for r in runs
    }
    found: Counter[str] = Counter()
    total: Counter[str] = Counter()
    kind: str
    matched: set[str] = set()
    for label in labels:
        kind = plants[label.label_id].kind
        total[kind] += 1
        hits = [
            f
            for f in findings.get((label.paper_id, PaperVariant.INJECTED.value), [])
            if _matches(label, f)
        ]
        if hits:
            found[kind] += 1
            matched |= {f.finding_id for f in hits}
    papers = {label.paper_id for label in labels}
    caught = {
        label.paper_id
        for label in labels
        if any(
            _matches(label, f)
            for f in findings.get((label.paper_id, PaperVariant.INJECTED.value), [])
        )
    }
    print("found / planted, by kind:")
    for kind in sorted(total):
        print(f"  {kind}: {found[kind]} / {total[kind]}")
    print(
        f"  all: {sum(found.values())} / {sum(total.values())}; papers with one found: "
        f"{len(caught)} / {len(papers)}"
    )
    classes = {
        ErrorClass.UNFINISHED_CONTENT,
        ErrorClass.RESULT_INCONSISTENCY,
        ErrorClass.CODE_DEFECT,
    }
    control = Counter(
        f.error_class.value
        for (pid, variant), fs in findings.items()
        if variant == PaperVariant.ORIGINAL.value
        for f in fs
        if f.error_class in classes
    )
    extra = Counter(
        f.error_class.value
        for (pid, variant), fs in findings.items()
        if variant == PaperVariant.INJECTED.value
        for f in fs
        if f.error_class in classes and f.finding_id not in matched
    )
    print(f"findings on the originals (clean controls): {dict(control)}")
    print(f"findings on the copies that match no plant: {dict(extra)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--set-c", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--data", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--set-c", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument("--no-llm", action="store_true")
    score = commands.add_parser("score")
    score.add_argument("labels", type=Path)
    score.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        return asyncio.run(_build(args))
    if args.command == "run":
        return asyncio.run(_run(args))
    return _score(args)


if __name__ == "__main__":
    sys.exit(main())
