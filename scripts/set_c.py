"""Build and score set C, the paper-code consistency set (P3.8, P3.9, decision D18).

    uv run python scripts/set_c.py candidates paper.json --repo URL [--commit SHA] [--analyse-data]
    uv run python scripts/set_c.py inject-paper paper.json --sentence s12 --start 30 --end 34 \
        --key learning_rate --value 1e-3 --evidence-type code --rationale "..." \
        --evidence evidence.json --out-dir datasets/set_c
    uv run python scripts/set_c.py inject-code data/repos/lora patched/lora --path train.py \
        --line 40 --old 3e-4 --new 1e-3
    uv run python scripts/set_c.py validate datasets/set_c [--analyse-data]
    uv run python scripts/set_c.py pool scicoqa/real.parquet scicoqa/synthetic.parquet \
        --out data/set_c/pool.jsonl
    uv run python scripts/set_c.py collect data/set_c/pool.jsonl --data data/set_c --want 60
    uv run python scripts/set_c.py select data/set_c/pool.jsonl --data data/set_c \
        --out datasets/set_c --pairs 40
    uv run python scripts/set_c.py build datasets/set_c --data data/set_c
    uv run python scripts/set_c.py score datasets/set_c/labels.jsonl runs/<run_id>

`pool` lists SciCoQA's arXiv and GitHub paper-code pairs (needs `uv sync --extra data`).
`collect` downloads and parses each paper (GROBID must run) and records what the code
agent finds for each of its settings, resumably. `select` picks the pairs with enough
settings the repository confirms and proposes injected errors and clean controls
(deterministic), writing the committed files to `--out` and the papers and the batch
dataset file to `--data`. `build` recreates the injected papers and the dataset file
from the committed files, for anyone who has parsed the originals.

`candidates` prints one JSON line per paper setting with what the code agent finds and
which evidence types exist. `inject-paper` writes the injected ParsedPaper copy and
appends the gold label and the injection record to `labels.jsonl` and `injections.jsonl`
in the output folder. `validate` runs the D18 checks before freezing. `score` reports
precision, recall and F1 of the code and data classes for one run's predictions.
"""

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from pathlib import Path

from pydantic import BaseModel

from auditor.agents.code.set_c import (
    CandidateRow,
    EvidenceType,
    InjectionRecord,
    Pair,
    apply_injections,
    candidates,
    inject_code,
    inject_paper,
    pool_from_scicoqa,
    propose,
    score_code_module,
    validate,
)
from auditor.config import Settings
from auditor.experiments import load_paper, resolve_repository
from auditor.infra.context import RunContext
from auditor.infra.http import HttpClient
from auditor.parser.grobid import GrobidClient
from auditor.parser.pipeline import parse_pdf
from auditor.schemas import (
    DatasetEntry,
    Evidence,
    GoldLabel,
    ParsedPaper,
    Prediction,
    RepositoryRef,
    SentenceAnchor,
)

LABELS = "labels.jsonl"
INJECTIONS = "injections.jsonl"


def _append(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def _read_jsonl[T: BaseModel](path: Path, model: type[T]) -> list[T]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(model.model_validate_json(line))
    return rows


async def _candidates(args: argparse.Namespace) -> int:
    settings = Settings()
    if args.analyse_data:
        settings = settings.model_copy(update={"data_analysis_enabled": True})
    async with HttpClient(settings) as http:
        paper = await load_paper(args.paper, settings, http)
        repository = resolve_repository(args.repo, paper, args.commit)
        if repository is None:
            print("No repository given or found in the paper.", file=sys.stderr)
            return 2
        ctx = RunContext(
            settings=settings,
            paper=paper,
            http=http,
            llm=None,
            logger=logging.getLogger("set_c"),
            repository=repository,
        )
        for row in await candidates(paper, repository, ctx):
            print(
                json.dumps(
                    {
                        "key": row.key,
                        "sentence_id": row.sentence_id,
                        "start": row.anchor.start,
                        "end": row.anchor.end,
                        "paper_value": row.paper_value,
                        "outcome": row.outcome,
                        "agrees": row.agrees,
                        "reason": row.reason,
                        "repository_values": list(row.repository_values),
                        "evidence_types": [t.value for t in row.evidence_types],
                        "sentence": paper.sentence(row.sentence_id).text,
                    }
                )
            )
    return 0


def _inject_paper(args: argparse.Namespace) -> int:
    paper = ParsedPaper.model_validate_json(args.paper.read_bytes())
    evidence = [
        Evidence.model_validate(item)
        for item in json.loads(args.evidence.read_text(encoding="utf-8"))
    ]
    injected, label, record = inject_paper(
        paper,
        SentenceAnchor(sentence_id=args.sentence, start=args.start, end=args.end),
        args.value,
        key=args.key,
        rationale=args.rationale,
        evidence_type=EvidenceType(args.evidence_type),
        evidence=evidence,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paper_path = args.out_dir / "papers" / f"{paper.paper_id}-{label.label_id}.json"
    paper_path.parent.mkdir(parents=True, exist_ok=True)
    paper_path.write_text(injected.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    _append(args.out_dir / LABELS, label.model_dump_json())
    _append(args.out_dir / INJECTIONS, record.model_dump_json())
    print(f"{label.label_id}: wrote {paper_path}")
    return 0


def _inject_code(args: argparse.Namespace) -> int:
    patched = inject_code(args.repo, args.destination, args.path, args.line, args.old, args.new)
    print(f"patched copy at {patched}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    labels = _read_jsonl(args.folder / LABELS, GoldLabel)
    records = _read_jsonl(args.folder / INJECTIONS, InjectionRecord)
    problems = validate(labels, records, data_mode=args.analyse_data)
    for problem in problems:
        print(problem)
    print(f"{len(labels)} labels, {len(problems)} problems")
    return 1 if problems else 0


def _score(args: argparse.Namespace) -> int:
    labels = _read_jsonl(args.labels, GoldLabel)
    predictions = [
        prediction
        for path in sorted(args.run.rglob("predictions.jsonl"))
        for prediction in _read_jsonl(path, Prediction)
    ]
    scores = score_code_module(labels, predictions)
    for error_class, score in sorted(scores.per_class.items()):
        print(
            f"{error_class.value:30} P {score.precision:.3f}  R {score.recall:.3f}  "
            f"F1 {score.f1:.3f}  (tp {score.true_positives}, fp {score.false_positives}, "
            f"fn {score.false_negatives})"
        )
    print(f"macro-F1 {scores.macro_f1:.3f}")
    print(f"false-positive rate on clean controls {scores.false_positive_rate:.3f}")
    print(f"unlabelled flags {scores.unlabelled}")
    return 0


PAIRS = "pairs.jsonl"
MIN_USABLE = 3


def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8", newline="\n")


def _pool(args: argparse.Namespace) -> int:
    import pyarrow.parquet as pq

    rows = [row for path in args.parquet for row in pq.read_table(path).to_pylist()]
    pairs = pool_from_scicoqa(rows)
    _write_jsonl(args.out, [pair.model_dump_json() for pair in pairs])
    real = sum(1 for pair in pairs if pair.real_issue_ids)
    print(f"{len(pairs)} pairs ({real} with a real reported discrepancy) in {args.out}")
    return 0


def _usable(rows: list[CandidateRow]) -> int:
    return len(
        {
            row.key
            for row in rows
            if row.agrees is True and row.agreeing_evidence and row.anchor.start is not None
        }
    )


async def _collect(args: argparse.Namespace) -> int:
    settings = Settings()
    pairs = _read_jsonl(args.pool, Pair)
    data: Path = args.data
    found = 0
    async with HttpClient(settings) as http:
        grobid = GrobidClient(http, settings.grobid_url)
        for pair in pairs:
            if found >= args.want:
                break
            rows_path = data / "rows" / f"{pair.pair_id}.json"
            if rows_path.is_file():
                rows = [
                    CandidateRow.model_validate(r) for r in json.loads(rows_path.read_text("utf-8"))
                ]
                found += _usable(rows) >= MIN_USABLE
                continue
            pdf = data / "pdfs" / f"{pair.pair_id}.pdf"
            try:
                if not pdf.is_file():
                    await http.download_to(pair.pdf_url, pdf, max_bytes=60_000_000)
                paper = await parse_pdf(pdf, grobid=grobid, cache_dir=settings.cache_dir)
            except Exception as exc:  # one bad paper must not stop the collection
                print(f"{pair.pair_id}: skipped, {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            paper_path = data / "papers" / f"{pair.pair_id}.json"
            paper_path.parent.mkdir(parents=True, exist_ok=True)
            paper_path.write_text(paper.model_dump_json(), encoding="utf-8", newline="\n")
            repository = RepositoryRef(url=pair.repository, commit=pair.commit)
            ctx = RunContext(
                settings=settings,
                paper=paper,
                http=http,
                llm=None,
                logger=logging.getLogger("set_c"),
                repository=repository,
            )
            try:
                rows = await candidates(paper, repository, ctx)
            except Exception as exc:
                print(
                    f"{pair.pair_id}: candidates failed, {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                continue
            rows_path.parent.mkdir(parents=True, exist_ok=True)
            rows_path.write_text(
                json.dumps([row.model_dump(mode="json") for row in rows]),
                encoding="utf-8",
                newline="\n",
            )
            usable = _usable(rows)
            found += usable >= MIN_USABLE
            print(f"{pair.pair_id}: {len(rows)} settings, {usable} confirmed by the code")
    print(f"{found} pairs with at least {MIN_USABLE} confirmed settings")
    return 0


def _select(args: argparse.Namespace) -> int:
    data: Path = args.data
    out: Path = args.out
    chosen: list[Pair] = []
    labels: list[str] = []
    records: list[str] = []
    entries: list[str] = []
    for pair in _read_jsonl(args.pool, Pair):
        if len(chosen) >= args.pairs:
            break
        rows_path = data / "rows" / f"{pair.pair_id}.json"
        if not rows_path.is_file():
            continue
        rows = [CandidateRow.model_validate(r) for r in json.loads(rows_path.read_text("utf-8"))]
        if _usable(rows) < MIN_USABLE:
            continue
        paper = ParsedPaper.model_validate_json(
            (data / "papers" / f"{pair.pair_id}.json").read_bytes()
        )
        seed = int(hashlib.sha256(pair.pair_id.encode()).hexdigest()[:8], 16)
        proposal = propose(paper, rows, errors=args.errors, controls=args.controls, seed=seed)
        if not proposal.records:
            continue
        pdf_sha = hashlib.sha256((data / "pdfs" / f"{pair.pair_id}.pdf").read_bytes()).hexdigest()
        chosen.append(pair.model_copy(update={"pdf_sha256": pdf_sha}))
        labels += [label.model_dump_json() for label in proposal.labels]
        records += [record.model_dump_json() for record in proposal.records]
        injected_path = data / "injected" / f"{pair.pair_id}.json"
        injected_path.parent.mkdir(parents=True, exist_ok=True)
        injected_path.write_text(
            proposal.injected.model_dump_json(), encoding="utf-8", newline="\n"
        )
        entries += _entries(pair, data)
    _write_jsonl(out / PAIRS, [pair.model_dump_json() for pair in chosen])
    _write_jsonl(out / LABELS, labels)
    _write_jsonl(out / INJECTIONS, records)
    _write_jsonl(data / "dataset.jsonl", entries)
    errors = sum(1 for line in labels if '"error_class":"none"' not in line)
    print(f"{len(chosen)} pairs, {errors} injected errors, {len(labels) - errors} control labels")
    return 0


def _entries(pair: Pair, data: Path) -> list[str]:
    return [
        DatasetEntry(
            paper=str((data / folder / f"{pair.pair_id}.json").resolve()),
            repository=pair.repository,
            repository_commit=pair.commit,
        ).model_dump_json()
        for folder in ("papers", "injected")
    ]


def _build(args: argparse.Namespace) -> int:
    data: Path = args.data
    records = _read_jsonl(args.folder / INJECTIONS, InjectionRecord)
    entries: list[str] = []
    for pair in _read_jsonl(args.folder / PAIRS, Pair):
        pdf = data / "pdfs" / f"{pair.pair_id}.pdf"
        if pair.pdf_sha256 and hashlib.sha256(pdf.read_bytes()).hexdigest() != pair.pdf_sha256:
            print(f"{pair.pair_id}: the PDF differs from the one the labels were made on")
            return 1
        paper = ParsedPaper.model_validate_json(
            (data / "papers" / f"{pair.pair_id}.json").read_bytes()
        )
        mine = [r for r in records if r.paper_id == paper.paper_id]
        injected = apply_injections(paper, mine)
        path = data / "injected" / f"{pair.pair_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(injected.model_dump_json(), encoding="utf-8", newline="\n")
        entries += _entries(pair, data)
    _write_jsonl(data / "dataset.jsonl", entries)
    print(f"rebuilt {len(entries) // 2} pairs; dataset file {data / 'dataset.jsonl'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    commands = parser.add_subparsers(dest="command", required=True)

    found = commands.add_parser("candidates")
    found.add_argument("paper", type=Path)
    found.add_argument("--repo")
    found.add_argument("--commit")
    found.add_argument("--analyse-data", action="store_true")

    paper = commands.add_parser("inject-paper")
    paper.add_argument("paper", type=Path)
    paper.add_argument("--sentence", required=True)
    paper.add_argument("--start", type=int, required=True)
    paper.add_argument("--end", type=int, required=True)
    paper.add_argument("--key", required=True)
    paper.add_argument("--value", required=True)
    paper.add_argument("--rationale", required=True)
    paper.add_argument("--evidence-type", required=True, choices=[t.value for t in EvidenceType])
    paper.add_argument("--evidence", type=Path, required=True, help="JSON list of Evidence")
    paper.add_argument("--out-dir", type=Path, required=True)

    code = commands.add_parser("inject-code")
    code.add_argument("repo", type=Path)
    code.add_argument("destination", type=Path)
    code.add_argument("--path", required=True)
    code.add_argument("--line", type=int, required=True)
    code.add_argument("--old", required=True)
    code.add_argument("--new", required=True)

    check = commands.add_parser("validate")
    check.add_argument("folder", type=Path)
    check.add_argument("--analyse-data", action="store_true")

    score = commands.add_parser("score")
    score.add_argument("labels", type=Path)
    score.add_argument("run", type=Path, help="A run folder holding predictions.jsonl files")

    pool = commands.add_parser("pool")
    pool.add_argument("parquet", type=Path, nargs="+")
    pool.add_argument("--out", type=Path, required=True)

    collect = commands.add_parser("collect")
    collect.add_argument("pool", type=Path)
    collect.add_argument("--data", type=Path, required=True)
    collect.add_argument("--want", type=int, default=60)

    select = commands.add_parser("select")
    select.add_argument("pool", type=Path)
    select.add_argument("--data", type=Path, required=True)
    select.add_argument("--out", type=Path, required=True)
    select.add_argument("--pairs", type=int, default=40)
    select.add_argument("--errors", type=int, default=2)
    select.add_argument("--controls", type=int, default=2)

    build = commands.add_parser("build")
    build.add_argument("folder", type=Path)
    build.add_argument("--data", type=Path, required=True)

    args = parser.parse_args()
    match args.command:
        case "pool":
            return _pool(args)
        case "collect":
            return asyncio.run(_collect(args))
        case "select":
            return _select(args)
        case "build":
            return _build(args)
        case "candidates":
            return asyncio.run(_candidates(args))
        case "inject-paper":
            return _inject_paper(args)
        case "inject-code":
            return _inject_code(args)
        case "validate":
            return _validate(args)
        case _:
            return _score(args)


if __name__ == "__main__":
    sys.exit(main())
