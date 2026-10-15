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
    uv run python scripts/set_c.py collect data/set_c/pool.jsonl --data data/set_c
    uv run python scripts/set_c.py select data/set_c/pool.jsonl --data data/set_c \
        --out datasets/set_c --pairs 40 --scicoqa scicoqa/real.parquet scicoqa/synthetic.parquet
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
import shutil
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel

from auditor.agents.code.set_c import (
    CandidateRow,
    EvidenceType,
    InjectionRecord,
    MethodIssue,
    Pair,
    apply_code_changes,
    apply_injections,
    candidates,
    inject_code,
    inject_paper,
    method_issue,
    method_labels,
    pool_from_scicoqa,
    propose,
    score_code_module,
    validate,
)
from auditor.config import Settings
from auditor.experiments import load_paper, resolve_repository
from auditor.infra.context import RunContext
from auditor.infra.http import HttpClient
from auditor.infra.snapshots import fetch_snapshot
from auditor.parser.grobid import GrobidClient
from auditor.parser.pipeline import parse_pdf
from auditor.schemas import (
    DatasetEntry,
    Evidence,
    GoldLabel,
    PaperVariant,
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
    method_ids = frozenset(
        issue.discrepancy_id for issue in _read_jsonl(args.folder / METHOD_ISSUES, MethodIssue)
    )
    problems = validate(labels, records, data_mode=args.analyse_data, method_ids=method_ids)
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
    scores = score_code_module(labels, predictions, before_verifier=args.before_verifier)
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
METHOD_ISSUES = "method_issues.jsonl"
MIN_USABLE = 2


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


def _load_rows(path: Path) -> list[CandidateRow]:
    return [CandidateRow.model_validate(r) for r in json.loads(path.read_text("utf-8"))]


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
    pairs = _read_jsonl(args.pool, Pair)[: args.limit]
    data: Path = args.data
    gate = asyncio.Semaphore(args.concurrency)
    async with HttpClient(settings) as http:
        grobid = GrobidClient(http, settings.grobid_url)

        async def one(pair: Pair) -> None:
            rows_path = data / "rows" / f"{pair.pair_id}.json"
            if rows_path.is_file():
                return
            # The download waits on arXiv's rate limit, so it does not hold a parsing slot.
            pdf = data / "pdfs" / f"{pair.pair_id}.pdf"
            try:
                if not pdf.is_file():
                    await http.download_to(pair.pdf_url, pdf, max_bytes=60_000_000)
            except Exception as exc:
                print(f"{pair.pair_id}: download failed, {type(exc).__name__}: {exc}", flush=True)
                return
            async with gate:
                try:
                    paper = await parse_pdf(pdf, grobid=grobid, cache_dir=settings.cache_dir)
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
                    rows = await candidates(paper, repository, ctx)
                except Exception as exc:  # one bad pair must not stop the collection
                    print(f"{pair.pair_id}: skipped, {type(exc).__name__}: {exc}", flush=True)
                    return
            rows_path.parent.mkdir(parents=True, exist_ok=True)
            rows_path.write_text(
                json.dumps([row.model_dump(mode="json") for row in rows]),
                encoding="utf-8",
                newline="\n",
            )
            print(f"{pair.pair_id}: {len(rows)} settings, {_usable(rows)} confirmed", flush=True)

        await asyncio.gather(*(one(pair) for pair in pairs))
    done = [
        _usable([CandidateRow.model_validate(r) for r in json.loads(path.read_text("utf-8"))])
        for path in sorted((data / "rows").glob("*.json"))
    ]
    print(f"{len(done)} pairs collected, {sum(n >= MIN_USABLE for n in done)} usable")
    return 0


def _issues_by_pair(paths: list[Path], pairs: list[Pair]) -> dict[str, list[MethodIssue]]:
    import pyarrow.parquet as pq

    ids = {i: pair.pair_id for pair in pairs for i in pair.scicoqa_ids}
    found: dict[str, list[MethodIssue]] = {}
    for path in paths:
        for row in pq.read_table(path).to_pylist():
            pair_id = ids.get(str(row["discrepancy_id"]))
            if pair_id is not None:
                found.setdefault(pair_id, []).append(method_issue(pair_id, row))
    return found


async def _patch(
    pair: Pair, issues: list[MethodIssue], data: Path, http: HttpClient, settings: Settings
) -> tuple[Path | None, set[str]]:
    """A copy of the pair's snapshot with its synthetic method issues applied."""
    synthetic = [i for i in issues if not i.real and i.changes]
    if not synthetic:
        return None, set()
    snapshot = await fetch_snapshot(
        RepositoryRef(url=pair.repository, commit=pair.commit), http, settings
    )
    target = data / "patched" / pair.pair_id
    shutil.rmtree(target, ignore_errors=True)
    applied = apply_code_changes(snapshot.root, target, synthetic)
    if not applied:
        shutil.rmtree(target, ignore_errors=True)
        return None, set()
    return target, applied


def _injected_copy(paper: ParsedPaper) -> ParsedPaper:
    data = paper.model_dump()
    data["variant"] = PaperVariant.INJECTED.value
    return ParsedPaper.model_validate(data)


async def _select(args: argparse.Namespace) -> int:
    settings = Settings()
    data: Path = args.data
    out: Path = args.out
    pool = _read_jsonl(args.pool, Pair)
    issues = _issues_by_pair(args.scicoqa, pool) if args.scicoqa else {}

    # Pairs whose code confirms enough settings come first, since settings are what the
    # set mainly tests; the rest (method issues only) fill up to --pairs.
    def confirmed(pair: Pair) -> bool:
        path = data / "rows" / f"{pair.pair_id}.json"
        return path.is_file() and _usable(_load_rows(path)) >= MIN_USABLE

    pool = sorted(pool, key=lambda pair: not confirmed(pair))
    chosen: list[Pair] = []
    labels: list[str] = []
    records: list[str] = []
    method_rows: list[str] = []
    entries: list[str] = []
    async with HttpClient(settings) as http:
        for pair in pool:
            if len(chosen) >= args.pairs:
                break
            rows_path = data / "rows" / f"{pair.pair_id}.json"
            if not rows_path.is_file():
                continue
            rows = [
                CandidateRow.model_validate(r) for r in json.loads(rows_path.read_text("utf-8"))
            ]
            paper = ParsedPaper.model_validate_json(
                (data / "papers" / f"{pair.pair_id}.json").read_bytes()
            )
            seed = int(hashlib.sha256(pair.pair_id.encode()).hexdigest()[:8], 16)
            proposal = (
                propose(paper, rows, errors=args.errors, controls=args.controls, seed=seed)
                if _usable(rows) >= MIN_USABLE
                else None
            )
            mine = issues.get(pair.pair_id, [])
            patched, applied = await _patch(pair, mine, data, http, settings)
            method = method_labels(paper.paper_id, mine, patched=applied)
            if (proposal is None or not proposal.records) and not method:
                continue
            injected = proposal.injected if proposal and proposal.records else _injected_copy(paper)
            pdf_sha = hashlib.sha256(
                (data / "pdfs" / f"{pair.pair_id}.pdf").read_bytes()
            ).hexdigest()
            chosen.append(pair.model_copy(update={"pdf_sha256": pdf_sha}))
            if proposal and proposal.records:
                labels += [label.model_dump_json() for label in proposal.labels]
                records += [record.model_dump_json() for record in proposal.records]
            labels += [label.model_dump_json() for label in method]
            method_rows += [
                issue.model_dump_json()
                for issue in mine
                if issue.real or issue.discrepancy_id in applied
            ]
            injected_path = data / "injected" / f"{pair.pair_id}.json"
            injected_path.parent.mkdir(parents=True, exist_ok=True)
            injected_path.write_text(injected.model_dump_json(), encoding="utf-8", newline="\n")
            entries += _entries(pair, data, patched)
    _write_jsonl(out / PAIRS, [pair.model_dump_json() for pair in chosen])
    _write_jsonl(out / LABELS, labels)
    _write_jsonl(out / INJECTIONS, records)
    _write_jsonl(out / METHOD_ISSUES, method_rows)
    _write_jsonl(data / "dataset.jsonl", entries)
    print(_summary(out))
    return 0


def _summary(folder: Path) -> str:
    labels = _read_jsonl(folder / LABELS, GoldLabel)
    by_class = Counter(str(label.error_class) for label in labels)
    pairs = len(_read_jsonl(folder / PAIRS, Pair))
    return f"{pairs} pairs, {len(labels)} labels: " + ", ".join(
        f"{name} {count}" for name, count in sorted(by_class.items())
    )


def _entries(pair: Pair, data: Path, patched: Path | None) -> list[str]:
    original = DatasetEntry(
        paper=str((data / "papers" / f"{pair.pair_id}.json").resolve()),
        repository=pair.repository,
        repository_commit=pair.commit,
    )
    injected = DatasetEntry(
        paper=str((data / "injected" / f"{pair.pair_id}.json").resolve()),
        repository=str(patched.resolve()) if patched else pair.repository,
        repository_commit=None if patched else pair.commit,
    )
    return [original.model_dump_json(), injected.model_dump_json()]


async def _build(args: argparse.Namespace) -> int:
    settings = Settings()
    data: Path = args.data
    records = _read_jsonl(args.folder / INJECTIONS, InjectionRecord)
    issues = _read_jsonl(args.folder / METHOD_ISSUES, MethodIssue)
    entries: list[str] = []
    async with HttpClient(settings) as http:
        for pair in _read_jsonl(args.folder / PAIRS, Pair):
            pdf = data / "pdfs" / f"{pair.pair_id}.pdf"
            digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
            if pair.pdf_sha256 and digest != pair.pdf_sha256:
                print(f"{pair.pair_id}: the PDF differs from the one the labels were made on")
                return 1
            paper = ParsedPaper.model_validate_json(
                (data / "papers" / f"{pair.pair_id}.json").read_bytes()
            )
            mine = [r for r in records if r.paper_id == paper.paper_id]
            injected = apply_injections(paper, mine) if mine else _injected_copy(paper)
            path = data / "injected" / f"{pair.pair_id}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(injected.model_dump_json(), encoding="utf-8", newline="\n")
            pair_issues = [i for i in issues if i.pair_id == pair.pair_id]
            patched, _ = await _patch(pair, pair_issues, data, http, settings)
            entries += _entries(pair, data, patched)
    _write_jsonl(data / "dataset.jsonl", entries)
    print(f"rebuilt {len(entries) // 2} pairs; dataset file {data / 'dataset.jsonl'}")
    return 0


REVIEW = "review.jsonl"


def _review_sample(args: argparse.Namespace) -> int:
    """A random sample of labels (fixed seed) for a second person to check (D18)."""
    import random

    labels = _read_jsonl(args.folder / LABELS, GoldLabel)
    rng = random.Random(args.seed)
    count = max(1, round(len(labels) * args.share))
    sample = sorted(rng.sample(labels, min(count, len(labels))), key=lambda lbl: lbl.label_id)
    rows = []
    for label in sample:
        papers = args.data / ("injected" if label.variant is PaperVariant.INJECTED else "papers")
        paper = next(
            (
                ParsedPaper.model_validate_json(path.read_bytes())
                for path in sorted(papers.glob("*.json"))
                if path.stem == label.paper_id
            ),
            None,
        )
        sentence = None
        if paper is not None and isinstance(label.anchor, SentenceAnchor):
            sentence = paper.sentence(label.anchor.sentence_id).text
        rows.append(
            json.dumps(
                {
                    "label_id": label.label_id,
                    "paper_id": label.paper_id,
                    "variant": label.variant.value,
                    "error_class": str(label.error_class),
                    "target": label.target.model_dump(mode="json"),
                    "sentence": sentence,
                    "rationale": label.rationale,
                    "evidence": [e.model_dump(mode="json") for e in label.evidence],
                    "reviewer": None,
                    "agree": None,
                    "note": "",
                }
            )
        )
    _write_jsonl(args.folder / REVIEW, rows)
    print(f"{len(rows)} of {len(labels)} labels in {args.folder / REVIEW}; fill reviewer and agree")
    return 0


def _apply_review(args: argparse.Namespace) -> int:
    """Record the second reviews in the labels; report disagreements and the agreement rate."""
    reviews = [
        json.loads(line)
        for line in (args.folder / REVIEW).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    done = {r["label_id"]: r for r in reviews if r.get("reviewer") and r.get("agree") is not None}
    labels = _read_jsonl(args.folder / LABELS, GoldLabel)
    updated = []
    for label in labels:
        review = done.get(label.label_id)
        if review is not None and review["reviewer"] not in label.reviewed_by:
            label = label.model_copy(
                update={"reviewed_by": (*label.reviewed_by, review["reviewer"])}
            )
        updated.append(label.model_dump_json())
    _write_jsonl(args.folder / LABELS, updated)
    agreed = sum(1 for r in done.values() if r["agree"])
    for review in done.values():
        if not review["agree"]:
            print(f"disagreement on {review['label_id']}: {review.get('note', '')}")
    rate = agreed / len(done) if done else 0.0
    print(f"{len(done)} reviews recorded, agreement {rate:.0%}")
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
    score.add_argument(
        "--before-verifier",
        action="store_true",
        help="Count every finding not dropped (module-level scores without the verifier)",
    )

    pool = commands.add_parser("pool")
    pool.add_argument("parquet", type=Path, nargs="+")
    pool.add_argument("--out", type=Path, required=True)

    collect = commands.add_parser("collect")
    collect.add_argument("pool", type=Path)
    collect.add_argument("--data", type=Path, required=True)
    collect.add_argument("--limit", type=int, default=None, help="Only the first N pairs")
    collect.add_argument("--concurrency", type=int, default=4)

    select = commands.add_parser("select")
    select.add_argument("pool", type=Path)
    select.add_argument("--data", type=Path, required=True)
    select.add_argument("--out", type=Path, required=True)
    select.add_argument("--pairs", type=int, default=40)
    select.add_argument("--errors", type=int, default=2)
    select.add_argument("--controls", type=int, default=2)
    select.add_argument("--scicoqa", type=Path, nargs="*", default=[], help="SciCoQA parquet files")

    build = commands.add_parser("build")
    build.add_argument("folder", type=Path)
    build.add_argument("--data", type=Path, required=True)

    sample = commands.add_parser("review-sample")
    sample.add_argument("folder", type=Path)
    sample.add_argument("--data", type=Path, required=True)
    sample.add_argument("--share", type=float, default=0.2)
    sample.add_argument("--seed", type=int, default=20261015)

    reviewed = commands.add_parser("apply-review")
    reviewed.add_argument("folder", type=Path)

    args = parser.parse_args()
    match args.command:
        case "review-sample":
            return _review_sample(args)
        case "apply-review":
            return _apply_review(args)
        case "pool":
            return _pool(args)
        case "collect":
            return asyncio.run(_collect(args))
        case "select":
            return asyncio.run(_select(args))
        case "build":
            return asyncio.run(_build(args))
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
