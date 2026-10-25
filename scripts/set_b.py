"""Build set B, the claim-source support set (P2.8, decision D18).

    uv run python scripts/set_b.py candidates data/set_b/papers/<id>.json \
        --source-text data/set_b/sources/<id>.txt
    uv run python scripts/set_b.py inject data/set_b/papers/<id>.json --sentence s12 \
        --start 30 --end 34 --kind number --value 88.4 --bib b3 --rationale "..." \
        --evidence evidence.json --source-text sources/<id>.txt --out-dir datasets/set_b
    uv run python scripts/set_b.py validate datasets/set_b --sources data/set_b/sources.jsonl
    uv run python scripts/set_b.py score datasets/set_b/labels.jsonl runs/<run_id>

`candidates` lists the claim-source pairs of one parsed paper: one line per citation task
the planner builds, with the claim sentence, and (given the recorded source text of the
cited work) the gold passage for Recall@5. The annotator picks pairs and fills the
injection arguments. `inject` applies one injection to a copy of the paper (number, claim,
metadata or not_found) and appends the gold label and the injection record to labels.jsonl
and injections.jsonl. `validate` runs the D18 checks before freezing. `score` reports
precision, recall and F1 of the citation classes for one run's predictions. Parsing the
papers is done with `uv run python -m auditor batch` on a dataset file; the source texts
of the cited works are fetched with `scripts/citation_eval.py fetch`.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import cast

from auditor.agents.citation.set_b import (
    InjectionRecord,
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
from auditor.planner.claims import citation_candidates
from auditor.planner.tasks import citation_tasks
from auditor.schemas import Evidence, GoldLabel, ParsedPaper, Prediction, SentenceAnchor


def _load(path: Path) -> ParsedPaper:
    return ParsedPaper.model_validate_json(path.read_bytes())


def _evidence(path: Path | None) -> tuple[Evidence, ...]:
    if path is None:
        return ()
    items = json.loads(path.read_text(encoding="utf-8"))
    return tuple(Evidence.model_validate(item) for item in items)


def _candidates(args: argparse.Namespace) -> int:
    paper = _load(args.paper)
    source_text = args.source_text.read_text(encoding="utf-8") if args.source_text else None
    tasks = citation_tasks(paper, citation_candidates(paper))
    for task in tasks:
        if not isinstance(task.anchor, SentenceAnchor):
            continue
        row: dict[str, object] = {
            "task_id": task.task_id,
            "sentence_id": task.anchor.sentence_id,
            "target": task.target.model_dump(mode="json"),
            "claim": task.claim,
            "sentence": paper.sentence(task.anchor.sentence_id).text,
        }
        if source_text and task.claim:
            try:
                passage = gold_passage_for(source_text, task.claim)
                row["gold_passage"] = passage.text
                row["gold_passage_start"] = passage.char_start
                row["gold_passage_end"] = passage.char_end
            except ValueError:
                pass
        print(json.dumps(row, ensure_ascii=False))
    return 0


def _inject(args: argparse.Namespace) -> int:
    paper = _load(args.paper)
    anchor = SentenceAnchor(sentence_id=args.sentence, start=args.start, end=args.end)
    evidence = _evidence(args.evidence)
    source_text = args.source_text.read_text(encoding="utf-8") if args.source_text else None
    if args.kind in ("number", "claim"):
        if args.value is None:
            args.error(f"--kind {args.kind} needs --value")
        gold = (
            gold_passage_for(source_text, paper.sentence(args.sentence).text)
            if source_text
            else None
        )
        inject = inject_number if args.kind == "number" else inject_claim
        injected, label, record = inject(
            paper,
            anchor,
            args.value,
            bib_id=args.bib,
            rationale=args.rationale,
            evidence=evidence,
            gold_passage=gold,
            source_text=source_text,
        )
    elif args.kind == "metadata":
        if args.field is None or args.value is None:
            args.error("--kind metadata needs --field and --value")
        injected, label, record = inject_metadata(
            paper,
            args.bib,
            field_name=args.field,
            new_value=args.value,
            rationale=args.rationale,
            evidence=evidence,
        )
    else:
        if args.value is None:
            args.error("--kind not_found needs --value (the fabricated title)")
        injected, label, record = inject_not_found(
            paper,
            args.bib,
            new_title=args.value,
            rationale=args.rationale,
            evidence=evidence,
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paper_path = write_paper(injected, args.out_dir)
    from auditor.agents.citation.set_b import append_jsonl

    append_jsonl(args.out_dir / "labels.jsonl", label)
    append_jsonl(args.out_dir / "injections.jsonl", record)
    print(f"{label.label_id}: wrote {paper_path}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    labels = read_jsonl(args.folder / "labels.jsonl", GoldLabel)
    records = read_jsonl(args.folder / "injections.jsonl", InjectionRecord)
    sources: dict[tuple[str, str], str] = {}
    if args.sources and args.sources.exists():
        for path in sorted(args.sources.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    sources[(row["paper_id"], row["bib_id"])] = row["source_text"]
    problems = validate(labels, records, sources=sources or None)
    for problem in problems:
        print(problem)
    print(f"{len(labels)} labels, {len(problems)} problems")
    return 1 if problems else 0


def _score(args: argparse.Namespace) -> int:
    labels = read_jsonl(args.labels, GoldLabel)
    predictions = [
        prediction
        for path in sorted(args.run.rglob("predictions.jsonl"))
        for prediction in read_jsonl(path, Prediction)
    ]
    scores = score_citation_module(labels, predictions, before_verifier=args.before_verifier)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    candidates_cmd = commands.add_parser("candidates", help="list a paper's claim-source pairs")
    candidates_cmd.add_argument("paper", type=Path)
    candidates_cmd.add_argument("--source-text", type=Path, default=None)
    candidates_cmd.set_defaults(func=_candidates)

    inject_cmd = commands.add_parser("inject", help="apply one injection to a paper copy")
    inject_cmd.add_argument("paper", type=Path)
    inject_cmd.add_argument("--sentence", required=True)
    inject_cmd.add_argument("--start", type=int, required=True)
    inject_cmd.add_argument("--end", type=int, required=True)
    inject_cmd.add_argument(
        "--kind", choices=["number", "claim", "metadata", "not_found"], required=True
    )
    inject_cmd.add_argument("--value", help="the new value, or the fabricated title")
    inject_cmd.add_argument("--field", choices=["title", "venue", "year", "doi"])
    inject_cmd.add_argument("--bib", required=True, help="the bibliography entry id")
    inject_cmd.add_argument("--rationale", required=True)
    inject_cmd.add_argument("--evidence", type=Path, default=None)
    inject_cmd.add_argument("--source-text", type=Path, default=None)
    inject_cmd.add_argument("--out-dir", type=Path, default=Path("datasets/set_b"))
    inject_cmd.set_defaults(func=_inject)

    validate_cmd = commands.add_parser("validate", help="the D18 checks before freezing")
    validate_cmd.add_argument("folder", type=Path)
    validate_cmd.add_argument("--sources", type=Path, default=None)
    validate_cmd.set_defaults(func=_validate)

    score_cmd = commands.add_parser("score", help="score one run's predictions (offline)")
    score_cmd.add_argument("labels", type=Path)
    score_cmd.add_argument("run", type=Path)
    score_cmd.add_argument("--before-verifier", action="store_true")
    score_cmd.set_defaults(func=_score)

    args = parser.parse_args()
    return cast(int, args.func(args))


if __name__ == "__main__":
    sys.exit(main())
