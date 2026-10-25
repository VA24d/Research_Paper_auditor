"""Citation module evaluation (P2.9): Recall@5, module scores, tool-call accuracy.

    uv run python scripts/citation_eval.py fetch data/set_b/papers/<id>.json \
        --out data/set_b/sources/<id>.txt          # network: resolves and retrieves
    uv run python scripts/citation_eval.py recall --injections datasets/set_b/injections.jsonl \
        --sources data/set_b/sources.jsonl [--k 5]  # offline
    uv run python scripts/citation_eval.py score --labels datasets/set_b/labels.jsonl \
        --run runs/<run_id> [--abstract-only]       # offline
    uv run python scripts/citation_eval.py tool-accuracy --file lookups.jsonl --gold gold.jsonl

`fetch` resolves each of a paper's cited works and writes the text retrieved at the best
level available (one JSON line per paper: paper_id, bib_id, level, source_text), so the
offline commands never need the network. `recall` reports Recall@k of the gold evidence
passages recorded with the injections. `score` reports the citation classes' P/R/F1 for
one run's predictions; with --abstract-only it scores the run made with
AUDITOR_CITATION_ABSTRACT_ONLY=1 so the two runs can be compared. `tool-accuracy` scores
labelled reference lookups ({key, resolved_title} vs {key, gold_title}).
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import cast

from auditor.agents.citation.eval import (
    LookupGold,
    LookupOutcome,
    recall_at_k,
    tool_call_accuracy,
)
from auditor.agents.citation.set_b import InjectionRecord, read_jsonl, score_citation_module
from auditor.config import Settings
from auditor.schemas import GoldLabel, Prediction


def _sources(folder: Path) -> dict[tuple[str, str], str]:
    sources: dict[tuple[str, str], str] = {}
    for path in sorted(folder.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                sources[(row["paper_id"], row["bib_id"])] = row["source_text"]
    return sources


def _original_claim(record: InjectionRecord, originals: Path) -> str | None:
    """The claim sentence as it stood before the injection.

    A number injection's `original` is only the number span; the sentence is rebuilt
    from the kept original paper (papers/original/<id>.json) with the original value
    spliced back in.
    """
    from auditor.schemas import ParsedPaper

    if record.sentence_id is None:
        return None
    path = originals / f"{record.paper_id}.json"
    if not path.exists():
        return None
    paper = ParsedPaper.model_validate_json(path.read_bytes())
    try:
        text = paper.sentence(record.sentence_id).text
    except KeyError:
        return None
    if record.start is None or record.end is None:
        return text
    return text[: record.start] + record.original + text[record.end :]


def _recall(args: argparse.Namespace) -> int:
    records = read_jsonl(args.injections, InjectionRecord)
    sources = _sources(args.sources)
    originals = args.originals
    hits = misses = no_passage = 0
    for record in records:
        if not record.gold_passage:
            no_passage += 1
            continue
        source_text = sources.get((record.paper_id, record.bib_id))
        if source_text is None:
            print(f"{record.label_id}: no recorded source text", file=sys.stderr)
            continue
        claim = _original_claim(record, originals)
        if claim is None:
            print(f"{record.label_id}: no original claim", file=sys.stderr)
            continue
        if recall_at_k(source_text, claim, record.gold_passage, k=args.k):
            hits += 1
        else:
            misses += 1
    total = hits + misses
    print(f"Recall@{args.k} {hits / total:.3f} ({hits} of {total})" if total else "Recall@k n/a")
    if no_passage:
        print(f"records without a gold passage {no_passage}")
    return 0


def _score(args: argparse.Namespace) -> int:
    labels = read_jsonl(args.labels, GoldLabel)
    predictions = [
        prediction
        for path in sorted(args.run.rglob("predictions.jsonl"))
        for prediction in read_jsonl(path, Prediction)
    ]
    scores = score_citation_module(labels, predictions, before_verifier=args.before_verifier)
    name = "abstract-only run" if args.abstract_only else "full-text run"
    print(f"--- {name} ---")
    for error_class, score in sorted(scores.per_class.items()):
        print(
            f"{error_class.value:30} P {score.precision:.3f}  R {score.recall:.3f}  "
            f"F1 {score.f1:.3f}"
        )
    print(f"macro-F1 {scores.macro_f1:.3f}")
    print(f"false-positive rate on clean controls {scores.false_positive_rate:.3f}")
    return 0


def _tool_accuracy(args: argparse.Namespace) -> int:
    outcomes = [
        LookupOutcome(row["key"], row.get("resolved_title"))
        for line in args.file.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    ]
    golds = [
        LookupGold(row["key"], row.get("gold_title"))
        for line in args.gold.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    ]
    print(f"tool-call accuracy {tool_call_accuracy(outcomes, golds):.3f}")
    return 0


async def _fetch(args: argparse.Namespace) -> int:
    """Resolve every bibliography entry of one paper and write the text retrieved at the
    best level available, one JSON line per entry."""
    from auditor.infra.context import RunContext
    from auditor.infra.http import HttpClient
    from auditor.retrieval.fulltext import retrieve_source
    from auditor.retrieval.resolve import resolve_entry
    from auditor.schemas import ParsedPaper

    paper = ParsedPaper.model_validate_json(args.paper.read_bytes())
    settings = Settings()
    lines: list[str] = []
    async with HttpClient(settings) as http:
        ctx = RunContext(
            settings=settings,
            paper=paper,
            http=http,
            llm=None,
            logger=logging.getLogger("citation-eval"),
        )
        for entry in paper.bibliography:
            resolution = await resolve_entry(ctx.http, entry)
            if not resolution.found or resolution.work is None:
                lines.append(
                    json.dumps(
                        {
                            "paper_id": paper.paper_id,
                            "bib_id": entry.id,
                            "level": "none",
                            "reason": resolution.reason,
                        },
                        ensure_ascii=False,
                    )
                )
                continue
            source = await retrieve_source(
                ctx.http,
                resolution.work,
                settings,
                abstract_only=args.abstract_only,
            )
            lines.append(
                json.dumps(
                    {
                        "paper_id": paper.paper_id,
                        "bib_id": entry.id,
                        "level": source.level.value,
                        "source_text": source.text,
                    },
                    ensure_ascii=False,
                )
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line + "\n")
    print(f"wrote {args.out} ({len(lines)} entries)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    fetch = commands.add_parser("fetch", help="retrieve the cited works' texts (network)")
    fetch.add_argument("paper", type=Path)
    fetch.add_argument("--out", type=Path, required=True)
    fetch.add_argument("--abstract-only", action="store_true")
    fetch.set_defaults(func=_fetch)

    recall = commands.add_parser("recall", help="Recall@k of the gold passages (offline)")
    recall.add_argument("--injections", type=Path, required=True)
    recall.add_argument("--sources", type=Path, required=True)
    recall.add_argument("--originals", type=Path, default=Path("datasets/set_b/papers/original"))
    recall.add_argument("--k", type=int, default=5)
    recall.set_defaults(func=_recall)

    score = commands.add_parser("score", help="module P/R/F1 for one run (offline)")
    score.add_argument("--labels", type=Path, required=True)
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--abstract-only", action="store_true")
    score.add_argument("--before-verifier", action="store_true")
    score.set_defaults(func=_score)

    accuracy = commands.add_parser("tool-accuracy", help="labelled lookup accuracy (offline)")
    accuracy.add_argument("--file", type=Path, required=True)
    accuracy.add_argument("--gold", type=Path, required=True)
    accuracy.set_defaults(func=_tool_accuracy)

    args = parser.parse_args()
    if asyncio.iscoroutinefunction(args.func):
        return cast(int, asyncio.run(args.func(args)))
    return cast(int, args.func(args))


if __name__ == "__main__":
    sys.exit(main())
