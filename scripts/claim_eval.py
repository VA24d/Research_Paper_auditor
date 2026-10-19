"""Run the code and data claim loop on set C pairs and judge its findings against the
SciCoQA paper-code discrepancies (`datasets/set_c/method_issues.jsonl`).

    uv run python scripts/claim_eval.py run --set-c data/set_c --pairs 10 --out data/claims.jsonl
    uv run python scripts/claim_eval.py judge data/claims.jsonl \
        --issues datasets/set_c/method_issues.jsonl

`run` checks the original and the injected copy of each pair with one strategy: the claim
loop on experiment claims (`loop`), the claim loop on the method's implementation details
(`loop-method`), or the method check (`method-check`), for comparison.
The original is checked against the repository as released, which holds the real
discrepancies; the injected copy against the patched repository, which also holds the
synthetic ones. `judge` asks a model, once per discrepancy, whether a finding of that
paper and variant describes it, and counts the findings no discrepancy explains.
"""

import argparse
import ast
import asyncio
import json
import logging
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from auditor.agents.code.claim_loop import check_claim
from auditor.agents.code.method_check import check_method
from auditor.config import Settings
from auditor.infra.context import RunContext
from auditor.infra.http import HttpClient
from auditor.infra.llm import StructuredLLM, closing_llm
from auditor.infra.scope import Scope, enter_scope
from auditor.infra.usage import UsageLog
from auditor.planner.code_claims import claim_task, claim_tasks
from auditor.planner.tasks import method_tasks
from auditor.schemas import AuditTask, DatasetEntry, ParsedPaper, RepositoryRef, SentenceAnchor


def _entries(set_c: Path) -> list[DatasetEntry]:
    lines = (set_c / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    return [DatasetEntry.model_validate_json(line) for line in lines if line.strip()]


def _load(path: str) -> ParsedPaper:
    return ParsedPaper.model_validate_json(Path(path).read_bytes())


def _oracle_tasks(paper: ParsedPaper, repository: RepositoryRef, issues: Path) -> list[AuditTask]:
    """One claim task per known discrepancy of the paper, on the sentence it quotes: the
    explorer's part of the loop, measured without the planner's."""
    rows = [json.loads(line) for line in issues.read_text(encoding="utf-8").splitlines()]
    tasks = []
    for issue in rows:
        if issue["pair_id"] != paper.paper_id:
            continue
        if str(issue["real"]) != "True" and paper.variant.value != "injected":
            continue
        quotes = issue["paper_quotes"]
        quotes = ast.literal_eval(quotes) if isinstance(quotes, str) else quotes
        for quote in quotes[:1]:
            words = " ".join(quote.split())[:80]
            sentence = next(
                (
                    s
                    for s in paper.sentences
                    if words[:40] and words[:40] in " ".join(s.text.split())
                ),
                None,
            )
            if sentence is None:
                continue
            anchor = SentenceAnchor(sentence_id=sentence.id)
            tasks.append(claim_task(paper, repository, anchor, "training", quote[:300]))
    return tasks


def _done(out: Path) -> set[tuple[str, str]]:
    if not out.exists():
        return set()
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    return {(r["paper_id"], r["variant"]) for r in rows}


async def _run(args: argparse.Namespace) -> int:
    settings = Settings().model_copy(update={"claim_loop_enabled": True})
    entries = _entries(args.set_c)
    pairs = sorted({_load(e.paper).paper_id for e in entries if "/papers/" in e.paper})
    chosen = set(pairs[args.skip : args.skip + args.pairs])
    done = _done(args.out)
    llm = StructuredLLM.from_settings(settings)
    gate = asyncio.Semaphore(args.concurrency)
    async with HttpClient(settings) as http, closing_llm(llm):

        async def one(entry: DatasetEntry) -> None:
            paper = _load(entry.paper)
            if paper.paper_id not in chosen or (paper.paper_id, paper.variant.value) in done:
                return
            repository = RepositoryRef(url=entry.repository or "", commit=entry.repository_commit)
            ctx = RunContext(
                settings=settings,
                paper=paper,
                http=http,
                llm=llm,
                logger=logging.getLogger(),
                repository=repository,
                review_llm=llm.review,
            )
            usage = UsageLog(args.out.parent / f"{args.out.stem}_usage")
            async with gate:
                with enter_scope(Scope.for_paper("claim-eval", paper, usage)):
                    await _paper(paper, repository, ctx)
            return

        async def _paper(paper: ParsedPaper, repository: RepositoryRef, ctx: RunContext) -> None:
            if args.strategy == "method-check":
                tasks = method_tasks(paper, repository)
                checker = check_method
            elif args.strategy == "oracle":
                tasks = _oracle_tasks(paper, repository, args.issues)
                checker = check_claim
            else:
                tasks = await claim_tasks(
                    paper,
                    repository,
                    llm,
                    limit=settings.claim_loop_max_claims,
                    max_output=settings.claim_loop_plan_max_output,
                    focus="method" if args.strategy == "loop-method" else "experiments",
                )
                checker = check_claim
            findings, outcomes = [], []
            for task in tasks:
                try:
                    result = await checker(task, ctx)
                except Exception as exc:
                    outcomes.append(f"error: {type(exc).__name__}: {exc}"[:200])
                    continue
                outcomes.append(result.outcome.value)
                findings += [json.loads(f.model_dump_json()) for f in result.findings]
            record = {
                "paper_id": paper.paper_id,
                "variant": paper.variant.value,
                "claims": len(tasks),
                "outcomes": outcomes,
                "findings": findings,
            }
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(paper.paper_id, paper.variant.value, len(tasks), len(findings), flush=True)

        await asyncio.gather(*(one(e) for e in entries if e.repository))
    print(f"cost ${llm.spend.spent_usd:.3f}, {llm.calls} main calls")
    return 0


JUDGE_PROMPT = """You compare one known discrepancy between a paper and its code with the
issues an automatic checker reported for that paper. Give the numbers of the reported
issues that describe the same discrepancy (the same component and the same mismatch);
wording may differ. Give an empty list when none does."""


class Match(BaseModel):
    matches: list[int]
    reason: str = ""


async def _judge(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in args.run.read_text(encoding="utf-8").splitlines()]
    issues = [json.loads(line) for line in args.issues.read_text(encoding="utf-8").splitlines()]
    llm = StructuredLLM.from_settings(Settings())
    found = total = 0
    explained: dict[tuple[str, str], set[int]] = {}
    for row in rows:
        key = (row["paper_id"], row["variant"])
        explained[key] = set()
        wanted = [
            i
            for i in issues
            if i["pair_id"] == row["paper_id"]
            and (str(i["real"]) == "True" or row["variant"] == "injected")
        ]
        listing = "\n".join(f"{n}. {f['summary']}" for n, f in enumerate(row["findings"], start=1))
        for issue in wanted:
            total += 1
            if not row["findings"]:
                continue
            verdict = await llm.generate(
                Match,
                [
                    SystemMessage(JUDGE_PROMPT),
                    HumanMessage(f"Discrepancy:\n{issue['description']}\n\nReported:\n{listing}"),
                ],
                purpose="claim-judge",
            )
            hits = {n for n in verdict.value.matches if 1 <= n <= len(row["findings"])}
            explained[key] |= hits
            found += bool(hits)
    findings = sum(len(r["findings"]) for r in rows)
    unexplained = sum(
        len(r["findings"]) - len(explained[(r["paper_id"], r["variant"])]) for r in rows
    )
    original = sum(len(r["findings"]) for r in rows if r["variant"] == "original")
    print(
        json.dumps(
            {
                "discrepancies": total,
                "found": found,
                "findings": findings,
                "findings_matching_a_discrepancy": findings - unexplained,
                "findings_on_originals": original,
                "claims": sum(r["claims"] for r in rows),
            }
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--set-c", type=Path, required=True)
    run.add_argument("--pairs", type=int, default=10)
    run.add_argument("--skip", type=int, default=0, help="pairs to leave out first")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--concurrency", type=int, default=4)
    run.add_argument(
        "--strategy",
        choices=["loop", "loop-method", "method-check", "oracle"],
        default="loop",
    )
    run.add_argument("--issues", type=Path, default=Path("datasets/set_c/method_issues.jsonl"))
    judge = commands.add_parser("judge")
    judge.add_argument("run", type=Path)
    judge.add_argument("--issues", type=Path, required=True)
    args = parser.parse_args()
    return asyncio.run(_run(args) if args.command == "run" else _judge(args))


if __name__ == "__main__":
    sys.exit(main())
