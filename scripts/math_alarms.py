"""Run the theory checker alone (triage, deep checks, challenge) on papers that are not
from SPOT, to count the math findings it makes on ordinary papers.

    uv run python scripts/math_alarms.py data/set_c/papers/1711.10703v1.json ... \
        --out data/math_alarms.jsonl

Each paper gives one JSON line with the findings that survive the challenge. The papers
are published versions with no known math error, so most findings are false alarms; each
needs a look before it is called one.
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from auditor.agents.theory import check
from auditor.agents.theory.review import review_math
from auditor.config import Settings
from auditor.infra.context import RunContext
from auditor.infra.http import HttpClient
from auditor.infra.llm import StructuredLLM, closing_llm
from auditor.planner.theory import triage_tasks
from auditor.schemas import CandidateFinding, ParsedPaper, VerificationState, VerifiedFinding


def _load(path: Path) -> ParsedPaper:
    return ParsedPaper.model_validate_json(path.read_bytes())


def _done(out: Path) -> set[str]:
    if not out.exists():
        return set()
    return {json.loads(line)["paper_id"] for line in out.read_text(encoding="utf-8").splitlines()}


async def _main(args: argparse.Namespace) -> int:
    settings = Settings()
    llm = StructuredLLM.from_settings(settings)
    done = _done(args.out)
    gate = asyncio.Semaphore(args.concurrency)
    async with HttpClient(settings) as http, closing_llm(llm):

        async def one(path: Path) -> None:
            paper = _load(path)
            if paper.paper_id in done:
                return
            ctx = RunContext(
                settings=settings,
                paper=paper,
                http=http,
                llm=llm,
                logger=logging.getLogger(),
                review_llm=llm.review,
            )
            async with gate:
                tasks = await triage_tasks(
                    paper,
                    ctx.require_review_llm(),
                    limit=settings.theory_deep_per_paper,
                    max_output=settings.theory_triage_max_output,
                )
                candidates: list[CandidateFinding] = []
                for task in tasks:
                    candidates += (await check(task, ctx)).findings
                verified = [
                    VerifiedFinding(
                        finding=f, state=VerificationState.NEEDS_HUMAN_REVIEW, reason="candidate"
                    )
                    for f in candidates
                ]
                kept = [
                    v
                    for v in await review_math(verified, ctx)
                    if v.state is not VerificationState.DROPPED
                ]
            record = {
                "paper_id": paper.paper_id,
                "suspects": sum(len(t.claims or ()) for t in tasks),
                "candidates": len(candidates),
                "kept": [json.loads(v.finding.model_dump_json()) for v in kept],
            }
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(paper.paper_id, record["suspects"], len(candidates), len(kept), flush=True)

        await asyncio.gather(*(one(p) for p in args.papers))
    print(f"cost ${llm.spend.spent_usd:.3f}, {llm.calls} calls")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("papers", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    return asyncio.run(_main(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
