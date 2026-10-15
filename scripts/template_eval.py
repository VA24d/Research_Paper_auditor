"""Run the template checker on its fixture cases and print per-rule accuracy (P3.9).

Each case is a hand-built paper with one rule and the expected conclusion: the rule passes,
is violated (with the expected page or document anchor), or cannot be checked.

    uv run python scripts/template_eval.py
"""

import asyncio
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from auditor.agents.template import check
from auditor.agents.template.fixtures import judge, observed, template_cases
from auditor.planner.tasks import template_tasks
from auditor.schemas import VenueProfile
from auditor.testing import offline_context


async def evaluate(scratch: Path) -> int:
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    wrong: list[str] = []
    for case in template_cases():
        venue = VenueProfile(id="fixtures", name="Template fixtures", rules=(case.rule,))
        [task] = template_tasks(case.paper, venue)
        result = await check(task, offline_context(scratch, paper=case.paper))
        right = judge(case, result)
        totals[case.rule.check][0] += right
        totals[case.rule.check][1] += 1
        if not right:
            wrong.append(f"- {case.name}: expected {case.expected}, got {observed(result)}")
    print("| Check | Cases | Correct | Accuracy |\n|---|---|---|---|")
    for name, (correct, cases) in sorted(totals.items()):
        print(f"| {name} | {cases} | {correct} | {100 * correct / cases:.0f}% |")
    correct = sum(c for c, _ in totals.values())
    cases = sum(n for _, n in totals.values())
    print(f"| all | {cases} | {correct} | {100 * correct / cases:.0f}% |")
    if wrong:
        print("\nWrong:\n" + "\n".join(wrong))
    return 1 if wrong else 0


def main() -> int:
    with tempfile.TemporaryDirectory() as scratch:
        return asyncio.run(evaluate(Path(scratch)))


if __name__ == "__main__":
    sys.exit(main())
