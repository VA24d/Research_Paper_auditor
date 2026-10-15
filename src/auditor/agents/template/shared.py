"""The template checker's shared tool functions (decision D7).

The specialist and B1's tools call these same functions, so every rule check is counted
as a tool call in both systems (decision D8).
"""

from collections.abc import Mapping

from pydantic import JsonValue

from auditor.agents.template.rules import RuleReport, describe_layout, evaluate
from auditor.infra.tools import tracked_tool
from auditor.schemas import ParsedPaper


@tracked_tool()
async def check_template_rule(
    paper: ParsedPaper, check: str, params: Mapping[str, JsonValue]
) -> RuleReport:
    """Run the deterministic check `check` with `params` on the paper."""
    return evaluate(paper, check, params)


@tracked_tool()
async def layout_summary(paper: ParsedPaper) -> str:
    """Page count, page sizes, smallest margins, body font size, main-text pages and
    section titles."""
    return describe_layout(paper)


def format_report(check: str, report: RuleReport) -> str:
    """A rule report as text, for B1's tool replies."""
    if report.unverifiable is not None:
        lines = [f"Rule could not be checked ({check}): {report.unverifiable}"]
    elif report.violations:
        lines = [f"Rule violated ({check})."]
    else:
        lines = [f"Rule passed ({check})."]
    if report.measured:
        lines.append(f"Measured: {report.measured}")
    for violation in report.violations:
        hint = " (uncertain)" if violation.uncertain else ""
        pages = f" Pages: {', '.join(map(str, violation.pages))}." if violation.pages else ""
        lines.append(f"- {violation.summary}{hint}{pages}")
    return "\n".join(lines)
