"""Template tools for baseline B1 (decision D7).

`RunContext` carries the paper but not the venue: B1 sees the venue rules in its prompt,
rendered as `[rule-id] description (check: name; key=value, ...)`. So
`check_template_rule` takes the check name and its params as shown there, either as a JSON
object or in that `key=value` form.
"""

import ast
import json
import re

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import JsonValue, TypeAdapter, ValidationError

from auditor.agents.template.shared import check_template_rule, format_report, layout_summary
from auditor.infra.context import RunContext

_KEY = re.compile(r"(?:^|,)\s*([A-Za-z_]\w*)\s*=")
_PARAMS = TypeAdapter(dict[str, JsonValue])


def parse_params(text: str) -> dict[str, JsonValue]:
    """Rule params from a JSON object or from `key=value, key=value` text."""
    text = text.strip()
    if not text:
        return {}
    if text.startswith("{"):
        return _PARAMS.validate_python(json.loads(text))
    keys = list(_KEY.finditer(text))
    if not keys or keys[0].start() != 0:
        raise ValueError("expected a JSON object or key=value pairs")
    params: dict[str, JsonValue] = {}
    for key, following in zip(keys, [*keys[1:], None], strict=True):
        end = following.start() if following is not None else len(text)
        params[key[1]] = _value(text[key.end() : end].strip())
    return params


def _value(raw: str) -> JsonValue:
    for parse in (json.loads, ast.literal_eval):
        try:
            value = parse(raw)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, tuple):
            value = list(value)
        try:
            return _PARAMS.validate_python({"v": value})["v"]
        except ValidationError:
            continue
    return raw


def make_tools(ctx: RunContext) -> list[BaseTool]:
    paper = ctx.paper

    async def check_rule(check: str, params: str = "{}") -> str:
        try:
            parsed = parse_params(params)
        except (ValueError, ValidationError) as exc:
            return f"Could not read the params {params!r}: {exc}"
        report = await check_template_rule(paper, check, parsed)
        return format_report(check, report)

    async def summary() -> str:
        return await layout_summary(paper)

    return [
        StructuredTool.from_function(
            coroutine=check_rule,
            name="check_template_rule",
            description=(
                "Check one venue formatting rule on the paper, deterministically. `check` is "
                "the rule's check name and `params` its parameters, both as listed with the "
                'venue rules, e.g. check="max_pages", params=\'{"max_pages": 9}\' or '
                'params="max_pages=9". Returns passed, violated (with the measured value '
                "and pages) or could not be checked (with the reason)."
            ),
        ),
        StructuredTool.from_function(
            coroutine=summary,
            name="layout_summary",
            description=(
                "The paper's layout measurements: page count, page sizes, smallest margins, "
                "body font size, main-text pages and section titles."
            ),
        ),
    ]
