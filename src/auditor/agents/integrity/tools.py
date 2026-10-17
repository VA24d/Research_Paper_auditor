"""Tools for baseline B1 (decision D7). The integrity screen reads only the paper, which B1
already has in its prompt, so there is nothing to add."""

from langchain_core.tools import BaseTool

from auditor.infra.context import RunContext


def make_tools(ctx: RunContext) -> list[BaseTool]:
    return []
