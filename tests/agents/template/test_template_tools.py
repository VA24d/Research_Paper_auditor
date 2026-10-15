from pathlib import Path

from auditor.agents.template.fixtures import build_paper
from auditor.agents.template.tools import make_tools, parse_params
from auditor.orchestration import registry
from auditor.testing import offline_context


def tools_by_name(tmp_path: Path) -> dict[str, object]:
    ctx = offline_context(tmp_path, paper=build_paper())
    return {tool.name: tool for tool in make_tools(ctx)}


async def test_check_template_rule_takes_the_rule_as_shown_in_the_prompt(tmp_path: Path) -> None:
    [check_tool] = [
        t
        for t in make_tools(offline_context(tmp_path, paper=build_paper()))
        if t.name == "check_template_rule"
    ]
    text = await check_tool.ainvoke({"check": "max_pages", "params": '{"max_pages": 8}'})
    assert "violated" in text
    assert "9 main-text pages" in text
    text = await check_tool.ainvoke({"check": "max_pages", "params": "max_pages=9"})
    assert "passed" in text


async def test_bad_tool_arguments_come_back_as_messages(tmp_path: Path) -> None:
    [check_tool] = [
        t
        for t in make_tools(offline_context(tmp_path, paper=build_paper()))
        if t.name == "check_template_rule"
    ]
    text = await check_tool.ainvoke({"check": "max_pages", "params": "{not json"})
    assert "could not read" in text.lower()
    text = await check_tool.ainvoke({"check": "kerning", "params": "{}"})
    assert "could not be checked" in text


async def test_layout_summary(tmp_path: Path) -> None:
    [summary] = [
        t
        for t in make_tools(offline_context(tmp_path, paper=build_paper()))
        if t.name == "layout_summary"
    ]
    text = await summary.ainvoke({})
    assert "612 x 792" in text
    assert "Sections:" in text


def test_tools_are_registered_for_b1(tmp_path: Path) -> None:
    names = {tool.name for tool in registry.load_tools(offline_context(tmp_path))}
    assert {"check_template_rule", "layout_summary"} <= names
    assert set(tools_by_name(tmp_path)) == {"check_template_rule", "layout_summary"}


def test_params_are_read_as_json_or_as_rendered_in_the_prompt() -> None:
    assert parse_params('{"max_pages": 9}') == {"max_pages": 9}
    rendered = "max_pages=9, exclude_sections=['references', 'appendix'], strict=True"
    assert parse_params(rendered) == {
        "max_pages": 9,
        "exclude_sections": ["references", "appendix"],
        "strict": True,
    }
    assert parse_params("width=612.0, height=792") == {"width": 612.0, "height": 792}
    assert parse_params("") == {}
