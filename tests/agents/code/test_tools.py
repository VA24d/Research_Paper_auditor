import dataclasses
from pathlib import Path

import pytest
from langchain_core.tools import BaseTool

from auditor.agents.code.tools import make_tools
from auditor.infra.context import RunContext
from auditor.infra.scope import Scope, enter_scope
from auditor.infra.usage import UsageLog
from auditor.schemas import RepositoryRef
from auditor.testing import offline_context

FILES = {
    "train.py": 'import argparse\nparser.add_argument("--lr", default=3e-4)\n',
    "configs/base.yaml": "batch_size: 32\n",
    "data/train.csv": "text,label\na,0\n",
    "data/test.csv": "text,label\na,0\n",
}


def context(tmp_path: Path, data_mode: bool = False) -> RunContext:
    root = tmp_path / "repo"
    for path, text in FILES.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")
    ctx = offline_context(tmp_path)
    settings = ctx.settings.model_copy(update={"data_analysis_enabled": data_mode})
    return dataclasses.replace(
        ctx, settings=settings, repository=RepositoryRef(url=root.resolve().as_uri())
    )


def tools_by_name(ctx: RunContext) -> dict[str, BaseTool]:
    return {tool.name: tool for tool in make_tools(ctx)}


async def test_tools_list_search_read_and_find(tmp_path: Path) -> None:
    tools = tools_by_name(context(tmp_path))
    assert set(tools) == {
        "list_repository_files",
        "search_repository",
        "read_repository_file",
        "find_repository_setting",
        "analyse_data",
    }
    listing = await tools["list_repository_files"].ainvoke({})
    assert "train.py" in listing
    assert "data/train.csv" in listing
    found = await tools["search_repository"].ainvoke({"pattern": "--lr"})
    assert 'train.py:2: parser.add_argument("--lr", default=3e-4)' in found
    text = await tools["read_repository_file"].ainvoke(
        {"path": "train.py", "start_line": 2, "end_line": 2}
    )
    assert text.strip() == '2: parser.add_argument("--lr", default=3e-4)'
    setting = await tools["find_repository_setting"].ainvoke({"key": "batch_size"})
    assert "32 (configs/base.yaml:1)" in setting


async def test_tool_calls_are_counted(tmp_path: Path) -> None:
    ctx = context(tmp_path)
    usage = UsageLog()
    with enter_scope(Scope(run_id="r", usage=usage)):
        await tools_by_name(ctx)["search_repository"].ainvoke({"pattern": "lr"})
    assert [r.tool for r in usage.tool_calls] == ["search_repository"]


async def test_reading_outside_the_snapshot_is_refused(tmp_path: Path) -> None:
    tools = tools_by_name(context(tmp_path))
    text = await tools["read_repository_file"].ainvoke({"path": "../secret"})
    assert "not in the repository snapshot" in text


async def test_data_analysis_tool_follows_the_mode(tmp_path: Path) -> None:
    off = await tools_by_name(context(tmp_path))["analyse_data"].ainvoke({})
    assert "data analysis mode is off" in off.lower()
    on = await tools_by_name(context(tmp_path / "on", data_mode=True))["analyse_data"].ainvoke({})
    assert '"rows": 1' in on
    assert "overlapping_rows" in on


async def test_no_repository(tmp_path: Path) -> None:
    tools = {tool.name: tool for tool in make_tools(offline_context(tmp_path))}
    assert "no code repository" in await tools["list_repository_files"].ainvoke({})


async def test_search_repository_caches_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from auditor.agents.code.tools import _snapshot_lines_cache
    from auditor.infra.snapshots.snapshot import Snapshot

    tools = tools_by_name(context(tmp_path))
    await tools["search_repository"].ainvoke({"pattern": "lr"})
    assert len(_snapshot_lines_cache) > 0

    # On second search, snapshot.read_text is not called because lines are cached
    read_calls = 0
    orig_read_text = Snapshot.read_text

    def tracking_read_text(self: Snapshot, path: str) -> str:
        nonlocal read_calls
        read_calls += 1
        return orig_read_text(self, path)

    monkeypatch.setattr(Snapshot, "read_text", tracking_read_text)
    res = await tools["search_repository"].ainvoke({"pattern": "lr"})
    assert read_calls == 0
    assert "train.py:2:" in res
