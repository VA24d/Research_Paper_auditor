"""The UI's audit helpers: the command it runs, the limits it sets, progress and reports."""

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from auditor.config import Settings
from auditor.schemas import LLMCallRecord
from auditor.ui.audit import (
    AuditRequest,
    command,
    environment,
    new_run_id,
    progress,
    report_of,
    reports,
    save_upload,
    tail,
)


def test_the_command_is_the_cli_audit_with_the_requested_inputs(tmp_path: Path) -> None:
    request = AuditRequest(
        paper=tmp_path / "p.json",
        repository="https://github.com/o/r",
        commit="abc",
        venue=tmp_path / "v.json",
        use_llm=False,
    )
    assert command(request, "ui-1") == [
        sys.executable,
        "-m",
        "auditor.cli",
        "audit",
        str(tmp_path / "p.json"),
        "--run-id",
        "ui-1",
        "--repo",
        "https://github.com/o/r",
        "--repo-commit",
        "abc",
        "--venue",
        str(tmp_path / "v.json"),
        "--no-llm",
    ]
    # A commit without a repository is not passed on (the CLI would refuse it).
    assert "--repo-commit" not in command(AuditRequest(paper=Path("p"), commit="abc"), "ui-1")


def test_the_environment_sets_the_model_and_both_spend_limits(tmp_path: Path) -> None:
    request = AuditRequest(
        paper=Path("p.json"),
        model="gemini-2.5-flash",
        project="proj",
        max_cost_usd=0.25,
        ledger=tmp_path / "ledger.jsonl",
        ledger_limit_usd=3.0,
        skip_citation=True,
    )
    env = environment(request, base={"KEEP": "1"})
    assert env["KEEP"] == "1"
    assert env["AUDITOR_LLM_MODEL"] == "gemini-2.5-flash"
    assert json.loads(env["AUDITOR_LLM_OPTIONS"])["project"] == "proj"
    assert env["AUDITOR_LLM_MAX_COST_USD"] == "0.25"
    assert env["AUDITOR_LLM_LEDGER"] == str(tmp_path / "ledger.jsonl")
    assert env["AUDITOR_LLM_LEDGER_LIMIT_USD"] == "3"
    assert json.loads(env["AUDITOR_SKIP_PARTS"]) == ["citation"]
    assert env["AUDITOR_CITATION_LLM_ENABLED"] == "false"
    assert "AUDITOR_EMBED_URL" not in env


def test_without_the_model_no_model_settings_are_added() -> None:
    env = environment(AuditRequest(paper=Path("p"), use_llm=False, skip_citation=False), base={})
    assert not any(key.startswith("AUDITOR_LLM") for key in env)
    assert json.loads(env["AUDITOR_SKIP_PARTS"]) == []


def test_progress_counts_calls_and_cost_from_the_usage_log(tmp_path: Path) -> None:
    def call(hit: bool) -> str:
        return LLMCallRecord(
            run_id="ui-1",
            model="google_genai:test-model",
            input_tokens=1_000_000,
            output_tokens=100_000,
            started_at=datetime(2026, 11, 8, tzinfo=UTC),
            duration_s=1.0,
            status="ok",  # type: ignore[arg-type]
            cache_hit=hit,
        ).model_dump_json()

    log = tmp_path / "usage" / "llm_calls.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(call(False) + "\n" + call(True) + "\n" + '{"partial', encoding="utf-8")
    settings = Settings(llm_prices={"test-model": (1.0, 10.0)})
    now = progress(tmp_path, settings)
    assert (now.calls, now.cost_usd, now.input_tokens) == (1, 2.0, 1_000_000)
    assert progress(tmp_path / "missing", settings).calls == 0


def test_reports_are_found_newest_first(tmp_path: Path) -> None:
    old = tmp_path / "ui-1" / "p-original" / "report.json"
    new = tmp_path / "ui-2" / "p-injected" / "report.json"
    for path, stamp in ((old, 1_000_000), (new, 2_000_000)):
        path.parent.mkdir(parents=True)
        path.write_text("{}", encoding="utf-8")
        os.utime(path, (stamp, stamp))
    assert reports(tmp_path) == [new, old]
    assert report_of(tmp_path / "ui-2") == new
    assert report_of(tmp_path / "ui-3") is None


def test_uploads_are_stored_once_by_content(tmp_path: Path) -> None:
    first = save_upload("Paper.PDF", b"%PDF", tmp_path)
    again = save_upload("other.pdf", b"%PDF", tmp_path)
    assert first == again
    assert first.suffix == ".pdf"
    assert first.read_bytes() == b"%PDF"


def test_tail_drops_the_logging_prefix() -> None:
    lines = [
        "2026-11-08 03:32:21,123 WARNING auditor.x [ui-1 code-1] embeddings failed\n",
        "plain line\n",
    ]
    assert tail(lines) == "WARNING embeddings failed\nplain line"


def test_run_ids_sort_by_time() -> None:
    assert new_run_id(datetime(2026, 11, 8, 3, 4, 5, tzinfo=UTC)) == "ui-20261108-030405"
