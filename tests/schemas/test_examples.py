from pathlib import Path

import pytest
from pydantic import BaseModel, TypeAdapter

from auditor.schemas import (
    AuditTask,
    GoldLabel,
    ParsedPaper,
    Prediction,
    RunManifest,
    TaskResult,
    VenueProfile,
    VerifiedFinding,
)
from auditor.schemas.examples import all_examples, render, write_examples

FIXTURES = Path(__file__).parents[1] / "fixtures" / "schemas"

MODELS: dict[str, TypeAdapter[object]] = {
    "parsed_paper": TypeAdapter(ParsedPaper),
    "venue_profile": TypeAdapter(VenueProfile),
    "audit_tasks": TypeAdapter(list[AuditTask]),
    "task_results": TypeAdapter(list[TaskResult]),
    "verified_findings": TypeAdapter(list[VerifiedFinding]),
    "predictions": TypeAdapter(list[Prediction]),
    "gold_labels": TypeAdapter(list[GoldLabel]),
    "run_manifest": TypeAdapter(RunManifest),
}


def test_every_example_has_a_model() -> None:
    assert set(all_examples()) == set(MODELS)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_fixture_file_is_up_to_date(name: str) -> None:
    on_disk = (FIXTURES / f"{name}.json").read_bytes().decode("utf-8")
    assert on_disk == render(all_examples()[name]), (
        "Regenerate with: uv run python -m auditor.schemas.examples tests/fixtures/schemas"
    )


@pytest.mark.parametrize("name", sorted(MODELS))
def test_fixture_validates_and_round_trips(name: str) -> None:
    raw = (FIXTURES / f"{name}.json").read_bytes()
    adapter = MODELS[name]
    value = adapter.validate_json(raw)
    assert adapter.validate_json(adapter.dump_json(value)) == value


def test_write_examples_uses_lf_line_endings(tmp_path: Path) -> None:
    for path in write_examples(tmp_path):
        assert b"\r\n" not in path.read_bytes()


def test_examples_are_models() -> None:
    for value in all_examples().values():
        items = value if isinstance(value, list) else [value]
        assert all(isinstance(item, BaseModel) for item in items)
