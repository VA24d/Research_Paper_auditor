from pathlib import Path

import pytest

from auditor.agents.template import check
from auditor.agents.template.fixtures import build_paper
from auditor.agents.template.rules import RULE_CHECKS, params_error
from auditor.experiments import load_venue
from auditor.planner.tasks import template_tasks
from auditor.schemas import TaskOutcome, VenueProfile
from auditor.testing import offline_context

VENUES = Path(__file__).resolve().parents[3] / "configs" / "venues"
PROFILES = sorted(VENUES.glob("*.json"))


def test_the_main_venues_have_profiles() -> None:
    assert {path.stem for path in PROFILES} >= {
        "neurips-2025",
        "iclr-2026",
        "icml-2025",
        "acl-2025",
        "emnlp-2025",
    }


@pytest.mark.parametrize("path", PROFILES, ids=[path.stem for path in PROFILES])
def test_profile_loads_and_every_rule_is_implemented(path: Path) -> None:
    venue = load_venue(path)
    assert isinstance(venue, VenueProfile)
    assert venue.id == path.stem
    assert venue.rules
    for rule in venue.rules:
        assert rule.check in RULE_CHECKS, rule.id
        assert params_error(rule.check, rule.params) is None, rule.id
        # Every limit records where it was verified.
        assert "https://" in rule.description, rule.id


@pytest.mark.parametrize("path", PROFILES, ids=[path.stem for path in PROFILES])
def test_profiles_are_documented(path: Path) -> None:
    readme = (VENUES / "README.md").read_text(encoding="utf-8")
    assert path.name in readme


@pytest.mark.parametrize("path", PROFILES, ids=[path.stem for path in PROFILES])
async def test_profile_runs_on_a_paper(path: Path, tmp_path: Path) -> None:
    venue = load_venue(path)
    assert venue is not None
    paper = build_paper()
    ctx = offline_context(tmp_path, paper=paper)
    for task in template_tasks(paper, venue):
        result = await check(task, ctx)
        assert task.rule is not None
        if result.outcome is TaskOutcome.UNVERIFIABLE:
            assert task.rule.needs_visual_inspection or task.rule.check == "anonymity"
