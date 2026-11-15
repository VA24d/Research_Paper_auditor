"""Offline tests for the set B builder's proposal verification (scripts/set_b_build.py).

The number-injection verification must reject a proposal whose recorded source does not
state the claim's ORIGINAL value (the D18 second review found such a label had slipped
through: TSPLIB's abstract states no numbers at all).
"""

import importlib.util
import sys
from pathlib import Path

from auditor.agents.citation.set_b import InjectionKind, Proposal
from auditor.schemas import SentenceAnchor
from auditor.schemas.examples import example_paper
from auditor.testing import offline_context

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("set_b_build", REPO / "scripts" / "set_b_build.py")
assert _spec is not None
assert _spec.loader is not None
set_b_build = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("set_b_build", set_b_build)
_spec.loader.exec_module(set_b_build)

PAPER = example_paper()


def number_proposal() -> Proposal:
    sentence = PAPER.sentence("s2")
    start = sentence.text.index("92.1%")
    return Proposal(
        kind=InjectionKind.CLAIM_NUMBER_MISMATCH,
        bib_id="b1",
        sentence_id="s2",
        anchor=SentenceAnchor(sentence_id="s2", start=start, end=start + len("92.1%")),
        new_value="88.4",
    )


async def test_number_proposal_rejected_when_the_source_states_the_perturbed_value(
    tmp_path: Path,
) -> None:
    ctx = offline_context(tmp_path / "cache")
    problem = await set_b_build._verify(
        number_proposal(),
        PAPER,
        "Prior work reports 88.4 exact match. Nothing else is stated.",
        ctx.http,
    )
    assert problem == "the perturbed value is also stated by the source"


async def test_number_proposal_rejected_when_the_source_lacks_the_original_value(
    tmp_path: Path,
) -> None:
    ctx = offline_context(tmp_path / "cache")
    problem = await set_b_build._verify(
        number_proposal(),
        PAPER,
        "The dataset holds 100,000 questions. No accuracy is stated anywhere.",
        ctx.http,
    )
    assert problem == "the source does not state the claim's original value"


async def test_number_proposal_accepted_when_the_source_states_the_original_only(
    tmp_path: Path,
) -> None:
    ctx = offline_context(tmp_path / "cache")
    problem = await set_b_build._verify(
        number_proposal(),
        PAPER,
        "Prior work reports 92.1% exact match on the SQuAD benchmark.",
        ctx.http,
    )
    assert problem is None


async def test_a_fabricated_title_is_not_accepted_while_the_indexes_are_down(
    tmp_path: Path,
) -> None:
    """ "Nothing resolves" is only verified when an index answered the search."""
    import httpx

    from auditor.config import Settings
    from auditor.infra.http import HttpClient

    async def no_sleep(_seconds: float) -> None:
        return None

    proposal = Proposal(
        kind=InjectionKind.REFERENCE_NOT_FOUND,
        bib_id="b0",
        sentence_id=None,
        anchor=None,
        new_value="A Fabricated Study of Nothing Real",
    )
    for status, expected in (
        (503, "the indexes could not be asked whether the fabricated title resolves"),
        (404, None),
    ):
        http = HttpClient(
            Settings(_env_file=None, cache_dir=tmp_path / str(status)),
            transport=httpx.MockTransport(lambda request, s=status: httpx.Response(s)),
            sleep=no_sleep,
        )
        assert await set_b_build._verify(proposal, PAPER, None, http) == expected
