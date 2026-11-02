from collections import Counter
from pathlib import Path

import pytest

from auditor.infra.llm import StructuredLLM
from auditor.orchestration.graph import AuditPipeline, run_audit
from auditor.planner.claims import (
    ClaimCandidate,
    ClaimJudgements,
    citation_candidates,
    quote_in,
    refine_candidates,
)
from auditor.planner.repository import find_repository, repository_url, snapshot_ref
from auditor.planner.tasks import PlannerOptions, citation_tasks, make_planner, plan_paper
from auditor.schemas import (
    BibTarget,
    Link,
    LinkSource,
    MarkerTarget,
    RepositoryRef,
    SystemName,
    TaskKind,
    TaskOutcome,
)
from auditor.schemas.examples import example_paper, example_venue
from auditor.testing import ScriptedChatModel, offline_context, tool_reply


def scripted(*judgement_lists: list[dict[str, object]]) -> tuple[StructuredLLM, ScriptedChatModel]:
    model = ScriptedChatModel(
        replies=[tool_reply(ClaimJudgements, {"judgements": j}) for j in judgement_lists]
    )
    return StructuredLLM(model, name="test:scripted"), model


def test_every_mention_and_linked_entry_becomes_a_candidate() -> None:
    candidates = citation_candidates(example_paper())
    assert [(c.mention_id, c.bib_id, c.marker) for c in candidates] == [
        ("c0", "b0", "[1]"),
        ("c1", "b1", "[2, 3]"),
        ("c1", "b2", "[2, 3]"),
        ("c2", None, "[7]"),
    ]
    assert all(c.claim == example_paper().sentence(c.sentence_id).text for c in candidates)


async def test_refinement_narrows_claims_and_marks_check_worthiness() -> None:
    paper = example_paper()
    candidates = citation_candidates(paper)
    llm, model = scripted(
        [
            {"candidate": 0, "check_worthy": True, "claim": "replaced recurrent networks"},
            {"candidate": 1, "check_worthy": True, "claim": "92.1% exact match on SQuAD"},
            {"candidate": 2, "check_worthy": False, "claim": "Prior work"},
            {"candidate": 3, "check_worthy": True, "claim": "an invented quote"},
            {"candidate": 9, "check_worthy": False, "claim": "unknown candidate"},
        ]
    )
    refined = await refine_candidates(paper, candidates, llm)
    assert [(c.claim, c.check_worthy) for c in refined] == [
        ("replaced recurrent networks", True),
        ("92.1% exact match on SQuAD", True),
        ("Prior work", False),
        (paper.sentence("s3").text, True),
    ]
    prompt = str(model.received[0][-1].content)
    assert "Candidate 3" in prompt
    assert "not linked to the bibliography" in prompt
    assert "Attention Is All You Need (2017)" in prompt


def spans(sentence: str, *markers: str) -> list[tuple[int, int]]:
    return [(sentence.index(m), sentence.index(m) + len(m)) for m in markers]


# From the live run on arXiv 2005.11401: Gemini left markers out of most of its quotes.
REALM = (
    "REALM [20] and ORQA [31], two recently introduced models that combine masked language "
    "models [8] with a differentiable retriever, have shown promising results"
)


def test_a_quote_without_the_citation_markers_is_the_sentence_text_with_them() -> None:
    markers = spans(REALM, "[20]", "[31]", "[8]")
    claim = "REALM and ORQA, two recently introduced models that combine masked language models"
    assert quote_in(REALM, claim, markers) == REALM[: REALM.index(" [8]")]
    # Only some markers left out.
    partly = "REALM [20] and ORQA, two recently introduced models"
    assert (
        quote_in(REALM, partly, markers)
        == "REALM [20] and ORQA [31], two recently introduced models"
    )


def test_a_quote_that_is_not_in_the_sentence_is_refused() -> None:
    markers = spans(REALM, "[20]", "[31]", "[8]")
    assert quote_in(REALM, "REALM and ORQA ... promising results", markers) is None
    assert quote_in(REALM, "REALM and ORQA, two new models", markers) is None
    assert quote_in(REALM, "", markers) is None
    # Without the parser's marker spans, only an exact quote counts.
    assert quote_in(REALM, "REALM and ORQA", []) is None
    assert quote_in(REALM, "ORQA [31]", []) == "ORQA [31]"


async def test_refinement_runs_in_batches() -> None:
    paper = example_paper()
    candidates = citation_candidates(paper)
    llm, model = scripted(
        [{"candidate": 0, "check_worthy": True, "claim": "Transformers"}],
        [{"candidate": 0, "check_worthy": True, "claim": "Prior work"}],
    )
    refined = await refine_candidates(paper, candidates, llm, batch_size=2)
    assert len(model.received) == 2
    assert refined[0].claim == "Transformers"
    assert refined[2].claim == "Prior work"


def test_repository_urls_are_canonicalised() -> None:
    assert repository_url("https://github.com/owner/name.git") == "https://github.com/owner/name"
    assert repository_url("https://www.github.com/owner/name/") == "https://github.com/owner/name"
    assert repository_url("https://github.com/owner/name#readme") == "https://github.com/owner/name"
    assert repository_url("https://github.com/owner") is None
    assert repository_url("https://example.org/owner/name") is None


@pytest.mark.parametrize(
    ("url", "ref"),
    [
        ("https://github.com/owner/name/tree/0123abc", "0123abc"),
        ("https://github.com/owner/name/commit/0123abc", "0123abc"),
        ("https://github.com/owner/name.git/tree/v1.0", "v1.0"),
        ("https://gitlab.com/owner/name/-/tree/v1.0/src", "v1.0"),
        ("https://github.com/owner/name/blob/main/train.py", "main"),
        ("https://github.com/owner/name", None),
        ("https://github.com/owner/name/issues/3", None),
    ],
)
def test_snapshot_ref_reads_the_commit_a_link_names(url: str, ref: str | None) -> None:
    assert snapshot_ref(url) == ref


def test_repository_next_to_code_words_is_preferred() -> None:
    paper = example_paper()
    extra = Link(url="https://github.com/someone/else", page=1, source=LinkSource.ANNOTATION)
    data = paper.model_dump()
    data["links"] = [extra.model_dump(), *data["links"]]
    paper = type(paper).model_validate(data)
    assert find_repository(paper) == RepositoryRef(url="https://github.com/example/auditor-demo")


def test_detected_repository_keeps_the_snapshot_its_link_names() -> None:
    paper = example_paper()
    links = tuple(
        link.model_copy(update={"url": link.url + "/tree/0123abc"})
        if link.url == "https://github.com/example/auditor-demo"
        else link
        for link in paper.links
    )
    found = find_repository(paper.model_copy(update={"links": links}))
    assert found == RepositoryRef(url="https://github.com/example/auditor-demo", commit="0123abc")


def test_no_repository_without_repository_links() -> None:
    data = example_paper().model_dump()
    data["links"] = []
    assert find_repository(type(example_paper()).model_validate(data)) is None


async def test_plan_without_llm_covers_every_kind(tmp_path: Path) -> None:
    paper = example_paper()
    tasks = await plan_paper(
        paper, offline_context(tmp_path), PlannerOptions(venue=example_venue())
    )
    kinds = Counter(task.kind for task in tasks)
    assert kinds == {
        TaskKind.REFERENCE: 3,
        TaskKind.CITATION: 4,
        TaskKind.CODE: 4,  # two settings, the method check and the code-defect check
        TaskKind.TEMPLATE: 3,
        TaskKind.INTEGRITY: 2,
    }
    for task in tasks:
        paper.validate_anchor(task.anchor)
    assert len({t.task_id for t in tasks}) == len(tasks)
    targets = {t.target for t in tasks if t.kind is TaskKind.CITATION}
    assert MarkerTarget(marker="[7]") in targets
    code = [t for t in tasks if t.kind is TaskKind.CODE]
    assert {t.repository.url for t in code if t.repository} == {
        "https://github.com/example/auditor-demo"
    }


def test_different_claims_about_one_work_in_a_sentence_stay_separate() -> None:
    paper = example_paper()

    def candidate(mention_id: str, claim: str) -> ClaimCandidate:
        return ClaimCandidate(
            mention_id=mention_id, sentence_id="s1", bib_id="b0", marker="[1]", claim=claim
        )

    tasks = citation_tasks(
        paper,
        [
            candidate("c0", "Transformers"),
            candidate("c9", "replaced recurrent networks"),
            candidate("c10", "Transformers"),
        ],
    )
    ids = [t.task_id for t in tasks]
    assert ids[0] != ids[1]
    assert ids[0] == ids[2]


async def test_plan_skips_claims_judged_not_check_worthy(tmp_path: Path) -> None:
    llm, _ = scripted(
        [
            {"candidate": 0, "check_worthy": False, "claim": "Transformers"},
            {"candidate": 1, "check_worthy": True, "claim": "92.1% exact match on SQuAD"},
            {"candidate": 2, "check_worthy": True, "claim": "92.1% exact match on SQuAD"},
            {"candidate": 3, "check_worthy": True, "claim": "Retrieval"},
        ]
    )
    ctx = offline_context(tmp_path, llm=llm)
    tasks = await plan_paper(example_paper(), ctx, PlannerOptions(detect_repository=False))
    citations = [t for t in tasks if t.kind is TaskKind.CITATION]
    assert BibTarget(bib_id="b0") not in {t.target for t in citations}
    assert {t.claim for t in citations} == {"92.1% exact match on SQuAD", "Retrieval"}
    assert not [t for t in tasks if t.kind is TaskKind.CODE]


async def test_planner_runs_inside_the_audit_graph(tmp_path: Path) -> None:
    pipeline = AuditPipeline(
        ctx=offline_context(tmp_path),
        planner=make_planner(PlannerOptions(venue=example_venue())),
    )
    outcome = await run_audit(pipeline, example_paper(), run_id="r1", system=SystemName.FULL)
    assert len(outcome.tasks) == 16
    assert len(outcome.results) == 16
    assert all(r.outcome is TaskOutcome.CHECKED for r in outcome.results)


@pytest.mark.parametrize(
    ("sentence", "found"),
    [
        ("Code is available at https://github.com/a/b.", True),
        ("We provide our implementations at https://github.com/a/b.", True),
        ("Code: https://github.com/a/b", True),
        ("Scripts to run experiments can be found at https://github.com/a/b.", True),
        ("The original dataset is from https://github.com/a/b.", False),
        ("Our implementation is based on the open-sourced code in https://github.com/a/b.", False),
        ("https://github.com/a/b", False),
    ],
)
def test_only_sentences_announcing_code_give_the_repository(sentence: str, found: bool) -> None:
    data = example_paper().model_dump()
    data["sentences"][5]["text"] = sentence
    data["links"] = [
        {
            "url": "https://github.com/a/b/tree/main/src",
            "page": 2,
            "sentence_id": "s5",
            "source": "text",
        }
    ]
    paper = type(example_paper()).model_validate(data)
    expected = RepositoryRef(url="https://github.com/a/b") if found else None
    assert find_repository(paper) == expected


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://anonymous.4open.science/r/paper-code-1A2B/README.md",
            RepositoryRef(url="https://anonymous.4open.science/r/paper-code-1A2B"),
        ),
        (
            "https://gitlab.com/group/sub/proj",
            RepositoryRef(url="https://gitlab.com/group/sub/proj"),
        ),
        # Model and data hosts are not "our code", even in a sentence announcing code.
        ("https://huggingface.co/org/model", None),
        ("https://zenodo.org/records/42", None),
    ],
)
def test_code_hosts_in_announcing_sentences_give_the_repository(
    url: str, expected: RepositoryRef | None
) -> None:
    data = example_paper().model_dump()
    data["sentences"][5]["text"] = f"Our code is available at {url}."
    data["links"] = [{"url": url, "page": 2, "sentence_id": "s5", "source": "text"}]
    assert find_repository(type(example_paper()).model_validate(data)) == expected
