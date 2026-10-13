from typing import Any

from auditor.parser.accuracy import context, identity, score
from auditor.schemas import ParsedPaper
from auditor.schemas.examples import example_paper


def gold(paper: ParsedPaper, mention_id: str, *bib_ids: str) -> dict[str, Any]:
    mention = next(m for m in paper.citations if m.id == mention_id)
    return {
        "paper_id": paper.paper_id,
        "marker": mention.marker,
        "context": context(paper, mention.sentence_id, mention.start, mention.end),
        "expected": [identity(paper.bib_entry(b)) for b in bib_ids] or None,
    }


def test_all_links_correct() -> None:
    paper = example_paper()
    items = [gold(paper, "c0", "b0"), gold(paper, "c1", "b1", "b2")]
    result = score({paper.paper_id: paper}, items)
    assert (result["precision"], result["recall"]) == (1.0, 1.0)


def test_a_marker_missing_from_the_parse_counts_as_a_miss() -> None:
    paper = example_paper()
    items = [gold(paper, "c0", "b0"), gold(paper, "c1", "b1", "b2")]
    without_c1 = paper.model_copy(
        update={"citations": tuple(m for m in paper.citations if m.id != "c1")}
    )
    result = score({paper.paper_id: without_c1}, items)
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["not_found"] == [f"{paper.paper_id} '[2, 3]'"]


def test_a_partial_link_is_wrong() -> None:
    paper = example_paper()
    items = [gold(paper, "c1", "b1")]
    result = score({paper.paper_id: paper}, items)
    assert result["correct_links"] == 0
    assert result["wrong"] == [f"{paper.paper_id} '[2, 3]'"]


def test_unlinked_marker_lowers_recall_only() -> None:
    paper = example_paper()
    item = gold(paper, "c2")
    item["expected"] = [identity(paper.bib_entry("b0"))]
    result = score({paper.paper_id: paper}, [gold(paper, "c0", "b0"), item])
    assert (result["precision"], result["recall"]) == (1.0, 0.5)
    assert result["unlinked"] == [f"{paper.paper_id} '[7]'"]


def test_identity_reads_the_printed_year_not_an_arxiv_id() -> None:
    entry = (
        example_paper()
        .bib_entry("b0")
        .model_copy(
            update={"raw": "Vaswani et al. Attention is all you need. arXiv:1706.03762, June 2017."}
        )
    )
    assert identity(entry)["year"] == "2017"
