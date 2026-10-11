from collections.abc import Sequence

from auditor.parser.linking import link_unresolved
from auditor.schemas import (
    BibEntry,
    CitationMention,
    LinkMethod,
    ParsedPaper,
    ParserInfo,
    Sentence,
)

ENTRIES = (
    BibEntry(id="b0", authors=("Ashish Vaswani",), year=2017),
    BibEntry(id="b1", authors=("John Smith",), year=2020),
    BibEntry(id="b2", authors=("Jane Smith",), year=2021),
    BibEntry(id="b3", authors=("María Núñez",), year=2019),
    BibEntry(id="b4", authors=("Wei Chen",), year=2022),
    BibEntry(id="b5", authors=("Wei Chen",), year=2022),
)


def paper(markers: Sequence[tuple[str, tuple[str, ...]]]) -> ParsedPaper:
    sentences = []
    mentions = []
    for index, (marker, bib_ids) in enumerate(markers):
        text = f"See {marker} here."
        sentences.append(Sentence(id=f"s{index}", text=text))
        mentions.append(
            CitationMention(
                id=f"c{index}",
                sentence_id=f"s{index}",
                start=4,
                end=4 + len(marker),
                marker=marker,
                bib_ids=bib_ids,
                link_method=LinkMethod.PARSER if bib_ids else LinkMethod.NONE,
            )
        )
    return ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sentences=tuple(sentences),
        bibliography=ENTRIES,
        citations=tuple(mentions),
    )


def links(result: ParsedPaper) -> dict[str, tuple[tuple[str, ...], LinkMethod]]:
    return {c.marker: (c.bib_ids, c.link_method) for c in result.citations}


NUMBERED = [("[1]", ("b0",)), ("[2]", ("b1",)), ("[3]", ("b2",))]


def test_numeric_markers_are_linked_when_numbering_is_consistent() -> None:
    result = link_unresolved(paper([*NUMBERED, ("[4]", ()), ("[5-6]", ())]))
    assert links(result)["[4]"] == (("b3",), LinkMethod.HEURISTIC)
    assert links(result)["[5-6]"] == (("b4", "b5"), LinkMethod.HEURISTIC)
    assert links(result)["[1]"] == (("b0",), LinkMethod.PARSER)


def test_en_dash_ranges_are_understood() -> None:
    marker = "[4" + chr(0x2013) + "5]"
    result = link_unresolved(paper([*NUMBERED, (marker, ())]))
    assert links(result)[marker] == (("b3", "b4"), LinkMethod.HEURISTIC)


def test_numeric_markers_stay_unlinked_without_enough_evidence() -> None:
    result = link_unresolved(paper([("[1]", ("b0",)), ("[4]", ())]))
    assert links(result)["[4]"] == ((), LinkMethod.NONE)


def test_numeric_markers_stay_unlinked_when_numbering_disagrees() -> None:
    shifted = [("[1]", ("b1",)), ("[2]", ("b2",)), ("[3]", ("b3",)), ("[4]", ())]
    assert links(link_unresolved(paper(shifted)))["[4]"] == ((), LinkMethod.NONE)


def test_number_past_the_bibliography_is_not_linked() -> None:
    result = link_unresolved(paper([*NUMBERED, ("[40]", ())]))
    assert links(result)["[40]"] == ((), LinkMethod.NONE)


def test_author_year_marker_with_unique_match_is_linked() -> None:
    result = link_unresolved(paper([("(Smith et al., 2020)", ()), ("Nunez (2019)", ())]))
    assert links(result)["(Smith et al., 2020)"] == (("b1",), LinkMethod.HEURISTIC)
    assert links(result)["Nunez (2019)"] == (("b3",), LinkMethod.HEURISTIC)


def test_ambiguous_author_year_marker_is_left_alone() -> None:
    result = link_unresolved(paper([("Chen et al. (2022)", ()), ("Smith (2019)", ())]))
    assert links(result)["Chen et al. (2022)"] == ((), LinkMethod.NONE)
    assert links(result)["Smith (2019)"] == ((), LinkMethod.NONE)


def test_paper_without_unresolved_markers_is_returned_unchanged() -> None:
    original = paper(NUMBERED)
    assert link_unresolved(original) is original


LETTERED = (
    # Parsed without a year, as GROBID did for this entry in BERT's bibliography.
    BibEntry(
        id="b0", authors=("Matthew Peters",), raw="Matthew Peters. 2018a. Deep contextualized."
    ),
    BibEntry(
        id="b1", authors=("Matthew Peters",), year=2018, raw="Matthew Peters. 2018b. Dissecting."
    ),
    BibEntry(id="b2", authors=("Wenlong Huang",), year=2022, raw="W. Huang. Planners. 2022a."),
    BibEntry(id="b3", authors=("Wenlong Huang",), year=2022, raw="W. Huang. Monologue. 2022b."),
    BibEntry(id="b4", authors=("Shunyu Yao",), year=2020, raw="S. Yao. Keep calm. 2020."),
    BibEntry(id="b5", authors=("James Thorne",), year=2018, raw="J. Thorne. FEVER. 2018."),
)


def lettered(markers: Sequence[str]) -> dict[str, tuple[str, ...]]:
    base = paper([(m, ()) for m in markers])
    data = base.model_dump()
    data["bibliography"] = [e.model_dump() for e in LETTERED]
    result = link_unresolved(ParsedPaper.model_validate(data))
    return {c.marker: c.bib_ids for c in result.citations}


def test_year_letters_must_match_the_reference_text() -> None:
    links = lettered(["(Peters et al., 2018a)", "Peters et al. (2018b)", "Huang et al. (2022b)"])
    assert links == {
        "(Peters et al., 2018a)": ("b0",),
        "Peters et al. (2018b)": ("b1",),
        "Huang et al. (2022b)": ("b3",),
    }


def test_letter_without_a_matching_entry_is_not_guessed() -> None:
    # The BERT failure: only the 2018b entry carried a year, so "2018a" used to link to it.
    data = paper([("(Peters et al., 2018a)", ())]).model_dump()
    data["bibliography"] = [LETTERED[1].model_dump()]
    result = link_unresolved(ParsedPaper.model_validate(data))
    assert result.citations[0].bib_ids == ()


def test_year_without_letter_is_ambiguous_when_entries_have_letters() -> None:
    assert lettered(["Huang et al. (2022)"]) == {"Huang et al. (2022)": ()}


def test_every_part_of_a_multi_citation_marker_must_resolve() -> None:
    links = lettered(
        [
            "(Yao et al., 2020;Huang et al., 2022a)",
            "(Huang et al., 2022a,b)",
            "(Yao et al., 2020; Nobody et al., 2019)",
        ]
    )
    assert links == {
        "(Yao et al., 2020;Huang et al., 2022a)": ("b4", "b2"),
        "(Huang et al., 2022a,b)": ("b2", "b3"),
        "(Yao et al., 2020; Nobody et al., 2019)": (),
    }


def test_surname_is_taken_from_before_et_al() -> None:
    assert lettered(["(Fever, Thorne et al., 2018"]) == {"(Fever, Thorne et al., 2018": ("b5",)}


def test_author_count_separates_same_year_works() -> None:
    entries = [
        BibEntry(id="b0", authors=("Sarah Wiegreffe", "Ana Marasovic"), year=2021, raw="2021."),
        BibEntry(
            id="b1",
            authors=("Sarah Wiegreffe", "Ana Marasovic", "Noah Smith"),
            year=2021,
            raw="2021.",
        ),
    ]
    markers = ["(Wiegreffe and Marasovic, 2021)", "Wiegreffe et al. (2021)", "Wiegreffe (2021)"]
    data = paper([(m, ()) for m in markers]).model_dump()
    data["bibliography"] = [e.model_dump() for e in entries]
    result = link_unresolved(ParsedPaper.model_validate(data))
    assert {c.marker: c.bib_ids for c in result.citations} == {
        "(Wiegreffe and Marasovic, 2021)": ("b0",),
        "Wiegreffe et al. (2021)": ("b1",),
        "Wiegreffe (2021)": (),
    }
