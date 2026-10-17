from auditor.parser.layout_tables import add_layout_tables
from auditor.schemas import Box, ParsedPaper, ParserInfo, Sentence, SentenceKind

PAGE_WITH_TABLE = """\
Our results are below.
        Table 1: Comparison of detection accuracy.
        Model              V2XSet           OPV2V          # Params
        No Fusion       0.754 / 0.602    0.659 / 0.545        -
        Late Fusion     0.885 / 0.764    0.852 / 0.768      6.58M
The proposed method improves on every baseline in the table above, as the text explains.
"""
PAGE_WITH_AN_IMAGE_TABLE = """\
        Table 2: Ablation of the sub-modules.
  (the table is an image)
"""
PAGE_WITH_MERGED_CELLS = """\
 Table 3: Token pruning results.
 Method                  GQA         SQAIMG      POPE       MME
 BAT + Inverse          58.4\u22120.2 69.1\u22120.2 85.0\u22120.3          1734\u22123
 BAT + Ours             58.8+0.2 69.4+0.1 86.8+0.9          1782+45
"""
PROSE = "A page of prose with numbers like 3 and 4 but no table caption at all.\n"
LEFT = [
    "mutation is applied if random number is less than",
    "probability of mutation. Different types of",
    "mutations are used in this study:",
    "a) Mutation on Boolean operator: randomly",
    "exchanging one operator to another.",
    "b) Mutation on term node (leaf node): in",
    "Boolean model one term is selected randomly",
    "from the offspring and replace by any other",
    "one from the terms in a given collection of",
    "documents. But in fuzzy model the term is",
    "not replaced, only the term weight is changed.",
    "c) Mutation by inserting or deleting operator",
    "between two nodes in the offsprings, as shown.",
    "As a result we create six different strategies",
    "for the Boolean and fuzzy models, which are the",
    "ones the experiments below compare with care.",
]
RIGHT = [
    "the second column of the page carries on here",
    "Table 1: Results with precision as the fitness.",
    "Recall      Traditional      GA(M1)      GA(M3)",
    "0.1         0.156            0.161       0.146",
    "0.2         0.162            0.162       0.187",
    "0.3         0.166            0.167       0.174",
    "Average     0.19777          0.19377     0.19744",
    "and more text of the right column follows the",
    "table, as running text in a two-column paper",
    "usually does, line after line after line here",
    "so that the page has enough prose to show the",
    "two columns and where the second one starts.",
    "The results for the strategies are shown in",
    "the table, and the text discusses them below",
    "with the improvements over the user query of",
    "every strategy, one strategy after the other.",
]
TWO_COLUMNS = "\n".join(f"  {a:<50}    {b}" for a, b in zip(LEFT, RIGHT, strict=True))


def paper(*extra: Sentence) -> ParsedPaper:
    return ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sentences=(Sentence(id="s1", text="Results are in Table 1."), *extra),
    )


def rows_of(out: ParsedPaper, index: int = 0) -> list[str]:
    by_id = {s.id: s for s in out.sentences}
    return [by_id[r].text for r in out.tables[index].row_ids]


def test_rows_under_a_table_caption_become_a_layout_table() -> None:
    out = add_layout_tables(paper(), [PAGE_WITH_TABLE, PROSE])
    (table,) = out.tables
    assert (table.page, table.label) == (1, "Table 1")
    rows = rows_of(out)
    assert rows[0] == "Model | V2XSet | OPV2V | # Params"
    assert "No Fusion | 0.754 / 0.602 | 0.659 / 0.545 | -" in rows
    by_id = {s.id: s for s in out.sentences}
    assert all(by_id[r].kind is SentenceKind.TABLE for r in table.row_ids)


def test_a_table_beside_the_other_columns_text_keeps_only_its_own_column() -> None:
    out = add_layout_tables(paper(), [TWO_COLUMNS])
    (table,) = out.tables
    assert table.label == "Table 1"
    rows = rows_of(out)
    assert rows[0] == "Recall | Traditional | GA(M1) | GA(M3)"
    assert rows[-1] == "Average | 0.19777 | 0.19377 | 0.19744"
    assert not any("mutation" in row.lower() for row in rows)


def test_a_captioned_page_without_readable_rows_is_left_unread() -> None:
    out = add_layout_tables(paper(), [PROSE, PAGE_WITH_AN_IMAGE_TABLE])
    (table,) = out.tables
    assert (table.page, table.label, table.row_ids) == (2, "Table 2", ())


def test_a_table_whose_columns_run_together_is_left_unread() -> None:
    out = add_layout_tables(paper(), [PAGE_WITH_MERGED_CELLS])
    (table,) = out.tables
    assert (table.page, table.label, table.row_ids) == (1, "Table 3", ())


def test_without_layout_text_the_caption_pages_are_left_unread() -> None:
    caption = Sentence(
        id="c1",
        kind=SentenceKind.CAPTION,
        text="Table 3: Results on the test set.",
        boxes=(Box(page=4, x=0, y=0, width=10, height=10),),
    )
    out = add_layout_tables(paper(caption), None)
    (table,) = out.tables
    assert (table.page, table.label, table.row_ids) == (4, "Table 3", ())


def test_a_caption_set_beside_running_text_is_found() -> None:
    page = (
        "In this paper, we explored an approach to full- Table 6: Human ratings of the\n"
        "duplex conversation centered on causal rea- rationales across audio sources.\n"
    )
    out = add_layout_tables(paper(), [page])
    (table,) = out.tables
    assert (table.label, table.row_ids) == ("Table 6", ())


def test_a_reference_to_a_table_in_running_text_is_not_a_caption() -> None:
    out = add_layout_tables(paper(), ["The results are shown in Table 2. They are good.\n"])
    assert out.tables == ()
