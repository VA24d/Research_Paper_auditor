import pytest
from pydantic import JsonValue

from auditor.agents.template.fixtures import APPENDIX, LETTER, MAIN_SECTIONS, build_paper
from auditor.agents.template.rules import (
    RULE_CHECKS,
    describe_layout,
    dominant_font_size,
    evaluate,
    main_text_pages,
    params_error,
)


def test_main_text_ends_before_the_references() -> None:
    measured = main_text_pages(build_paper())
    assert measured is not None
    assert measured.pages == 9
    assert measured.located


def test_letter_numbered_sections_count_as_appendix() -> None:
    measured = main_text_pages(build_paper(bibliography_pages=()))
    assert measured is not None
    assert measured.pages == 9
    assert "Proofs" in measured.boundary


def test_roman_numerals_are_not_an_appendix() -> None:
    roman = [
        (title, numeral, pages)
        for (title, _, pages), numeral in zip(MAIN_SECTIONS, ("I", "II", "III", "IV"), strict=True)
    ]
    measured = main_text_pages(build_paper(sections=roman, bibliography_pages=()))
    assert measured is not None
    assert measured.pages == 9
    assert not measured.located


def test_pages_after_the_references_start_do_not_count() -> None:
    undetected = (*MAIN_SECTIONS, ("Extra results", None, (11, 12)))
    measured = main_text_pages(build_paper(sections=undetected))
    assert measured is not None
    assert measured.pages == 9


def test_excluded_sections_are_configurable() -> None:
    sections = (*MAIN_SECTIONS, ("Limitations", None, (10,)))
    paper = build_paper(sections=sections, bibliography_pages=(11,))
    counted = main_text_pages(paper)
    excluded = main_text_pages(paper, ("limitations", "references"))
    assert counted is not None
    assert excluded is not None
    assert counted.pages == 10
    assert excluded.pages == 9
    assert "Limitations" in excluded.boundary


def test_page_limit_reports_the_measured_pages() -> None:
    report = evaluate(build_paper(), "max_pages", {"max_pages": 8})
    assert report.unverifiable is None
    [violation] = report.violations
    assert "9" in violation.measured
    assert not violation.pages
    assert not violation.uncertain
    assert evaluate(build_paper(), "max_pages", {"max_pages": 9}).passed


def test_page_limit_is_uncertain_when_the_end_of_the_main_text_is_unknown() -> None:
    sections = [(title, None, pages) for title, _, pages in (*MAIN_SECTIONS, *APPENDIX)]
    report = evaluate(
        build_paper(sections=sections, bibliography_pages=()), "max_pages", {"max_pages": 9}
    )
    [violation] = report.violations
    assert violation.uncertain
    assert "12" in violation.measured


def test_page_limit_without_positions_falls_back_to_the_page_count() -> None:
    short = build_paper(with_boxes=False, page_count=8)
    report = evaluate(short, "max_pages", {"max_pages": 9})
    assert report.passed
    assert "8" in report.measured

    long = build_paper(with_boxes=False, page_count=14)
    report = evaluate(long, "max_pages", {"max_pages": 9})
    assert report.unverifiable is not None
    assert "14" in report.unverifiable
    assert not report.violations

    report = evaluate(
        build_paper(with_boxes=False, with_layout=False), "max_pages", {"max_pages": 9}
    )
    assert report.unverifiable is not None


def test_dominant_font_size_is_weighted_by_characters() -> None:
    paper = build_paper(fonts=(("A", 9.0, 100), ("B", 10.0, 300), ("C", 10.0, 50)))
    assert paper.layout is not None
    dominant = dominant_font_size(paper.layout)
    assert dominant is not None
    assert dominant[0] == 10.0
    assert dominant[1] == pytest.approx(350 / 450)


def test_font_size_has_a_small_tolerance() -> None:
    near = build_paper(fonts=(("Times", 9.95, 1000),))
    assert evaluate(near, "min_body_font_size", {"min_size": 10}).passed
    small = build_paper(fonts=(("Times", 9.0, 1000), ("Times", 10.0, 10)))
    [violation] = evaluate(small, "min_body_font_size", {"min_size": 10}).violations
    assert "9" in violation.measured


def test_font_size_needs_font_measurements() -> None:
    report = evaluate(build_paper(fonts=()), "min_body_font_size", {"min_size": 10})
    assert report.unverifiable is not None
    report = evaluate(build_paper(with_layout=False), "min_body_font_size", {"min_size": 10})
    assert report.unverifiable is not None


def test_paper_size_accepts_landscape_pages_and_flags_other_sizes() -> None:
    landscape = {5: {"width": LETTER[1], "height": LETTER[0]}}
    params = {"width": 612, "height": 792}
    assert evaluate(build_paper(page_overrides=landscape), "paper_size", params).passed
    a4 = build_paper(page_size=(595.0, 842.0), page_count=3, sections=(), bibliography_pages=())
    [violation] = evaluate(a4, "paper_size", params).violations
    assert violation.pages == (1, 2, 3)
    assert "595" in violation.measured


def test_margins_skip_pages_without_text_and_measure_each_side() -> None:
    empty = {2: {"text_left": None, "text_top": None, "text_right": None, "text_bottom": None}}
    params = {"min_left": 108, "min_right": 108, "min_top": 72, "min_bottom": 72}
    assert evaluate(build_paper(page_overrides=empty), "margins", params).passed

    wide = build_paper(page_overrides={4: {"text_right": 530.0}})
    [violation] = evaluate(wide, "margins", params).violations
    assert violation.pages == (4,)
    assert "right 82.0" in violation.measured
    # A side without a minimum is not checked.
    assert evaluate(wide, "margins", {"min_left": 108}).passed


def test_margins_need_at_least_one_side() -> None:
    assert params_error("margins", {}) is not None
    assert params_error("margins", {"min_top": 72}) is None


def test_required_sections_match_titles_case_insensitively() -> None:
    sections = (*MAIN_SECTIONS, ("LIMITATIONS and broader impact", None, (9,)))
    params: dict[str, JsonValue] = {"sections": ["limitations"]}
    assert evaluate(build_paper(sections=sections), "required_sections", params).passed
    [violation] = evaluate(
        build_paper(), "required_sections", {"sections": ["^limitations", "ethics"]}
    ).violations
    assert "^limitations" in violation.measured
    assert "ethics" in violation.measured


def test_required_sections_need_parsed_headings() -> None:
    paper = build_paper(sections=(), bibliography_pages=())
    report = evaluate(paper, "required_sections", {"sections": ["limitations"]})
    assert report.unverifiable is not None


def test_forbidden_sections_are_reported() -> None:
    sections = (*MAIN_SECTIONS, ("Acknowledgements", None, (9,)))
    params: dict[str, JsonValue] = {"sections": ["acknowledg"]}
    [violation] = evaluate(build_paper(sections=sections), "forbidden_sections", params).violations
    assert "Acknowledgements" in violation.measured
    assert evaluate(build_paper(), "forbidden_sections", params).passed


def test_references_can_be_a_section_without_parsed_entries() -> None:
    sections = (*MAIN_SECTIONS, ("References", None, (10,)))
    paper = build_paper(sections=sections, bibliography_pages=())
    assert evaluate(paper, "references_present", {}).passed
    no_refs = build_paper(bibliography_pages=())
    assert len(evaluate(no_refs, "references_present", {}).violations) == 1


def test_anonymity_is_unverifiable_with_uncertain_hints() -> None:
    clean = evaluate(build_paper(authors=["Anonymous Author(s)"]), "anonymity", {})
    assert clean.unverifiable is not None
    assert not clean.violations

    paper = build_paper(
        authors=["Ada Lovelace"],
        links=["https://github.com/ada/engine", "https://anonymous.4open.science/r/x"],
    )
    report = evaluate(paper, "anonymity", {})
    assert report.unverifiable is not None
    assert len(report.violations) == 2
    assert all(v.uncertain for v in report.violations)
    assert any("github.com/ada" in v.measured for v in report.violations)


def test_unknown_checks_and_bad_params_are_unverifiable() -> None:
    report = evaluate(build_paper(), "line_spacing", {})
    assert report.unverifiable is not None
    assert "line_spacing" in report.unverifiable
    report = evaluate(build_paper(), "max_pages", {"max_page": 9})
    assert report.unverifiable is not None
    assert "max_page" in report.unverifiable


def test_layout_rules_need_a_layout() -> None:
    paper = build_paper(with_layout=False)
    for check, params in [
        ("paper_size", {"width": 612, "height": 792}),
        ("margins", {"min_left": 108}),
    ]:
        report = evaluate(paper, check, params)
        assert report.unverifiable is not None
        assert "layout" in report.unverifiable


def test_every_check_validates_its_params() -> None:
    assert set(RULE_CHECKS) >= {
        "max_pages",
        "min_body_font_size",
        "paper_size",
        "margins",
        "required_sections",
        "forbidden_sections",
        "references_present",
        "anonymity",
    }
    assert params_error("max_pages", {"max_pages": 0}) is not None
    assert params_error("required_sections", {"sections": ["("]}) is not None
    assert params_error("unknown", {}) is not None


def test_layout_description_uses_the_measurement_names() -> None:
    text = describe_layout(build_paper())
    assert "Pages: 12" in text
    assert "612 x 792" in text
    assert "left 108.0" in text
    assert "10 pt" in text
    assert "Main text: 9 pages" in text
    assert describe_layout(build_paper(with_layout=False)).startswith("No layout")
