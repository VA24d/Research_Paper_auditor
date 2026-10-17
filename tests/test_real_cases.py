import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load() -> ModuleType:
    path = Path(__file__).parent.parent / "scripts" / "real_cases.py"
    spec = importlib.util.spec_from_file_location("real_cases", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["real_cases"] = module
    spec.loader.exec_module(module)
    return module


real_cases = _load()

FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2002.12856v2</id>
    <title>A  Paper
      Title</title>
    <arxiv:comment>Withdrawn due to an error in the proof of Lemma 3.4</arxiv:comment>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2001.00001v1</id>
    <title>First version</title>
    <arxiv:comment>Withdrawn due to an error in the proof of Lemma 2</arxiv:comment>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2001.00002v2</id>
    <title>Copyright</title>
    <arxiv:comment>Withdrawn for copyright reasons, error in table</arxiv:comment>
  </entry>
</feed>"""


def test_parse_arxiv_points_at_the_version_before_the_fix() -> None:
    found = list(real_cases.parse_arxiv(FEED))
    assert [c.case_id for c in found] == ["arxiv:2002.12856v1"]
    candidate = found[0]
    assert candidate.pdf_url == "https://arxiv.org/pdf/2002.12856v1"
    assert candidate.paper_title == "A Paper Title"
    assert candidate.error_type == "math_error"
    assert candidate.location == "Lemma 3.4"


def test_classify() -> None:
    assert real_cases.classify("Results in Table 2 are incorrect") == "result_inconsistency"
    assert real_cases.classify("hallucinated references throughout") == "reference_not_found"
    assert real_cases.classify("exceeds the page limit") == "template_violation"
    assert real_cases.classify("thanks to the reviewers") is None


def test_clean_keeps_the_suggestive_phrase() -> None:
    text = "x " * 500 + "the proof of Lemma 2 is wrong " + "y " * 500
    cleaned = real_cases.clean(text)
    assert len(cleaned) <= real_cases.QUOTE_LIMIT
    assert "proof of Lemma 2 is wrong" in cleaned


def test_parse_openreview_reads_venue_notes_only() -> None:
    submission = json.loads('{"id": "abc", "forum": "abc", "content": {"title": {"value": "T"}}}')
    replies = json.loads(
        """[
        {"id": "abc", "invitations": ["x/-/Submission"], "content": {}},
        {"id": "n1", "invitations": ["x/Submission1/-/Desk_Rejection"],
         "content": {"desk_reject_comments": {"value": "This paper exceeds the page limit."}}},
        {"id": "n2", "invitations": ["x/Submission1/-/Official_Comment"],
         "content": {"comment": {"value": "The proof is wrong."}}}
        ]"""
    )
    found = list(real_cases.parse_openreview(submission, replies))
    assert [c.error_type for c in found] == ["template_violation"]
    assert found[0].paper_url == "https://openreview.net/forum?id=abc"
    assert found[0].location == "page limit"
