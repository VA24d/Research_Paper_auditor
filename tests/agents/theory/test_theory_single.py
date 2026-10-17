from auditor.agents.theory.single import SingleReview, check_paper
from auditor.infra.llm import StructuredLLM
from auditor.schemas import ErrorClass, ParsedPaper, ParserInfo, Section, Sentence
from auditor.testing import ScriptedChatModel, tool_reply

LEMMA = "Lemma 2. For every n >= 1 we have S(n) = n^2 / 2 exactly."


def paper() -> ParsedPaper:
    return ParsedPaper(
        paper_id="p",
        parser=ParserInfo(name="test", version="1"),
        sections=(Section(id="main", title="Sums"),),
        sentences=(Sentence(id="s1", section_id="main", text=LEMMA),),
    )


async def test_one_call_gives_math_errors_that_pass_the_evidence_gate() -> None:
    issues = {
        "errors": [
            {
                "quote": "we have S(n) = n^2 / 2 exactly",
                "location": "Lemma 2",
                "explanation": "S(n) = n(n+1)/2.",
                "witness": "n = 1 gives 1/2, not 1.",
                "confidence": "high",
            },
            {"quote": "we have S(n) = n^2 / 2 exactly", "witness": ""},
            {"quote": "a sentence that is not in the paper at all", "witness": "w"},
        ]
    }
    model = ScriptedChatModel(replies=[tool_reply(SingleReview, issues)])
    (only,) = await check_paper(
        paper(), StructuredLLM(model, name="t"), run_id="r", effort="medium"
    )
    assert only.error_class is ErrorClass.MATH_ERROR
    assert only.summary.startswith("Lemma 2:")
    assert only.state is None
    assert f"[s1] {LEMMA}" in " ".join(str(m.content) for m in model.received[0])
