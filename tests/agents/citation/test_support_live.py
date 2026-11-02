"""A live regression test for the citation agent's support judgement (decision P2.5).

Marked `llm` so the default suite skips it; run with `pytest tests -m llm --override... `
(see pyproject's marker setup). Costs ONE small model call (`effort="low"`, ~1k tokens).

    .\\.venv\\Scripts\\python.exe -m pytest tests\\agents\\citation\\test_support_live.py -m llm -q
"""

from pathlib import Path

import pytest

from auditor.agents.citation.support import check_citation
from auditor.config import Settings
from auditor.infra.llm import StructuredLLM, create_chat_model, model_id
from auditor.retrieval.clients import Work
from auditor.retrieval.fulltext import SourceLevel, SourceText
from auditor.retrieval.resolve import Resolution
from auditor.schemas import (
    AuditTask,
    BibTarget,
    ErrorClass,
    EvidenceKind,
    SentenceAnchor,
    TaskKind,
    TaskOutcome,
)
from auditor.schemas.examples import example_paper
from auditor.testing import offline_context

SOURCE_TEXT = (
    "The Transformer reaches 28.4 BLEU on WMT 2014 English to German. " * 60
    + "Recurrence and convolution were the dominant sequence models before it. " * 60
)
WORK = Work(id="arxiv:1706.03762", source="arxiv", title="Attention Is All You Need")
ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


def citation_task(claim: str) -> AuditTask:
    paper = example_paper()
    return AuditTask.create(
        paper_id=paper.paper_id,
        variant=paper.variant,
        kind=TaskKind.CITATION,
        anchor=SentenceAnchor(sentence_id="s2"),
        target=BibTarget(bib_id="b1"),
        claim=claim,
    )


@pytest.mark.llm
async def test_support_judgement_against_the_live_model(tmp_path: Path) -> None:
    # The suite's isolated_environment fixture clears AUDITOR_* and chdirs away, so the
    # .env is passed explicitly (read into Settings only, never printed or logged).
    settings = Settings(llm_cache=False, _env_file=ENV_FILE if ENV_FILE.exists() else None)
    if not settings.llm_provider or not settings.llm_model:
        pytest.skip("no LLM configured (AUDITOR_LLM_PROVIDER/MODEL)")
    llm = StructuredLLM(create_chat_model(settings), name=model_id(settings))
    ctx = offline_context(tmp_path, llm=llm)
    # The claim attributes 51.0 BLEU to the cited work; the canned source states 28.4,
    # so the code-side number check must produce a mismatch finding whose evidence
    # carries a quote anchored in the source text.
    task = citation_task("The Transformer reaches 51.0 BLEU on WMT 2014 English to German. [1]")
    result = await check_citation(
        task,
        ctx,
        Resolution(work=WORK),
        SourceText(work=WORK, level=SourceLevel.FULL_HTML, text=SOURCE_TEXT),
    )
    assert result.outcome is TaskOutcome.CHECKED
    classes = {finding.error_class for finding in result.findings}
    assert classes & {ErrorClass.CLAIM_NUMBER_MISMATCH, ErrorClass.CLAIM_NOT_SUPPORTED}
    quotes = [
        evidence
        for finding in result.findings
        for evidence in finding.evidence
        if evidence.kind is EvidenceKind.QUOTE
    ]
    assert quotes, "no quote came back from the live model"
    anchored = any(quote.char_start is not None or "28.4" in quote.text for quote in quotes)
    assert anchored, f"the quote is not from the canned source: {[q.text[:80] for q in quotes]}"
