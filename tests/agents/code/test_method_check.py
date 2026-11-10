import dataclasses
from pathlib import Path
from typing import Any

from auditor.agents.code import check
from auditor.agents.code.method_check import MethodReview, PairJudgement, _Chunk, method_pairs
from auditor.infra.context import RunContext
from auditor.infra.llm import StructuredLLM
from auditor.planner.tasks import method_tasks
from auditor.schemas import (
    AuditTask,
    DocumentAnchor,
    ErrorClass,
    EvidenceKind,
    ParsedPaper,
    RepositoryRef,
    TaskOutcome,
)
from auditor.schemas.examples import example_paper
from auditor.testing import ScriptedChatModel, offline_context, tool_reply

LOSS = "We train with the focal loss with gamma set to two, which down-weights easy examples."
CODE = (
    "import torch\n\n\ndef loss_fn(logits, target):\n"
    "    return torch.nn.functional.cross_entropy(logits, target)\n"
)


def method_paper() -> ParsedPaper:
    data = example_paper().model_dump()
    data["sections"] = [*data["sections"], {"id": "sec3", "title": "Method", "number": "3"}]
    data["sentences"][4]["text"] = LOSS
    data["sentences"][4]["section_id"] = "sec3"
    return ParsedPaper.model_validate(data)


def repo(tmp_path: Path) -> str:
    root = tmp_path / "repo"
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.py").write_text(CODE, encoding="utf-8")
    (root / "README.md").write_text("# Focal loss demo\n", encoding="utf-8")
    return root.resolve().as_uri()


def task(paper: ParsedPaper, url: str) -> AuditTask:
    (only,) = method_tasks(paper, RepositoryRef(url=url))
    return only


def context(
    tmp_path: Path, paper: ParsedPaper, replies: list[dict[str, Any]]
) -> tuple[RunContext, ScriptedChatModel]:
    model = ScriptedChatModel(replies=[tool_reply(MethodReview, reply) for reply in replies])
    llm = StructuredLLM(model, name="test:scripted")
    ctx = offline_context(tmp_path, paper=paper, llm=llm)
    settings = ctx.settings.model_copy(update={"method_check_mode": "whole"})
    return dataclasses.replace(ctx, settings=settings), model


DISCREPANCY = {
    "paper_quote": "We train with the focal loss with gamma set to two",
    "code_path": "model.py",
    "code_start_line": 4,
    "code_end_line": 5,
    "description": "The paper uses focal loss; the code uses plain cross-entropy.",
    "category": "Loss",
    "confidence": "high",
}


def test_a_method_task_per_paper_with_a_repository() -> None:
    paper = method_paper()
    (only,) = method_tasks(paper, RepositoryRef(url="https://github.com/o/r"))
    assert only.setting is not None
    assert only.setting.key == "method"
    assert only.anchor.sentence_id == "s4"  # type: ignore[union-attr]
    assert method_tasks(paper, None) == []


async def test_quoted_discrepancy_becomes_a_method_mismatch(tmp_path: Path) -> None:
    paper = method_paper()
    ctx, model = context(tmp_path, paper, [{"discrepancies": [DISCREPANCY]}])
    result = await check(task(paper, repo(tmp_path)), ctx)
    assert result.outcome is TaskOutcome.CHECKED
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.METHOD_MISMATCH
    assert finding.anchor == DocumentAnchor()
    assert not finding.uncertain
    quote, code = finding.evidence
    assert quote.kind is EvidenceKind.QUOTE
    assert quote.text == LOSS
    assert code.kind is EvidenceKind.CODE
    assert (code.path, code.line_start, code.line_end) == ("model.py", 4, 5)
    assert "cross_entropy" in code.text
    prompt = " ".join(str(m.content) for m in model.received[0])
    assert "model.py" in prompt
    assert LOSS in prompt


async def test_unquoted_or_unlocated_claims_are_dropped(tmp_path: Path) -> None:
    paper = method_paper()
    invented = {**DISCREPANCY, "paper_quote": "We use a hinge loss."}
    elsewhere = {**DISCREPANCY, "code_path": "missing.py"}
    past_end = {**DISCREPANCY, "code_start_line": 40, "code_end_line": 41}
    ctx, _ = context(tmp_path, paper, [{"discrepancies": [invented, elsewhere, past_end]}])
    result = await check(task(paper, repo(tmp_path)), ctx)
    assert result.outcome is TaskOutcome.CHECKED
    assert result.findings == ()


async def test_low_confidence_is_uncertain(tmp_path: Path) -> None:
    paper = method_paper()
    ctx, _ = context(
        tmp_path, paper, [{"discrepancies": [{**DISCREPANCY, "confidence": "medium"}]}]
    )
    (finding,) = (await check(task(paper, repo(tmp_path)), ctx)).findings
    assert finding.uncertain


async def test_without_an_llm_the_method_is_unverifiable(tmp_path: Path) -> None:
    paper = method_paper()
    result = await check(task(paper, repo(tmp_path)), offline_context(tmp_path, paper=paper))
    assert result.outcome is TaskOutcome.UNVERIFIABLE
    assert result.reason is not None
    assert "LLM" in result.reason


async def test_short_quotes_do_not_count_as_evidence(tmp_path: Path) -> None:
    paper = method_paper()
    ctx, _ = context(
        tmp_path, paper, [{"discrepancies": [{**DISCREPANCY, "paper_quote": "focal loss"}]}]
    )
    result = await check(task(paper, repo(tmp_path)), ctx)
    assert result.findings == ()


# --- pair mode: one small judgement per paper sentence and code span -------------------


def pairs_context(
    tmp_path: Path, paper: ParsedPaper, replies: list[dict[str, Any]]
) -> tuple[RunContext, ScriptedChatModel]:
    model = ScriptedChatModel(replies=[tool_reply(PairJudgement, reply) for reply in replies])
    llm = StructuredLLM(model, name="test:scripted")
    ctx = offline_context(tmp_path, paper=paper, llm=llm)
    settings = ctx.settings.model_copy(update={"method_check_mode": "pairs"})
    return dataclasses.replace(ctx, settings=settings), model


async def test_pairs_mode_judges_small_snippets(tmp_path: Path) -> None:
    paper = method_paper()
    verdict = {
        "verdict": "inconsistent",
        "paper_quote": "We train with the focal loss with gamma set to two",
        "code_start_line": 4,
        "code_end_line": 5,
        "description": "Cross-entropy instead of focal loss.",
        "confidence": "high",
    }
    ctx, model = pairs_context(tmp_path, paper, [verdict])
    result = await check(task(paper, repo(tmp_path)), ctx)
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.METHOD_MISMATCH
    code = finding.evidence[1]
    assert (code.path, code.line_start, code.line_end) == ("model.py", 4, 5)
    (prompt,) = model.received
    sent = " ".join(str(m.content) for m in prompt)
    assert "def loss_fn" in sent
    assert len(sent) < 4000  # one sentence and one function, not whole files


async def test_pairs_mode_drops_lines_outside_the_snippet(tmp_path: Path) -> None:
    paper = method_paper()
    verdict = {
        "verdict": "inconsistent",
        "paper_quote": "We train with the focal loss with gamma set to two",
        "code_start_line": 1,
        "code_end_line": 1,
        "description": "Imports torch.",
        "confidence": "high",
    }
    ctx, _ = pairs_context(tmp_path, paper, [verdict])
    assert (await check(task(paper, repo(tmp_path)), ctx)).findings == ()


# --- compute budget per section ---------------------------------------------------------


def two_section_paper() -> ParsedPaper:
    data = example_paper().model_dump()
    data["sections"] = [
        {"id": "m", "title": "Model", "number": "3"},
        {"id": "t", "title": "Training details", "number": "4"},
    ]
    model = [
        f"The encoder layer {i} computes h_{i} = softmax(W_{i} x + b_{i}) over attention scores."
        for i in range(6)
    ]
    training = [
        "The optimizer schedule warms up slowly and then decays the learning rate steadily.",
        "The trainer saves a checkpoint after every epoch of training on the corpus.",
    ]
    data["sentences"] = [
        {"id": f"m{i}", "section_id": "m", "text": text} for i, text in enumerate(model)
    ] + [{"id": f"t{i}", "section_id": "t", "text": text} for i, text in enumerate(training)]
    data["citations"] = []
    data["links"] = []
    data["tables"] = []
    return ParsedPaper.model_validate(data)


def chunk(name: str, line: int, words: str) -> _Chunk:
    return _Chunk(name, line, line + 5, frozenset(words.split()))


def test_the_pair_budget_is_shared_by_section_density() -> None:
    paper = two_section_paper()
    chunks = [
        # Training chunks share more words with their sentences than the model chunks, so
        # a budget by overlap alone would spend every call on the training section.
        chunk("train.py", 1, "optimizer schedule warms slowly decays learning rate steadily"),
        chunk("train.py", 20, "trainer saves checkpoint every epoch training corpus"),
        chunk("model.py", 1, "encoder layer computes softmax"),
        chunk("model.py", 20, "encoder attention scores"),
        chunk("model.py", 40, "layer softmax attention"),
    ]
    pairs = method_pairs(paper, chunks, 3)
    sections = sorted(p.sentence.section_id or "" for p in pairs)
    assert sections == ["m", "m", "t"]


def test_every_section_with_matching_code_gets_a_pair_when_the_budget_allows() -> None:
    paper = two_section_paper()
    chunks = [
        chunk("model.py", 1, "encoder layer computes softmax"),
        chunk("model.py", 20, "encoder attention scores"),
        chunk("train.py", 1, "trainer saves checkpoint"),
    ]
    pairs = method_pairs(paper, chunks, 2)
    assert {p.sentence.section_id for p in pairs} == {"m", "t"}


def test_an_unused_share_goes_to_the_next_best_pairs() -> None:
    paper = two_section_paper()
    chunks = [
        chunk("model.py", 1, "encoder layer computes softmax"),
        chunk("model.py", 20, "encoder attention scores"),
        chunk("model.py", 40, "layer softmax attention"),
        chunk("train.py", 1, "trainer saves checkpoint"),
    ]
    assert len(method_pairs(paper, chunks, 4)) == 4


def rag_context(
    tmp_path: Path, paper: ParsedPaper, replies: list[dict[str, Any]]
) -> tuple[RunContext, ScriptedChatModel]:
    ctx, model = context(tmp_path, paper, replies)
    settings = ctx.settings.model_copy(update={"method_check_mode": "rag", "embed_url": None})
    return dataclasses.replace(ctx, settings=settings), model


async def test_rag_mode_shows_the_code_each_sentence_retrieves(tmp_path: Path) -> None:
    paper = method_paper()
    ctx, model = rag_context(tmp_path, paper, [{"discrepancies": [DISCREPANCY]}])
    result = await check(task(paper, repo(tmp_path)), ctx)
    (finding,) = result.findings
    assert finding.error_class is ErrorClass.METHOD_MISMATCH
    code = [e for e in finding.evidence if e.kind is EvidenceKind.CODE]
    assert code[0].path == "model.py"
    prompt = " ".join(str(m.content) for m in model.received[0])
    # Without an embeddings endpoint the code is ranked by BM25, shown with line numbers.
    assert "(bm25)" in prompt
    assert "    5|     return torch.nn.functional.cross_entropy" in prompt


async def test_rag_mode_drops_a_discrepancy_on_lines_it_did_not_show(tmp_path: Path) -> None:
    paper = method_paper()
    elsewhere = {**DISCREPANCY, "code_start_line": 40, "code_end_line": 41}
    ctx, _ = rag_context(tmp_path, paper, [{"discrepancies": [elsewhere]}])
    assert (await check(task(paper, repo(tmp_path)), ctx)).findings == ()


def test_a_quote_that_runs_past_its_sentence_is_anchored_on_where_it_starts() -> None:
    from auditor.agents.code.method_check import quoted_sentence

    paper = method_paper()
    across = "We train with the focal loss with gamma set to two, as follows: where p is"
    found = quoted_sentence(paper, across)
    assert found is not None
    assert found.text == LOSS
    # A quote the paper does not contain is still refused.
    assert quoted_sentence(paper, "the weights are drawn from a heavy tailed prior always") is None


async def test_evidence_guards_against_start_line_below_one(tmp_path: Path) -> None:
    from auditor.agents.code.method_check import Discrepancy, _evidence
    from auditor.infra.snapshots import fetch_snapshot

    paper = method_paper()
    url = repo(tmp_path)
    ctx = offline_context(tmp_path, paper=paper)
    snapshot = await fetch_snapshot(RepositoryRef(url=url), ctx.http, ctx.settings)
    disc = Discrepancy.model_construct(
        paper_quote=LOSS,
        code_path="model.py",
        code_start_line=0,
        code_end_line=2,
        description="test",
        category="Loss",
        confidence="high",
    )
    assert _evidence(paper, snapshot, ["model.py"], disc) is None


async def test_method_check_deduplicates_overlapping_code_spans(tmp_path: Path) -> None:
    paper = method_paper()
    d1 = {**DISCREPANCY, "code_start_line": 4, "code_end_line": 5, "description": "Desc 1"}
    d2 = {
        **DISCREPANCY,
        "code_start_line": 5,
        "code_end_line": 5,
        "description": "Desc 2 on overlap",
    }
    ctx, _ = context(tmp_path, paper, [{"discrepancies": [d1, d2]}])
    result = await check(task(paper, repo(tmp_path)), ctx)
    assert result.outcome is TaskOutcome.CHECKED
    assert len(result.findings) == 1
    assert result.findings[0].summary == "Desc 1"


async def test_method_check_deduplicates_identical_summaries(tmp_path: Path) -> None:
    paper = method_paper()
    d1 = {
        **DISCREPANCY,
        "code_start_line": 1,
        "code_end_line": 1,
        "description": "The paper uses focal loss; the code uses plain cross-entropy.",
    }
    d2 = {
        **DISCREPANCY,
        "code_start_line": 4,
        "code_end_line": 5,
        "description": "  The paper uses focal loss; the code uses plain cross-entropy.  ",
    }
    ctx, _ = context(tmp_path, paper, [{"discrepancies": [d1, d2]}])
    result = await check(task(paper, repo(tmp_path)), ctx)
    assert result.outcome is TaskOutcome.CHECKED
    assert len(result.findings) == 1
    assert result.findings[0].evidence[1].line_start == 1


async def test_rag_mode_deduplicates_overlapping_discrepancies(tmp_path: Path) -> None:
    paper = method_paper()
    d1 = {**DISCREPANCY, "code_start_line": 4, "code_end_line": 5, "description": "RAG Desc 1"}
    d2 = {**DISCREPANCY, "code_start_line": 4, "code_end_line": 4, "description": "RAG Desc 2"}
    ctx, _ = rag_context(tmp_path, paper, [{"discrepancies": [d1, d2]}])
    result = await check(task(paper, repo(tmp_path)), ctx)
    assert result.outcome is TaskOutcome.CHECKED
    assert len(result.findings) == 1
    assert result.findings[0].summary == "RAG Desc 1"
