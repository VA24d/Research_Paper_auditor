from pathlib import Path
from typing import Any

from auditor.agents.code import check
from auditor.agents.code.method_check import MethodReview
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
    return offline_context(tmp_path, paper=paper, llm=llm), model


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
