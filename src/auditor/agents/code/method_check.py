"""Checks that the code implements the method the paper describes (`method_mismatch`).

Settings can be compared value by value; a loss, an algorithm step or a data pipeline
cannot. Here the paper's method sections and the repository files most related to them
(retrieved by term overlap) go to the LLM, which lists discrepancies. Each one must quote
the paper verbatim and point at existing lines of a retrieved file; anything else is
dropped before it becomes a finding (evidence gating, as the verifier requires). Findings
are anchored on the document: a method discrepancy is about the paper as a whole.
"""

import re
from collections import Counter
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from auditor.agents.code.data_checks import paper_quote_text
from auditor.infra.context import RunContext
from auditor.infra.snapshots import (
    Snapshot,
    SnapshotUnavailableError,
    SourceError,
    fetch_snapshot,
)
from auditor.schemas import (
    AuditTask,
    CandidateFinding,
    DocumentAnchor,
    ErrorClass,
    Evidence,
    EvidenceKind,
    ParsedPaper,
    SentenceKind,
    SourceKind,
    SourceRef,
    TaskOutcome,
    TaskResult,
)

METHOD_SECTION = re.compile(
    r"method|approach|model|algorithm|framework|architecture|propos|training|implement"
    r"|objective|loss|learning|setup|experiment",
    re.IGNORECASE,
)
MAX_PAPER_CHARS = 15_000
MAX_FILES = 6
MAX_FILE_LINES = 400
MAX_CODE_CHARS = 60_000
# Shorter quotes ("loss") match too many sentences to pin the claim down.
MIN_QUOTE_CHARS = 20
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]{3,}")
_STOPWORDS = frozenset(
    {
        "this",
        "that",
        "with",
        "from",
        "which",
        "where",
        "their",
        "these",
        "each",
        "then",
        "than",
        "when",
        "into",
        "also",
        "such",
        "used",
        "using",
        "have",
        "been",
        "were",
        "they",
        "self",
        "return",
        "import",
    }
)

SYSTEM_PROMPT = """You check whether research code implements the method its paper describes.
You get the paper's method text and the repository files most related to it, with line
numbers. List every place where the code does something different from what the paper says
(a different loss, algorithm step, model component, data processing, training or evaluation
procedure), or where one of them leaves out something the other relies on.

For each discrepancy give:
- paper_quote: a verbatim quote of the paper text you rely on (copy it exactly);
- code_path, code_start_line, code_end_line: the lines of one shown file;
- description: one or two sentences on what differs;
- category: Algorithm, Loss, Model, Data, Training or Evaluation;
- confidence: high only if the quoted lines clearly contradict the quote.

Do not report differences in hyperparameter values, style, or efficiency. An empty list
is a valid answer."""


class Discrepancy(BaseModel):
    paper_quote: str
    code_path: str
    code_start_line: int = Field(ge=1)
    code_end_line: int = Field(ge=1)
    description: str
    category: Literal["Algorithm", "Loss", "Model", "Data", "Training", "Evaluation"]
    confidence: Literal["low", "medium", "high"]


class MethodReview(BaseModel):
    discrepancies: list[Discrepancy] = Field(default_factory=list)


async def check_method(task: AuditTask, ctx: RunContext) -> TaskResult:
    repository = task.repository
    if repository is None:
        raise ValueError(f"{task.task_id} has no repository")
    if ctx.llm is None:
        return TaskResult.for_task(
            task,
            TaskOutcome.UNVERIFIABLE,
            reason="comparing the method with the code needs an LLM, and none is configured",
        )
    try:
        snapshot = await fetch_snapshot(repository, ctx.http, ctx.settings)
    except (SnapshotUnavailableError, SourceError) as exc:
        return TaskResult.for_task(
            task, TaskOutcome.UNVERIFIABLE, reason=f"the repository could not be fetched: {exc}"
        )
    method_text = paper_method_text(ctx.paper)
    files = related_files(snapshot, method_text)
    if not files:
        return TaskResult.for_task(
            task, TaskOutcome.UNVERIFIABLE, reason="the repository has no Python files to compare"
        )
    prompt = (
        f"Paper method text:\n{method_text}\n\nRepository {snapshot.source_id}, related files:\n"
        + "\n\n".join(_numbered(snapshot, path) for path in files)
    )
    result = await ctx.require_llm().generate(
        MethodReview, [SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)]
    )
    findings: list[CandidateFinding] = []
    for discrepancy in result.value.discrepancies:
        evidence = _evidence(ctx.paper, snapshot, files, discrepancy)
        if evidence is None:
            ctx.logger.info("dropping a method discrepancy without verbatim evidence")
            continue
        findings.append(
            CandidateFinding.for_task(
                task,
                len(findings),
                error_class=ErrorClass.METHOD_MISMATCH,
                summary=discrepancy.description,
                anchor=DocumentAnchor(),
                explanation=(
                    f"{discrepancy.category} discrepancy, {discrepancy.confidence} confidence."
                ),
                evidence=evidence,
                uncertain=discrepancy.confidence != "high",
            )
        )
    return TaskResult.for_task(task, TaskOutcome.CHECKED, findings=findings)


def paper_method_text(paper: ParsedPaper) -> str:
    """The body text of the paper's method-like sections (or of the whole body if none)."""
    titles = {s.id: s.title for s in paper.sections}
    body = [s for s in paper.sentences if s.kind is SentenceKind.BODY]
    chosen = [s for s in body if METHOD_SECTION.search(titles.get(s.section_id or "", ""))]
    text = " ".join(s.text for s in (chosen or body))
    return text[:MAX_PAPER_CHARS]


def related_files(snapshot: Snapshot, text: str) -> list[str]:
    """The snapshot's Python files that share the most terms with `text`."""
    terms = {w.lower() for w in _WORD.findall(text)} - _STOPWORDS
    scored = []
    for file in snapshot.manifest.files:
        if not file.path.endswith(".py"):
            continue
        words = Counter(w.lower() for w in _WORD.findall(snapshot.read_text(file.path)))
        score = sum(1 for term in terms if words[term])
        name_bonus = sum(1 for part in re.split(r"[/_.]", file.path.lower()) if part in terms)
        scored.append((score + 3 * name_bonus, file.path))
    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    chosen: list[str] = []
    size = 0
    for score, path in scored:
        if len(chosen) >= MAX_FILES or size >= MAX_CODE_CHARS or (score == 0 and chosen):
            break
        chosen.append(path)
        size += len(snapshot.read_text(path))
    return chosen


def _numbered(snapshot: Snapshot, path: str) -> str:
    lines = snapshot.read_text(path).splitlines()[:MAX_FILE_LINES]
    body = "\n".join(f"{number:5}| {line}" for number, line in enumerate(lines, start=1))
    return f"--- {path}\n{body}"


def _normalised(text: str) -> str:
    return " ".join(text.split())


def _evidence(
    paper: ParsedPaper, snapshot: Snapshot, files: list[str], discrepancy: Discrepancy
) -> list[Evidence] | None:
    quote = _normalised(discrepancy.paper_quote)
    if len(quote) < MIN_QUOTE_CHARS:
        return None
    sentence = next((s for s in paper.sentences if quote and quote in _normalised(s.text)), None)
    if sentence is None or discrepancy.code_path not in files:
        return None
    lines = snapshot.read_text(discrepancy.code_path).splitlines()
    start, end = discrepancy.code_start_line, discrepancy.code_end_line
    if end < start or end > len(lines):
        return None
    return [
        paper_quote_text(paper.paper_id, sentence.text),
        Evidence(
            kind=EvidenceKind.CODE,
            source=SourceRef(
                kind=SourceKind.REPOSITORY, id=snapshot.source_id, url=snapshot.manifest.url
            ),
            text="\n".join(lines[start - 1 : end]),
            path=discrepancy.code_path,
            line_start=start,
            line_end=end,
        ),
    ]
