"""Checks that the code implements the method the paper describes (`method_mismatch`).

Settings can be compared value by value; a loss, an algorithm step or a data pipeline
cannot. Here the paper's method sections and the repository files most related to them
(retrieved by term overlap) go to the LLM, which lists discrepancies. Each one must quote
the paper verbatim and point at existing lines of a retrieved file; anything else is
dropped before it becomes a finding (evidence gating, as the verifier requires). Findings
are anchored on the document: a method discrepancy is about the paper as a whole.
"""

import ast
import asyncio
import re
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from auditor.agents.code.code_retrieval import Chunk, method_sentences, retrieve
from auditor.agents.code.data_checks import paper_quote_text
from auditor.infra.context import RunContext
from auditor.infra.llm import StructuredLLM
from auditor.infra.snapshots import (
    Snapshot,
    SnapshotUnavailableError,
    SourceError,
    fetch_snapshot,
)
from auditor.planner.theory import formal_density
from auditor.schemas import (
    AuditTask,
    CandidateFinding,
    DocumentAnchor,
    ErrorClass,
    Evidence,
    EvidenceKind,
    ParsedPaper,
    Sentence,
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
You get the paper's method text and the repository code most related to it, with line
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


def _method_llm(ctx: RunContext) -> StructuredLLM:
    """The review model when one is configured and the settings ask for it, else the main."""
    if ctx.settings.method_check_model == "review" and ctx.review_llm is not None:
        return ctx.review_llm
    return ctx.require_llm()


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
    if ctx.settings.method_check_mode == "rag":
        return await _check_rag(task, ctx, snapshot, method_text)
    files = related_files(snapshot, method_text)
    if not files:
        return TaskResult.for_task(
            task, TaskOutcome.UNVERIFIABLE, reason="the repository has no Python files to compare"
        )
    if ctx.settings.method_check_mode == "pairs":
        return await _check_pairs(task, ctx, snapshot, files)
    prompt = (
        f"Paper method text:\n{method_text}\n\nRepository {snapshot.source_id}, related files:\n"
        + "\n\n".join(_numbered(snapshot, path) for path in files)
    )
    result = await _method_llm(ctx).generate(
        MethodReview, [SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)], purpose="method-whole"
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


# --- rag mode ----------------------------------------------------------------------------


async def _check_rag(
    task: AuditTask, ctx: RunContext, snapshot: Snapshot, method_text: str
) -> TaskResult:
    """The method text with the code each of its sentences retrieves (code_retrieval),
    in one call. A discrepancy must cite lines inside a shown chunk."""
    group = ctx.settings.method_rag_sentences
    sentences = method_sentences(method_text)
    parts = (
        [" ".join(sentences[i : i + group]) for i in range(0, len(sentences), group)]
        if group and sentences
        else [method_text]
    )
    budget = MAX_CODE_CHARS if len(parts) == 1 else MAX_CODE_CHARS // 2

    async def review(part: str) -> tuple[list[Chunk], list[Discrepancy]]:
        chunks, how = await retrieve(snapshot, part, ctx.http, ctx.settings, budget)
        if not chunks:
            return [], []
        shown = "\n\n".join(f"--- {c.path} (lines {c.start}-{c.end})\n{c.numbered}" for c in chunks)
        prompt = (
            f"Paper method text:\n{part}\n\nRepository {snapshot.source_id}, the code "
            f"most related to the method sentences ({how}):\n{shown}"
        )
        result = await _method_llm(ctx).generate(
            MethodReview,
            [SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)],
            purpose="method-rag",
        )
        return chunks, result.value.discrepancies

    reviewed = await asyncio.gather(*(review(p) for p in parts))
    if not any(chunks for chunks, _ in reviewed):
        return TaskResult.for_task(
            task, TaskOutcome.UNVERIFIABLE, reason="the repository has no code to compare"
        )
    findings: list[CandidateFinding] = []
    for chunks, discrepancies in reviewed:
        for discrepancy in discrepancies:
            inside = any(
                c.path == discrepancy.code_path
                and c.start <= discrepancy.code_start_line <= discrepancy.code_end_line <= c.end
                for c in chunks
            )
            evidence = (
                _evidence(ctx.paper, snapshot, [discrepancy.code_path], discrepancy)
                if inside
                else None
            )
            if evidence is None:
                ctx.logger.info("dropping a method discrepancy without shown evidence")
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


# --- pair mode ---------------------------------------------------------------------------

PAIR_PROMPT = """You compare one passage of a research paper with one piece of its code.
Answer "inconsistent" only if the code clearly does something different from what the
passage says (a different loss, formula, algorithm step, model component or data
processing). Answer "unrelated" if they are about different things, and "consistent"
otherwise. For "inconsistent", copy a quote of at least five words from the passage and give
the line numbers of the code that differs. Differences in hyperparameter values do not count."""
MAX_CHUNK_LINES = 80
_CONCURRENCY = 4
_IDENTIFIER_PART = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


class PairJudgement(BaseModel):
    verdict: Literal["consistent", "inconsistent", "unrelated"]
    paper_quote: str = ""
    code_start_line: int = Field(default=1, ge=1)
    code_end_line: int = Field(default=1, ge=1)
    description: str = ""
    confidence: Literal["low", "medium", "high"] = "low"


@dataclass(frozen=True)
class _Chunk:
    path: str
    start: int
    end: int
    terms: frozenset[str]


@dataclass(frozen=True)
class _Pair:
    sentence: Sentence
    passage: str
    chunk: _Chunk
    score: int


def _terms(text: str) -> set[str]:
    words = set()
    for word in _WORD.findall(text):
        words.add(word.lower())
        words |= {part.lower() for part in _IDENTIFIER_PART.findall(word) if len(part) > 3}
    return words - _STOPWORDS


def code_chunks(snapshot: Snapshot, files: list[str]) -> list[_Chunk]:
    """Functions and methods of the related files (or slices of files without any)."""
    chunks = []
    for path in files:
        text = snapshot.read_text(path)
        lines = text.splitlines()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError):
            tree = None
        spans = []
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    end = node.end_lineno or node.lineno
                    spans.append((node.lineno, min(end, node.lineno + MAX_CHUNK_LINES - 1)))
        if not spans:
            spans = [
                (start, min(start + MAX_CHUNK_LINES - 1, len(lines)))
                for start in range(1, len(lines) + 1, MAX_CHUNK_LINES)
            ]
        for start, end in spans:
            body = "\n".join(lines[start - 1 : end])
            chunks.append(_Chunk(path, start, end, frozenset(_terms(body))))
    return chunks


def method_pairs(paper: ParsedPaper, chunks: list[_Chunk], limit: int) -> list[_Pair]:
    """The best-matching (paper sentence, code chunk) pairs, at most one per chunk."""
    titles = {s.id: s.title for s in paper.sections}
    body = [s for s in paper.sentences if s.kind is SentenceKind.BODY]
    method = [s for s in body if METHOD_SECTION.search(titles.get(s.section_id or "", ""))] or body
    sentences = [s for s in method if len(s.text.split()) >= 8]
    index = {s.id: i for i, s in enumerate(body)}
    scored = [(s, frozenset(_terms(s.text))) for s in sentences]
    pairs = []
    for chunk in chunks:
        best = max(scored, key=lambda item: len(item[1] & chunk.terms), default=None)
        if best is None or not best[1] & chunk.terms:
            continue
        sentence, terms = best
        position = index[sentence.id]
        passage = " ".join(s.text for s in body[max(0, position - 1) : position + 2])
        pairs.append(_Pair(sentence, passage, chunk, len(terms & chunk.terms)))
    pairs.sort(key=_pair_order)
    return _within_budget(pairs, method, limit)


def _pair_order(pair: _Pair) -> tuple[int, str, int]:
    return (-pair.score, pair.chunk.path, pair.chunk.start)


def _within_budget(pairs: list[_Pair], method: Sequence[Sentence], limit: int) -> list[_Pair]:
    """At most `limit` pairs, shared out by section (as Google's Paper Assistant Tool gives
    dense sections more compute; arXiv 2606.28277). A section's share grows with its
    candidate pairs and with how formal its text is; every section with a candidate gets
    one pair first while the budget lasts, and a share a section cannot use goes to the
    next best pairs."""
    groups: dict[str, list[_Pair]] = {}
    for pair in pairs:
        groups.setdefault(pair.sentence.section_id or "", []).append(pair)
    by_section: dict[str, list[Sentence]] = {}
    for sentence in method:
        by_section.setdefault(sentence.section_id or "", []).append(sentence)
    weights = {
        key: len(group) * formal_density([s.text for s in by_section.get(key, [])])
        for key, group in groups.items()
    }
    order = sorted(groups, key=lambda key: (-weights[key], key))
    quota = dict.fromkeys(order, 0)
    for key in order[:limit]:
        quota[key] = 1
    spare = limit - sum(quota.values())
    if spare > 0:
        total = sum(weights.values())
        shares = {key: spare * weights[key] / total for key in order}
        for key in order:
            quota[key] += int(shares[key])
        left = limit - sum(quota.values())
        for key in sorted(order, key=lambda k: (int(shares[k]) - shares[k], k))[:left]:
            quota[key] += 1
    chosen = [pair for key in order for pair in groups[key][: quota[key]]]
    rest = sorted((p for key in order for p in groups[key][quota[key] :]), key=_pair_order)
    chosen += rest[: limit - len(chosen)]
    return sorted(chosen, key=_pair_order)


async def _check_pairs(
    task: AuditTask, ctx: RunContext, snapshot: Snapshot, files: list[str]
) -> TaskResult:
    pairs = method_pairs(ctx.paper, code_chunks(snapshot, files), ctx.settings.method_check_pairs)
    if not pairs:
        return TaskResult.for_task(
            task, TaskOutcome.UNVERIFIABLE, reason="no code shares terms with the method text"
        )
    gate = asyncio.Semaphore(_CONCURRENCY)
    llm = _method_llm(ctx)

    async def judge(pair: _Pair) -> PairJudgement:
        lines = snapshot.read_text(pair.chunk.path).splitlines()
        code = "\n".join(
            f"{n:5}| {lines[n - 1]}" for n in range(pair.chunk.start, pair.chunk.end + 1)
        )
        prompt = f"Paper passage:\n{pair.passage}\n\nCode ({pair.chunk.path}):\n{code}"
        async with gate:
            result = await llm.generate(
                PairJudgement,
                [SystemMessage(PAIR_PROMPT), HumanMessage(prompt)],
                purpose="method-pair",
            )
        return result.value

    judgements = await asyncio.gather(*(judge(pair) for pair in pairs))
    findings: list[CandidateFinding] = []
    taken: list[tuple[str, int, int]] = []
    for pair, judgement in zip(pairs, judgements, strict=True):
        if judgement.verdict != "inconsistent":
            continue
        chunk = pair.chunk
        start, end = judgement.code_start_line, judgement.code_end_line
        if not chunk.start <= start <= end <= chunk.end:
            continue
        if any(p == chunk.path and s <= end and start <= e for p, s, e in taken):
            continue
        discrepancy = Discrepancy(
            paper_quote=judgement.paper_quote,
            code_path=chunk.path,
            code_start_line=start,
            code_end_line=end,
            description=judgement.description or "The code differs from the paper.",
            category="Algorithm",
            confidence=judgement.confidence,
        )
        evidence = _evidence(ctx.paper, snapshot, [chunk.path], discrepancy)
        if evidence is None:
            continue
        taken.append((chunk.path, start, end))
        findings.append(
            CandidateFinding.for_task(
                task,
                len(findings),
                error_class=ErrorClass.METHOD_MISMATCH,
                summary=discrepancy.description,
                anchor=DocumentAnchor(),
                explanation=f"{judgement.confidence} confidence.",
                evidence=evidence,
                uncertain=judgement.confidence != "high",
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


_QUOTE_WORD = re.compile(r"[A-Za-z]{3,}|\d+")
# Share of a quote's first words that must be in the sentence it is anchored on.
MIN_QUOTE_OVERLAP = 0.8
QUOTE_HEAD_WORDS = 12


def quoted_sentence(paper: ParsedPaper, quote: str) -> Sentence | None:
    """The sentence a quote comes from. Models often quote across a sentence boundary or
    an equation the parser dropped ("as follows: where: ..."), so a quote that is not one
    sentence's text is anchored on the sentence holding its opening words: first exactly,
    then by most of its first words. On set C, 7 of 17 discrepancies the model reported on
    the right lines were dropped because their quotes ran past one sentence."""
    kinds = (SentenceKind.BODY, SentenceKind.ABSTRACT, SentenceKind.CAPTION)
    body = [s for s in paper.sentences if s.kind in kinds]
    found = next((s for s in paper.sentences if quote in _normalised(s.text)), None)
    if found is not None:
        return found
    head = " ".join(quote.split()[:QUOTE_HEAD_WORDS])
    if len(head) >= MIN_QUOTE_CHARS:
        found = next((s for s in body if head in _normalised(s.text)), None)
        if found is not None:
            return found
    words = [w.lower() for w in _QUOTE_WORD.findall(quote)][:QUOTE_HEAD_WORDS]
    if len(words) < 6:
        return None
    best, score = None, 0.0
    for sentence in body:
        have = {w.lower() for w in _QUOTE_WORD.findall(sentence.text)}
        share = sum(1 for w in words if w in have) / len(words)
        if share > score:
            best, score = sentence, share
    return best if score >= MIN_QUOTE_OVERLAP else None


def _evidence(
    paper: ParsedPaper, snapshot: Snapshot, files: list[str], discrepancy: Discrepancy
) -> list[Evidence] | None:
    quote = _normalised(discrepancy.paper_quote)
    if len(quote) < MIN_QUOTE_CHARS:
        return None
    sentence = quoted_sentence(paper, quote)
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
