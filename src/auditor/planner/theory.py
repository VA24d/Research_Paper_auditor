"""Theory tasks: the formal claims of a paper whose mathematics is checked (decision D21).

The first stage of the theory cascade. One cheap call with minimal thinking reads the
paper and lists its formal claims, whatever the field: what each says, the sentences that
state and argue it, the sentences it builds on, how risky it looks and whether checking it
takes real reasoning. Which claims count, and how risky they are, is the model's reading of
the paper, not a list of keywords. The claims are then grouped into tasks of a few claims
each; when there are more than the budget allows, the least risky are left out first, so
the later, costlier stages only see claims worth their time.
"""

import asyncio
import logging
import re
from collections.abc import Sequence
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from auditor.infra.llm import StructuredLLM, StructuredOutputError
from auditor.schemas import (
    AuditTask,
    ParsedPaper,
    Sentence,
    SentenceAnchor,
    SentenceKind,
    SettingTarget,
    TaskKind,
    TheoryClaim,
)

logger = logging.getLogger(__name__)

THEORY_KEY = "math"
# The listing sent to the model is cut here (about 50k tokens).
MAX_LISTING_CHARS = 200_000
# A paper is planned in chunks of about this many characters, in parallel: one long answer
# per paper made a small model repeat itself until it hit the output cap.
MAX_CHUNK_CHARS = 15_000
MAX_PLAN_TOKENS = 8192
# From this density on a chunk counts as formal (1 is plain prose).
FORMAL_DENSITY = 1.5
# The most sentences a fallback claim covers.
MAX_FALLBACK_SENTENCES = 40
# Tokens that mark formal content: operators, sums, integrals, partial derivatives,
# inequalities, arrows, Greek letters, digits, camelCase and snake_case identifiers.
_FORMAL_TOKEN = re.compile(
    r"[=+^<>\\\u2211\u220f\u222b\u2202\u2207\u00d7\u00b7\u2264\u2265\u2248\u2192\u0370-\u03ff]"
    r"|\d|[a-z][A-Z]|\w_\w"
)

PLAN_PROMPT = """You prepare the mathematical check of a research paper. List the paper's
formal claims that could be wrong: theorems, lemmas, propositions, corollaries, derived
formulas, bounds, estimates, identities, complexity results and similar statements that an
argument or calculation in the paper supports. Leave out definitions, assumptions,
citations of other work stated without proof, and prose. For each claim give:
- kind: a word for it, such as "lemma" or "bound";
- statement: what it says, in at most 20 words;
- sentence_ids: the ids of the sentences that state it and that argue or derive it;
- depends_on: the ids of the sentences with the definitions and earlier results it uses;
- risk: "high" for long or delicate arguments, "low" for routine steps;
- needs_reasoning: false if reading is enough to check it, true if it takes working through.
List at most 15 claims, the riskiest first if there are more. Use only sentence ids from
the listing."""


class ClaimEntry(BaseModel):
    kind: str = ""
    statement: str = ""
    sentence_ids: list[str]
    depends_on: list[str] = []
    risk: Literal["low", "medium", "high"] = "medium"
    needs_reasoning: bool = True


class ClaimInventory(BaseModel):
    claims: list[ClaimEntry]


def formal_density(texts: Sequence[str]) -> float:
    """How formal a text is: 1 for plain prose, up to 5 when every token is formal."""
    tokens = [token for text in texts for token in text.split()]
    if not tokens:
        return 1.0
    formal = sum(1 for token in tokens if _FORMAL_TOKEN.search(token))
    return 1.0 + 4.0 * formal / len(tokens)


def _readable(paper: ParsedPaper) -> list[Sentence]:
    kinds = (SentenceKind.ABSTRACT, SentenceKind.BODY)
    return [s for s in paper.sentences if s.kind in kinds]


def paper_listing(paper: ParsedPaper, sentences: Sequence[Sentence] | None = None) -> str:
    """The abstract and body (or the given sentences), one `[id] text` line per sentence,
    with section headings: every sentence once, so a model can point at sentences by id
    instead of quoting."""
    titles = {s.id: s.title for s in paper.sections}
    lines: list[str] = []
    current: str | None = None
    for sentence in _readable(paper) if sentences is None else sentences:
        if sentence.section_id != current:
            current = sentence.section_id
            if titles.get(current or ""):
                lines.append(f"## {titles[current or '']}")
        lines.append(f"[{sentence.id}] {sentence.text}")
    return "\n".join(lines)[:MAX_LISTING_CHARS]


_RISK = {"high": 0, "medium": 1, "low": 2}

PAPER_CONTEXT = """You review the mathematics of a research paper for a referee. The paper
follows, one sentence per line with its id in brackets; the task comes after it.

The paper:
{listing}"""


def paper_context(paper: ParsedPaper) -> SystemMessage:
    """The paper as the first message of every review-model call about it (triage, deep
    checks, challenge), so a provider's prompt cache can reuse it across all of them; the
    instructions come after it."""
    return SystemMessage(PAPER_CONTEXT.format(listing=paper_listing(paper)))


def _chunks(paper: ParsedPaper) -> list[list[Sentence]]:
    """The readable sentences in pieces of about MAX_CHUNK_CHARS, cut at section starts
    where possible."""
    chunks: list[list[Sentence]] = []
    size = 0
    previous: str | None = None
    for sentence in _readable(paper):
        new_section = sentence.section_id != previous
        previous = sentence.section_id
        over = size + len(sentence.text) > MAX_CHUNK_CHARS
        if not chunks or (over and (new_section or size >= MAX_CHUNK_CHARS)):
            chunks.append([])
            size = 0
        chunks[-1].append(sentence)
        size += len(sentence.text)
    return chunks


def _fallback(chunk: Sequence[Sentence]) -> ClaimEntry | None:
    """A claim covering the formal sentences of a chunk the plan said nothing about, so
    formal text is never left unchecked; None for prose."""
    formal = [s for s in chunk if formal_density([s.text]) >= FORMAL_DENSITY]
    if formal_density([s.text for s in chunk]) < FORMAL_DENSITY and len(formal) < 3:
        return None
    return ClaimEntry(
        kind="section",
        statement="the formal content of this part of the paper",
        sentence_ids=[s.id for s in formal[:MAX_FALLBACK_SENTENCES]],
        risk="high",
        needs_reasoning=True,
    )


async def _plan_chunk(
    paper: ParsedPaper, chunk: Sequence[Sentence], llm: StructuredLLM
) -> list[ClaimEntry]:
    try:
        result = await llm.generate(
            ClaimInventory,
            [SystemMessage(PLAN_PROMPT), HumanMessage(paper_listing(paper, chunk))],
            effort="low",
            purpose="theory-plan",
            max_output_tokens=MAX_PLAN_TOKENS,
        )
        entries = result.value.claims
    except StructuredOutputError as exc:
        logger.warning(
            "planning %s failed (%s); checking its formal text whole", paper.paper_id, exc
        )
        entries = []
    ids = {s.id for s in chunk}
    if not any(i in ids for e in entries for i in e.sentence_ids):
        fallback = _fallback(chunk)
        return [fallback] if fallback is not None else []
    return entries


async def theory_tasks(
    paper: ParsedPaper, llm: StructuredLLM, *, batch_size: int, max_tasks: int
) -> list[AuditTask]:
    planned = await asyncio.gather(*(_plan_chunk(paper, c, llm) for c in _chunks(paper)))
    order = {s.id: i for i, s in enumerate(paper.sentences)}
    claims: list[TheoryClaim] = []
    for entry in (e for entries in planned for e in entries):
        ids = tuple(dict.fromkeys(i for i in entry.sentence_ids if i in order))
        if not ids:
            continue
        claims.append(
            TheoryClaim(
                claim_id=f"c{len(claims) + 1}",
                kind=entry.kind.strip(),
                statement=entry.statement.strip()[:300],
                sentence_ids=ids,
                depends_on=tuple(dict.fromkeys(i for i in entry.depends_on if i in order)),
                risk=entry.risk,
                needs_reasoning=entry.needs_reasoning,
            )
        )
    # Over the budget, the least risky claims (and among those, the ones that only need
    # reading) are left out first.
    budget = batch_size * max_tasks
    ranked = sorted(
        range(len(claims)),
        key=lambda i: (_RISK[claims[i].risk], not claims[i].needs_reasoning, i),
    )
    kept = [claims[i] for i in sorted(ranked[:budget])]
    kept.sort(key=lambda c: order[c.sentence_ids[0]])

    batches: list[list[TheoryClaim]] = []
    for item in kept:
        # A batch starts on its own sentence, which keeps task ids apart.
        starts_anew = not batches or len(batches[-1]) >= batch_size
        if starts_anew and batches and batches[-1][0].sentence_ids[0] == item.sentence_ids[0]:
            starts_anew = False
        if starts_anew:
            batches.append([item])
        else:
            batches[-1].append(item)
    return [
        AuditTask.create(
            paper_id=paper.paper_id,
            variant=paper.variant,
            kind=TaskKind.THEORY,
            anchor=SentenceAnchor(sentence_id=batch[0].sentence_ids[0]),
            target=SettingTarget(key=THEORY_KEY),
            claims=batch,
        )
        for batch in batches
    ]


TRIAGE_PROMPT = """You are a careful referee. Read the whole paper and name the formal
claims most likely to contain a real mathematical error: a false statement, a step that
does not follow, a missing case, a result used outside its assumptions, a wrong constant,
exponent or bound, or a conclusion stronger than the argument gives. Rank them, most
likely first, and give at most {limit}. For each give:
- reason: in at most 30 words, what looks wrong and what a check should work out;
- kind: one word for it, such as "lemma" or "bound";
- sentence_ids: the ids of the sentences that state it and the step you doubt;
- depends_on: the ids of the sentences with the definitions and results it uses.
Do not restate the claim or write formulas outside the reason.
Name a claim only for a concrete doubt, not because it is long. Use only sentence ids
from the listing; an empty list is a valid answer."""
MAX_TRIAGE_TOKENS = 16_384


class SuspectEntry(BaseModel):
    # Few free-text fields, the reason first: with a restated claim as well, 3.8 Flash
    # repeated phrases inside it until the output cap in two of three papers.
    reason: str = ""
    kind: str = ""
    sentence_ids: list[str]
    depends_on: list[str] = []


class Suspects(BaseModel):
    suspects: list[SuspectEntry]


async def triage_tasks(paper: ParsedPaper, llm: StructuredLLM, *, limit: int) -> list[AuditTask]:
    """One task with the claims a referee reading the whole paper suspects most, at most
    `limit`. One call to the review model replaces the cheap listing and screen, which
    passed nearly every claim, wrong ones included."""
    if limit <= 0:
        return []
    try:
        result = await llm.generate(
            Suspects,
            [paper_context(paper), HumanMessage(TRIAGE_PROMPT.format(limit=limit))],
            effort="medium",
            purpose="theory-triage",
            max_output_tokens=MAX_TRIAGE_TOKENS,
            cache_prefix=True,
        )
    except StructuredOutputError as exc:
        logger.warning("triage of %s failed (%s)", paper.paper_id, exc)
        return []
    order = {s.id: i for i, s in enumerate(paper.sentences)}
    claims: list[TheoryClaim] = []
    for entry in result.value.suspects:
        ids = tuple(dict.fromkeys(i for i in entry.sentence_ids if i in order))
        if not ids or len(claims) >= limit:
            continue
        claims.append(
            TheoryClaim(
                claim_id=f"c{len(claims) + 1}",
                kind=entry.kind.strip()[:40],
                statement=entry.reason.strip()[:300],
                sentence_ids=ids,
                depends_on=tuple(dict.fromkeys(i for i in entry.depends_on if i in order)),
                risk="high",
                suspicion=entry.reason.strip()[:300] or "named by the triage step",
            )
        )
    if not claims:
        return []
    return [
        AuditTask.create(
            paper_id=paper.paper_id,
            variant=paper.variant,
            kind=TaskKind.THEORY,
            anchor=SentenceAnchor(sentence_id=claims[0].sentence_ids[0]),
            target=SettingTarget(key=THEORY_KEY),
            claims=claims,
        )
    ]
