"""Theory tasks: the formal claims of a paper whose mathematics is checked (decision D21).

The first stage of the theory cascade. One cheap call with minimal thinking reads the
paper and lists its formal claims, whatever the field: what each says, the sentences that
state and argue it, the sentences it builds on, how risky it looks and whether checking it
takes real reasoning. Which claims count, and how risky they are, is the model's reading of
the paper, not a list of keywords. The claims are then grouped into tasks of a few claims
each; when there are more than the budget allows, the least risky are left out first, so
the later, costlier stages only see claims worth their time.
"""

import re
from collections.abc import Sequence
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from auditor.infra.llm import StructuredLLM
from auditor.schemas import (
    AuditTask,
    ParsedPaper,
    SentenceAnchor,
    SentenceKind,
    SettingTarget,
    TaskKind,
    TheoryClaim,
)

THEORY_KEY = "math"
# The listing sent to the model is cut here (about 50k tokens).
MAX_LISTING_CHARS = 200_000
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
- statement: what it says, in at most 25 words;
- sentence_ids: the ids of the sentences that state it and that argue or derive it;
- depends_on: the ids of the sentences with the definitions and earlier results it uses;
- risk: "high" for long or delicate arguments, "low" for routine steps;
- needs_reasoning: false if reading is enough to check it, true if it takes working through.
Use only sentence ids from the listing."""


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


def paper_listing(paper: ParsedPaper) -> str:
    """The abstract and body, one `[id] text` line per sentence, with section headings:
    every sentence once, so a model can point at sentences by id instead of quoting."""
    titles = {s.id: s.title for s in paper.sections}
    lines: list[str] = []
    current: str | None = None
    for sentence in paper.sentences:
        if sentence.kind not in (SentenceKind.ABSTRACT, SentenceKind.BODY):
            continue
        if sentence.section_id != current:
            current = sentence.section_id
            if titles.get(current or ""):
                lines.append(f"## {titles[current or '']}")
        lines.append(f"[{sentence.id}] {sentence.text}")
    return "\n".join(lines)[:MAX_LISTING_CHARS]


_RISK = {"high": 0, "medium": 1, "low": 2}


async def theory_tasks(
    paper: ParsedPaper, llm: StructuredLLM, *, batch_size: int, max_tasks: int
) -> list[AuditTask]:
    result = await llm.generate(
        ClaimInventory,
        [SystemMessage(PLAN_PROMPT), HumanMessage(paper_listing(paper))],
        effort="minimal",
        purpose="theory-plan",
    )
    order = {s.id: i for i, s in enumerate(paper.sentences)}
    claims: list[TheoryClaim] = []
    for entry in result.value.claims:
        ids = tuple(dict.fromkeys(i for i in entry.sentence_ids if i in order))
        if not ids:
            continue
        claims.append(
            TheoryClaim(
                claim_id=f"c{len(claims) + 1}",
                kind=entry.kind.strip(),
                statement=entry.statement.strip(),
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
