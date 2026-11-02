"""Claims attributed to cited works: one candidate per citation mention and cited entry.

The deterministic pass keeps every mention, including markers the parser could not link,
so nothing silently drops out (decision D6). The optional LLM pass decides which
candidates make a checkable statement about the cited work and narrows the claim text;
whatever it returns must be an exact substring of the sentence, or it is ignored. Models
tend to leave the citation markers out of a quote ("REALM and ORQA" for "REALM [20] and
ORQA [31]"), so a quote also counts when it matches the sentence with the markers the
parser found taken out; the claim is then the sentence's own text, markers included.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from auditor.infra.llm import StructuredLLM
from auditor.schemas import ParsedPaper

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClaimCandidate:
    mention_id: str
    sentence_id: str
    bib_id: str | None
    marker: str
    claim: str
    check_worthy: bool = True


def citation_candidates(paper: ParsedPaper) -> list[ClaimCandidate]:
    candidates = []
    for mention in paper.citations:
        sentence = paper.sentence(mention.sentence_id).text
        targets: Sequence[str | None] = mention.bib_ids or (None,)
        for bib_id in targets:
            candidates.append(
                ClaimCandidate(
                    mention_id=mention.id,
                    sentence_id=mention.sentence_id,
                    bib_id=bib_id,
                    marker=mention.marker,
                    claim=sentence,
                )
            )
    return candidates


class ClaimJudgement(BaseModel):
    candidate: int = Field(description="Number of the candidate being judged")
    check_worthy: bool = Field(
        description="True if the sentence states something specific about the cited work "
        "that could be checked against that work's text"
    )
    claim: str = Field(
        description="The part of the sentence that is attributed to the cited work, copied "
        "exactly, character for character"
    )


class ClaimJudgements(BaseModel):
    judgements: list[ClaimJudgement]


SYSTEM_PROMPT = """You help audit citations in a research paper.

For each numbered candidate you get a sentence from the paper, one citation marker in it,
and the cited work. Decide whether the sentence makes a specific statement about that
cited work which could be checked against the work's own text (a result, a number, a
method, a finding). Statements that only point to related work in general ("see [3] for a
survey") are not check-worthy.

Copy into "claim" the part of the sentence that is attributed to the cited work. Copy it
exactly as it appears in the sentence, without rewording, fixing typos or adding words.
If you cannot isolate a part, copy the whole sentence. Answer for every candidate."""


def _describe(paper: ParsedPaper, index: int, candidate: ClaimCandidate) -> str:
    if candidate.bib_id is not None:
        entry = paper.bib_entry(candidate.bib_id)
        work = entry.title or entry.raw or candidate.bib_id
        if entry.year:
            work = f"{work} ({entry.year})"
    else:
        work = "unknown (the marker is not linked to the bibliography)"
    return (
        f"Candidate {index}\n"
        f"Sentence: {paper.sentence(candidate.sentence_id).text}\n"
        f"Marker: {candidate.marker}\n"
        f"Cited work: {work}"
    )


def quote_in(sentence: str, claim: str, markers: Sequence[tuple[int, int]]) -> str | None:
    """The part of `sentence` that `claim` quotes, or None. The claim may leave out any of
    the citation markers at the `markers` spans (start and end in the sentence), each with
    the space before it."""
    if not claim:
        return None
    if claim in sentence:
        return claim
    # Both texts without the markers; for each character of the sentence that is kept, its
    # index in the sentence.
    spans = [
        (start - 1 if start > 0 and sentence[start - 1] == " " else start, end)
        for start, end in markers
    ]
    dropped = {i for start, end in spans for i in range(start, end)}
    kept = [i for i in range(len(sentence)) if i not in dropped]
    bare = claim
    for text in sorted({sentence[start:end] for start, end in spans}, key=len, reverse=True):
        bare = bare.replace(text, "").replace(text.strip(), "")
    bare = bare.strip()
    at = "".join(sentence[i] for i in kept).find(bare) if bare else -1
    if at < 0:
        return None
    return sentence[kept[at] : kept[at + len(bare) - 1] + 1]


async def refine_candidates(
    paper: ParsedPaper,
    candidates: list[ClaimCandidate],
    llm: StructuredLLM,
    *,
    batch_size: int = 20,
) -> list[ClaimCandidate]:
    refined = list(candidates)
    for offset in range(0, len(candidates), batch_size):
        batch = candidates[offset : offset + batch_size]
        prompt = "\n\n".join(_describe(paper, i, c) for i, c in enumerate(batch))
        result = await llm.generate(
            ClaimJudgements,
            [SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)],
            purpose="citation-refine",
        )
        for judgement in result.value.judgements:
            if not 0 <= judgement.candidate < len(batch):
                logger.warning("ignoring judgement for unknown candidate %d", judgement.candidate)
                continue
            original = batch[judgement.candidate]
            sentence = paper.sentence(original.sentence_id).text
            spans = [
                (m.start, m.end) for m in paper.citations if m.sentence_id == original.sentence_id
            ]
            quoted = quote_in(sentence, judgement.claim.strip(), spans)
            if quoted is not None:
                narrowed = quoted
            elif not judgement.claim.strip():
                # Usually a candidate judged not check-worthy; the sentence stands.
                narrowed = original.claim
            else:
                logger.warning(
                    "claim for %s is not an exact quote of the sentence; keeping the sentence",
                    original.mention_id,
                )
                narrowed = original.claim
            refined[offset + judgement.candidate] = replace(
                original, claim=narrowed, check_worthy=judgement.check_worthy
            )
    return refined
