"""A second look at uncertain setting mismatches, one small question per code line.

When the repository holds several values for a setting, rules cannot tell which one the
paper's experiment used. A small model can: it sees the paper sentence and a few lines
around one candidate value, and answers whether that line sets the value for the
experiment the sentence describes. Each question is a single yes/no decision on a short
prompt, which suits small local models. Values it rejects are dropped; values it confirms
make the finding firm.
"""

import asyncio
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from auditor.agents.code.data_checks import Candidate
from auditor.infra.context import RunContext
from auditor.schemas import ExtractedSetting, RepositorySource

# Candidate values asked about per finding; the rest keep the finding uncertain.
MAX_QUESTIONS = 4
CONTEXT_LINES = 3

SYSTEM_PROMPT = """You check research code. You get a sentence from a paper and a few lines
of the paper's code repository. Answer "yes" if the marked line sets the value the paper's
experiment in that sentence uses, "no" if it belongs to something else (another model,
dataset or experiment, a library default, test or debugging code), and "unsure" if the lines
do not tell."""


class Applies(BaseModel):
    answer: Literal["yes", "no", "unsure"]
    reason: str = ""


async def second_look(
    ctx: RunContext, setting: ExtractedSetting, sentence: str, candidates: list[Candidate]
) -> tuple[list[Candidate], list[Candidate], bool]:
    """The candidates the model confirms, those it does not reject, and whether every
    asked candidate was rejected."""
    llm = ctx.require_llm()
    asked = candidates[:MAX_QUESTIONS]

    async def ask(candidate: Candidate) -> str:
        source = candidate.setting.source
        assert isinstance(source, RepositorySource)
        lines = candidate.snapshot.read_text(source.path).splitlines()
        first = max(1, source.line - CONTEXT_LINES)
        last = min(len(lines), source.line + CONTEXT_LINES)
        shown = "\n".join(
            f"{'>>' if n == source.line else '  '} {n:5}| {lines[n - 1]}"
            for n in range(first, last + 1)
        )
        prompt = (
            f"Paper sentence: {sentence}\n\nSetting: {setting.key} "
            f"(the paper says {setting.raw_value})\n\n{source.path}:\n{shown}"
        )
        result = await llm.generate(
            Applies, [SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)], purpose="second-look"
        )
        return result.value.answer

    answers = await asyncio.gather(*(ask(c) for c in asked))
    confirmed = [c for c, a in zip(asked, answers, strict=True) if a == "yes"]
    kept = [c for c, a in zip(asked, answers, strict=True) if a != "no"]
    kept += candidates[MAX_QUESTIONS:]
    rejected_all = not kept
    return confirmed, kept, rejected_all
