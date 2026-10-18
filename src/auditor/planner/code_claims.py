"""Code claim tasks: what a paper says about its experiments that its released code or data
can confirm or refute (the planning step of the code and data claim loop).

One cheap call to the main model reads the paper and lists up to N such claims: the
dataset and split used, preprocessing, the number of training examples, model
architecture choices, the training procedure and the evaluation protocol. Each claim must
quote its sentence verbatim and is anchored on that sentence by its parser id (D3). Plain
hyperparameter values are left to the settings check, and a claim about a sentence that
already has a settings or data task for the same thing is dropped, so the loop adds checks
instead of repeating them.
"""

import logging
from collections.abc import Sequence
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from auditor.infra.llm import StructuredLLM, StructuredOutputError
from auditor.planner.theory import paper_listing
from auditor.schemas import (
    AuditTask,
    ExtractedSetting,
    PaperSource,
    ParsedPaper,
    RepositoryRef,
    SentenceAnchor,
    SentenceKind,
    SettingTarget,
    TaskKind,
)

logger = logging.getLogger(__name__)

CLAIM_KEY = "code_claim"
ClaimCategory = Literal[
    "dataset_split", "preprocessing", "training_examples", "architecture", "training", "evaluation"
]
# Shorter quotes match too many sentences to pin the claim down.
MIN_QUOTE_CHARS = 20
MAX_STATEMENT_CHARS = 300
# The setting keys of the settings and data checks that already cover a category on the
# same sentence (the same names as in auditor.agents.code.data_checks; importing them here
# would be circular).
COVERING_KEYS: dict[str, frozenset[str]] = {
    "dataset_split": frozenset({"data_split", "test_fraction", "validation_fraction", "dataset"}),
    "training_examples": frozenset({"train_examples", "total_examples"}),
    "evaluation": frozenset({"eval_metric"}),
}

PLAN_PROMPT = """You prepare a check of a research paper against its released code and data.
The paper follows, one sentence per line with its id in brackets. List the claims about the
paper's own experiments that reading the repository could confirm or refute:
- dataset_split: which dataset, version or split is used, how it is split or filtered;
- preprocessing: how inputs or labels are processed before training or evaluation;
- training_examples: how many training examples are used;
- architecture: model components and how they are connected (not their sizes);
- training: the training procedure (objective, schedule, stages, sampling, augmentation);
- evaluation: the evaluation protocol (test data, metric computation, averaging, checkpoint
  selection).
Leave out single hyperparameter values (learning rate, batch size, epochs, optimizer, seed,
layer sizes), results, claims about other work, and motivation. For each claim give:
- sentence_id: the id of the sentence that states it;
- quote: at least five consecutive words of that sentence, copied exactly;
- category: one of the categories above;
- claim: what the code or data should show, in at most 25 words.
List at most {limit} claims, those the code is most likely to settle first. An empty list
is a valid answer.

The paper:
{listing}"""


class PlannedClaim(BaseModel):
    sentence_id: str
    quote: str
    category: ClaimCategory
    claim: str


class ClaimPlan(BaseModel):
    claims: list[PlannedClaim] = []


def _normalised(text: str) -> str:
    return " ".join(text.split())


def claim_anchor(paper: ParsedPaper, entry: PlannedClaim) -> SentenceAnchor | None:
    """The anchor of a planned claim: its sentence, narrowed to the quote when the quote is
    found exactly; None when the quote is too short or not in the sentence."""
    quote = _normalised(entry.quote)
    if len(quote) < MIN_QUOTE_CHARS:
        return None
    try:
        sentence = paper.sentence(entry.sentence_id)
    except KeyError:
        return None
    if sentence.kind not in (SentenceKind.ABSTRACT, SentenceKind.BODY):
        return None
    if quote not in _normalised(sentence.text):
        return None
    start = sentence.text.find(entry.quote.strip())
    if start < 0:
        return SentenceAnchor(sentence_id=sentence.id)
    return SentenceAnchor(
        sentence_id=sentence.id, start=start, end=start + len(entry.quote.strip())
    )


def _covered(existing: Sequence[AuditTask]) -> set[tuple[str, str]]:
    """(sentence id, setting key) of the code tasks already planned."""
    pairs = set()
    for task in existing:
        anchor, target = task.anchor, task.target
        if (
            task.kind is TaskKind.CODE
            and isinstance(anchor, SentenceAnchor)
            and isinstance(target, SettingTarget)
        ):
            pairs.add((anchor.sentence_id, target.key))
    return pairs


async def claim_tasks(
    paper: ParsedPaper,
    repository: RepositoryRef | None,
    llm: StructuredLLM,
    *,
    limit: int,
    max_output: int = 4096,
    existing: Sequence[AuditTask] = (),
) -> list[AuditTask]:
    """At most `limit` code tasks, one per claim the code or data can settle."""
    if repository is None or limit <= 0:
        return []
    listing = paper_listing(paper)
    if not listing:
        return []
    try:
        result = await llm.generate(
            ClaimPlan,
            [
                SystemMessage(PLAN_PROMPT.format(limit=limit, listing=listing)),
                HumanMessage("List the claims."),
            ],
            effort="low",
            purpose="claim-plan",
            max_output_tokens=max_output,
        )
    except StructuredOutputError as exc:
        logger.warning("planning the code claims of %s failed (%s)", paper.paper_id, exc)
        return []
    covered = _covered(existing)
    tasks: dict[str, AuditTask] = {}
    for entry in result.value.claims:
        if len(tasks) >= limit:
            break
        anchor = claim_anchor(paper, entry)
        statement = _normalised(entry.claim)[:MAX_STATEMENT_CHARS]
        if anchor is None or not statement:
            logger.info("dropping a planned code claim without a verbatim quote")
            continue
        keys = COVERING_KEYS.get(entry.category, frozenset())
        if any((anchor.sentence_id, key) in covered for key in keys):
            continue
        task = AuditTask.create(
            paper_id=paper.paper_id,
            variant=paper.variant,
            kind=TaskKind.CODE,
            anchor=anchor,
            target=SettingTarget(key=CLAIM_KEY),
            setting=ExtractedSetting(
                key=CLAIM_KEY,
                raw_name=entry.category,
                raw_value=statement,
                source=PaperSource(anchor=anchor),
            ),
            repository=repository,
        )
        tasks.setdefault(task.task_id, task)
    return list(tasks.values())
