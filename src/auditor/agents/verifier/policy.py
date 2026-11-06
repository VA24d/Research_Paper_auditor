"""What each error class needs before it counts as a verified flag (team plan P4.2).

A finding states that something in the paper disagrees with something else: the cited
work, the repository, a venue rule, or another part of the paper. It is verified only when
both sides are shown by evidence that was found again in its source (`grounding`):

- the paper side is a quote of the paper, or for checks that start from one sentence
  (a citation, a stated setting) the flagged sentence itself;
- the other side is the source the class compares against.

Rules for the decision, in order:

1. A piece of evidence that is not in its source is a false accusation in the making: if
   it was the only support for a side, the finding is dropped.
2. A finding with no evidence at all is dropped; from a check that did not finish it goes
   to human review instead (D6).
3. A side that is not shown, or a source that could not be loaded, sends the finding to
   human review.
4. Absence is never proof: a reference that no index lists goes to human review, since a
   missing record alone never makes a verified flag.
5. A finding its specialist marked uncertain, one from a check that did not finish, and
   one whose quotes are all from somewhere other than the flagged sentence, go to review.
"""

from dataclasses import dataclass
from enum import StrEnum

from auditor.schemas import ErrorClass, SourceKind


class PaperSide(StrEnum):
    QUOTE = "quote"  # a quote of the paper, found in its text
    ANCHOR = "anchor"  # a quote, or else the flagged sentence or bibliography entry
    NONE = "none"  # the finding is about the repository or the layout alone


@dataclass(frozen=True)
class Need:
    paper: PaperSide
    # Source kinds the other side may come from; empty when the paper is compared with
    # itself.
    other: frozenset[SourceKind] = frozenset()
    # Whether a tool record (an index lookup, a rule) shows the other side. Otherwise only
    # evidence found again in its source does.
    records: bool = False
    # The finding rests on something not being found.
    absence: bool = False


_REPOSITORY = frozenset({SourceKind.REPOSITORY})
_CITED = frozenset({SourceKind.CITED_WORK})
_INDEX = frozenset({SourceKind.SCHOLARLY_INDEX})

NEEDS: dict[ErrorClass, Need] = {
    ErrorClass.REFERENCE_NOT_FOUND: Need(PaperSide.ANCHOR, _INDEX, records=True, absence=True),
    ErrorClass.REFERENCE_METADATA_MISMATCH: Need(PaperSide.ANCHOR, _INDEX, records=True),
    ErrorClass.CLAIM_NOT_SUPPORTED: Need(PaperSide.ANCHOR, _CITED),
    ErrorClass.CLAIM_NUMBER_MISMATCH: Need(PaperSide.ANCHOR, _CITED),
    ErrorClass.SETTING_MISMATCH: Need(PaperSide.ANCHOR, _REPOSITORY),
    ErrorClass.DATA_SPLIT_MISMATCH: Need(PaperSide.ANCHOR, _REPOSITORY),
    ErrorClass.METRIC_MISMATCH: Need(PaperSide.ANCHOR, _REPOSITORY),
    ErrorClass.DATASET_SIZE_MISMATCH: Need(PaperSide.ANCHOR, _REPOSITORY),
    ErrorClass.LABEL_DISTRIBUTION_MISMATCH: Need(PaperSide.ANCHOR, _REPOSITORY),
    ErrorClass.DATA_LEAKAGE: Need(PaperSide.ANCHOR, _REPOSITORY),
    ErrorClass.METHOD_MISMATCH: Need(PaperSide.ANCHOR, _REPOSITORY),
    ErrorClass.CODE_DEFECT: Need(PaperSide.NONE, _REPOSITORY),
    ErrorClass.MATH_ERROR: Need(PaperSide.QUOTE),
    ErrorClass.RESULT_INCONSISTENCY: Need(PaperSide.QUOTE),
    ErrorClass.UNFINISHED_CONTENT: Need(PaperSide.QUOTE),
    ErrorClass.TEMPLATE_VIOLATION: Need(
        PaperSide.NONE, frozenset({SourceKind.VENUE_RULE}), records=True
    ),
}

# Classes whose verdict is a model's reading (does this passage support the claim, does
# this code do what the sentence says); the optional recheck asks a second model about
# these. Math errors get their own review step (D21).
MODEL_JUDGED = frozenset({ErrorClass.CLAIM_NOT_SUPPORTED, ErrorClass.METHOD_MISMATCH})
