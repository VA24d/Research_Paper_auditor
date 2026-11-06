"""Re-checking each piece of evidence against its source (team plan P4.1).

Specialists build their evidence from what they read, but nothing downstream checked it.
Here every piece is looked up again in the source it names, with code and not a model:

- a quote from the paper must occur in the paper's sentences after the agreed
  normalisation (Unicode compatibility forms, so ligatures; hyphenation at line ends;
  quote marks and dashes; whitespace). A quote may run across sentences;
- a quote from a cited work counts when the citation agent's code located it in the
  retrieved text, which it records as character offsets. Without offsets the model's
  quote was not found there;
- code evidence must match the named lines of the repository snapshot;
- a repository lookup that names a file must name one the snapshot holds;
- a venue rule must name its rule; other lookups are the record of a tool call and are
  kept as such.
"""

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from auditor.infra.snapshots.snapshot import Snapshot
from auditor.schemas import Evidence, EvidenceKind, ParsedPaper, SourceKind

# Quotes shorter than this match too easily by chance ("we", "0.5").
MIN_GROUNDED_CHARS = 12

# Curly quotes, dashes and the minus sign compare as their ASCII forms; soft hyphens and
# zero-width spaces are dropped.
_TRANSLATE: dict[int, str | None] = {
    **dict.fromkeys((0x2018, 0x2019), "'"),
    **dict.fromkeys((0x201C, 0x201D), '"'),
    **dict.fromkeys((0x2010, 0x2011, 0x2012, 0x2013, 0x2014, 0x2212), "-"),
    **dict.fromkeys((0x00AD, 0x200B)),
}
_LINE_HYPHEN = re.compile(r"(\w)-\s+(\w)")


def normalise(text: str) -> str:
    """The agreed normalisation for comparing a quote with its source."""
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE)
    text = _LINE_HYPHEN.sub(r"\1\2", text)
    return " ".join(text.split())


class Grounding(StrEnum):
    FOUND = "found"  # located in its source
    RECORD = "record"  # the record of a tool call or a rule; nothing to quote
    NOT_FOUND = "not_found"  # the source does not contain it
    UNCHECKED = "unchecked"  # the source is not available to check against


@dataclass(frozen=True)
class Check:
    evidence: Evidence
    grounding: Grounding
    note: str = ""
    # For a paper quote that was found: the sentences it spans.
    sentence_ids: tuple[str, ...] = ()


class PaperText:
    """The paper's sentences joined in order, normalised, with each sentence's span, so a
    quote that runs across a sentence boundary is still found."""

    def __init__(self, paper: ParsedPaper) -> None:
        parts: list[str] = []
        self._spans: list[tuple[int, int, str]] = []
        offset = 0
        for sentence in paper.sentences:
            text = normalise(sentence.text)
            if not text:
                continue
            self._spans.append((offset, offset + len(text), sentence.id))
            parts.append(text)
            offset += len(text) + 1
        self.text = " ".join(parts)
        self.ids = {sentence.id for sentence in paper.sentences}
        self.raw = {sentence.id: sentence.text for sentence in paper.sentences}

    def find(self, quote: str) -> tuple[str, ...] | None:
        """The ids of the sentences a quote spans, or None when it is not in the paper."""
        needle = normalise(quote)
        start = self.text.find(needle)
        if start < 0:
            return None
        end = start + len(needle)
        return tuple(sid for s, e, sid in self._spans if s < end and start < e)


def _paper_quote(evidence: Evidence, paper: PaperText, sentence_id: str | None) -> Check:
    # A span of the flagged sentence, given by offsets, is checked exactly, however short
    # ("??" in an unfinished sentence).
    if sentence_id is not None and evidence.char_start is not None:
        sentence = paper.raw.get(sentence_id, "")
        if sentence[evidence.char_start : evidence.char_end] == evidence.text:
            return Check(evidence, Grounding.FOUND, sentence_ids=(sentence_id,))
    if len(normalise(evidence.text)) < MIN_GROUNDED_CHARS:
        return Check(evidence, Grounding.NOT_FOUND, "the quote is too short to check")
    sentences = paper.find(evidence.text)
    if sentences is None:
        return Check(evidence, Grounding.NOT_FOUND, "the quote is not in the paper")
    return Check(evidence, Grounding.FOUND, sentence_ids=sentences)


def _cited_quote(evidence: Evidence) -> Check:
    if evidence.char_start is None:
        return Check(evidence, Grounding.NOT_FOUND, "the quote was not found in the cited work")
    return Check(evidence, Grounding.FOUND)


def _code(evidence: Evidence, snapshot: Snapshot | None) -> Check:
    if snapshot is None:
        return Check(evidence, Grounding.UNCHECKED, "the repository snapshot is not available")
    assert evidence.path is not None
    assert evidence.line_start is not None
    try:
        lines = snapshot.read_text(evidence.path).splitlines()
    except (KeyError, OSError, UnicodeDecodeError):
        return Check(evidence, Grounding.NOT_FOUND, f"{evidence.path} is not in the repository")
    end = evidence.line_end or evidence.line_start
    if end > len(lines):
        return Check(
            evidence,
            Grounding.NOT_FOUND,
            f"{evidence.path} has {len(lines)} lines, not {end}",
        )
    window = normalise("\n".join(lines[evidence.line_start - 1 : end]))
    if normalise(evidence.text) not in window:
        return Check(
            evidence,
            Grounding.NOT_FOUND,
            f"{evidence.path}:{evidence.line_start} does not hold the quoted code",
        )
    return Check(evidence, Grounding.FOUND)


def _repository_lookup(evidence: Evidence, snapshot: Snapshot | None) -> Check:
    if evidence.path is None:
        return Check(evidence, Grounding.RECORD)
    if snapshot is None:
        return Check(evidence, Grounding.UNCHECKED, "the repository snapshot is not available")
    manifest = snapshot.manifest
    paths = {f.path for f in manifest.files} | {d.path for d in manifest.data_files}
    if evidence.path not in paths:
        return Check(evidence, Grounding.NOT_FOUND, f"{evidence.path} is not in the repository")
    return Check(evidence, Grounding.FOUND)


def ground(
    evidence: Evidence,
    paper: PaperText,
    snapshots: Mapping[str, Snapshot],
    sentence_id: str | None = None,
) -> Check:
    """How one piece of evidence stands against its source. `snapshots` holds the
    repository snapshots that could be loaded, by source id; `sentence_id` is the flagged
    sentence, which character offsets of a paper quote refer to."""
    source = evidence.source.kind
    snapshot = snapshots.get(evidence.source.id)
    match evidence.kind:
        case EvidenceKind.QUOTE if source is SourceKind.SUBMISSION:
            return _paper_quote(evidence, paper, sentence_id)
        case EvidenceKind.QUOTE if source is SourceKind.CITED_WORK:
            return _cited_quote(evidence)
        case EvidenceKind.QUOTE if source is SourceKind.REPOSITORY:
            return _repository_lookup(evidence, snapshot)
        case EvidenceKind.CODE:
            return _code(evidence, snapshot)
        case EvidenceKind.LOOKUP if source is SourceKind.REPOSITORY:
            return _repository_lookup(evidence, snapshot)
        case EvidenceKind.RULE:
            if source is SourceKind.VENUE_RULE:
                return Check(evidence, Grounding.RECORD)
            return Check(evidence, Grounding.NOT_FOUND, "rule evidence must name a venue rule")
        case _:
            return Check(evidence, Grounding.RECORD)
