"""Theory tasks: the segments of a paper whose mathematics is checked (decision D21).

Google's Paper Assistant Tool splits a paper into segments with a shared theme and gives
dense ones (proofs) more compute than prose (arXiv 2606.28277). Here a segment is a run
of neighbouring sections with formal content: formulas, symbols, identifiers, or the
statement of a theorem, lemma or proof. Each segment becomes one task, so its check sees
the whole paper but spends its effort on that part; when there are more segments than the
budget allows, the densest are kept.
"""

import re
from collections.abc import Sequence

from auditor.schemas import (
    AuditTask,
    ParsedPaper,
    Sentence,
    SentenceAnchor,
    SentenceKind,
    SettingTarget,
    TaskKind,
)

THEORY_KEY = "math"
# Tokens that mark formal content: operators, sums, integrals, partial derivatives,
# inequalities, arrows, Greek letters, digits, camelCase and snake_case identifiers.
_FORMAL_TOKEN = re.compile(
    r"[=+^<>\\\u2211\u220f\u222b\u2202\u2207\u00d7\u00b7\u2264\u2265\u2248\u2192\u0370-\u03ff]"
    r"|\d|[a-z][A-Z]|\w_\w"
)
_STATEMENT = re.compile(
    r"\b(theorem|lemma|proposition|corollary|proof|claim|conjecture|definition)\b",
    re.IGNORECASE,
)
# A section counts as formal from this density on (1 is plain prose).
MIN_DENSITY = 1.5
# The most text one segment's check focuses on.
MAX_SEGMENT_CHARS = 12_000


def formal_density(texts: Sequence[str]) -> float:
    """How formal a text is: 1 for plain prose, up to 5 when every token is formal."""
    tokens = [token for text in texts for token in text.split()]
    if not tokens:
        return 1.0
    formal = sum(1 for token in tokens if _FORMAL_TOKEN.search(token))
    return 1.0 + 4.0 * formal / len(tokens)


def _units(paper: ParsedPaper) -> list[tuple[str, list[Sentence]]]:
    """The body sentences of each section, in reading order."""
    units: dict[str, list[Sentence]] = {}
    for sentence in paper.sentences:
        if sentence.kind is SentenceKind.BODY:
            units.setdefault(sentence.section_id or "", []).append(sentence)
    return list(units.items())


def _formal(sentences: Sequence[Sentence]) -> bool:
    texts = [s.text for s in sentences]
    return formal_density(texts) >= MIN_DENSITY or any(_STATEMENT.search(t) for t in texts)


def theory_tasks(paper: ParsedPaper, max_segments: int) -> list[AuditTask]:
    segments: list[list[tuple[str, list[Sentence]]]] = []
    current: list[tuple[str, list[Sentence]]] = []
    size = 0
    for key, sentences in _units(paper):
        length = sum(len(s.text) for s in sentences)
        if not _formal(sentences):
            if current:
                segments.append(current)
            current, size = [], 0
            continue
        if current and size + length > MAX_SEGMENT_CHARS:
            segments.append(current)
            current, size = [], 0
        current.append((key, sentences))
        size += length
    if current:
        segments.append(current)

    def weight(segment: list[tuple[str, list[Sentence]]]) -> float:
        texts = [s.text for _, sentences in segment for s in sentences]
        return formal_density(texts) * sum(len(t) for t in texts)

    ranked = sorted(range(len(segments)), key=lambda i: (-weight(segments[i]), i))
    kept = sorted(ranked[:max_segments])
    return [
        AuditTask.create(
            paper_id=paper.paper_id,
            variant=paper.variant,
            kind=TaskKind.THEORY,
            anchor=SentenceAnchor(sentence_id=segments[i][0][1][0].id),
            target=SettingTarget(key=THEORY_KEY),
            sections=[key for key, _ in segments[i]],
        )
        for i in kept
    ]
