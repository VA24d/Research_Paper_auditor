"""Chunking and passage retrieval with exact character offsets (decision P2.4).

A source's text is split into overlapping chunks; the chunks relevant to a claim are
ranked with BM25, a pure-Python ranker that needs no model and runs on every operating
system. Each passage keeps the exact `char_start`/`char_end` offsets of its text inside
the source, so an Evidence quote can point at the place it came from.
"""

import logging
import math
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Passage:
    """One chunk of a source's text, with its exact place in it."""

    text: str
    char_start: int
    char_end: int  # exclusive


class Embedder(Protocol):
    """A semantic ranker: cosine similarity of each chunk against a claim.

    Implementations may be expensive or unavailable; `rank_passages` falls back to BM25
    when the embedder raises or returns None, so an Embedder never has to degrade
    gracefully itself.
    """

    def score(self, chunks: list["Passage"], claim: str) -> list[float] | None:
        """One similarity per chunk (same order), or None when it cannot score."""
        ...


def chunk_text(text: str, *, size: int = 1200, overlap: int = 200) -> list[Passage]:
    """Split `text` into chunks of about `size` characters, `overlap` shared.

    Chunks break at whitespace so a sentence is not cut in half; offsets are exact
    character positions in `text`.
    """
    if not text:
        return []
    chunks: list[Passage] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            # Break at the last whitespace inside the chunk, so words stay whole.
            cut = text.rfind(" ", start + size // 2, end)
            if cut > start:
                end = cut
        chunk = text[start:end]
        chunks.append(Passage(text=chunk, char_start=start, char_end=end))
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class _BM25:
    """Okapi BM25 over a fixed set of chunks (k1=1.5, b=0.75)."""

    def __init__(self, chunks: list[Passage]) -> None:
        self._chunks = chunks
        self._doc_counts: list[dict[str, int]] = []
        self._doc_freq: dict[str, int] = {}
        total = 0
        for chunk in chunks:
            counts: dict[str, int] = {}
            for token in _tokens(chunk.text):
                counts[token] = counts.get(token, 0) + 1
            self._doc_counts.append(counts)
            for token in counts:
                self._doc_freq[token] = self._doc_freq.get(token, 0) + 1
            total += sum(counts.values())
        self._avg_len = total / len(chunks) if chunks else 1.0

    def score(self, query_tokens: list[str]) -> list[float]:
        n = len(self._chunks)
        k1, b = 1.5, 0.75
        scores = []
        for counts in self._doc_counts:
            length = sum(counts.values()) or 1
            score = 0.0
            for token in query_tokens:
                freq = counts.get(token, 0)
                if not freq:
                    continue
                df = self._doc_freq.get(token, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += idf * freq * k1 / (freq + k1 * (1 - b + b * length / self._avg_len))
            scores.append(score)
        return scores


def _query_tokens(claim: str) -> list[str]:
    """The claim's content words, with duplicates kept (they carry weight)."""
    return _tokens(claim)


def rank_passages(
    text: str,
    claim: str,
    *,
    size: int = 1200,
    overlap: int = 200,
    top: int = 5,
    embedder: Embedder | None = None,
) -> list[Passage]:
    """The `top` chunks of `text` most relevant to `claim`, best first.

    With `embedder`, the chunks are ranked by cosine similarity against the claim; when
    the embedder is unavailable or fails, BM25 is used instead (logged), so the same
    call never fails because of the ranker. Deterministic: equal scores keep the text
    order, so the same claim and source always give the same passages.
    """
    chunks = chunk_text(text, size=size, overlap=overlap)
    if not chunks:
        return []
    scores: list[float] | None = None
    if embedder is not None:
        try:
            scores = embedder.score(chunks, claim)
        except Exception:
            logger.warning("the embedding ranker failed; falling back to BM25", exc_info=True)
        if scores is not None and len(scores) != len(chunks):
            logger.warning(
                "the embedding ranker scored %d of %d chunks; using BM25", len(scores), len(chunks)
            )
            scores = None
    if scores is None:
        scores = _BM25(chunks).score(_query_tokens(claim))
    order = sorted(range(len(chunks)), key=lambda i: (-scores[i], i))
    return [chunks[i] for i in order[:top]]


def passage_fingerprint(passage: Passage) -> str:
    """A stable id for one passage, used to deduplicate evidence across runs."""
    return sha256(passage.text.encode("utf-8")).hexdigest()[:16]
