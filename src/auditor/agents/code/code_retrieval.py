"""The code a paper's method sentences most likely describe, as line-numbered chunks.

The method check used to send the repository files that share the most words with the
whole method text. On the SciCoQA synthetic discrepancies of set C, that showed the
changed line in 35 of 84 cases. Here each method sentence retrieves its own code:

1. Python files are cut into functions (long ones into pieces) and the code between them;
   configuration and shell files into fixed windows.
2. Each sentence ranks the chunks, by embedding similarity when an embedding endpoint is
   configured (`AUDITOR_EMBED_URL`, any OpenAI-compatible `/v1/embeddings`, such as LM
   Studio running nomic-embed-text locally), else by BM25 over identifier words.
3. The sentences take turns adding their best chunk not yet chosen, until the character
   budget is spent, so every sentence gets some of its code in.

On the same 84 cases, with a 60,000-character budget, BM25 showed the changed line in 48
and embeddings in 59.
"""

import ast
import hashlib
import json
import logging
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from auditor.config import Settings
from auditor.infra.http import HttpClient, HttpError
from auditor.infra.snapshots import Snapshot

logger = logging.getLogger(__name__)

CODE_SUFFIXES = (".py", ".yaml", ".yml", ".json", ".sh", ".cfg", ".toml", ".ini")
MAX_FUNCTION_LINES = 150
PIECE_LINES = 80
WINDOW_LINES = 60
MAX_SENTENCES = 120
MAX_EMBED_CHARS = 2000
EMBED_BATCH = 32
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunk:
    path: str
    start: int
    end: int
    text: str

    @property
    def numbered(self) -> str:
        lines = self.text.splitlines()
        return "\n".join(f"{self.start + i:5}| {line}" for i, line in enumerate(lines))


def words(text: str) -> list[str]:
    """Identifier-aware words: `batch_norm` and `BatchNorm` give "batch", "norm"."""
    out = []
    for token in _WORD.findall(text):
        spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", token).replace("_", " ")
        out += [part.lower() for part in spaced.split() if len(part) > 2]
    return out


def _spans(text: str, path: str) -> list[tuple[int, int]]:
    lines = text.splitlines()
    if path.endswith(".py"):
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            tree = None
        if tree is not None:
            spans: list[tuple[int, int]] = []
            covered: set[int] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    start, end = node.lineno, node.end_lineno or node.lineno
                    if end - start > MAX_FUNCTION_LINES:
                        spans += [
                            (s, min(end, s + PIECE_LINES - 1))
                            for s in range(start, end + 1, PIECE_LINES)
                        ]
                    else:
                        spans.append((start, end))
                    covered.update(range(start, end + 1))
            run: list[int] = []
            for number in range(1, len(lines) + 1):
                if number not in covered:
                    run.append(number)
                elif run:
                    spans.append((run[0], run[-1]))
                    run = []
            if run:
                spans.append((run[0], run[-1]))
            if spans:
                return spans
    return [
        (s, min(len(lines), s + WINDOW_LINES - 1)) for s in range(1, len(lines) + 1, WINDOW_LINES)
    ]


def code_chunks(snapshot: Snapshot) -> list[Chunk]:
    chunks = []
    for file in snapshot.manifest.files:
        if not file.path.endswith(CODE_SUFFIXES):
            continue
        text = snapshot.read_text(file.path)
        lines = text.splitlines()
        for start, end in _spans(text, file.path):
            body = "\n".join(lines[start - 1 : end])
            if body.strip():
                chunks.append(Chunk(file.path, start, end, body))
    return chunks


def method_sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text) if len(s.split()) >= 6][:MAX_SENTENCES]


def bm25_ranks(queries: Sequence[str], chunks: Sequence[Chunk]) -> list[list[int]]:
    docs = [words(f"{c.path} {c.text}") for c in chunks]
    size = len(docs)
    average = sum(map(len, docs)) / max(size, 1)
    frequency = Counter(w for doc in docs for w in set(doc))
    counts = [Counter(doc) for doc in docs]
    ranks = []
    for query in queries:
        terms = words(query)
        scores = []
        for doc, tf in zip(docs, counts, strict=True):
            score = 0.0
            for term in terms:
                if term in tf:
                    idf = math.log(1 + (size - frequency[term] + 0.5) / (frequency[term] + 0.5))
                    norm = tf[term] + 1.5 * (0.25 + 0.75 * len(doc) / max(average, 1))
                    score += idf * tf[term] * 2.5 / norm
            scores.append(score)
        ranks.append(sorted(range(size), key=lambda i: -scores[i]))
    return ranks


class Embedder:
    """Embeddings from an OpenAI-compatible endpoint, cached on disk by text."""

    def __init__(self, http: HttpClient, settings: Settings) -> None:
        self.http = http
        self.url = (settings.embed_url or "").rstrip("/") + "/v1/embeddings"
        self.model = settings.embed_model
        self.cache = settings.cache_dir / "embeddings"

    def _path(self, text: str) -> Path:
        digest = hashlib.sha256(f"{self.model}\0{text}".encode()).hexdigest()
        return self.cache / digest[:2] / f"{digest}.json"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        found: dict[str, list[float]] = {}
        missing = []
        for text in dict.fromkeys(texts):
            path = self._path(text)
            if path.is_file():
                found[text] = json.loads(path.read_text())
            else:
                missing.append(text)
        for i in range(0, len(missing), EMBED_BATCH):
            batch = missing[i : i + EMBED_BATCH]
            response = await self.http.post(self.url, json={"model": self.model, "input": batch})
            if not response.ok:
                raise HttpError(f"embeddings: HTTP {response.status_code}")
            for text, item in zip(batch, json.loads(response.content)["data"], strict=True):
                vector = [float(v) for v in item["embedding"]]
                found[text] = vector
                path = self._path(text)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(vector))
        return [found[t] for t in texts]


def _unit(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


async def embedding_ranks(
    queries: Sequence[str], chunks: Sequence[Chunk], embedder: Embedder
) -> list[list[int]]:
    # nomic-embed-text expects these task prefixes; other models ignore them.
    documents = await embedder.embed(
        [f"search_document: {c.path}\n{c.text}"[:MAX_EMBED_CHARS] for c in chunks]
    )
    asked = await embedder.embed([f"search_query: {q}"[:MAX_EMBED_CHARS] for q in queries])
    docs = [_unit(d) for d in documents]
    ranks = []
    for query in asked:
        q = _unit(query)
        scores = [sum(a * b for a, b in zip(q, d, strict=True)) for d in docs]
        ranks.append(sorted(range(len(docs)), key=lambda i: -scores[i]))
    return ranks


def take_turns(ranks: Sequence[Sequence[int]], chunks: Sequence[Chunk], budget: int) -> list[Chunk]:
    """The sentences take turns adding their best chunk not yet chosen, within budget."""
    chosen: list[int] = []
    size = 0
    depth = 0
    longest = max((len(r) for r in ranks), default=0)
    while size < budget and depth < longest:
        for rank in ranks:
            if depth >= len(rank) or rank[depth] in chosen:
                continue
            chosen.append(rank[depth])
            size += len(chunks[rank[depth]].text)
            if size >= budget:
                break
        depth += 1
    ordered = sorted(chosen, key=lambda i: (chunks[i].path, chunks[i].start))
    return [chunks[i] for i in ordered]


async def retrieve(
    snapshot: Snapshot, text: str, http: HttpClient, settings: Settings, budget: int
) -> tuple[list[Chunk], str]:
    """The chunks for the method `text`, and how they were ranked ("embeddings" or
    "bm25"; BM25 when no endpoint is configured or it fails)."""
    chunks = code_chunks(snapshot)
    queries = method_sentences(text)
    if not chunks or not queries:
        return [], "none"
    if settings.embed_url:
        try:
            ranks = await embedding_ranks(queries, chunks, Embedder(http, settings))
            return take_turns(ranks, chunks, budget), "embeddings"
        except (HttpError, KeyError, ValueError) as exc:
            logger.warning("embeddings failed (%s); ranking code by BM25", exc)
    return take_turns(bm25_ranks(queries, chunks), chunks, budget), "bm25"
