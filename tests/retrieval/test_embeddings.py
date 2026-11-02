"""Offline tests for the optional embedding ranker (P2.4 amended).

The real MiniLM model needs the `embed` extra and a download; those paths are covered by
the network-marked test at the bottom. Everything here runs offline with fakes.
"""

import logging
from pathlib import Path

import pytest

from auditor.config import Settings
from auditor.retrieval import embeddings as embeddings_module
from auditor.retrieval.passages import Passage, chunk_text, rank_passages

TEXT = (
    "The model reaches 92.1 exact match on the SQuAD benchmark. " * 30
    + "Training used a batch size of 256 examples on one GPU. " * 30
)


class FakeEmbedder:
    """Scores every chunk 0.0 except the one whose text contains `needle` (0.9)."""

    def __init__(self, needle: str, fail: bool = False, width: int | None = None) -> None:
        self.needle = needle
        self.fail = fail
        self.width = width

    def score(self, chunks: list[Passage], claim: str) -> list[float] | None:
        if self.fail:
            raise RuntimeError("embedder is down")
        if self.width is not None:
            return [0.0] * self.width  # wrong length -> fallback
        return [0.9 if self.needle in chunk.text else 0.0 for chunk in chunks]


def test_rank_passages_uses_the_embedder_when_given() -> None:
    chunks = chunk_text(TEXT)
    target = chunks[3].text[:60]  # the chunk BM25 would not rank first
    embedder = FakeEmbedder(target[:30])
    ranked = rank_passages(TEXT, "a claim about nothing in particular", top=5, embedder=embedder)
    assert target[:30] in ranked[0].text


def test_rank_passages_falls_back_when_the_embedder_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    bm25 = rank_passages(TEXT, "exact match on the SQuAD benchmark", top=5)
    with caplog.at_level(logging.WARNING):
        fallback = rank_passages(
            TEXT, "exact match on the SQuAD benchmark", top=5, embedder=FakeEmbedder("", fail=True)
        )
    assert fallback == bm25  # the same ranking the embedder-free call gives
    assert any("falling back to BM25" in record.message for record in caplog.records)


def test_rank_passages_falls_back_on_a_wrong_score_width() -> None:
    bm25 = rank_passages(TEXT, "exact match on the SQuAD benchmark", top=5)
    fallback = rank_passages(
        TEXT, "exact match on the SQuAD benchmark", top=5, embedder=FakeEmbedder("", width=2)
    )
    assert fallback == bm25


def test_get_embedder_returns_none_for_the_default(tmp_path: Path) -> None:
    import asyncio

    from auditor.infra.http import HttpClient

    async def _no_sleep(_seconds: float) -> None:
        return None

    settings = Settings(cache_dir=tmp_path / "cache")

    async def run() -> None:
        http = HttpClient(settings, sleep=_no_sleep)
        assert await embeddings_module.get_embedder(settings, http) is None

    asyncio.run(run())


def test_get_embedder_without_the_extra_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`citation_embedder=minilm` without onnxruntime installed -> None + BM25, no raise."""
    import asyncio
    import builtins

    from auditor.infra.http import HttpClient

    real_import = builtins.__import__

    def refuse(name: str, *args: object, **kwargs: object) -> object:
        if name == "onnxruntime" or name == "tokenizers":
            raise ImportError(f"no {name}")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", refuse)
    settings = Settings(cache_dir=tmp_path / "cache", citation_embedder="minilm")

    async def _no_sleep(_seconds: float) -> None:
        return None

    async def run() -> object:
        http = HttpClient(settings, sleep=_no_sleep)
        return await embeddings_module.get_embedder(settings, http)

    assert asyncio.run(run()) is None


@pytest.mark.network
async def test_minilm_embedder_ranks_a_paraphrase_first(tmp_path: Path) -> None:
    """The real model (needs the `embed` extra and a ~23 MB download, once)."""
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")

    from auditor.infra.http import HttpClient

    settings = Settings(cache_dir=tmp_path / "cache", citation_embedder="minilm")

    async def _no_sleep(_seconds: float) -> None:
        return None

    http = HttpClient(settings, sleep=_no_sleep)
    embedder = await embeddings_module.get_embedder(settings, http)
    assert embedder is not None
    claim = "The authors train on hardware used by everyday consumers."
    ranked = rank_passages(
        TEXT + " Gaming GPUs trained the network cheaply. " * 30,
        claim,
        top=1,
        embedder=embedder,
    )
    assert "Gaming GPUs" in ranked[0].text
