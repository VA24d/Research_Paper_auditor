from pathlib import Path

from auditor.agents.code.code_retrieval import (
    Chunk,
    _spans,
    bm25_ranks,
    method_sentences,
    take_turns,
    words,
)

SOURCE = """import torch


def focal_loss(logits, target, gamma=2.0):
    p = torch.sigmoid(logits)
    return -((1 - p) ** gamma) * torch.log(p)


def build_optimizer(params):
    return torch.optim.SGD(params, lr=0.1)
"""


def test_identifiers_are_split_into_words() -> None:
    assert words("BatchNorm2d batch_norm lr") == ["batch", "norm2d", "batch", "norm"]


def test_python_files_are_cut_into_functions_and_the_code_between() -> None:
    spans = _spans(SOURCE, "loss.py")
    assert (4, 6) in spans  # focal_loss
    assert (9, 10) in spans  # build_optimizer
    assert any(start == 1 for start, _ in spans)  # the import


def test_each_sentence_ranks_its_own_code_and_they_take_turns() -> None:
    chunks = [
        Chunk("loss.py", 4, 6, "def focal_loss(logits, target, gamma=2.0): ..."),
        Chunk("optim.py", 1, 2, "def build_optimizer(params): torch.optim.SGD(lr)"),
        Chunk("data.py", 1, 2, "def load_images(path): ..."),
    ]
    sentences = [
        "We train with the focal loss and gamma set to two.",
        "The optimizer is SGD with a learning rate of 0.1 throughout.",
    ]
    ranks = bm25_ranks(sentences, chunks)
    assert ranks[0][0] == 0
    assert ranks[1][0] == 1
    # A budget for two chunks gives each sentence its best one.
    chosen = take_turns(ranks, chunks, budget=len(chunks[0].text) + 1)
    assert {c.path for c in chosen} == {"loss.py", "optim.py"}


def test_method_sentences_leave_out_fragments() -> None:
    text = "See above. We train the model with the focal loss for ten epochs."
    assert method_sentences(text) == ["We train the model with the focal loss for ten epochs."]


async def test_embedding_ranks_follow_similarity() -> None:
    from collections.abc import Sequence

    from auditor.agents.code.code_retrieval import Embedder, embedding_ranks

    class Fake(Embedder):
        def __init__(self) -> None:
            pass

        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            return [[1.0, 0.0] if "loss" in t else [0.0, 1.0] for t in texts]

    chunks = [Chunk("a.py", 1, 1, "optimizer"), Chunk("b.py", 1, 1, "focal loss")]
    ranks = await embedding_ranks(["the loss we use"], chunks, Fake())
    assert ranks == [[1, 0]]


def test_bm25_ranks_edge_cases() -> None:
    # Empty chunks
    assert bm25_ranks(["query words here"], []) == [[]]
    # Empty queries
    assert bm25_ranks([], [Chunk("a.py", 1, 1, "text")]) == []
    # Queries with words not in docs
    ranks = bm25_ranks(["nonexistent terms totally absent"], [Chunk("a.py", 1, 1, "hello world")])
    assert ranks == [[0]]


def test_take_turns_deduplication_and_budget() -> None:
    chunks = [
        Chunk("a.py", 1, 1, "chunk 0 text"),
        Chunk("b.py", 1, 1, "chunk 1 text"),
        Chunk("c.py", 1, 1, "chunk 2 text"),
    ]
    # Multiple ranks prefer chunk 0 first
    ranks = [[0, 1, 2], [0, 2, 1], [0, 1, 2]]
    chosen = take_turns(ranks, chunks, budget=100)
    # Deduplicated: every chunk appears at most once
    assert len(chosen) == 3
    assert [c.path for c in chosen] == ["a.py", "b.py", "c.py"]


async def test_retrieve_uses_supplied_chunks(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from auditor.agents.code.code_retrieval import retrieve
    from auditor.config import Settings
    from auditor.infra.http import HttpClient
    from auditor.infra.snapshots.snapshot import Snapshot, SnapshotManifest
    from auditor.infra.snapshots.sources import SourceKind

    custom_chunks = [
        Chunk("custom.py", 1, 5, "def custom_optimizer(): lr = 0.01"),
    ]
    manifest = SnapshotManifest(
        source=SourceKind.GITHUB,
        url="https://github.com/org/repo",
        identity="org/repo",
        requested_ref=None,
        revision="main",
        fetched_at=datetime.now(UTC),
        limits={},
        files=(),
        skipped=(),
        skipped_count=0,
        incomplete=(),
        metadata={},
    )
    snap = Snapshot(tmp_path, manifest)
    text = "We use a custom optimizer with learning rate 0.01."
    settings = Settings()
    retrieved, how = await retrieve(
        snap, text, HttpClient(settings), settings, budget=1000, chunks=custom_chunks
    )
    assert how == "bm25"
    assert len(retrieved) == 1
    assert retrieved[0].path == "custom.py"
