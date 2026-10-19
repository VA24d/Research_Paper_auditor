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
