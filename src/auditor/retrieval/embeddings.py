"""An optional embedding ranker for passage retrieval (decision P2.4, amended).

`MiniLMEmbedder` runs all-MiniLM-L6-v2 as a quantised ONNX model (~23 MB) on the CPU, so
it runs on low-end systems and on every operating system; it ranks chunks by cosine
similarity against the claim, which helps when the claim paraphrases the source (BM25
only matches tokens). It is OPT-IN: `AUDITOR_CITATION_EMBEDDER=minilm` and the `embed`
extra (`uv sync --extra embed`) — the default stays BM25, which needs no model at all.

The model and its tokenizer are downloaded once through the shared HTTP client (the
Hugging Face hosts are on the allowlist) and cached under `cache_dir/embeddings/`.
`get_embedder` never raises: without the extra or the network it returns None and the
caller keeps BM25.
"""

import logging
from pathlib import Path
from typing import Any

from auditor.config import Settings
from auditor.infra.http import HttpClient
from auditor.retrieval.passages import Embedder, Passage

logger = logging.getLogger(__name__)

MODEL_URL = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx"
TOKENIZER_URL = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json"
MODEL_BYTES = 40_000_000  # the quantised model is ~23 MB; allow headroom
DIMENSIONS = 384
MAX_TOKENS = 512


class MiniLMEmbedder:
    """all-MiniLM-L6-v2 as a quantised ONNX model, CPU, mean-pooled and L2-normalised."""

    def __init__(self, model_path: Path, tokenizer_path: Path) -> None:
        import onnxruntime
        from tokenizers import Tokenizer

        self._session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._input_names = {inp.name for inp in self._session.get_inputs()}
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.no_padding()
        self._tokenizer.enable_truncation(max_length=MAX_TOKENS)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One L2-normalised embedding per text, mean-pooled over the tokens."""
        import numpy as np

        encodings = [self._tokenizer.encode(text) for text in texts]
        width = max(len(encoding.ids) for encoding in encodings)
        ids = np.zeros((len(encodings), width), dtype=np.int64)
        mask = np.zeros((len(encodings), width), dtype=np.int64)
        for row, encoding in enumerate(encodings):
            ids[row, : len(encoding.ids)] = encoding.ids
            mask[row, : len(encoding.attention_mask)] = encoding.attention_mask
        feeds: dict[str, Any] = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = self._session.run(None, feeds)[0]  # (batch, tokens, DIMENSIONS)
        mask_float = mask[:, :, None].astype(hidden.dtype)
        pooled = (hidden * mask_float).sum(axis=1) / np.clip(
            mask_float.sum(axis=1), a_min=1, a_max=None
        )
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        normalised = pooled / np.clip(norms, a_min=1e-12, a_max=None)
        return [[float(value) for value in row] for row in normalised]

    def score(self, chunks: list[Passage], claim: str) -> list[float] | None:
        """Cosine similarity of each chunk against the claim (dot of unit vectors)."""
        vectors = self.embed([chunk.text for chunk in chunks] + [claim])
        claim_vector = vectors[-1]
        return [
            float(sum(a * b for a, b in zip(vector, claim_vector, strict=False)))
            for vector in vectors[:-1]
        ]


async def get_embedder(settings: Settings, http: HttpClient) -> Embedder | None:
    """The configured embedder, or None (use BM25).

    Downloads the model and tokenizer once, through the shared HTTP client, into
    `cache_dir/embeddings/`. Never raises: a missing `embed` extra or a failed download
    is a logged warning and a None, and BM25 keeps working.
    """
    if settings.citation_embedder != "minilm":
        return None
    try:
        import onnxruntime  # noqa: F401
        from tokenizers import Tokenizer  # noqa: F401
    except ImportError:
        logger.warning(
            "AUDITOR_CITATION_EMBEDDER=minilm but the `embed` extra is not installed; "
            "install it with `uv sync --extra embed` and use BM25 for now."
        )
        return None
    directory = settings.cache_dir / "embeddings"
    model_path = directory / "minilm-model_quantized.onnx"
    tokenizer_path = directory / "minilm-tokenizer.json"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            await http.download_to(MODEL_URL, model_path, max_bytes=MODEL_BYTES)
        if not tokenizer_path.exists():
            await http.download_to(TOKENIZER_URL, tokenizer_path, max_bytes=10_000_000)
    except Exception as exc:
        logger.warning("the embedding model could not be downloaded (%s); using BM25", exc)
        return None
    try:
        return MiniLMEmbedder(model_path, tokenizer_path)
    except Exception as exc:
        logger.warning("the embedding model could not be loaded (%s); using BM25", exc)
        return None
