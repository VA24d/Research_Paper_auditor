"""Offline tests for scripts/citation_eval.py: the recall CLI and its --embedder flag."""

import importlib.util
import sys
from pathlib import Path

import pytest

from auditor.retrieval.passages import Passage

REPO = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "citation_eval_script", REPO / "scripts" / "citation_eval.py"
)
assert _spec is not None
assert _spec.loader is not None
citation_eval = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("citation_eval_script", citation_eval)
_spec.loader.exec_module(citation_eval)

SOURCE_TEXT = (
    "Prior work reports 92.1% exact match on the SQuAD benchmark. " * 30
    + "The dataset holds 100,000 questions. " * 30
)
GOLD = "Prior work reports 92.1% exact match on the SQuAD benchmark."


class GoldFirstEmbedder:
    """Ranks the chunk holding the gold text first, everything else 0.0."""

    def score(self, chunks: list[Passage], claim: str) -> list[float]:
        return [0.9 if GOLD in chunk.text else 0.0 for chunk in chunks]


def _write_dataset(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    from auditor.schemas import PaperVariant
    from auditor.schemas.examples import example_paper

    paper = example_paper()
    paper = paper.model_copy(
        update={
            "paper_id": "2501.00001",
            "variant": PaperVariant.ORIGINAL,
        }
    )
    originals = tmp_path / "originals"
    originals.mkdir()
    (originals / "2501.00001.json").write_bytes(paper.model_dump_json().encode("utf-8"))
    sentence = paper.sentence("s2")
    # rebuild the record's original value into the claim the eval rebuilds: a record
    # without start/end uses the whole original sentence as the claim
    injections = tmp_path / "injections.jsonl"
    sources_dir = tmp_path / "sources"
    sources_dir.mkdir()
    sources = sources_dir / "2501.00001.jsonl"
    record = {
        "label_id": "b-test0001",
        "paper_id": "2501.00001",
        "kind": "claim_not_supported",
        "bib_id": "b1",
        "sentence_id": "s2",
        "start": None,
        "end": None,
        "original": sentence.text,
        "injected": "changed",
        "gold_passage": GOLD,
        "gold_passage_start": 0,
        "gold_passage_end": len(GOLD),
        "source_text_sha256": None,
    }
    injections.write_text(
        citation_eval.json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    sources.write_text(
        citation_eval.json.dumps(
            {
                "paper_id": "2501.00001",
                "bib_id": "b1",
                "level": "full_html",
                "source_text": SOURCE_TEXT,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return injections, sources_dir, originals


def _run_recall(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    embedder: str,
    fake: object | None,
) -> str:
    injections, sources, originals = _write_dataset(tmp_path)
    argv = [
        "citation_eval.py",
        "recall",
        "--injections",
        str(injections),
        "--sources",
        str(sources),
        "--originals",
        str(originals),
        "--k",
        "5",
        "--embedder",
        embedder,
    ]
    if fake is not None:
        import auditor.retrieval.embeddings as embeddings_module

        async def _fake_get_embedder(_settings: object, _http: object) -> object:
            return fake

        monkeypatch.setattr(embeddings_module, "get_embedder", _fake_get_embedder)
    monkeypatch.setattr(sys, "argv", argv)
    assert citation_eval.main() == 0
    return capsys.readouterr().out


def test_recall_defaults_to_bm25_and_scores_a_hit(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = _run_recall(capsys, monkeypatch, tmp_path, "bm25", None)
    assert "ranker bm25" in out
    assert "Recall@5 1.000 (1 of 1)" in out


def test_recall_with_the_embedder_flag_uses_the_ranker(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out = _run_recall(capsys, monkeypatch, tmp_path, "minilm", GoldFirstEmbedder())
    assert "ranker minilm" in out
    assert "Recall@5 1.000 (1 of 1)" in out


def test_recall_reports_a_miss_without_the_embedder(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    injections, sources_dir, originals = _write_dataset(tmp_path)
    # a gold passage the chunker cannot cut out of the source misses at any k
    injections.write_text(
        citation_eval.json.dumps(
            {
                "label_id": "b-test0001",
                "paper_id": "2501.00001",
                "kind": "claim_not_supported",
                "bib_id": "b1",
                "sentence_id": "s2",
                "start": None,
                "end": None,
                "original": "x",
                "injected": "y",
                "gold_passage": "this passage is nowhere in the source at all",
                "gold_passage_start": 0,
                "gold_passage_end": 10,
                "source_text_sha256": None,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    argv = [
        "citation_eval.py",
        "recall",
        "--injections",
        str(injections),
        "--sources",
        str(sources_dir),
        "--originals",
        str(originals),
        "--embedder",
        "bm25",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert citation_eval.main() == 0
    out = capsys.readouterr().out
    assert "Recall@5 0.000 (0 of 1)" in out
