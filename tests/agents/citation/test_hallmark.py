"""Offline tests for the HALLMARK loader and existence-check scoring (P2.7)."""

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from auditor.agents.citation.hallmark import (
    ExistenceResult,
    Flag,
    HallmarkEntry,
    TypeScore,
    bib_entry_for,
    load_split,
    paper_for,
    read_results,
    score,
    write_results,
)
from auditor.config import Settings

VALID = {
    "bibtex_key": "ba6f8800e25a",
    "bibtex_type": "inproceedings",
    "fields": {
        "title": "Real-Time Image Demoiréing on Mobile Devices",
        "author": "Yuxin Zhang and Mingbao Lin",
        "year": "2023",
        "booktitle": "ICLR",
    },
    "label": "VALID",
    "hallucination_type": None,
    "difficulty_tier": None,
    "generation_method": "scraped",
    "subtests": {"doi_resolves": None, "title_exists": True},
    "raw_bibtex": None,
    "schema_version": "1.0",
}
HALLUCINATED = {
    "bibtex_key": "a80e0803bdbf",
    "bibtex_type": "inproceedings",
    "fields": {
        "title": "simpleKT: A Simple But Tough-to-Beat Baseline",
        "author": "Zitao Liu and Qiongqiong Liu",
        "year": "2031",
        "booktitle": "ICLR",
    },
    "label": "HALLUCINATED",
    "hallucination_type": "future_date",
    "difficulty_tier": 1,
    "explanation": "Publication year 2031 is in the future",
    "generation_method": "perturbation",
    "subtests": {"doi_resolves": None, "title_exists": True},
    "raw_bibtex": None,
    "schema_version": "1.0",
}
CANARY = {
    "bibtex_key": "__canary__abc",
    "bibtex_type": "article",
    "fields": {"title": "canary"},
    "label": "VALID",
}


def write_split(tmp_path: Path, *records: Mapping[str, object]) -> Path:
    path = tmp_path / "test_public.jsonl"
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def test_load_split_drops_canaries_and_rejects_duplicates(tmp_path: Path) -> None:
    path = write_split(tmp_path, VALID, HALLUCINATED, CANARY)
    entries = load_split(path)
    assert [e.bibtex_key for e in entries] == ["ba6f8800e25a", "a80e0803bdbf"]

    duplicated = write_split(tmp_path, VALID, VALID)
    with pytest.raises(ValueError, match="duplicate bibtex_key"):
        load_split(duplicated)


def test_bib_entry_for_maps_the_bibtex_fields() -> None:
    entry = bib_entry_for(HallmarkEntry.model_validate(HALLUCINATED))
    assert entry.id == "a80e0803bdbf"
    assert entry.title == "simpleKT: A Simple But Tough-to-Beat Baseline"
    assert entry.authors == ("Zitao Liu", "Qiongqiong Liu")
    assert entry.year == 2031
    assert entry.venue == "ICLR"
    # D1: the raw text is rendered from the same fields, so the two never disagree.
    assert entry.raw is not None
    assert entry.title in entry.raw
    assert "2031" in entry.raw


def test_paper_for_builds_a_one_entry_paper() -> None:
    paper = paper_for(HallmarkEntry.model_validate(HALLUCINATED))
    assert paper.paper_id == "hallmark-a80e0803bdbf"
    assert [b.id for b in paper.bibliography] == ["a80e0803bdbf"]
    assert len(paper.sentences) == 1


def test_score_counts_flags_against_the_labels() -> None:
    results = [
        ExistenceResult("k1", "HALLUCINATED", "future_date", 1, Flag.NOT_FOUND),
        ExistenceResult("k2", "HALLUCINATED", "future_date", 1, Flag.CLEAN),
        ExistenceResult("k3", "HALLUCINATED", "chimeric_title", 2, Flag.METADATA_MISMATCH),
        ExistenceResult("k4", "VALID", None, None, Flag.CLEAN),
        ExistenceResult("k5", "VALID", None, None, Flag.NOT_FOUND),
        ExistenceResult("k6", "VALID", None, None, Flag.UNVERIFIABLE),
    ]
    scores = score(results)
    assert scores.overall == TypeScore(true_positives=2, false_positives=1, false_negatives=1)
    assert scores.per_type["future_date"] == TypeScore(true_positives=1, false_negatives=1)
    assert scores.per_type["chimeric_title"] == TypeScore(true_positives=1)
    assert (scores.valid_entries, scores.valid_flagged) == (2, 1)
    assert scores.false_positive_rate == pytest.approx(0.5)
    assert scores.unverifiable == 1
    assert scores.overall.precision == pytest.approx(2 / 3)
    assert scores.overall.recall == pytest.approx(2 / 3)
    assert scores.overall.f1 == pytest.approx(2 / 3)


def test_results_round_trip_through_jsonl(tmp_path: Path) -> None:
    results = [
        ExistenceResult("k1", "HALLUCINATED", "future_date", 1, Flag.NOT_FOUND, reason="r"),
        ExistenceResult("k2", "VALID", None, None, Flag.CLEAN),
    ]
    path = tmp_path / "results.jsonl"
    write_results(results, path)
    assert read_results(path) == results


async def _no_sleep(_seconds: float) -> None:
    return None


async def test_check_entry_flags_a_fabricated_entry_and_passes_a_real_one(
    tmp_path: Path,
) -> None:
    """End to end through the agent's reference check, with canned Crossref replies."""
    import httpx

    from auditor.agents.citation.hallmark import check_entry
    from auditor.infra.context import RunContext
    from auditor.infra.http import HttpClient

    hit = {
        "message": {
            "DOI": "10.1234/real",
            "title": ["A Real Work About Real Things"],
            "author": [{"given": "Ada", "family": "Example"}],
            "issued": {"date-parts": [[2020]]},
        }
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if "10.1234/real" in str(request.url):
            return httpx.Response(200, json=hit)
        return httpx.Response(404)

    settings = Settings(cache_dir=tmp_path / "cache")
    http = HttpClient(settings, transport=httpx.MockTransport(handler), sleep=_no_sleep)
    ctx = RunContext(settings=settings, paper=None, http=http, llm=None, logger=None)  # type: ignore[arg-type]

    real = HallmarkEntry.model_validate(
        {
            "bibtex_key": "real1",
            "fields": {
                "title": "A Real Work About Real Things",
                "author": "Ada Example",
                "year": "2020",
                "doi": "10.1234/real",
            },
            "label": "VALID",
        }
    )
    fabricated = HallmarkEntry.model_validate(
        {
            "bibtex_key": "fake1",
            "fields": {
                "title": "A Fabricated Work About Nothing Real",
                "author": "Nobody Person",
                "year": "2031",
                "doi": "10.9999/fake",
            },
            "label": "HALLUCINATED",
            "hallucination_type": "fabricated_doi",
            "difficulty_tier": 1,
        }
    )
    found = await check_entry(ctx, real)
    assert found.flag is Flag.CLEAN
    missing = await check_entry(ctx, fabricated)
    assert missing.flag is Flag.NOT_FOUND
    assert score([found, missing]).overall.f1 == 1.0
