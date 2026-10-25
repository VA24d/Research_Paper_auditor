"""Build set B in one pass: propose, verify, inject (P2.8, decision D18).

    uv run python scripts/set_b_build.py --papers data/set_b/papers \
        --sources data/set_b/sources --out datasets/set_b [--pairs 10] [--controls 3] \
        [--seed 20260101]

For every parsed paper (data/set_b/papers/<id>.json) the deterministic picker proposes
injections (number perturbations, direction-swapped claims, metadata edits, fabricated
titles) and clean controls, each proposal is VERIFIED first (the perturbed value must not
be stated by the recorded source; the source must state the claim's original direction;
a metadata edit must disagree with the work the index resolves; a fabricated title must
resolve to nothing), and only verified proposals are applied: the injected paper copy is
written to <out>/papers/, the label and the injection record appended to labels.jsonl and
injections.jsonl, the original paper copied to <out>/papers/original/ (D14 runs it too).
"""

import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

from auditor.agents.citation.set_b import (
    InjectionKind,
    InjectionRecord,
    Proposal,
    append_jsonl,
    clean_label,
    gold_passage_for,
    inject_claim,
    inject_metadata,
    inject_not_found,
    inject_number,
    propose,
    write_paper,
)
from auditor.config import Settings
from auditor.infra.http import HttpClient
from auditor.retrieval.clients import Work
from auditor.retrieval.resolve import resolve_entry
from auditor.schemas import (
    BibEntry,
    Evidence,
    EvidenceKind,
    GoldLabel,
    ParsedPaper,
    SourceKind,
    SourceRef,
)


def _gold_evidence(gold_passage: object, bib_id: str) -> tuple[Evidence, ...]:
    from auditor.retrieval.passages import Passage

    assert isinstance(gold_passage, Passage)
    return (
        Evidence(
            kind=EvidenceKind.QUOTE,
            source=SourceRef(kind=SourceKind.CITED_WORK, id=bib_id),
            text=gold_passage.text,
            char_start=gold_passage.char_start,
            char_end=gold_passage.char_end,
        ),
    )


def _stated_in(source_text: str, value: str) -> bool:
    from auditor.agents.citation.support import extract_numbers, numbers_agree

    return any(numbers_agree(value, written) for written in extract_numbers(source_text))


async def _resolved_work(http: HttpClient, entry: BibEntry) -> Work | None:
    resolution = await resolve_entry(http, entry)
    return resolution.work


async def _verify(
    proposal: Proposal, paper: ParsedPaper, source_text: str | None, http: HttpClient
) -> str | None:
    """Why the proposal should NOT be applied, or None when it checks out.

    The perturbed number must not be stated by the recorded source; the claim swap is
    only applied when the source states the ORIGINAL direction word; a metadata edit
    must disagree with the work the index resolves; a fabricated title must resolve to
    nothing.
    """
    from auditor.agents.citation.set_b import _DIRECTION_SWAPS

    entry = paper.bib_entry(proposal.bib_id)
    kind = proposal.kind.value
    if kind == "claim_number_mismatch":
        if not source_text or _stated_in(source_text, proposal.new_value):
            return "the perturbed value is also stated by the source"
        return None
    if kind == "claim_not_supported":
        sentence = paper.sentence(proposal.sentence_id or "").text
        for old, _new in _DIRECTION_SWAPS:
            if (
                re.search(rf"\b{old}\b", sentence, re.IGNORECASE)
                and old in (source_text or "").lower()
            ):
                return None
        return "no direction swap verified against the source"
    if kind == "reference_metadata_mismatch":
        if not source_text:
            return "no source text"
        work = await _resolved_work(http, entry)
        if work is None:
            return "the original entry does not resolve (would be a not_found)"
        if (
            proposal.field_name == "year"
            and work.year is not None
            and str(work.year) == proposal.new_value
        ):
            return "the resolved work has the injected year"
        return None
    if kind == "reference_not_found":
        fabricated = entry.model_copy(
            update={"title": proposal.new_value, "doi": None, "arxiv_id": None}
        )
        if await _resolved_work(http, fabricated) is not None:
            return "the fabricated title resolves to a real work"
        return None
    return "unknown proposal kind"


async def _apply(
    proposal: Proposal,
    paper: ParsedPaper,
    source_text: str | None,
    sentence_text: str,
    original: str,
) -> tuple[ParsedPaper, GoldLabel, InjectionRecord]:
    if proposal.kind is InjectionKind.CLAIM_NUMBER_MISMATCH:
        assert proposal.anchor is not None
        gold = gold_passage_for(source_text or "", sentence_text)
        return inject_number(
            paper,
            proposal.anchor,
            proposal.new_value,
            bib_id=proposal.bib_id,
            rationale=f"the source states {original}, not {proposal.new_value}",
            evidence=_gold_evidence(gold, proposal.bib_id),
            gold_passage=gold,
            source_text=source_text,
        )
    if proposal.kind is InjectionKind.CLAIM_NOT_SUPPORTED:
        assert proposal.anchor is not None
        gold = gold_passage_for(source_text or "", sentence_text)
        return inject_claim(
            paper,
            proposal.anchor,
            proposal.new_value,
            bib_id=proposal.bib_id,
            rationale=(
                "the claim's direction was reversed; the source states the original direction"
            ),
            evidence=_gold_evidence(gold, proposal.bib_id),
            gold_passage=gold,
            source_text=source_text,
        )
    if proposal.kind is InjectionKind.REFERENCE_METADATA_MISMATCH:
        return inject_metadata(
            paper,
            proposal.bib_id,
            field_name=proposal.field_name or "year",  # type: ignore[arg-type]
            new_value=proposal.new_value,
            rationale=(
                f"the {proposal.field_name} was changed to {proposal.new_value}; "
                "the index says otherwise"
            ),
            evidence=(),
        )
    return inject_not_found(
        paper,
        proposal.bib_id,
        new_title=proposal.new_value,
        rationale=("the entry's title was replaced and its identifiers cleared; nothing resolves"),
        evidence=(),
    )


async def _build(args: argparse.Namespace) -> int:
    settings = Settings()
    labels_path = args.out / "labels.jsonl"
    records_path = args.out / "injections.jsonl"
    papers_dir = args.out / "papers"
    originals_dir = papers_dir / "original"
    rng = random.Random(args.seed)
    applied_total = 0

    async with HttpClient(settings) as http:
        for paper_path in sorted(args.papers.glob("*.json")):
            paper = ParsedPaper.model_validate_json(paper_path.read_bytes())
            sources_path = args.sources / f"{paper.paper_id}.jsonl"
            sources_by_bib: dict[str, str] = {}
            if sources_path.exists():
                for line in sources_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        if row.get("source_text"):
                            sources_by_bib[row["bib_id"]] = row["source_text"]
            proposals, clean_pairs = propose(
                paper, sources_by_bib, pairs=args.pairs, controls=args.controls, rng=rng
            )
            injected_paper = paper
            applied = 0
            for proposal in proposals:
                source_text = sources_by_bib.get(proposal.bib_id)
                if await _verify(proposal, injected_paper, source_text, http) is not None:
                    continue
                sentence_text = (
                    injected_paper.sentence(proposal.sentence_id or "").text
                    if proposal.sentence_id
                    else ""
                )
                original = (
                    sentence_text[proposal.anchor.start : proposal.anchor.end]
                    if proposal.anchor
                    else ""
                )
                try:
                    injected_paper, label, record = await _apply(
                        proposal, injected_paper, source_text, sentence_text, original
                    )
                except ValueError as exc:
                    print(f"{paper.paper_id}: skip ({exc})", file=sys.stderr)
                    continue
                append_jsonl(labels_path, label)
                append_jsonl(records_path, record)
                applied += 1
            for bib_id, anchor in clean_pairs:
                gold = gold_passage_for(
                    sources_by_bib[bib_id], paper.sentence(anchor.sentence_id).text
                )
                label = clean_label(
                    paper,
                    anchor,
                    bib_id,
                    _gold_evidence(gold, bib_id),
                    "the source states this claim as written",
                )
                append_jsonl(labels_path, label)
            originals_dir.mkdir(parents=True, exist_ok=True)
            (originals_dir / paper_path.name).write_bytes(paper_path.read_bytes())
            if applied:
                write_paper(injected_paper, args.out_dir)
            applied_total += applied
            print(f"{paper.paper_id}: {applied} injections, {len(clean_pairs)} clean")
    print(f"total: {applied_total} injections")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("datasets/set_b"))
    parser.add_argument("--pairs", type=int, default=10, help="injections per paper")
    parser.add_argument("--controls", type=int, default=3, help="clean pairs per paper")
    parser.add_argument("--seed", type=int, default=20260101)
    args = parser.parse_args()
    return asyncio.run(_build(args))


if __name__ == "__main__":
    sys.exit(main())
