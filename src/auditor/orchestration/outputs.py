"""What a run leaves on disk, and the predictions the evaluation reads (decision D3).

runs/<run_id>/
    manifest.json
    usage/{tool_calls,http_requests,llm_calls}.jsonl
    <paper_id>-<variant>/
        tasks.jsonl, results.jsonl, verified.jsonl    (full system)
        report.md, report.json                        (full system; auditor.reporting)
        predictions.jsonl                             (every system)
        predictions.incomplete.jsonl                  (instead, when model calls ran out)

A paper whose checks ran out of model calls has its predictions written under the second
name. Its missing findings would read as clean results, and a batch takes a paper with
`predictions.jsonl` as done, so only complete papers get that name.
"""

from collections.abc import Iterable, Sequence
from pathlib import Path

from pydantic import BaseModel

from auditor.infra.files import atomic_write
from auditor.orchestration.graph import AuditOutcome
from auditor.reporting import build_report, write_report
from auditor.schemas import ParsedPaper, Prediction, SystemName, VerifiedFinding

PREDICTIONS = "predictions.jsonl"
INCOMPLETE_PREDICTIONS = "predictions.incomplete.jsonl"


def predictions_from_verified(
    verified: Sequence[VerifiedFinding], *, run_id: str, system: SystemName
) -> list[Prediction]:
    """Every verifier decision becomes a prediction; only VERIFIED_FLAG counts as positive."""
    return [
        Prediction(
            run_id=run_id,
            system=system,
            paper_id=decision.finding.paper_id,
            variant=decision.finding.variant,
            error_class=decision.finding.error_class,
            anchor=decision.finding.anchor,
            target=decision.finding.target,
            state=decision.state,
            summary=decision.finding.summary,
            evidence=decision.accepted_evidence,
            finding_id=decision.finding.finding_id,
            severity=decision.severity,
            duplicate_of=decision.duplicate_of,
        )
        for decision in verified
    ]


def write_jsonl(path: Path, records: Iterable[BaseModel]) -> Path:
    lines = "".join(record.model_dump_json() + "\n" for record in records)
    atomic_write(path, lines.encode("utf-8"))
    return path


class RunDirectory:
    def __init__(self, runs_dir: Path, run_id: str) -> None:
        self.path = runs_dir / run_id
        self.path.mkdir(parents=True, exist_ok=True)

    @property
    def usage_dir(self) -> Path:
        return self.path / "usage"

    def paper_dir(self, paper: ParsedPaper) -> Path:
        return self.path / f"{paper.paper_id}-{paper.variant.value}"

    def save_audit(
        self,
        paper: ParsedPaper,
        outcome: AuditOutcome,
        predictions: Sequence[Prediction],
        *,
        complete: bool = True,
    ) -> Path:
        """Writes one paper's outputs; `complete` is False when some of its checks ran out
        of model calls."""
        folder = self.paper_dir(paper)
        write_jsonl(folder / "tasks.jsonl", outcome.tasks)
        write_jsonl(folder / "results.jsonl", outcome.results)
        write_jsonl(folder / "verified.jsonl", outcome.verified)
        name, other = (
            (PREDICTIONS, INCOMPLETE_PREDICTIONS)
            if complete
            else (INCOMPLETE_PREDICTIONS, PREDICTIONS)
        )
        write_jsonl(folder / name, predictions)
        (folder / other).unlink(missing_ok=True)
        report = build_report(
            paper, outcome.tasks, outcome.results, outcome.verified, run_id=self.path.name
        )
        write_report(folder, report)
        return folder

    def save_predictions(self, paper: ParsedPaper, predictions: Sequence[Prediction]) -> Path:
        return write_jsonl(self.paper_dir(paper) / PREDICTIONS, predictions)
