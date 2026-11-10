"""Audit reports (part 4, team plan P4.4): what a human reviewer reads.

`auditor audit` writes `report.md` and `report.json` next to a paper's other outputs.
`auditor report <paper folder> --paper <ParsedPaper JSON>` rebuilds them for a finished run.
"""

from pathlib import Path

from auditor.infra.files import atomic_write
from auditor.reporting.report import AuditReport, build_report, render_markdown

REPORT_JSON = "report.json"
REPORT_MARKDOWN = "report.md"


def write_report(folder: Path, report: AuditReport) -> Path:
    """Writes the report as JSON and Markdown into `folder`; returns the Markdown path."""
    atomic_write(folder / REPORT_JSON, (report.model_dump_json(indent=1) + "\n").encode("utf-8"))
    markdown = folder / REPORT_MARKDOWN
    atomic_write(markdown, render_markdown(report).encode("utf-8"))
    return markdown


__all__ = ["AuditReport", "build_report", "render_markdown", "write_report"]
