"""Integrity tasks: one per paper, for problems in its own text (decision D22)."""

from auditor.schemas import AuditTask, DocumentAnchor, ParsedPaper, RuleTarget, TaskKind

RULE_ID = "unfinished_content"
# Statements about the paper's own results checked against its tables.
RESULTS_RULE_ID = "result_consistency"


def integrity_tasks(paper: ParsedPaper) -> list[AuditTask]:
    return [
        AuditTask.create(
            paper_id=paper.paper_id,
            variant=paper.variant,
            kind=TaskKind.INTEGRITY,
            anchor=DocumentAnchor(),
            target=RuleTarget(rule_id=rule),
        )
        for rule in (RULE_ID, RESULTS_RULE_ID)
    ]
