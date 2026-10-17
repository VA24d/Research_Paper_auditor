"""Integrity tasks: one per paper, for problems in its own text (decision D22)."""

from auditor.schemas import AuditTask, DocumentAnchor, ParsedPaper, RuleTarget, TaskKind

RULE_ID = "unfinished_content"


def integrity_tasks(paper: ParsedPaper) -> list[AuditTask]:
    return [
        AuditTask.create(
            paper_id=paper.paper_id,
            variant=paper.variant,
            kind=TaskKind.INTEGRITY,
            anchor=DocumentAnchor(),
            target=RuleTarget(rule_id=RULE_ID),
        )
    ]
