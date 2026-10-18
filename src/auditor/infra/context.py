"""What a specialist's `check(task, ctx)` receives (decision D6)."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from auditor.config import Settings
from auditor.infra.http import HttpClient
from auditor.infra.llm import LLMNotConfiguredError, StructuredLLM
from auditor.schemas import ParsedPaper, RepositoryRef


@dataclass(frozen=True)
class RunContext:
    settings: Settings
    paper: ParsedPaper
    http: HttpClient
    llm: StructuredLLM | None
    logger: logging.Logger
    # The code repository of this run, when one is known; B1's code tools inspect it.
    repository: RepositoryRef | None = None
    # A stronger model for the review steps that judge other checks' findings, such as the
    # merge of the theory findings (AUDITOR_LLM_REVIEW_MODEL); None means the main model.
    review_llm: StructuredLLM | None = None
    # The PDF the paper was parsed from, when the input was a PDF; checks that need to look
    # at a page (a table the parser could not read) use it.
    source_pdf: Path | None = None
    # Counts the paper's tasks share, by name, such as the theory checker's deep checks
    # left for the paper. A context belongs to one paper, so they are per paper.
    allowances: dict[str, int] = field(default_factory=dict, compare=False)

    def require_llm(self) -> StructuredLLM:
        if self.llm is None:
            raise LLMNotConfiguredError("this run has no LLM configured")
        return self.llm

    def require_review_llm(self) -> StructuredLLM:
        return self.review_llm or self.require_llm()
