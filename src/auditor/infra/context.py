"""What a specialist's `check(task, ctx)` receives (decision D6)."""

import logging
from dataclasses import dataclass

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

    def require_llm(self) -> StructuredLLM:
        if self.llm is None:
            raise LLMNotConfiguredError("this run has no LLM configured")
        return self.llm

    def require_review_llm(self) -> StructuredLLM:
        return self.review_llm or self.require_llm()
