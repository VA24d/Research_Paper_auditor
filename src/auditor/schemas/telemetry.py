"""Per-call records used for the cost metrics (decision D8).

`paper_id` and `variant` say which paper a call was for, so costs can be split per
paper and between original and injected copies; they are empty only for calls made
outside any paper.
"""

from enum import StrEnum

from pydantic import AwareDatetime, Field

from auditor.schemas.base import NonEmptyStr, Record
from auditor.schemas.paper import PaperVariant


class CallStatus(StrEnum):
    OK = "ok"
    ERROR = "error"


class ToolCallRecord(Record):
    run_id: NonEmptyStr
    paper_id: str | None = None
    variant: PaperVariant | None = None
    task_id: str | None = None
    tool: NonEmptyStr
    args_hash: NonEmptyStr
    started_at: AwareDatetime
    duration_s: float = Field(ge=0)
    status: CallStatus
    error: str | None = None


class HttpRequestRecord(Record):
    run_id: NonEmptyStr
    paper_id: str | None = None
    variant: PaperVariant | None = None
    task_id: str | None = None
    method: NonEmptyStr
    host: NonEmptyStr
    path: str
    status_code: int | None = None
    # One logical request: `attempts` counts tries and `redirects` the extra hops across
    # all of them, so `attempts + redirects` requests were sent.
    attempts: int = Field(ge=1)
    redirects: int = Field(default=0, ge=0)
    cache_hit: bool
    started_at: AwareDatetime
    duration_s: float = Field(ge=0)
    error: str | None = None


class LLMCallRecord(Record):
    run_id: NonEmptyStr
    paper_id: str | None = None
    variant: PaperVariant | None = None
    task_id: str | None = None
    model: NonEmptyStr
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    cache_hit: bool = False
    started_at: AwareDatetime
    duration_s: float = Field(ge=0)
    status: CallStatus
    error: str | None = None
