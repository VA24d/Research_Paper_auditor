"""Collects per-call usage records for a run and appends them to JSON Lines files."""

from pathlib import Path

from auditor.schemas import HttpRequestRecord, LLMCallRecord, ToolCallRecord

UsageRecord = ToolCallRecord | HttpRequestRecord | LLMCallRecord

_FILE_NAMES: dict[type[UsageRecord], str] = {
    ToolCallRecord: "tool_calls.jsonl",
    HttpRequestRecord: "http_requests.jsonl",
    LLMCallRecord: "llm_calls.jsonl",
}


class UsageLog:
    """In-memory record of every call in a run, optionally mirrored to `directory`."""

    def __init__(self, directory: Path | None = None) -> None:
        self.directory = directory
        self.tool_calls: list[ToolCallRecord] = []
        self.http_requests: list[HttpRequestRecord] = []
        self.llm_calls: list[LLMCallRecord] = []
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    def add(self, record: UsageRecord) -> None:
        match record:
            case ToolCallRecord():
                self.tool_calls.append(record)
            case HttpRequestRecord():
                self.http_requests.append(record)
            case LLMCallRecord():
                self.llm_calls.append(record)
        if self.directory is not None:
            path = self.directory / _FILE_NAMES[type(record)]
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(record.model_dump_json() + "\n")
