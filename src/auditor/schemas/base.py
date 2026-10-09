from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

SCHEMA_VERSION: Final = 1
SchemaVersion = Literal[1]

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]


class Record(BaseModel):
    """Base for every contract model: immutable, and unknown fields are rejected."""

    model_config = ConfigDict(frozen=True, extra="forbid")
