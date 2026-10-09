"""Experimental settings stated in a paper or found in its repository."""

from typing import Annotated, Literal

from pydantic import Field

from auditor.schemas.anchors import SentenceAnchor
from auditor.schemas.base import NonEmptyStr, Record

SettingValue = bool | int | float | str


class RepositoryRef(Record):
    url: NonEmptyStr
    commit: str | None = None


class PaperSource(Record):
    kind: Literal["paper"] = "paper"
    anchor: SentenceAnchor


class RepositorySource(Record):
    kind: Literal["repository"] = "repository"
    repository: RepositoryRef
    path: NonEmptyStr
    line: int = Field(ge=1)


SettingSource = Annotated[PaperSource | RepositorySource, Field(discriminator="kind")]


class ExtractedSetting(Record):
    """One setting, such as a learning rate, as written and in normalised form.

    `key` is the canonical name shared by the paper side and the repository side
    (for example `learning_rate` for both "lr" and "learning rate").
    """

    key: NonEmptyStr = Field(pattern=r"^[a-z][a-z0-9_]*$")
    raw_name: NonEmptyStr
    raw_value: NonEmptyStr
    value: SettingValue | None = None
    unit: str | None = None
    source: SettingSource
