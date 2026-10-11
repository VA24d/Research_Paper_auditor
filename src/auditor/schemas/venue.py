"""Venue formatting rules checked by the template checker."""

from typing import Self

from pydantic import Field, JsonValue, model_validator

from auditor.schemas.base import SCHEMA_VERSION, NonEmptyStr, Record, SchemaVersion


class VenueRule(Record):
    """One formatting requirement.

    `check` names the deterministic check that implements the rule and `params` holds its
    arguments, for example `{"max_pages": 9}` for a page limit.
    """

    id: NonEmptyStr
    description: NonEmptyStr
    check: NonEmptyStr
    params: dict[str, JsonValue] = Field(default_factory=dict)
    needs_visual_inspection: bool = False


class VenueProfile(Record):
    schema_version: SchemaVersion = SCHEMA_VERSION
    id: NonEmptyStr
    name: NonEmptyStr
    rules: tuple[VenueRule, ...] = ()

    @model_validator(mode="after")
    def _check_unique_rules(self) -> Self:
        ids = [rule.id for rule in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("rule ids must be unique within a venue profile")
        return self

    def rule(self, rule_id: str) -> VenueRule:
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        raise KeyError(rule_id)
