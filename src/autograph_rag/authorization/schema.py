from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from autograph_rag.authorization.filter import And, Filter, Match, Not, Or
from autograph_rag.types import AttributeValue


class AttributeType(StrEnum):
    """Value type an access attribute accepts; each maps to one exact-match payload index."""

    KEYWORD = "keyword"
    INTEGER = "integer"
    BOOLEAN = "bool"


_PYTHON_TYPE: dict[AttributeType, type] = {
    AttributeType.KEYWORD: str,
    AttributeType.INTEGER: int,
    AttributeType.BOOLEAN: bool,
}


class Attribute(BaseModel):
    """One declared access attribute."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="Payload field name the attribute is indexed under")
    type: AttributeType = Field(description="Value type the attribute accepts")
    multi: bool = Field(default=False, description="Whether a chunk may carry several values")


class AccessSchema:
    """The closed vocabulary of access attributes a deployment declares.

    One declaration, three consumers: it validates what the labeler writes at ingestion,
    tells each index which payload fields to index, and rejects a filter naming an
    attribute nobody declared — so an undeclared attribute has no silent path through.
    """

    def __init__(self, attributes: Sequence[Attribute]) -> None:
        self.attributes = tuple(attributes)
        self._by_name = {attribute.name: attribute for attribute in self.attributes}
        if len(self._by_name) != len(self.attributes):
            raise ValueError("duplicate attribute name in the access schema")

    def attribute(self, name: str) -> Attribute:
        """The declared attribute under ``name``; an undeclared name never passes."""
        if name not in self._by_name:
            raise ValueError(f"undeclared access attribute: {name!r}")
        return self._by_name[name]

    def validate_access(
        self, access: Mapping[str, object]
    ) -> dict[str, AttributeValue | list[AttributeValue]]:
        """Checks what the labeler wrote, before it reaches the store and the payloads."""
        validated: dict[str, AttributeValue | list[AttributeValue]] = {}
        for name, value in access.items():
            attribute = self.attribute(name)
            if attribute.multi:
                if isinstance(value, str | bytes) or not isinstance(value, Iterable):
                    raise ValueError(f"attribute {name!r} is multi-valued: expected a collection")
                validated[name] = [self._checked(attribute, item) for item in value]
            else:
                validated[name] = self._checked(attribute, value)
        return validated

    def validate_filter(self, predicate: Filter) -> None:
        """Walks the predicate and rejects any attribute or value the schema doesn't declare."""
        match predicate:
            case Match(attribute=name, values=values):
                attribute = self.attribute(name)
                for value in values:
                    self._checked(attribute, value)
            case And(clauses=clauses) | Or(clauses=clauses):
                for clause in clauses:
                    self.validate_filter(clause)
            case Not(clause=clause):
                self.validate_filter(clause)
            case _:
                raise ValueError(f"unsupported filter node: {type(predicate).__name__}")

    def _checked(self, attribute: Attribute, value: object) -> AttributeValue:
        # bool is an int subclass, so it needs a check of its own in both directions
        expected_bool = attribute.type is AttributeType.BOOLEAN
        if isinstance(value, bool) is not expected_bool or not isinstance(
            value, _PYTHON_TYPE[attribute.type]
        ):
            raise ValueError(
                f"attribute {attribute.name!r} expects {attribute.type}, got {value!r}"
            )
        return value
