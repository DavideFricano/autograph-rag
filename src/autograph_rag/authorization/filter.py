from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from autograph_rag.types import AttributeValue


class Filter(BaseModel):
    """Backend-neutral authorization predicate: what the PEP hands to the indexes.

    Not the vector backend's filter type — the policy engine's dialect and the backend's
    dialect both stay outside the library, so each index translates this into its own and
    nothing else has to know either.
    """

    model_config = ConfigDict(frozen=True)


class Match(Filter):
    """The chunk's ``attribute`` holds one of ``values``; a single value is equality.

    No values matches nothing, which is the honest reading of a policy that authorized
    none of them.
    """

    attribute: str = Field(description="Name of the access attribute to test")
    values: frozenset[AttributeValue] = Field(description="Values authorized for it")


class And(Filter):
    """Every clause must hold.

    At least one clause is required: an empty conjunction is vacuously true, so it would
    silently authorize the whole corpus.
    """

    clauses: tuple[Filter, ...] = Field(min_length=1, description="Clauses to conjoin")

    def __init__(self, *clauses: Filter) -> None:
        super().__init__(clauses=clauses)


class Or(Filter):
    """At least one clause must hold, and at least one must be given: a deny is written as
    a ``Match`` with no values, so it keeps a single spelling."""

    clauses: tuple[Filter, ...] = Field(min_length=1, description="Clauses to disjoin")

    def __init__(self, *clauses: Filter) -> None:
        super().__init__(clauses=clauses)


class Not(Filter):
    """The clause must not hold."""

    clause: Filter = Field(description="Clause to negate")

    def __init__(self, clause: Filter) -> None:
        super().__init__(clause=clause)
