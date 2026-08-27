from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field

from autograph_rag.types import AttributeValue


class Filter(BaseModel):
    """Backend-neutral authorization predicate: what the PEP hands to the indexes.

    Not the vector backend's filter type — the policy engine's dialect and the backend's
    dialect both stay outside the library, so each index translates this into its own and
    nothing else has to know either.

    Unknown keys are refused: a predicate may be deserialized from what the PDP returned,
    and a field silently dropped there would quietly widen or narrow the authorization.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


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


class Allow(Filter):
    """Everything is authorized: the explicit spelling of "no restriction".

    Carries no field because it is the constant of the algebra — "no restriction" has no
    parameters; the semantics live in ``evaluate`` and in each backend's translation,
    which dispatch on the type. Same runtime effect as passing no filter at all, opposite
    meaning to whoever reads the code or the audit log: ``None`` is an omission,
    ``Allow()`` is a decision. It is what a deployment that declares an ``AccessSchema``
    writes when a call genuinely has no restriction (a reindex job, an internal
    evaluation), since there omitting the filter is refused. Its counterpart is the
    canonical deny, a ``Match`` with no values.

    That an empty ``And``/``Or`` is refused is not in tension with this: the objection
    there was that authorizing the whole corpus must never happen *by accident*.

    It waives the *policy*, not the schema's integrity: a chunk missing an attribute the
    schema requires stays out even under ``Allow()``, because that check says the data
    doesn't honour the declaration, not that the caller lacks a right.
    """


def evaluate(
    predicate: Filter, access: Mapping[str, AttributeValue | list[AttributeValue]]
) -> bool:
    """Whether these access attributes satisfy the predicate.

    The reference semantics of the algebra: a backend that pushes a filter down must
    agree with this, which makes it both the fallback for backends that cannot filter and
    the oracle a pushdown is tested against. An attribute the chunk doesn't carry never
    matches, so an unlabeled chunk is denied rather than allowed. Values are compared by
    type as well, since ``bool`` subclasses ``int`` and ``True`` would otherwise satisfy a
    filter asking for ``1``.
    """
    match predicate:
        case Match(attribute=name, values=values):
            if name not in access:
                return False
            held = access[name]
            candidates = held if isinstance(held, list) else [held]
            return any(
                type(value) is type(candidate) and value == candidate
                for candidate in candidates
                for value in values
            )
        case And(clauses=clauses):
            return all(evaluate(clause, access) for clause in clauses)
        case Or(clauses=clauses):
            return any(evaluate(clause, access) for clause in clauses)
        case Not(clause=clause):
            return not evaluate(clause, access)
        case Allow():
            return True
        case _:
            raise ValueError(f"unsupported filter node: {type(predicate).__name__}")
