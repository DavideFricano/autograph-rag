import pytest
from pydantic import ValidationError

from autograph_rag.authorization.filter import And, Match, Not, Or


def test_nested_clauses_keep_their_concrete_type():
    """Clauses are declared as the abstract Filter, so the tree must not be flattened to
    it — an index translating the predicate needs to see Match, And, Or, Not."""
    predicate = And(
        Match(attribute="tenant", values={"acme"}),
        Not(Match(attribute="classification", values={"confidential"})),
    )
    assert [type(clause).__name__ for clause in predicate.clauses] == ["Match", "Not"]


def test_filters_are_frozen_and_hashable():
    predicate = Match(attribute="tenant", values={"acme"})
    assert hash(predicate)
    with pytest.raises(ValidationError):
        predicate.attribute = "other"


def test_empty_conjunction_is_refused():
    """An empty AND is vacuously true: allowed through, it would authorize the whole
    corpus without raising anything."""
    with pytest.raises(ValidationError):
        And()


def test_empty_disjunction_is_refused():
    """Safe on its own (an empty OR denies everything) but refused anyway, so a deny has
    exactly one spelling: a Match with no values."""
    with pytest.raises(ValidationError):
        Or()


def test_match_with_no_values_is_the_canonical_deny():
    assert Match(attribute="tenant", values=set()).values == frozenset()


def test_booleans_survive_the_value_union():
    """str | int | bool must not coerce True into 1, or a bool attribute stops matching."""
    values = Match(attribute="exportable", values={True}).values
    assert type(next(iter(values))) is bool
