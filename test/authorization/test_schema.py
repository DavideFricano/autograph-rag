import pytest

from autograph_rag.authorization.filter import And, Match, Not, Or
from autograph_rag.authorization.schema import AccessSchema, Attribute, AttributeType


def _schema() -> AccessSchema:
    return AccessSchema(
        [
            Attribute(name="tenant", type=AttributeType.KEYWORD),
            Attribute(name="classification", type=AttributeType.KEYWORD),
            Attribute(name="care_team", type=AttributeType.KEYWORD, multi=True),
            Attribute(name="retention_years", type=AttributeType.INTEGER),
            Attribute(name="exportable", type=AttributeType.BOOLEAN),
        ]
    )


def test_duplicate_declaration_is_refused():
    with pytest.raises(ValueError):
        AccessSchema(
            [
                Attribute(name="tenant", type=AttributeType.KEYWORD),
                Attribute(name="tenant", type=AttributeType.INTEGER),
            ]
        )


def test_validate_access_accepts_declared_attributes():
    access = _schema().validate_access(
        {
            "tenant": "acme",
            "care_team": ["cardiology", "icu"],
            "retention_years": 10,
            "exportable": False,
        }
    )
    assert access["care_team"] == ["cardiology", "icu"]
    assert access["exportable"] is False


def test_validate_access_rejects_undeclared_attribute():
    """A typo must fail at ingestion, not become an attribute nothing ever filters on."""
    with pytest.raises(ValueError, match="undeclared"):
        _schema().validate_access({"tenat": "acme"})


def test_validate_access_rejects_wrong_type():
    with pytest.raises(ValueError, match="expects"):
        _schema().validate_access({"retention_years": "ten"})


def test_bool_and_int_are_not_interchangeable():
    """bool subclasses int in Python, so without an explicit check 0 would pass as a
    boolean and True as an integer."""
    with pytest.raises(ValueError):
        _schema().validate_access({"exportable": 0})
    with pytest.raises(ValueError):
        _schema().validate_access({"retention_years": True})


def test_multi_valued_attribute_wants_a_collection():
    with pytest.raises(ValueError, match="multi-valued"):
        _schema().validate_access({"care_team": "cardiology"})


def test_single_valued_attribute_refuses_a_collection():
    with pytest.raises(ValueError, match="expects"):
        _schema().validate_access({"tenant": ["acme", "globex"]})


def test_validate_filter_walks_the_whole_tree():
    """The undeclared attribute is buried under And -> Or -> Not, so a shallow check
    would let it reach the backend and silently match nothing."""
    predicate = And(
        Match(attribute="tenant", values={"acme"}),
        Or(
            Match(attribute="classification", values={"public"}),
            Not(Match(attribute="undeclared", values={"x"})),
        ),
    )
    with pytest.raises(ValueError, match="undeclared"):
        _schema().validate_filter(predicate)


def test_validate_filter_accepts_a_well_formed_predicate():
    predicate = And(
        Match(attribute="tenant", values={"acme"}),
        Not(Match(attribute="classification", values={"confidential"})),
    )
    assert _schema().validate_filter(predicate) is None


def test_validate_filter_checks_value_types_too():
    with pytest.raises(ValueError, match="expects"):
        _schema().validate_filter(Match(attribute="retention_years", values={"ten"}))
