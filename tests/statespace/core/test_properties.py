import pytest

from pymc_extras.statespace.core.properties import (
    CoordInfo,
    Data,
    DataInfo,
    Parameter,
    ParameterInfo,
    Shock,
    ShockInfo,
    State,
    StateInfo,
)
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    OBS_STATE_AUX_DIM,
    OBS_STATE_DIM,
    SHOCK_AUX_DIM,
    SHOCK_DIM,
)


def test_property_str_formats_fields():
    p = Parameter(name="alpha", shape=(2,), dims=("param",))
    s = str(p).splitlines()
    assert s == [
        "name: alpha",
        "shape: (2,)",
        "dims: ('param',)",
        "constraints: None",
    ]


def test_info_lookup_contains_and_missing_key():
    params = [
        Parameter("a", (1,), ("d",)),
        Parameter("b", (2,), ("d",)),
        Parameter("c", (3,), ("d",)),
    ]
    info = ParameterInfo(params)

    assert info.get("b").name == "b"
    assert info["a"].shape == (1,)
    assert "c" in info

    with pytest.raises(KeyError) as e:
        _ = info["missing"]
    assert "No name 'missing'" in str(e.value)


def test_data_info_needs_exogenous_and_str():
    data = [
        Data("price", (10,), ("time",), is_exogenous=False),
        Data("x", (10,), ("time",), is_exogenous=True),
    ]
    info = DataInfo(data)

    assert info.needs_exogenous_data is True
    s = str(info)
    assert "data: ['price', 'x']" in s
    assert "needs exogenous data: True" in s

    no_exog = DataInfo([Data("y", (10,), ("time",), is_exogenous=False)])
    assert no_exog.needs_exogenous_data is False


def test_coord_info_make_defaults_from_component_and_types():
    class DummyComponent:
        state_names = ("x1", "x2")
        observed_states = ("x2",)
        shock_names = ("eps1",)

    coord_info = CoordInfo.default_coords_from_model(DummyComponent())

    expected = [
        (ALL_STATE_DIM, ("x1", "x2")),
        (ALL_STATE_AUX_DIM, ("x1", "x2")),
        (OBS_STATE_DIM, ("x2",)),
        (OBS_STATE_AUX_DIM, ("x2",)),
        (SHOCK_DIM, ("eps1",)),
        (SHOCK_AUX_DIM, ("eps1",)),
    ]

    assert len(coord_info) == 6
    assert isinstance(coord_info.items, tuple)

    for dim, labels in expected:
        assert dim in coord_info
        assert coord_info[dim].labels == labels


def test_state_info_and_shockinfo_basic():
    states = (
        State("x1", observed=True, shared=False),
        State("x2", observed=False, shared=True),
    )

    state_info = StateInfo(states)
    assert state_info[("x1", True)].observed is True
    s = str(state_info)

    assert "states: ['x1', 'x2']" in s
    assert "observed: [True, False]" in s

    shocks = (Shock("s1"), Shock("s2"))
    shock_info = ShockInfo(shocks)

    assert "s1" in shock_info
    assert shock_info["s2"].name == "s2"


def test_info_is_iterable_and_unpackable():
    items = (Parameter("p1", (1,), ("d",)), Parameter("p2", (2,), ("d",)))
    info = ParameterInfo(items)

    names = info.names
    assert names == ("p1", "p2")

    a, b = info.items
    assert a.name == "p1" and b.name == "p2"


def test_info_add_method():
    a_param = Parameter(name="a", shape=(1,), dims=("dim",))
    param_info = ParameterInfo(parameters=(a_param,))

    b_param = Parameter(name="b", shape=(1,), dims=("dim",))

    new_param_info = param_info.add(new_item=b_param)

    assert new_param_info.names == ("a", "b")


def test_info_merge_method():
    a_param = Parameter(name="a", shape=(1,), dims=("dim",))
    a_param_info = ParameterInfo(parameters=(a_param,))

    b_param = Parameter(name="b", shape=(1,), dims=("dim",))
    b_param_info = ParameterInfo(parameters=(b_param,))

    new_param_info = a_param_info.merge(b_param_info)

    assert new_param_info.names == ("a", "b")


def test_state_info_compound_key_allows_same_name_different_observed():
    """StateInfo uses compound key (name, observed) so same name with different observed is allowed."""
    states = (
        State("data", observed=False),
        State("other", observed=False),
        State("data", observed=True),
    )
    state_info = StateInfo(states)

    assert len(state_info) == 3
    assert state_info.unobserved_state_names == ("data", "other")
    assert state_info.observed_state_names == ("data",)
    assert state_info.names == ("data", "other", "data")

    assert state_info[("data", False)].observed is False
    assert state_info[("data", True)].observed is True

    assert "data" in state_info
    assert "other" in state_info
    assert "missing" not in state_info


def test_state_info_rejects_true_duplicates():
    """StateInfo rejects states with same name AND same observed value."""
    with pytest.raises(ValueError, match="Duplicate"):
        StateInfo(
            (
                State("x", observed=False),
                State("x", observed=False),
            )
        )


def test_state_info_merge_with_compound_keys():
    """Merging StateInfo works with compound keys."""
    info1 = StateInfo((State("a", observed=False), State("b", observed=True)))
    info2 = StateInfo((State("c", observed=False), State("a", observed=True)))

    merged = info1.merge(info2)

    assert len(merged) == 4
    assert ("a", False) in merged._index
    assert ("a", True) in merged._index
    assert ("b", True) in merged._index
    assert ("c", False) in merged._index


def test_state_info_merge_overwrite_duplicates():
    """Merging with overwrite_duplicates=True skips items with same compound key."""
    info1 = StateInfo((State("a", observed=False, shared=False),))
    info2 = StateInfo((State("a", observed=False, shared=True),))

    merged = info1.merge(info2, overwrite_duplicates=True)

    assert len(merged) == 1
    # Original kept, duplicate skipped
    assert merged[("a", False)].shared is False
