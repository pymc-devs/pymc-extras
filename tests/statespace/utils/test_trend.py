import numpy as np
import pytest

from numpy.testing import assert_allclose

from pymc_extras.statespace.utils.trend import parse_trend, trend_design, trend_names


@pytest.mark.parametrize(
    "trend, expected",
    [
        (None, ()),
        ("n", ()),
        ("c", (0,)),
        ("ct", (0, 1)),
        ("ctt", (0, 1, 2)),
        ([1, 1, 0, 1], (0, 1, 3)),
    ],
    ids=["none", "n", "c", "ct", "ctt", "flags"],
)
def test_parse_trend(trend, expected):
    assert parse_trend(trend) == expected


@pytest.mark.parametrize(
    "trend, message",
    [("cttt", "must be one of"), ([1, 2], "only 0 and 1 flags"), ([[1, 0]], "only 0 and 1 flags")],
    ids=["string", "not_flags", "2d"],
)
def test_parse_trend_rejects_invalid_specs(trend, message):
    with pytest.raises(ValueError, match=message):
        parse_trend(trend)


def test_trend_names_are_descriptive_then_numbered():
    assert trend_names((0, 1, 3, 4)) == ("constant", "linear", "cubic", "degree_4")


def test_trend_design_evaluates_powers_from_the_offset():
    design = trend_design((0, 1, 2), 4, offset=3).eval()

    time = np.arange(4) + 3
    assert_allclose(design, np.stack([np.ones(4), time, time**2], axis=1))
