import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.xtensor import as_xtensor

from pymc_extras.statespace.filters import StandardFilter
from pymc_extras.statespace.filters.batched import FILTER_INPUT_DIMS, build_batched_filter_graph
from pymc_extras.statespace.utils.constants import SHORT_NAME_TO_LONG, TIME_DIM
from tests.statespace.shared_fixtures import (  # pylint: disable=unused-import
    rng,
)
from tests.statespace.test_utilities import make_test_inputs

floatX = pytensor.config.floatX
ATOL = 1e-6 if floatX.endswith("64") else 1e-3
RTOL = 1e-6 if floatX.endswith("64") else 1e-3

INPUT_NAMES = ["data", "x0", "P0", "c", "d", "T", "Z", "R", "H", "Q"]
N_TIMESTEPS, N_SERIES, N_SCENARIOS = 10, 3, 2


def to_xtensor(name, value, batch_dims):
    time_dims = (TIME_DIM,) if TIME_DIM in batch_dims else ()
    other_dims = tuple(dim for dim in batch_dims if dim != TIME_DIM)
    return as_xtensor(pt.as_tensor(value), dims=(*time_dims, *other_dims, *FILTER_INPUT_DIMS[name]))


def build_batched_outputs(values, batch_dims):
    xtensors = [to_xtensor(name, values[name], batch_dims.get(name, ())) for name in INPUT_NAMES]
    return build_batched_filter_graph(StandardFilter(), *xtensors)


def evaluate_batched(values, batch_dims):
    outputs = build_batched_outputs(values, batch_dims)
    return {name: variable.values.eval() for name, variable in outputs.items()}


def evaluate_reference(values, time_varying=()):
    kalman_filter = StandardFilter(
        time_varying_names=[SHORT_NAME_TO_LONG[name] for name in time_varying]
    )
    outputs = kalman_filter.build_graph(*(pt.as_tensor(values[name]) for name in INPUT_NAMES))
    return {
        variable.name: value for variable, value in zip(outputs, pytensor.function([], outputs)())
    }


def series_inputs(rng):
    data, *params = make_test_inputs(2, 3, 2, N_TIMESTEPS, rng)
    series = np.stack([data + rng.normal(size=data.shape) for _ in range(N_SERIES)], axis=1)
    for index in range(N_SERIES):
        series[rng.choice(N_TIMESTEPS, size=index + 1, replace=False), index, index % 2] = np.nan
    return dict(zip(INPUT_NAMES, [series.astype(floatX), *params]))


def test_batch_over_data_matches_per_series_filter(rng):
    values = series_inputs(rng)
    batched = evaluate_batched(values, batch_dims={"data": (TIME_DIM, "series")})

    for index in range(N_SERIES):
        expected = evaluate_reference({**values, "data": values["data"][:, index]})
        for name, value in expected.items():
            assert_allclose(batched[name][:, index], value, atol=ATOL, rtol=RTOL, err_msg=name)


def test_batch_dims_from_data_and_params_combine(rng):
    values = series_inputs(rng)
    values["H"] = np.stack([values["H"] * scale for scale in (0.5, 2.0)]).astype(floatX)
    batch_dims = {"data": (TIME_DIM, "series"), "H": ("scenario",)}

    outputs = build_batched_outputs(values, batch_dims)
    assert outputs["filtered_covariances"].dims == (
        TIME_DIM,
        "series",
        "scenario",
        "state",
        "state_aux",
    )

    batched = {name: variable.values.eval() for name, variable in outputs.items()}
    for series_index in range(N_SERIES):
        for scenario_index in range(N_SCENARIOS):
            expected = evaluate_reference(
                {
                    **values,
                    "data": values["data"][:, series_index],
                    "H": values["H"][scenario_index],
                }
            )
            for name, value in expected.items():
                assert_allclose(
                    batched[name][:, series_index, scenario_index],
                    value,
                    atol=ATOL,
                    rtol=RTOL,
                    err_msg=name,
                )


def test_time_varying_matrix_with_batch_dim(rng):
    values = series_inputs(rng)
    noise = rng.normal(scale=0.1, size=(N_TIMESTEPS, N_SERIES, *values["Z"].shape))
    values["Z"] = (values["Z"] + noise).astype(floatX)
    batched = evaluate_batched(
        values, batch_dims={"data": (TIME_DIM, "series"), "Z": (TIME_DIM, "series")}
    )

    for index in range(N_SERIES):
        expected = evaluate_reference(
            {**values, "data": values["data"][:, index], "Z": values["Z"][:, index]},
            time_varying=["Z"],
        )
        for name, value in expected.items():
            assert_allclose(batched[name][:, index], value, atol=ATOL, rtol=RTOL, err_msg=name)


@pytest.mark.parametrize(
    "static_shape", [True, False], ids=["static_core_shape", "unknown_core_shape"]
)
def test_gradient_matches_per_series_filter(rng, static_shape):
    values = series_inputs(rng)
    H = pt.tensor("H", shape=values["H"].shape if static_shape else (None, None), dtype=floatX)
    xtensors = [
        to_xtensor(
            name, H if name == "H" else values[name], (TIME_DIM, "series") if name == "data" else ()
        )
        for name in INPUT_NAMES
    ]

    outputs = build_batched_filter_graph(StandardFilter(), *xtensors)
    loglike = rewrite_graph(outputs["loglike_obs"].values.sum(axis=0), include=("lower_xtensor",))
    batched_grads = pytensor.function([H], pt.jacobian(loglike, H))(values["H"])

    for index in range(N_SERIES):
        series_values = {**values, "data": values["data"][:, index]}
        series_inputs_ = [
            H if name == "H" else pt.as_tensor(series_values[name]) for name in INPUT_NAMES
        ]
        series_loglike = StandardFilter().build_graph(*series_inputs_)[-1].sum()
        expected = pytensor.function([H], pt.grad(series_loglike, H))(values["H"])
        assert_allclose(batched_grads[index], expected, atol=ATOL, rtol=RTOL)


def test_data_without_time_dim_raises(rng):
    values = series_inputs(rng)
    with pytest.raises(ValueError, match="must have a 'time' dim"):
        build_batched_filter_graph(
            StandardFilter(),
            as_xtensor(pt.as_tensor(values["data"][0]), dims=("series", "observed_state")),
            *(to_xtensor(name, values[name], ()) for name in INPUT_NAMES[1:]),
        )
