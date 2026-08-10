import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose
from pymc.exceptions import ImputationWarning

from pymc_extras.statespace import BayesianETS, BayesianSARIMAX, BayesianVARMAX
from pymc_extras.statespace.core.statespace import FILTER_FACTORY, PyMCStateSpace
from pymc_extras.statespace.models import structural as st
from pymc_extras.statespace.models.DFM import BayesianDynamicFactor
from pymc_extras.statespace.utils.constants import MISSING_FILL
from tests.statespace.shared_fixtures import (
    rng,
)
from tests.statespace.test_utilities import (
    fast_eval,
    make_statespace_mod,
    make_test_inputs,
)

floatX = pytensor.config.floatX


def test_invalid_filter_name_raises():
    msg = "The following are valid filter types: " + ", ".join(list(FILTER_FACTORY.keys()))
    with pytest.raises(NotImplementedError, match=msg):
        mod = make_statespace_mod(k_endog=1, k_states=5, k_posdef=1, filter_type="invalid_filter")


def test_unpack_before_insert_raises(rng):
    p, m, r, n = 2, 5, 1, 10
    data, *inputs = make_test_inputs(p, m, r, n, rng, missing_data=0)
    mod = make_statespace_mod(
        k_endog=p, k_states=m, k_posdef=r, filter_type="standard", verbose=False
    )

    msg = "Cannot unpack the complete statespace system until PyMC model variables have been inserted."
    with pytest.raises(ValueError, match=msg):
        outputs = mod.unpack_statespace()


def test_unpack_matrices(rng):
    p, m, r, n = 2, 5, 1, 10
    data, *inputs = make_test_inputs(p, m, r, n, rng, missing_data=0)
    mod = make_statespace_mod(
        k_endog=p, k_states=m, k_posdef=r, filter_type="standard", verbose=False
    )

    # mod is a dummy statespace, so there are no placeholders to worry about. Monkey patch subbed_ssm with the defaults
    mod.subbed_ssm = mod._unpack_statespace_with_placeholders()

    outputs = mod.unpack_statespace()
    for x, y in zip(inputs, outputs):
        assert_allclose(np.zeros_like(x), fast_eval(y))


def test_base_class_raises():
    with pytest.raises(NotImplementedError):
        mod = PyMCStateSpace(
            k_endog=1, k_states=5, k_posdef=1, filter_type="standard", verbose=False
        )


def test_update_raises_if_missing_variables(ss_mod):
    with pm.Model() as mod:
        rho = pm.Normal("rho")
        msg = "The following required model parameters were not found in the PyMC model: zeta"
        with pytest.raises(ValueError, match=msg):
            ss_mod._insert_random_variables()


def test_build_statespace_graph_warns_if_data_has_nans():
    # Breaks tests if it uses the session fixtures because we can't call build_statespace_graph over and over
    ss_mod = st.LevelTrend(name="trend", order=1, innovations_order=0).build(verbose=False)

    with pm.Model() as pymc_mod:
        initial_trend = pm.Normal("initial_trend", shape=(1,))
        P0 = pm.Deterministic("P0", pt.eye(1, dtype=floatX))
        with pytest.warns(ImputationWarning):
            ss_mod.build_statespace_graph(
                data=np.full((10, 1), np.nan, dtype=floatX), register_data=False
            )


def test_build_statespace_graph_raises_if_data_has_missing_fill():
    # Breaks tests if it uses the session fixtures because we can't call build_statespace_graph over and over
    ss_mod = st.LevelTrend(name="trend", order=1, innovations_order=0).build(verbose=False)

    with pm.Model() as pymc_mod:
        initial_trend = pm.Normal("initial_trend", shape=(1,))
        P0 = pm.Deterministic("P0", pt.eye(1, dtype=floatX))
        with pytest.raises(ValueError, match=r"Provided data contains the value 1.0"):
            data = np.ones((10, 1), dtype=floatX)
            data[3] = np.nan
            ss_mod.build_statespace_graph(data=data, missing_fill_value=1.0, register_data=False)


def test_build_statespace_graph(pymc_mod):
    for name in [
        "filtered_states",
        "predicted_states",
        "predicted_covariances",
        "filtered_covariances",
    ]:
        assert name in [x.name for x in pymc_mod.deterministics]


def test_build_smoother_graph(pymc_mod):
    names = ["smoothed_states", "smoothed_covariances"]
    for name in names:
        assert name in [x.name for x in pymc_mod.deterministics]


def test_param_dims_coords(ss_mod_multi_component):
    for param in ss_mod_multi_component.param_names:
        shape = ss_mod_multi_component.param_info[param]["shape"]
        dims = ss_mod_multi_component.param_dims.get(param, None)
        if len(shape) == 0:
            assert dims is None
            continue
        for i, s in zip(shape, dims):
            assert i == len(ss_mod_multi_component.coords[s]), (
                f"Mismatch between shape {i} and dimension {s}"
            )


@pytest.mark.filterwarnings("ignore:No time index found on the supplied data")
def test_missing_fill_value_reaches_post_estimation_graphs(rng):
    """Post-estimation graphs must mask missing values with the sentinel the fit used.

    Every observation here is the library's default sentinel but is declared real, via a custom
    ``missing_fill_value``. A post-estimation graph that fell back to the default would treat the whole
    series as missing and return the prior rather than states tracking the data.
    """
    data = np.full((20, 1), MISSING_FILL, dtype=floatX)
    ss_mod = st.LevelTrend(name="trend", order=1, innovations_order=1).build(
        verbose=False, missing_fill_value=-1234.0
    )

    with pm.Model(coords=ss_mod.coords):
        pm.Normal("initial_trend", mu=0.0, sigma=1.0, shape=(1,))
        pm.Deterministic("P0", pt.eye(1, dtype=floatX))
        pm.Exponential("sigma_trend", 1, shape=(1,))
        ss_mod.build_statespace_graph(data=data)
        idata = pm.sample_prior_predictive(draws=5, random_seed=rng)

    conditional = ss_mod.sample_conditional_prior(idata, random_seed=rng)
    filtered_level = conditional["filtered_prior"].values[..., -1, 0]

    assert_allclose(filtered_level, MISSING_FILL, rtol=1e-4)


@pytest.mark.filterwarnings("ignore:No time index found on the supplied data")
def test_filter_config_is_overridable_at_build():
    """A build-time argument overrides the model's filter settings; others are left alone."""
    ss_mod = st.LevelTrend(name="trend", order=1, innovations_order=1).build(
        verbose=False, cov_jitter=1e-4, missing_fill_value=-1234.0
    )

    with pm.Model(coords=ss_mod.coords):
        pm.Normal("initial_trend", shape=(1,))
        pm.Deterministic("P0", pt.eye(1, dtype=floatX))
        pm.Exponential("sigma_trend", 1, shape=(1,))
        ss_mod.build_statespace_graph(data=np.zeros((20, 1), dtype=floatX), cov_jitter=1e-2)

    assert ss_mod.cov_jitter == 1e-2
    assert ss_mod.missing_fill_value == -1234.0


@pytest.mark.parametrize(
    "make_model",
    [
        pytest.param(
            lambda **kw: BayesianSARIMAX(order=(1, 0, 1), stationary_initialization=True, **kw),
            id="SARIMAX",
        ),
        pytest.param(
            lambda **kw: BayesianVARMAX(order=(1, 0), endog_names=["a", "b"], **kw), id="VARMAX"
        ),
        pytest.param(
            lambda **kw: BayesianETS(order=("A", "N", "N"), endog_names=["a"], **kw), id="ETS"
        ),
        pytest.param(
            lambda **kw: BayesianDynamicFactor(
                k_factors=1, factor_order=1, endog_names=["a", "b"], **kw
            ),
            id="DFM",
        ),
        pytest.param(
            lambda **kw: st.LevelTrend(name="trend", order=1, innovations_order=1).build(**kw),
            id="structural",
        ),
    ],
)
def test_shipped_models_accept_filter_config(make_model):
    """Every shipped model exposes the filter settings its post-estimation graphs will use."""
    ss_mod = make_model(verbose=False, cov_jitter=1e-3, missing_fill_value=-777.0)

    assert ss_mod.cov_jitter == 1e-3
    assert ss_mod.missing_fill_value == -777.0
