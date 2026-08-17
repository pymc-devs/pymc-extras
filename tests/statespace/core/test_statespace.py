import pickle

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose
from pymc.exceptions import ImputationWarning
from pytensor.graph.traversal import ancestors
from pytensor.scan.op import Scan

from pymc_extras.statespace import BayesianETS, BayesianSARIMAX, BayesianVARMAX
from pymc_extras.statespace.core.statespace import FILTER_FACTORY, PyMCStateSpace
from pymc_extras.statespace.models import structural as st
from pymc_extras.statespace.models.DFM import BayesianDynamicFactor
from pymc_extras.statespace.utils.constants import JITTER_DEFAULT, MISSING_FILL
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


def test_unpack_matrices(rng):
    p, m, r, n = 2, 5, 1, 10
    data, *inputs = make_test_inputs(p, m, r, n, rng, missing_data=0)
    mod = make_statespace_mod(
        k_endog=p, k_states=m, k_posdef=r, filter_type="standard", verbose=False
    )

    outputs = mod._unpack_statespace_with_placeholders()
    for x, y in zip(inputs, outputs, strict=True):
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
            ss_mod.build_statespace_graph(data=np.full((10, 1), np.nan, dtype=floatX))


def test_build_statespace_graph_raises_if_data_has_missing_fill():
    # Breaks tests if it uses the session fixtures because we can't call build_statespace_graph over and over
    ss_mod = st.LevelTrend(name="trend", order=1, innovations_order=0).build(verbose=False)

    with pm.Model() as pymc_mod:
        initial_trend = pm.Normal("initial_trend", shape=(1,))
        P0 = pm.Deterministic("P0", pt.eye(1, dtype=floatX))
        with pytest.raises(ValueError, match=r"Provided data contains the value 1.0"):
            data = np.ones((10, 1), dtype=floatX)
            data[3] = np.nan
            ss_mod.build_statespace_graph(data=data, missing_fill_value=1.0)


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


def test_filter_config_defaults_to_the_documented_values():
    """The constructor defaults are the values themselves, not a sentinel resolved further down."""
    ss_mod = st.LevelTrend(name="trend", order=1, innovations_order=1).build(verbose=False)

    assert ss_mod.cov_jitter == JITTER_DEFAULT
    assert ss_mod.missing_fill_value == MISSING_FILL


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


@pytest.mark.filterwarnings("ignore:No time index found on the supplied data")
def test_register_additional_statespace_variables_hook():
    """A subclass adds its own nodes through the hook, not by overriding build_statespace_graph."""

    class SubclassStateSpace(PyMCStateSpace):
        def make_symbolic_graph(self):
            rho = self.make_and_register_variable("rho", ())
            self.ssm["transition", 0, 0] = rho
            self.ssm["design", 0, 0] = 1.0
            self.ssm["selection", 0, 0] = 1.0
            self.ssm["state_cov", 0, 0] = 1.0
            self.ssm["initial_state_cov", 0, 0] = 1.0

        @property
        def param_names(self):
            return ["rho"]

        def _register_additional_statespace_variables(self):
            pm.Deterministic("subclass_det", pm.modelcontext(None)["data"].sum())
            pm.Potential("subclass_check", pt.zeros(()))

    ss_mod = SubclassStateSpace(
        k_endog=1, k_states=1, k_posdef=1, filter_type="standard", verbose=False
    )

    with pm.Model() as pymc_mod:
        pm.Normal("rho")
        ss_mod.build_statespace_graph(np.arange(10, dtype=floatX)[:, None])

    assert "subclass_det" in [x.name for x in pymc_mod.deterministics]
    assert "subclass_check" in [x.name for x in pymc_mod.potentials]
    assert np.isfinite(pymc_mod.compile_logp()(pymc_mod.initial_point()))


@pytest.mark.filterwarnings("ignore:No time index found on the supplied data")
def test_statespace_model_survives_pickling():
    """A statespace model holds no PyMC-model references, so it round-trips and still builds."""
    ss_mod = st.LevelTrend(name="trend", order=1, innovations_order=1).build(verbose=False)
    unpickled = pickle.loads(pickle.dumps(ss_mod))

    with pm.Model(coords=unpickled.coords) as pymc_mod:
        pm.Normal("initial_trend", shape=(1,))
        pm.Deterministic("P0", pt.eye(1, dtype=floatX))
        pm.Exponential("sigma_trend", 1, shape=(1,))
        unpickled.build_statespace_graph(np.arange(10, dtype=floatX)[:, None])

    assert np.isfinite(pymc_mod.compile_logp()(pymc_mod.initial_point()))


def test_make_filters_carries_the_models_filter_config():
    """Post-estimation graphs filter with the settings the fit used, not the library defaults."""
    ss_mod = st.LevelTrend(name="trend", order=1, innovations_order=1).build(
        verbose=False, cov_jitter=1e-3, missing_fill_value=-777.0
    )
    kalman_filter, kalman_smoother = ss_mod.make_filters()

    assert kalman_filter.cov_jitter == 1e-3
    assert kalman_filter.missing_fill_value == -777.0
    assert kalman_smoother.cov_jitter == 1e-3


def test_build_graph_does_not_mutate_the_filters(ss_mod):
    """Filters carry their settings from construction, so building a graph writes nothing back."""
    kalman_filter, kalman_smoother = ss_mod.make_filters()
    filter_state = dict(kalman_filter.__dict__)
    smoother_state = dict(kalman_smoother.__dict__)

    n_timesteps = 25
    matrices = ss_mod._insert_constant_timestep(
        list(ss_mod._unpack_statespace_with_placeholders()), n_timesteps
    )
    _, _, _, _, T, _, R, _, Q = matrices
    outputs = kalman_filter.build_graph(pt.zeros((n_timesteps, ss_mod.k_endog)), *matrices)
    kalman_smoother.build_graph(T, R, Q, outputs[0], outputs[3])

    assert kalman_filter.__dict__ == filter_state
    assert kalman_smoother.__dict__ == smoother_state


def test_unpack_statespace_binds_matrices_to_the_active_model(ss_mod, pymc_mod):
    """The escape hatch for building further deterministics off the fitted system."""
    with pymc_mod:
        matrices = ss_mod.unpack_statespace()
    x0, P0, c, d, T, Z, R, H, Q = matrices

    assert pymc_mod["rho"] in set(ancestors([T]))
    assert not set(ss_mod._name_to_variable.values()).intersection(ancestors(matrices))


@pytest.mark.filterwarnings("ignore:No time index found on the supplied data")
def test_unpack_statespace_does_not_depend_on_the_filter():
    """A matrix sized against n_timesteps takes its length from the data, not from the filter.

    Taking it from the observation instead would make every deterministic built off these matrices
    recompute the Kalman filter.
    """
    mod = (
        st.LevelTrend(order=1, innovations_order=1)
        + st.TimeSeasonality(season_length=4, duration=3, innovations=False, name="s")
    ).build(verbose=False)

    with pm.Model(coords=mod.coords) as pymc_mod:
        pm.Normal("initial_level_trend", dims=["state_level_trend"])
        pm.Deterministic("P0", pt.eye(mod.k_states))
        pm.Exponential("sigma_level_trend", 1, dims=["shock_level_trend"])
        pm.Normal("params_s", dims=["state_s"])
        mod.build_statespace_graph(np.zeros((30, 1), dtype=floatX))
        matrices = mod.unpack_statespace()

    assert pymc_mod["obs"] not in set(ancestors(matrices))
    assert not [
        variable
        for variable in ancestors(matrices)
        if variable.owner is not None and isinstance(variable.owner.op, Scan)
    ]


def test_unpack_statespace_outside_a_model_raises(ss_mod):
    with pytest.raises(TypeError, match="No model on context stack"):
        ss_mod.unpack_statespace()


@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    [
        ("forecast", {"start": -1, "periods": 10}),
        ("impulse_response_function", {"n_steps": 10}),
        ("sample_conditional_prior", {}),
    ],
    ids=["forecast", "irf", "sample_conditional_prior"],
)
def test_post_estimation_leaves_template_matrices_alone(
    method_name, kwargs, ss_mod, pymc_mod, idata, rng
):
    """Each of these substitutes into a throwaway model, so none may touch the template.

    The three cover distinct substitution paths. Identity rather than equality: a cached list of
    substituted matrices would still compare equal element-wise while pointing at nodes bound to a
    model that has gone out of scope.
    """
    before = list(ss_mod._unpack_statespace_with_placeholders())
    getattr(ss_mod, method_name)(idata, random_seed=rng, **kwargs)
    after = list(ss_mod._unpack_statespace_with_placeholders())

    assert all(x is y for x, y in zip(before, after, strict=True))
    assert not set(pymc_mod.basic_RVs).intersection(ancestors(after))
