from itertools import pairwise, product

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import statsmodels.api as sm

from numpy.testing import assert_allclose, assert_array_equal, assert_array_less
from pymc.model.transform.optimization import freeze_dims_and_data
from pymc.testing import mock_sample_setup_and_teardown

from pymc_extras.statespace import BayesianVARMAX
from pymc_extras.statespace.core.irf import DEFAULT_IRF_STEPS
from pymc_extras.statespace.utils.constants import SHORT_NAME_TO_LONG
from tests.statespace.shared_fixtures import (  # pylint: disable=unused-import
    rng,
)

mock_sample = pytest.fixture(scope="function")(mock_sample_setup_and_teardown)

floatX = pytensor.config.floatX
ps = [0, 1, 2, 3]
qs = [0, 1, 2, 3]
orders = list(product(ps, qs))[1:]
ids = [f"p={x[0]}, q={x[1]}" for x in orders]


@pytest.fixture(scope="session")
def data():
    df = pd.read_csv(
        "tests/statespace/_data/statsmodels_macrodata_processed.csv",
        index_col=0,
        parse_dates=True,
    ).astype(floatX)
    df.index.freq = df.index.inferred_freq
    return df


@pytest.fixture(scope="session")
def varma_mod(data):
    return BayesianVARMAX(
        endog_names=data.columns,
        order=(2, 0),
        stationary_initialization=True,
        verbose=False,
        measurement_error=True,
    )


@pytest.fixture(scope="session")
def pymc_mod(varma_mod, data):
    with pm.Model(coords=varma_mod.coords) as pymc_mod:
        # x0 = pm.Normal("x0", dims=["state"])
        # P0_diag = pm.Exponential("P0_diag", 1, size=varma_mod.k_states)
        # P0 = pm.Deterministic(
        #     "P0", pt.diag(P0_diag), dims=["state", "state_aux"]
        # )
        state_chol, *_ = pm.LKJCholeskyCov(
            "state_chol", n=varma_mod.k_posdef, eta=1, sd_dist=pm.Exponential.dist(1)
        )
        ar_params = pm.Normal(
            "ar_params", mu=0, sigma=0.1, dims=["observed_state", "lag_ar", "observed_state_aux"]
        )
        state_cov = pm.Deterministic(
            "state_cov", state_chol @ state_chol.T, dims=["shock", "shock_aux"]
        )
        sigma_obs = pm.Exponential("sigma_obs", 1, dims=["observed_state"])

        varma_mod.build_statespace_graph(data=data)

    return pymc_mod


@pytest.fixture(scope="session")
def idata(pymc_mod, rng):
    with pymc_mod:
        idata = pm.sample_prior_predictive(draws=10, random_seed=rng)

    return idata


def test_mode_argument():
    # Mode argument should be passed to the parent class
    mod = BayesianVARMAX(endog_names=["y1", "y2"], order=(3, 0), mode="FAST_RUN", verbose=False)
    assert mod.mode == "FAST_RUN"


@pytest.mark.parametrize("order", orders, ids=ids)
@pytest.mark.parametrize("var", ["AR", "MA", "state_cov"])
@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.EstimationWarning")
def test_VARMAX_param_counts_match_statsmodels(data, order, var):
    p, q = order

    mod = BayesianVARMAX(
        endog_names=["realgdp", "realcons", "realinv"], order=(p, q), verbose=False
    )
    sm_var = sm.tsa.VARMAX(data, order=(p, q))

    count = mod.param_counts[var]
    if var == "state_cov":
        # Statsmodels only counts the lower triangle
        count = mod.k_posdef * (mod.k_posdef - 1)
    assert count == sm_var.parameters[var.lower()]


@pytest.mark.parametrize("order", orders, ids=ids)
@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.EstimationWarning")
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_VARMAX_update_matches_statsmodels(data, order, rng):
    p, q = order

    sm_var = sm.tsa.VARMAX(data, order=(p, q))

    param_counts = [None, *np.cumsum(list(sm_var.parameters.values())).tolist()]
    param_slices = [slice(a, b) for a, b in pairwise(param_counts)]
    param_lists = [trend, ar, ma, reg, state_cov, obs_cov] = [
        sm_var.param_names[idx] for idx in param_slices
    ]
    param_d = {
        k: getattr(np, floatX)(rng.normal(scale=0.1) ** 2)
        for param_list in param_lists
        for k in param_list
    }

    res = sm_var.fit_constrained(param_d)

    mod = BayesianVARMAX(
        endog_names=["realgdp", "realcons", "realinv"],
        order=(p, q),
        verbose=False,
        measurement_error=False,
        stationary_initialization=False,
    )

    ar_shape = (mod.k_endog, mod.p, mod.k_endog)
    ma_shape = (mod.k_endog, mod.q, mod.k_endog)

    with pm.Model() as pm_mod:
        x0 = pm.Deterministic("x0", pt.zeros(mod.k_states, dtype=floatX))
        P0 = pm.Deterministic("P0", pt.eye(mod.k_states, dtype=floatX))
        ma_params = pm.Deterministic(
            "ma_params",
            pt.as_tensor_variable(np.array([param_d[var] for var in ma])).reshape(ma_shape),
        )
        ar_params = pm.Deterministic(
            "ar_params",
            pt.as_tensor_variable(np.array([param_d[var] for var in ar])).reshape(ar_shape),
        )
        state_chol = np.zeros((mod.k_posdef, mod.k_posdef), dtype=floatX)
        state_chol[np.tril_indices(mod.k_posdef)] = np.array([param_d[var] for var in state_cov])
        state_cov = pm.Deterministic("state_cov", pt.as_tensor_variable(state_chol @ state_chol.T))

        matrices = pm.draw(mod._insert_random_variables())
        matrix_dict = dict(zip(SHORT_NAME_TO_LONG.values(), matrices))

    for matrix in ["transition", "selection", "state_cov", "obs_cov", "design"]:
        assert_allclose(matrix_dict[matrix], sm_var.ssm[matrix])


def test_measurement_error_enters_obs_cov_as_a_variance():
    """``sigma_obs`` is a standard deviation, so the observation covariance holds its square."""
    mod = BayesianVARMAX(
        endog_names=["a", "b"],
        order=(1, 0),
        measurement_error=True,
        stationary_initialization=False,
        verbose=False,
    )
    sigma = np.array([0.5, 2.0], dtype=floatX)

    with pm.Model():
        pm.Deterministic("x0", pt.zeros(mod.k_states, dtype=floatX))
        pm.Deterministic("P0", pt.eye(mod.k_states, dtype=floatX))
        pm.Deterministic("ar_params", pt.zeros((mod.k_endog, mod.p, mod.k_endog), dtype=floatX))
        pm.Deterministic("state_cov", pt.eye(mod.k_posdef, dtype=floatX))
        pm.Deterministic("sigma_obs", pt.as_tensor_variable(sigma))
        matrices = pm.draw(mod._insert_random_variables())

    obs_cov = dict(zip(SHORT_NAME_TO_LONG.values(), matrices))["obs_cov"]
    assert_allclose(obs_cov, np.diag(sigma**2))


@pytest.mark.parametrize("filter_output", ["filtered", "predicted", "smoothed"])
def test_all_prior_covariances_are_PSD(filter_output, varma_mod, idata, rng):
    cov_name = f"{filter_output}_covariances"
    filter_idata = varma_mod.sample_filter_outputs(
        idata, filter_output_names=[cov_name], group="prior", random_seed=rng
    )
    cov_mats = filter_idata.posterior_predictive[cov_name].values.reshape(
        -1, varma_mod.k_states, varma_mod.k_states
    )
    w, v = np.linalg.eig(cov_mats)
    assert_array_less(0, w, err_msg=f"Smallest eigenvalue: {min(w.ravel())}")


@pytest.mark.skipif(floatX == "float32", reason="Impulse covariance not PSD if float32")
@pytest.mark.parametrize(
    "shock_kwargs",
    [
        {},
        {"shock_cov": np.array([[1.38, 0.58, -1.84], [0.58, 0.99, -0.82], [-1.84, -0.82, 2.51]])},
    ],
    ids=["from-posterior-cov", "user-cov"],
)
def test_impulse_response_from_covariance(varma_mod, idata, rng, shock_kwargs):
    """A covariance draws the impulse at random, so only states the shock cannot reach are known."""
    irf = varma_mod.impulse_response_function(
        idata, n_steps=10, group="prior", random_seed=rng, **shock_kwargs
    )

    # The selection matrix routes shocks to a subset of states; the rest can only move via the
    # intercept in the impact period, whatever impulse was drawn.
    R = varma_mod.ssm["selection"].eval()
    c = varma_mod.ssm["state_intercept"].eval()
    unreachable = ~R.any(axis=1)

    assert unreachable.any(), "Fixture no longer has states outside the shock's reach"

    impact = irf.irf.isel(chain=0, time=0).values[:, unreachable]
    assert_allclose(impact, np.broadcast_to(c[unreachable], impact.shape), atol=1e-12)


@pytest.mark.skipif(floatX == "float32", reason="Impulse covariance not PSD if float32")
def test_impulse_response_of_fixed_shock(varma_mod, idata, rng):
    """A fixed shock_size makes the impulse deterministic, so the impact period is known exactly."""
    shock_size = np.array([1.0, 0.0, 0.0], dtype=floatX)
    irf = varma_mod.impulse_response_function(
        idata, n_steps=5, shock_size=shock_size, group="prior", random_seed=rng
    )

    R = varma_mod.ssm["selection"].eval()
    impact = irf.irf.isel(chain=0, time=0).values
    assert_allclose(impact, np.broadcast_to(R @ shock_size, impact.shape), atol=1e-8)

    # The system is linear in the impulse, so doubling the shock doubles the whole path.
    doubled = varma_mod.impulse_response_function(
        idata, n_steps=5, shock_size=shock_size * 2, group="prior", random_seed=rng
    )
    assert_allclose(doubled.irf.values, 2 * irf.irf.values, rtol=1e-8)


@pytest.mark.skipif(floatX == "float32", reason="Impulse covariance not PSD if float32")
def test_impulse_response_respects_shock_trajectory_timing(varma_mod, idata, rng):
    """Nothing may move before the trajectory's first non-zero entry."""
    quiet_steps = 3
    shock_trajectory = np.zeros((8, varma_mod.k_posdef), dtype=floatX)
    shock_trajectory[quiet_steps, 0] = 1.0

    irf = varma_mod.impulse_response_function(
        idata, shock_trajectory=shock_trajectory, group="prior", random_seed=rng
    )

    # irf[t] is the state after shock t is applied, so the impulse lands on its own index.
    before = irf.irf.isel(time=slice(None, quiet_steps)).values
    assert_allclose(before, 0.0, atol=1e-12)

    R = varma_mod.ssm["selection"].eval()
    impact = irf.irf.isel(chain=0, time=quiet_steps).values
    expected = np.broadcast_to(R @ shock_trajectory[quiet_steps], impact.shape)
    assert_allclose(impact, expected, atol=1e-8)


@pytest.mark.skipif(floatX == "float32", reason="Cholesky factor not accurate enough in float32")
class TestOrthogonalizedImpulseResponse:
    """Recursive (Cholesky) identification of the reduced-form VAR shocks.

    VARMAX has a full state_cov, so the factorization has content and the ordering matters. The
    identifying assumption lives entirely in the impact period, so that is what these tests pin
    down; later periods follow from the transition matrix, which the tests above already cover.
    """

    @staticmethod
    def impact(irf, states=None):
        """Impact-period response, as (draw, state, structural_shock)."""
        response = irf.irf.isel(chain=0, time=0)
        if states is not None:
            response = response.sel(state=states)
        return response.transpose("draw", "state", "structural_shock").values

    def test_impact_is_cholesky_factor(self, varma_mod, idata, rng):
        irf = varma_mod.impulse_response_function(
            idata, n_steps=8, orthogonalize_shocks=True, group="prior", random_seed=rng
        )

        assert irf.irf.dims == ("chain", "draw", "structural_shock", "time", "state")
        assert list(irf.irf.coords["structural_shock"].values) == list(varma_mod.coords["shock"])

        Q = idata.prior["state_cov"].values[0]
        R = varma_mod.ssm["selection"].eval()
        expected = np.stack([R @ np.linalg.cholesky(Q_draw) for Q_draw in Q])

        assert_allclose(self.impact(irf), expected, atol=1e-8)

    def test_is_deterministic_given_parameters(self, varma_mod, idata, rng):
        """No random node in the graph, so the seed must not move the answer at all."""
        kwargs = dict(n_steps=5, orthogonalize_shocks=True, group="prior")
        first = varma_mod.impulse_response_function(idata, random_seed=rng, **kwargs)
        second = varma_mod.impulse_response_function(idata, random_seed=12345, **kwargs)

        assert_array_equal(first.irf.values, second.irf.values)

    def test_shock_order_changes_identification(self, varma_mod, idata, rng):
        reversed_order = list(varma_mod.coords["shock"])[::-1]
        irf = varma_mod.impulse_response_function(
            idata, n_steps=5, orthogonalize_shocks=True, group="prior", random_seed=rng
        )
        reordered = varma_mod.impulse_response_function(
            idata,
            n_steps=5,
            orthogonalize_shocks=True,
            shock_order=reversed_order,
            group="prior",
            random_seed=rng,
        )

        assert list(reordered.irf.coords["structural_shock"].values) == reversed_order
        assert not np.allclose(irf.irf.values, reordered.irf.values)

        # Whatever the ordering, the impact matrix must still reproduce the shock covariance --
        # this is what catches a botched un-permutation, which a shape check would sail past.
        Q = idata.prior["state_cov"].values[0]
        B = self.impact(reordered, states=list(varma_mod.coords["observed_state"]))
        assert_allclose(B @ B.transpose(0, 2, 1), Q, atol=1e-8)

    @pytest.mark.parametrize(
        "shock_order, expected",
        [
            ([...], ["realgdp", "realcons", "realinv"]),
            (["realinv", ...], ["realinv", "realgdp", "realcons"]),
            ([..., "realgdp"], ["realcons", "realinv", "realgdp"]),
            (["realinv", ..., "realgdp"], ["realinv", "realcons", "realgdp"]),
        ],
        ids=["all", "leading-name", "trailing-name", "both-sides"],
    )
    def test_ellipsis_fills_unnamed_shocks(self, varma_mod, idata, rng, shock_order, expected):
        """`...` takes whatever is left, in the order the fit dims give it."""
        irf = varma_mod.impulse_response_function(
            idata,
            n_steps=3,
            orthogonalize_shocks=True,
            shock_order=shock_order,
            group="prior",
            random_seed=rng,
        )

        assert list(irf.irf.coords["structural_shock"].values) == expected

        explicit = varma_mod.impulse_response_function(
            idata,
            n_steps=3,
            orthogonalize_shocks=True,
            shock_order=expected,
            group="prior",
            random_seed=rng,
        )
        assert_array_equal(irf.irf.values, explicit.irf.values)

    def test_diagonal_covariance_is_order_invariant(self, varma_mod, idata, rng):
        """With independent shocks the factorization is a rescaling, so ordering does nothing."""
        shock_cov = np.diag(np.array([1.0, 4.0, 9.0], dtype=floatX))
        kwargs = dict(
            n_steps=3,
            shock_cov=shock_cov,
            orthogonalize_shocks=True,
            group="prior",
            random_seed=rng,
        )
        irf = varma_mod.impulse_response_function(idata, **kwargs)
        reordered = varma_mod.impulse_response_function(
            idata, shock_order=list(varma_mod.coords["shock"])[::-1], **kwargs
        )

        flipped = reordered.irf.isel(structural_shock=slice(None, None, -1))
        assert_allclose(irf.irf.values, flipped.values, atol=1e-8)

    @pytest.mark.parametrize(
        "kwargs, error_msg",
        [
            ({"orthogonalize_shocks": True, "shock_size": 1.0}, "cannot be combined"),
            (
                {"orthogonalize_shocks": True, "shock_trajectory": np.zeros((3, 3))},
                "cannot be combined",
            ),
            ({"shock_order": ["realgdp", "realcons", "realinv"]}, "only meaningful"),
            (
                {"orthogonalize_shocks": True, "shock_order": ["realgdp", "realcons"]},
                "must name every shock",
            ),
            (
                {"orthogonalize_shocks": True, "shock_order": ["realgdp", "realcons", "nope"]},
                "does not have",
            ),
            (
                {"orthogonalize_shocks": True, "shock_order": ["realgdp", "realgdp", ...]},
                "more than once",
            ),
            (
                {"orthogonalize_shocks": True, "shock_order": [..., "realgdp", ...]},
                "at most one",
            ),
            (
                {"orthogonalize_shocks": True, "use_posterior_cov": False},
                "needs a shock covariance matrix",
            ),
        ],
        ids=[
            "with-shock-size",
            "with-trajectory",
            "order-without-flag",
            "short-order",
            "bad-name",
            "duplicate-name",
            "two-ellipsis",
            "no-covariance",
        ],
    )
    def test_invalid_arguments(self, varma_mod, idata, kwargs, error_msg):
        with pytest.raises(ValueError, match=error_msg):
            varma_mod.impulse_response_function(idata, n_steps=3, group="prior", **kwargs)


# A single unit shock to the first variable at t=3, quiet before and after.
SHOCK_TRAJECTORY = np.r_[
    np.zeros((3, 3), dtype=floatX),
    np.array([[1.0, 0.0, 0.0]]).astype(floatX),
    np.zeros((6, 3), dtype=floatX),
]


@pytest.mark.skipif(floatX == "float32", reason="Impulse covariance not PSD if float32")
class TestImpulseResponseHorizon:
    """The trajectory sets the horizon; n_steps is only a default when there is no trajectory."""

    trajectory_length = SHOCK_TRAJECTORY.shape[0]

    def test_trajectory_length_overrides_n_steps(self, varma_mod, idata, rng, caplog):
        irf = varma_mod.impulse_response_function(
            idata, n_steps=40, shock_trajectory=SHOCK_TRAJECTORY, group="prior", random_seed=rng
        )

        assert len(irf.irf.coords["time"]) == self.trajectory_length
        assert any("do not agree" in message for message in caplog.messages)

    def test_matching_n_steps_is_quiet(self, varma_mod, idata, rng, caplog):
        irf = varma_mod.impulse_response_function(
            idata,
            n_steps=self.trajectory_length,
            shock_trajectory=SHOCK_TRAJECTORY,
            group="prior",
            random_seed=rng,
        )

        assert len(irf.irf.coords["time"]) == self.trajectory_length
        assert not any("do not agree" in message for message in caplog.messages)

    def test_omitted_n_steps_is_quiet(self, varma_mod, idata, rng, caplog):
        irf = varma_mod.impulse_response_function(
            idata, shock_trajectory=SHOCK_TRAJECTORY, group="prior", random_seed=rng
        )

        # The trajectory wins over the default, which is longer than it is.
        assert len(irf.irf.coords["time"]) == self.trajectory_length
        assert not any("do not agree" in message for message in caplog.messages)

    def test_default_applies_without_a_trajectory(self, varma_mod, idata, rng):
        """Omitting n_steps entirely is the only way the module default is reached."""
        irf = varma_mod.impulse_response_function(idata, group="prior", random_seed=rng)

        assert len(irf.irf.coords["time"]) == DEFAULT_IRF_STEPS


def test_forecast(varma_mod, idata, rng):
    forecast = varma_mod.forecast(idata, periods=10, random_seed=rng, group="prior")

    assert np.isfinite(forecast.forecast_latent.values).all()
    assert np.isfinite(forecast.forecast_observed.values).all()


def test_varmax_workflow(rng, mock_sample):
    df = pd.read_csv(
        "tests/statespace/_data/statsmodels_macrodata_processed.csv",
        index_col=0,
        parse_dates=True,
    ).astype(floatX)
    df.index.freq = df.index.inferred_freq

    ss_mod = BayesianVARMAX(
        endog_names=df.columns,
        order=(1, 0),
        stationary_initialization=True,
        measurement_error=True,
        verbose=False,
    )

    with pm.Model(coords=ss_mod.coords) as m:
        state_cov_diag = pm.Exponential("state_cov_diag", 1, dims=["shock"])
        pm.Deterministic("state_cov", pt.diag(state_cov_diag), dims=["shock", "shock_aux"])
        pm.Normal("ar_params", sigma=0.1, dims=["observed_state", "lag_ar", "observed_state_aux"])
        pm.Exponential("sigma_obs", 1, dims=["observed_state"])

        ss_mod.build_statespace_graph(df)

        idata = pm.sample()

    post = ss_mod.sample_conditional_posterior(idata, mvn_method="svd")
    assert "filtered_posterior" in post
    assert "smoothed_posterior" in post
    assert "predicted_posterior" in post

    forecast = ss_mod.forecast(idata, periods=10, random_seed=rng)
    assert "forecast_latent" in forecast
    assert "forecast_observed" in forecast
    assert np.isfinite(forecast.forecast_latent.values).all()
    assert np.isfinite(forecast.forecast_observed.values).all()

    irf = ss_mod.impulse_response_function(idata, n_steps=10, random_seed=rng)
    assert "irf" in irf
    assert np.isfinite(irf.irf.values).all()


class TestVARMAXWithExogenous:
    def test_create_varmax_with_exogenous_list_of_names(self, data):
        mod = BayesianVARMAX(
            endog_names=["realgdp", "realcons", "realinv"],
            order=(1, 0),
            exog_state_names=["foo", "bar"],
            verbose=False,
            measurement_error=False,
            stationary_initialization=False,
        )
        assert mod.k_exog == 2
        assert mod.exog_state_names == ["foo", "bar"]
        assert mod.data_names == ("exogenous_data",)
        assert mod.param_dims["beta_exog"] == ("observed_state", "exogenous")
        assert mod.coords["exogenous"] == ("foo", "bar")
        assert mod.param_info["beta_exog"]["shape"] == (mod.k_endog, 2)
        assert mod.param_info["beta_exog"]["dims"] == ("observed_state", "exogenous")

    def test_create_varmax_with_exogenous_both_defined_correctly(self, data):
        mod = BayesianVARMAX(
            endog_names=["realgdp", "realcons", "realinv"],
            order=(1, 0),
            exog_state_names=["a", "b"],
            verbose=False,
            measurement_error=False,
            stationary_initialization=False,
        )
        assert mod.k_exog == 2
        assert mod.exog_state_names == ["a", "b"]
        assert mod.data_names == ("exogenous_data",)
        assert mod.param_dims["beta_exog"] == ("observed_state", "exogenous")
        assert mod.coords["exogenous"] == ("a", "b")
        assert mod.param_info["beta_exog"]["shape"] == (mod.k_endog, 2)
        assert mod.param_info["beta_exog"]["dims"] == ("observed_state", "exogenous")

    def test_create_varmax_with_exogenous_exog_names_dict(self, data):
        exog_state_names = {"observed_0": ["a", "b"], "observed_1": ["c"], "observed_2": []}
        mod = BayesianVARMAX(
            endog_names=["observed_0", "observed_1", "observed_2"],
            order=(1, 0),
            exog_state_names=exog_state_names,
            verbose=False,
            measurement_error=False,
            stationary_initialization=False,
        )
        assert mod.k_exog == {"observed_0": 2, "observed_1": 1, "observed_2": 0}
        assert mod.exog_state_names == exog_state_names
        assert mod.data_names == (
            "observed_0_exogenous_data",
            "observed_1_exogenous_data",
            "observed_2_exogenous_data",
        )
        assert mod.param_dims["beta_observed_0"] == ("exogenous_observed_0",)
        assert mod.param_dims["beta_observed_1"] == ("exogenous_observed_1",)
        assert (
            "beta_observed_2" not in mod.param_dims
            or mod.param_info.get("beta_observed_2") is None
            or mod.param_info.get("beta_observed_2", {}).get("shape", (0,))[0] == 0
        )

        assert mod.coords["exogenous_observed_0"] == ("a", "b")
        assert mod.coords["exogenous_observed_1"] == ("c",)
        assert "exogenous_observed_2" in mod.coords and mod.coords["exogenous_observed_2"] == ()

        assert mod.param_info["beta_observed_0"]["shape"] == (2,)
        assert mod.param_info["beta_observed_0"]["dims"] == ("exogenous_observed_0",)
        assert mod.param_info["beta_observed_1"]["shape"] == (1,)
        assert mod.param_info["beta_observed_1"]["dims"] == ("exogenous_observed_1",)

    def test_create_varmax_with_exogenous_dict_converts_to_list(self, data):
        exog_state_names = {
            "observed_0": ["a", "b"],
            "observed_1": ["a", "b"],
            "observed_2": ["a", "b"],
        }
        mod = BayesianVARMAX(
            endog_names=["observed_0", "observed_1", "observed_2"],
            order=(1, 0),
            exog_state_names=exog_state_names,
            verbose=False,
            measurement_error=False,
            stationary_initialization=False,
        )

        assert mod.k_exog == 2
        assert mod.exog_state_names == ["a", "b"]
        assert mod.data_names == ("exogenous_data",)
        assert mod.param_dims["beta_exog"] == ("observed_state", "exogenous")
        assert mod.coords["exogenous"] == ("a", "b")
        assert mod.param_info["beta_exog"]["shape"] == (mod.k_endog, 2)
        assert mod.param_info["beta_exog"]["dims"] == ("observed_state", "exogenous")

    def _build_varmax(self, df, exog_state_names, exog_data):
        endog_names = df.columns.values.tolist()

        mod = BayesianVARMAX(
            endog_names=endog_names,
            order=(1, 0),
            exog_state_names=exog_state_names,
            verbose=False,
            measurement_error=False,
            stationary_initialization=False,
            mode="JAX",
        )

        with pm.Model(coords=mod.coords) as m:
            for var_name, data in exog_data.items():
                pm.Data(var_name, data, dims=mod.data_info[var_name]["dims"])

            x0 = pm.Deterministic("x0", pt.zeros(mod.k_states), dims=mod.param_dims["x0"])
            P0_diag = pm.Exponential("P0_diag", 1.0, dims=mod.param_dims["P0"][0])
            P0 = pm.Deterministic("P0", pt.diag(P0_diag), dims=mod.param_dims["P0"])

            ar_params = pm.Normal("ar_params", mu=0, sigma=1, dims=mod.param_dims["ar_params"])
            state_cov_diag = pm.Exponential(
                "state_cov_diag", 1.0, dims=mod.param_dims["state_cov"][0]
            )
            state_cov = pm.Deterministic(
                "state_cov", pt.diag(state_cov_diag), dims=mod.param_dims["state_cov"]
            )

            # Exogenous priors
            if isinstance(mod.exog_state_names, list):
                beta_exog = pm.Normal("beta_exog", mu=0, sigma=1, dims=mod.param_dims["beta_exog"])
            elif isinstance(mod.exog_state_names, dict):
                for name in mod.exog_state_names:
                    if mod.exog_state_names.get(name):
                        pm.Normal(
                            f"beta_{name}", mu=0, sigma=1, dims=mod.param_dims[f"beta_{name}"]
                        )

            mod.build_statespace_graph(data=df)

        return mod, m

    @pytest.mark.parametrize(
        "exog_state_names",
        [
            (["foo", "bar"]),
            ({"y1": ["a", "b"], "y2": ["c"]}),
        ],
        ids=["exog_state_names_list", "exog_state_names_dict"],
    )
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_varmax_with_exog(self, rng, exog_state_names):
        endog_names = ["y1", "y2", "y3"]
        n_obs = 50
        time_idx = pd.date_range(start="2020-01-01", periods=n_obs, freq="D")

        y = rng.normal(size=(n_obs, len(endog_names)))
        df = pd.DataFrame(y, columns=endog_names, index=time_idx).astype(floatX)

        if isinstance(exog_state_names, dict):
            exog_data = {
                f"{name}_exogenous_data": pd.DataFrame(
                    rng.normal(size=(n_obs, len(exog_names))).astype(floatX),
                    columns=exog_names,
                    index=time_idx,
                )
                for name, exog_names in exog_state_names.items()
            }
        else:
            exog_data = {
                "exogenous_data": pd.DataFrame(
                    rng.normal(size=(n_obs, len(exog_state_names))).astype(floatX),
                    columns=exog_state_names,
                    index=time_idx,
                )
            }

        mod, m = self._build_varmax(df, exog_state_names, exog_data)

        with freeze_dims_and_data(m):
            prior = pm.sample_prior_predictive(
                draws=10, random_seed=rng, compile_kwargs={"mode": "JAX"}
            )

        prior_cond = mod.sample_conditional_prior(prior, mvn_method="eigh")
        beta_dot_data = prior_cond.filtered_prior_observed.values - prior_cond.filtered_prior.values

        if isinstance(exog_state_names, list):
            beta = prior.prior.beta_exog
            assert beta.shape == (1, 10, 3, 2)

            np.testing.assert_allclose(
                beta_dot_data,
                np.einsum("tx,...sx->...ts", exog_data["exogenous_data"].values, beta),
                atol=1e-2,
            )

        elif isinstance(exog_state_names, dict):
            assert prior.prior.beta_y1.shape == (1, 10, 2)
            assert prior.prior.beta_y2.shape == (1, 10, 1)

            obs_intercept = [
                np.einsum("tx,...x->...t", exog_data[f"{name}_exogenous_data"].values, beta)
                for name, beta in zip(["y1", "y2"], [prior.prior.beta_y1, prior.prior.beta_y2])
            ]

            # y3 has no exogenous variables
            obs_intercept.append(np.zeros_like(obs_intercept[0]))

            np.testing.assert_allclose(beta_dot_data, np.stack(obs_intercept, axis=-1), atol=1e-2)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_forecast_with_exog(self, rng):
        endog_names = ["y1", "y2", "y3"]
        n_obs = 50
        time_idx = pd.date_range(start="2020-01-01", periods=n_obs, freq="D")

        y = rng.normal(size=(n_obs, len(endog_names)))
        df = pd.DataFrame(y, columns=endog_names, index=time_idx).astype(floatX)

        mod, m = self._build_varmax(
            df,
            exog_state_names=["exogenous_0", "exogenous_1"],
            exog_data={
                "exogenous_data": pd.DataFrame(
                    rng.normal(size=(n_obs, 2)).astype(floatX),
                    columns=["exogenous_0", "exogenous_1"],
                    index=time_idx,
                )
            },
        )

        assert mod._needs_exog_data

        with freeze_dims_and_data(m):
            prior = pm.sample_prior_predictive(
                draws=10, random_seed=rng, compile_kwargs={"mode": "JAX"}
            )

        with pytest.raises(
            ValueError,
            match=r"This model was fit using exogenous data. Forecasting cannot be performed "
            r"without providing scenario data",
        ):
            mod.forecast(prior, periods=10, random_seed=rng, group="prior")

        forecast = mod.forecast(
            prior,
            periods=10,
            group="prior",
            random_seed=rng,
            scenario={
                "exogenous_data": pd.DataFrame(
                    rng.normal(size=(10, 2)).astype(floatX),
                    columns=["exogenous_0", "exogenous_1"],
                    index=pd.date_range(start=df.index[-1], periods=10, freq="D"),
                )
            },
        )

        assert np.isfinite(forecast.forecast_latent.values).all()
        assert np.isfinite(forecast.forecast_observed.values).all()
