import numpy as np
import pandas as pd
import pymc as pm
import pytest

from numpy.testing import assert_allclose
from pytensor import config
from pytensor import tensor as pt

from pymc_extras.statespace.models import structural as st
from tests.statespace.models.structural.conftest import _assert_basic_coords_correct
from tests.statespace.test_utilities import simulate_from_numpy_model

ATOL = 1e-8 if config.floatX.endswith("64") else 1e-4
RTOL = 0 if config.floatX.endswith("64") else 1e-6


@pytest.fixture
def regression_data(rng):
    """Generate test data for regression components (2 exogenous variables)."""
    return rng.normal(size=(100, 2)).astype(config.floatX)


@pytest.fixture
def multiple_regression_data(rng):
    """Generate test data for multiple regression components."""
    return {
        "data_1": rng.normal(size=(100, 2)).astype(config.floatX),
        "data_2": rng.normal(size=(100, 1)).astype(config.floatX),
    }


@pytest.fixture
def time_series_data(rng):
    """Generate time series data for PyMC integration tests."""
    time_idx = pd.date_range(start="2000-01-01", freq="D", periods=100)
    data = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"], index=time_idx)
    y = pd.DataFrame(rng.normal(size=(100, 1)), columns=["data"], index=time_idx)
    return data, y


class TestRegressionComponent:
    """Test basic regression component functionality."""

    @pytest.mark.parametrize("innovations", [False, True])
    def test_exogenous_component(self, rng, regression_data, innovations):
        """Test basic regression component with and without innovations."""
        mod = st.RegressionComponent(
            state_names=["feature_1", "feature_2"], name="exog", innovations=innovations
        )

        params = {"beta_exog": np.array([1.0, 2.0], dtype=config.floatX)}
        if innovations:
            params["sigma_beta_exog"] = np.array([0.1, 0.2], dtype=config.floatX)

        exog_data = {"data_exog": regression_data}
        x, y = simulate_from_numpy_model(mod, rng, params, exog_data)

        if not innovations:
            # Check that the generated data is just a linear regression
            assert_allclose(y, regression_data @ params["beta_exog"], atol=ATOL, rtol=RTOL)
        else:
            # With innovations, the coefficients should vary over time
            # The initial state should match the beta parameters
            assert_allclose(x[0], params["beta_exog"], atol=ATOL, rtol=RTOL)

        mod = mod.build(verbose=False)
        _assert_basic_coords_correct(mod)
        assert mod.coords["state_exog"] == ["feature_1", "feature_2"]

        if innovations:
            # Check that sigma_beta parameter is included
            assert "sigma_beta_exog" in mod.param_names

    @pytest.mark.parametrize("innovations", [False, True])
    def test_adding_exogenous_component(self, rng, regression_data, innovations):
        """Test adding regression component to other components."""
        reg = st.RegressionComponent(state_names=["a", "b"], name="exog", innovations=innovations)
        ll = st.LevelTrendComponent(name="level")
        seasonal = st.FrequencySeasonality(name="annual", season_length=12, n=4)
        mod = reg + ll + seasonal

        assert mod.ssm["design"].eval({"data_exog": regression_data}).shape == (100, 1, 2 + 2 + 8)
        assert_allclose(
            mod.ssm["design", 5, 0, :2].eval({"data_exog": regression_data}), regression_data[5]
        )

        if innovations:
            # Check that sigma_beta parameter is included in the combined model
            assert "sigma_beta_exog" in mod.param_names


class TestMultivariateRegression:
    """Test multivariate regression functionality."""

    @pytest.mark.parametrize("innovations", [False, True])
    def test_regression_with_multiple_observed_states(self, rng, regression_data, innovations):
        """Test multivariate regression with and without innovations."""
        from scipy.linalg import block_diag

        mod = st.RegressionComponent(
            state_names=["feature_1", "feature_2"],
            name="exog",
            observed_state_names=["data_1", "data_2"],
            innovations=innovations,
        )

        params = {"beta_exog": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX)}
        if innovations:
            params["sigma_beta_exog"] = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=config.floatX)

        exog_data = {"data_exog": regression_data}
        x, y = simulate_from_numpy_model(mod, rng, params, exog_data)

        assert x.shape == (100, 4)  # 2 features, 2 states
        assert y.shape == (100, 2)

        if not innovations:
            # Check that the generated data are two independent linear regressions
            assert_allclose(y[:, 0], regression_data @ params["beta_exog"][0], atol=ATOL, rtol=RTOL)
            assert_allclose(y[:, 1], regression_data @ params["beta_exog"][1], atol=ATOL, rtol=RTOL)
        else:
            # Check that initial states match the beta parameters
            assert_allclose(x[0, :2], params["beta_exog"][0], atol=ATOL, rtol=RTOL)
            assert_allclose(x[0, 2:], params["beta_exog"][1], atol=ATOL, rtol=RTOL)

        mod = mod.build(verbose=False)
        assert mod.coords["state_exog"] == ["feature_1", "feature_2"]

        Z = mod.ssm["design"].eval({"data_exog": regression_data})
        vec_block_diag = np.vectorize(block_diag, signature="(n,m),(o,p)->(q,r)")
        assert Z.shape == (100, 2, 4)
        assert np.allclose(
            Z,
            vec_block_diag(regression_data[:, None, :], regression_data[:, None, :]),
        )

        if innovations:
            # Check that sigma_beta parameter is included
            assert "sigma_beta_exog" in mod.param_names


class TestMultipleRegressionComponents:
    """Test multiple regression components functionality."""

    @pytest.mark.parametrize("innovations", [False, True])
    def test_add_regression_components_with_multiple_observed_states(
        self, rng, multiple_regression_data, innovations
    ):
        """Test adding multiple regression components with and without innovations."""
        from scipy.linalg import block_diag

        reg1 = st.RegressionComponent(
            state_names=["a", "b"],
            name="exog1",
            observed_state_names=["data_1", "data_2"],
            innovations=innovations,
        )
        reg2 = st.RegressionComponent(
            state_names=["c"],
            name="exog2",
            observed_state_names=["data_3"],
            innovations=innovations,
        )

        mod = (reg1 + reg2).build(verbose=False)
        assert mod.coords["state_exog1"] == ["a", "b"]
        assert mod.coords["state_exog2"] == ["c"]

        Z = mod.ssm["design"].eval(
            {
                "data_exog1": multiple_regression_data["data_1"],
                "data_exog2": multiple_regression_data["data_2"],
            }
        )
        vec_block_diag = np.vectorize(block_diag, signature="(n,m),(o,p)->(q,r)")
        assert Z.shape == (100, 3, 5)
        assert np.allclose(
            Z,
            vec_block_diag(
                vec_block_diag(
                    multiple_regression_data["data_1"][:, None, :],
                    multiple_regression_data["data_1"][:, None, :],
                ),
                multiple_regression_data["data_2"][:, None, :],
            ),
        )

        x0 = mod.ssm["initial_state"].eval(
            {
                "beta_exog1": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=config.floatX),
                "beta_exog2": np.array([5.0], dtype=config.floatX),
            }
        )
        np.testing.assert_allclose(x0, np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=config.floatX))

        if innovations:
            # Check that sigma_beta parameters are included
            assert "sigma_beta_exog1" in mod.param_names
            assert "sigma_beta_exog2" in mod.param_names


class TestPyMCIntegration:
    """Test PyMC integration functionality."""

    @pytest.mark.parametrize("innovations", [False, True])
    def test_filter_scans_time_varying_design_matrix(self, rng, time_series_data, innovations):
        """Test PyMC integration with and without innovations."""
        data, y = time_series_data

        reg = st.RegressionComponent(state_names=["a", "b"], name="exog", innovations=innovations)
        mod = reg.build(verbose=False)

        with pm.Model(coords=mod.coords) as m:
            data_exog = pm.Data("data_exog", data.values)

            x0 = pm.Normal("x0", dims=["state"])
            P0 = pm.Deterministic("P0", pt.eye(mod.k_states), dims=["state", "state_aux"])
            beta_exog = pm.Normal("beta_exog", dims=["state_exog"])

            if innovations:
                sigma_beta_exog = pm.Exponential("sigma_beta_exog", 1, dims=["state_exog"])

            mod.build_statespace_graph(y)
            x0, P0, c, d, T, Z, R, H, Q = mod.unpack_statespace()
            pm.Deterministic("Z", Z)

            prior = pm.sample_prior_predictive(draws=10)

        prior_Z = prior.prior.Z.values
        assert prior_Z.shape == (1, 10, 100, 1, 2)
        assert_allclose(prior_Z[0, :, :, 0, :], data.values[None].repeat(10, axis=0))

        if innovations:
            # Check that sigma_beta parameter is included in the prior
            assert "sigma_beta_exog" in prior.prior.data_vars


def test_regression_multiple_shared_construction():
    rc = st.RegressionComponent(
        state_names=["A"],
        observed_state_names=["data_1", "data_2"],
        innovations=True,
        share_states=True,
    )
    mod = rc.build(verbose=False)

    assert mod.k_endog == 2
    assert mod.k_states == 1
    assert mod.k_posdef == 1

    assert mod.coords["state_regression"] == ["A"]
    assert mod.coords["endog_regression"] == ["data_1", "data_2"]

    assert mod.state_names == [
        "A[regression_shared]",
    ]

    assert mod.shock_names == ["A_shared"]

    data = np.random.standard_normal(size=(10, 1))
    Z = mod.ssm["design"].eval({"data_regression": data})
    T = mod.ssm["transition"].eval()
    R = mod.ssm["selection"].eval()

    np.testing.assert_allclose(
        Z,
        np.hstack(
            [
                data,
                data,
            ]
        )[:, :, np.newaxis],
    )

    np.testing.assert_allclose(T, np.array([[1.0]]))
    np.testing.assert_allclose(R, np.array([[1.0]]))


def test_regression_multiple_shared_observed(rng):
    mod = st.RegressionComponent(
        state_names=["A"],
        observed_state_names=["data_1", "data_2", "data_3"],
        innovations=False,
        share_states=True,
    )
    data = np.random.standard_normal(size=(10, 1))

    params = {"beta_regression": np.array([1.0])}
    data_dict = {"data_regression": data}
    x, y = simulate_from_numpy_model(mod, rng, params, data_dict, steps=data.shape[0])
    np.testing.assert_allclose(y[:, 0], y[:, 1])
    np.testing.assert_allclose(y[:, 0], y[:, 2])


def test_regression_mixed_shared_and_not_shared():
    mod_1 = st.RegressionComponent(
        name="individual",
        state_names=["A"],
        observed_state_names=["data_1", "data_2"],
    )
    mod_2 = st.RegressionComponent(
        name="joint",
        state_names=["B", "C"],
        observed_state_names=["data_1", "data_2"],
        share_states=True,
    )

    mod = (mod_1 + mod_2).build(verbose=False)

    assert mod.k_endog == 2
    assert mod.k_states == 4
    assert mod.k_posdef == 4

    assert mod.state_names == ["A[data_1]", "A[data_2]", "B[joint_shared]", "C[joint_shared]"]
    assert mod.shock_names == ["A", "B_shared", "C_shared"]

    data_joint = np.random.standard_normal(size=(10, 2))
    data_individual = np.random.standard_normal(size=(10, 1))
    Z = mod.ssm["design"].eval({"data_joint": data_joint, "data_individual": data_individual})
    T = mod.ssm["transition"].eval()
    R = mod.ssm["selection"].eval()

    np.testing.assert_allclose(
        Z,
        np.concat(
            (
                pt.linalg.block_diag(
                    *[data_individual[:, None] for _ in range(mod.k_endog)]
                ).eval(),
                np.concat((data_joint[:, None], data_joint[:, None]), axis=1),
            ),
            axis=2,
        ),
    )

    np.testing.assert_allclose(T, np.eye(mod.k_states))
    np.testing.assert_allclose(R, np.eye(mod.k_states))
