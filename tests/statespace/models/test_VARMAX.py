from itertools import pairwise, product

import numpy as np
import pandas as pd
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import statsmodels.api as sm

from numpy.testing import assert_allclose, assert_array_less
from pymc.model.transform.optimization import freeze_dims_and_data

from pymc_extras.statespace import BayesianVARMAX
from pymc_extras.statespace.utils.constants import SHORT_NAME_TO_LONG
from tests.statespace.shared_fixtures import (  # pylint: disable=unused-import
    rng,
)

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

        varma_mod.build_statespace_graph(data=data, save_kalman_filter_outputs_in_idata=True)

    return pymc_mod


@pytest.fixture(scope="session")
def idata(pymc_mod, rng):
    with pymc_mod:
        idata = pm.sample_prior_predictive(draws=10, random_seed=rng)

    return idata


def test_mode_argument():
    # Mode argument should be passed to the parent class
    mod = BayesianVARMAX(k_endog=2, order=(3, 0), mode="FAST_RUN", verbose=False)
    assert mod.mode == "FAST_RUN"


@pytest.mark.parametrize("order", orders, ids=ids)
@pytest.mark.parametrize("var", ["AR", "MA", "state_cov"])
@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.EstimationWarning")
def test_VARMAX_param_counts_match_statsmodels(data, order, var):
    p, q = order

    mod = BayesianVARMAX(k_endog=data.shape[1], order=(p, q), verbose=False)
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
        k_endog=data.shape[1],
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
        mod._insert_random_variables()

        matrices = pm.draw(mod.subbed_ssm)
        matrix_dict = dict(zip(SHORT_NAME_TO_LONG.values(), matrices))

    for matrix in ["transition", "selection", "state_cov", "obs_cov", "design"]:
        assert_allclose(matrix_dict[matrix], sm_var.ssm[matrix])


@pytest.mark.parametrize("filter_output", ["filtered", "predicted", "smoothed"])
def test_all_prior_covariances_are_PSD(filter_output, pymc_mod, rng):
    rv = pymc_mod[f"{filter_output}_covariances"]
    cov_mats = pm.draw(rv, 100, random_seed=rng)
    w, v = np.linalg.eig(cov_mats)
    assert_array_less(0, w, err_msg=f"Smallest eigenvalue: {min(w.ravel())}")


parameters = [
    {"n_steps": 10, "shock_size": None},
    {"n_steps": 10, "shock_size": 1.0},
    {"n_steps": 10, "shock_size": np.array([1.0, 0.0, 0.0])},
    {
        "n_steps": 10,
        "shock_cov": np.array([[1.38, 0.58, -1.84], [0.58, 0.99, -0.82], [-1.84, -0.82, 2.51]]),
    },
    {
        "shock_trajectory": np.r_[
            np.zeros((3, 3), dtype=floatX),
            np.array([[1.0, 0.0, 0.0]]).astype(floatX),
            np.zeros((6, 3), dtype=floatX),
        ]
    },
]

ids = ["from-posterior-cov", "scalar_shock_size", "array_shock_size", "user-cov", "trajectory"]


@pytest.mark.parametrize("parameters", parameters, ids=ids)
@pytest.mark.skipif(floatX == "float32", reason="Impulse covariance not PSD if float32")
def test_impulse_response(parameters, varma_mod, idata, rng):
    irf = varma_mod.impulse_response_function(idata.prior, random_seed=rng, **parameters)

    assert not np.any(np.isnan(irf.irf.values))


class TestVARMAXWithExogenous:
    def test_create_varmax_with_exogenous_k_exog_int(self, data):
        mod = BayesianVARMAX(
            k_endog=data.shape[1],
            order=(1, 0),
            k_exog=2,
            verbose=False,
            measurement_error=False,
            stationary_initialization=False,
        )
        assert mod.k_exog == 2
        assert mod.exog_state_names == ["exogenous_0", "exogenous_1"]
        assert mod.data_names == ["exogenous_data"]
        assert mod.param_dims["beta_exog"] == ("observed_state", "exogenous")
        assert mod.coords["exogenous"] == ["exogenous_0", "exogenous_1"]
        assert mod.param_info["beta_exog"]["shape"] == (mod.k_endog, 2)
        assert mod.param_info["beta_exog"]["dims"] == ("observed_state", "exogenous")

    def test_create_varmax_with_exogenous_list_of_names(self, data):
        mod = BayesianVARMAX(
            k_endog=data.shape[1],
            order=(1, 0),
            exog_state_names=["foo", "bar"],
            verbose=False,
            measurement_error=False,
            stationary_initialization=False,
        )
        assert mod.k_exog == 2
        assert mod.exog_state_names == ["foo", "bar"]
        assert mod.data_names == ["exogenous_data"]
        assert mod.param_dims["beta_exog"] == ("observed_state", "exogenous")
        assert mod.coords["exogenous"] == ["foo", "bar"]
        assert mod.param_info["beta_exog"]["shape"] == (mod.k_endog, 2)
        assert mod.param_info["beta_exog"]["dims"] == ("observed_state", "exogenous")

    def test_create_varmax_with_exogenous_both_defined_correctly(self, data):
        mod = BayesianVARMAX(
            k_endog=data.shape[1],
            order=(1, 0),
            k_exog=2,
            exog_state_names=["a", "b"],
            verbose=False,
            measurement_error=False,
            stationary_initialization=False,
        )
        assert mod.k_exog == 2
        assert mod.exog_state_names == ["a", "b"]
        assert mod.data_names == ["exogenous_data"]
        assert mod.param_dims["beta_exog"] == ("observed_state", "exogenous")
        assert mod.coords["exogenous"] == ["a", "b"]
        assert mod.param_info["beta_exog"]["shape"] == (mod.k_endog, 2)
        assert mod.param_info["beta_exog"]["dims"] == ("observed_state", "exogenous")

    def test_create_varmax_with_exogenous_k_exog_dict(self, data):
        k_exog = {"observed_0": 2, "observed_1": 1, "observed_2": 0}
        mod = BayesianVARMAX(
            endog_names=["observed_0", "observed_1", "observed_2"],
            order=(1, 0),
            k_exog=k_exog,
            verbose=False,
            measurement_error=False,
            stationary_initialization=False,
        )
        assert mod.k_exog == k_exog
        assert mod.exog_state_names == {
            "observed_0": ["observed_0_exogenous_0", "observed_0_exogenous_1"],
            "observed_1": ["observed_1_exogenous_0"],
            "observed_2": [],
        }
        assert mod.data_names == [
            "observed_0_exogenous_data",
            "observed_1_exogenous_data",
            "observed_2_exogenous_data",
        ]
        assert mod.param_dims["beta_observed_0"] == ("exogenous_observed_0",)
        assert mod.param_dims["beta_observed_1"] == ("exogenous_observed_1",)
        assert (
            "beta_observed_2" not in mod.param_dims
            or mod.param_info.get("beta_observed_2") is None
            or mod.param_info.get("beta_observed_2", {}).get("shape", (0,))[0] == 0
        )

        assert mod.coords["exogenous_observed_0"] == [
            "observed_0_exogenous_0",
            "observed_0_exogenous_1",
        ]
        assert mod.coords["exogenous_observed_1"] == ["observed_1_exogenous_0"]
        assert "exogenous_observed_2" in mod.coords and mod.coords["exogenous_observed_2"] == []

        assert mod.param_info["beta_observed_0"]["shape"] == (2,)
        assert mod.param_info["beta_observed_0"]["dims"] == ("exogenous_observed_0",)
        assert mod.param_info["beta_observed_1"]["shape"] == (1,)
        assert mod.param_info["beta_observed_1"]["dims"] == ("exogenous_observed_1",)

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
        assert mod.data_names == [
            "observed_0_exogenous_data",
            "observed_1_exogenous_data",
            "observed_2_exogenous_data",
        ]
        assert mod.param_dims["beta_observed_0"] == ("exogenous_observed_0",)
        assert mod.param_dims["beta_observed_1"] == ("exogenous_observed_1",)
        assert (
            "beta_observed_2" not in mod.param_dims
            or mod.param_info.get("beta_observed_2") is None
            or mod.param_info.get("beta_observed_2", {}).get("shape", (0,))[0] == 0
        )

        assert mod.coords["exogenous_observed_0"] == ["a", "b"]
        assert mod.coords["exogenous_observed_1"] == ["c"]
        assert "exogenous_observed_2" in mod.coords and mod.coords["exogenous_observed_2"] == []

        assert mod.param_info["beta_observed_0"]["shape"] == (2,)
        assert mod.param_info["beta_observed_0"]["dims"] == ("exogenous_observed_0",)
        assert mod.param_info["beta_observed_1"]["shape"] == (1,)
        assert mod.param_info["beta_observed_1"]["dims"] == ("exogenous_observed_1",)

    def test_create_varmax_with_exogenous_both_dict_correct(self, data):
        k_exog = {"observed_0": 2, "observed_1": 1}
        exog_state_names = {"observed_0": ["a", "b"], "observed_1": ["c"]}
        mod = BayesianVARMAX(
            endog_names=["observed_0", "observed_1"],
            order=(1, 0),
            k_exog=k_exog,
            exog_state_names=exog_state_names,
            verbose=False,
            measurement_error=False,
            stationary_initialization=False,
        )
        assert mod.k_exog == k_exog
        assert mod.exog_state_names == exog_state_names
        assert mod.data_names == ["observed_0_exogenous_data", "observed_1_exogenous_data"]
        assert mod.param_dims["beta_observed_0"] == ("exogenous_observed_0",)
        assert mod.param_dims["beta_observed_1"] == ("exogenous_observed_1",)
        assert mod.coords["exogenous_observed_0"] == ["a", "b"]
        assert mod.coords["exogenous_observed_1"] == ["c"]
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
        assert mod.data_names == ["exogenous_data"]
        assert mod.param_dims["beta_exog"] == ("observed_state", "exogenous")
        assert mod.coords["exogenous"] == ["a", "b"]
        assert mod.param_info["beta_exog"]["shape"] == (mod.k_endog, 2)
        assert mod.param_info["beta_exog"]["dims"] == ("observed_state", "exogenous")

    def test_create_varmax_with_exogenous_raises_if_args_disagree(self, data):
        # List case
        with pytest.raises(
            ValueError, match="Length of exog_state_names does not match provided k_exog"
        ):
            BayesianVARMAX(
                k_endog=2,
                order=(1, 0),
                k_exog=3,
                exog_state_names=["a", "b"],
                verbose=False,
                measurement_error=False,
                stationary_initialization=False,
            )

        # Dict case
        with pytest.raises(
            ValueError,
            match="If k_exog is an int, exog_state_names must be a list of the same length",
        ):
            BayesianVARMAX(
                k_endog=2,
                order=(1, 0),
                k_exog=2,
                exog_state_names={"observed_0": ["a"], "observed_1": ["b"]},
                verbose=False,
                measurement_error=False,
                stationary_initialization=False,
            )

        # dict + list
        with pytest.raises(
            ValueError, match="If k_exog is a dict, exog_state_names must be a dict as well"
        ):
            BayesianVARMAX(
                endog_names=["observed_0", "observed_1"],
                order=(1, 0),
                k_exog={"observed_0": 1, "observed_1": 1},
                exog_state_names=["a", "b"],
                verbose=False,
                measurement_error=False,
                stationary_initialization=False,
            )

        # Dict/dict, key mismatch
        with pytest.raises(
            ValueError, match="Keys of k_exog and exog_state_names dicts must match"
        ):
            BayesianVARMAX(
                endog_names=["observed_0", "observed_1"],
                order=(1, 0),
                k_exog={"observed_0": 1, "observed_1": 1},
                exog_state_names={"observed_0": ["a"], "observed_2": ["b"]},
                verbose=False,
                measurement_error=False,
                stationary_initialization=False,
            )

        # Dict/dict, length mismatch
        with pytest.raises(ValueError, match="lengths of exog_state_names lists must match"):
            BayesianVARMAX(
                endog_names=["observed_0", "observed_1"],
                order=(1, 0),
                k_exog={"observed_0": 2, "observed_1": 1},
                exog_state_names={"observed_0": ["a"], "observed_1": ["b"]},
                verbose=False,
                measurement_error=False,
                stationary_initialization=False,
            )

    @pytest.mark.parametrize(
        "k_exog, exog_state_names",
        [
            (2, None),
            (None, ["foo", "bar"]),
            (None, {"y1": ["a", "b"], "y2": ["c"]}),
        ],
        ids=["k_exog_int", "exog_state_names_list", "exog_state_names_dict"],
    )
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_varmax_with_exog(self, rng, k_exog, exog_state_names):
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
            exog_names = exog_state_names or [f"exogenous_{i}" for i in range(k_exog)]
            exog_data = {
                "exogenous_data": pd.DataFrame(
                    rng.normal(size=(n_obs, k_exog or len(exog_state_names))).astype(floatX),
                    columns=exog_names,
                    index=time_idx,
                )
            }

        mod = BayesianVARMAX(
            endog_names=endog_names,
            order=(1, 0),
            k_exog=k_exog,
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

        with freeze_dims_and_data(m):
            prior = pm.sample_prior_predictive(
                draws=10, random_seed=rng, compile_kwargs={"mode": "JAX"}
            )

        prior_cond = mod.sample_conditional_prior(prior, mvn_method="eigh")
        beta_dot_data = prior_cond.filtered_prior_observed.values - prior_cond.filtered_prior.values

        if isinstance(exog_state_names, list) or k_exog is not None:
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
