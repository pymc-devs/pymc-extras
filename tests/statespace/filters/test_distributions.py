import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose
from scipy.stats import multivariate_normal

from pymc_extras.statespace import structural
from pymc_extras.statespace.filters.distributions import (
    LinearGaussianStateSpace,
    SequenceMvNormal,
    _LinearGaussianStateSpace,
)
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_DIM,
    OBS_STATE_DIM,
    TIME_DIM,
)
from tests.statespace.shared_fixtures import (  # pylint: disable=unused-import
    rng,
)
from tests.statespace.test_utilities import (
    delete_rvs_from_model,
    fast_eval,
    load_nile_test_data,
)

floatX = pytensor.config.floatX

# TODO: These are pretty loose because of all the stabilizing of covariance matrices that is done inside the kalman
#  filters. When that is improved, this should be tightened.
ATOL = 1e-5 if floatX.endswith("64") else 1e-4
RTOL = 1e-5 if floatX.endswith("64") else 1e-4

filter_names = [
    "standard",
    "cholesky",
    "univariate",
]


@pytest.fixture(scope="session")
def data():
    return load_nile_test_data()


@pytest.fixture(scope="session")
def pymc_model(data):
    with pm.Model() as mod:
        data = pm.Data("data", data.values)
        P0_diag = pm.Exponential("P0_diag", 1, shape=(2,))
        P0 = pm.Deterministic("P0", pt.diag(P0_diag))
        initial_trend = pm.Normal("initial_trend", shape=(2,))
        sigma_trend = pm.Exponential("sigma_trend", 1, shape=(2,))

    return mod


@pytest.fixture(scope="session")
def pymc_model_2(data):
    coords = {
        ALL_STATE_DIM: ["level", "trend"],
        OBS_STATE_DIM: ["level"],
        TIME_DIM: np.arange(101, dtype="int"),
    }

    with pm.Model(coords=coords) as mod:
        P0_diag = pm.Exponential("P0_diag", 1, shape=(2,))
        P0 = pm.Deterministic("P0", pt.diag(P0_diag))
        initial_trend = pm.Normal("initial_trend", shape=(2,))
        sigma_trend = pm.Exponential("sigma_trend", 1, shape=(2,))
        sigma_me = pm.Exponential("sigma_error", 1)

    return mod


@pytest.fixture(scope="session")
def ss_mod_me():
    ss_mod = structural.LevelTrendComponent(order=2)
    ss_mod += structural.MeasurementError(name="error")
    ss_mod = ss_mod.build("data", verbose=False)

    return ss_mod


@pytest.fixture(scope="session")
def ss_mod_no_me():
    ss_mod = structural.LevelTrendComponent(order=2)
    ss_mod = ss_mod.build("data", verbose=False)

    return ss_mod


@pytest.mark.parametrize("kfilter", filter_names, ids=filter_names)
def test_loglike_vectors_agree(kfilter, pymc_model):
    # TODO: This test might be flakey, I've gotten random failures
    ss_mod = structural.LevelTrendComponent(order=2).build(
        "data", verbose=False, filter_type=kfilter
    )
    with pymc_model:
        ss_mod._insert_random_variables()
        matrices = ss_mod.unpack_statespace()

        filter_outputs = ss_mod.kalman_filter.build_graph(pymc_model["data"], *matrices)
        filter_mus, pred_mus, obs_mu, filter_covs, pred_covs, obs_cov, ll = filter_outputs

    test_ll = fast_eval(ll)

    # TODO: BUG: Why does fast eval end up with a 2d output when filter is "single"?
    obs_mu_np = obs_mu.eval()
    obs_cov_np = fast_eval(obs_cov)
    data_np = fast_eval(pymc_model["data"])

    scipy_lls = []
    for y, mu, cov in zip(data_np, obs_mu_np, obs_cov_np):
        scipy_lls.append(multivariate_normal.logpdf(y, mean=mu, cov=cov))
    assert_allclose(test_ll, np.array(scipy_lls).ravel(), atol=ATOL, rtol=RTOL)


def test_sequence_mvn_distribution():
    # Base Case
    mu_sequence = pt.tensor("mu_sequence", shape=(100, 3))
    cov_sequence = pt.tensor("cov_sequence", shape=(100, 3, 3))
    logp = pt.tensor("logp", shape=(100,))

    dist = SequenceMvNormal.dist(mu_sequence, cov_sequence, logp)
    assert dist.type.shape == (100, 3)

    # With batch dimension
    mu_sequence = pt.tensor("mu_sequence", shape=(10, 100, 3))
    cov_sequence = pt.tensor("cov_sequence", shape=(10, 100, 3, 3))
    logp = pt.tensor(
        "logp",
        shape=(
            10,
            100,
        ),
    )

    dist = SequenceMvNormal.dist(mu_sequence, cov_sequence, logp)
    assert dist.type.shape == (10, 100, 3)


@pytest.mark.parametrize("output_name", ["states_latent", "states_observed"])
def test_lgss_distribution_from_steps(output_name, ss_mod_me, pymc_model_2):
    with pymc_model_2:
        ss_mod_me._insert_random_variables()
        matrices = ss_mod_me.unpack_statespace()

        # pylint: disable=unpacking-non-sequence
        latent_states, obs_states = LinearGaussianStateSpace("states", *matrices, steps=100)
        # pylint: enable=unpacking-non-sequence

        idata = pm.sample_prior_predictive(draws=10)
        delete_rvs_from_model(["states_latent", "states_observed", "states_combined"])

    assert idata.prior.coords["states_latent_dim_0"].shape == (101,)
    assert not np.any(np.isnan(idata.prior[output_name].values))


@pytest.mark.parametrize("output_name", ["states_latent", "states_observed"])
def test_lgss_distribution_with_dims(output_name, ss_mod_me, pymc_model_2):
    with pymc_model_2:
        ss_mod_me._insert_random_variables()
        matrices = ss_mod_me.unpack_statespace()

        # pylint: disable=unpacking-non-sequence
        latent_states, obs_states = LinearGaussianStateSpace(
            "states",
            *matrices,
            steps=100,
            dims=[TIME_DIM, ALL_STATE_DIM, OBS_STATE_DIM],
            sequence_names=[],
            k_endog=ss_mod_me.k_endog,
        )
        # pylint: enable=unpacking-non-sequence
        idata = pm.sample_prior_predictive(draws=10)
        delete_rvs_from_model(["states_latent", "states_observed", "states_combined"])

    assert idata.prior.coords["time"].shape == (101,)
    assert all(
        [dim in idata.prior.states_latent.coords.keys() for dim in [TIME_DIM, ALL_STATE_DIM]]
    )
    assert all(
        [dim in idata.prior.states_observed.coords.keys() for dim in [TIME_DIM, OBS_STATE_DIM]]
    )
    assert not np.any(np.isnan(idata.prior[output_name].values))


@pytest.mark.parametrize("output_name", ["states_latent", "states_observed"])
def test_lgss_with_time_varying_inputs(output_name, rng):
    X = rng.random(size=(10, 3), dtype=floatX)
    ss_mod = structural.LevelTrendComponent() + structural.RegressionComponent(
        name="exog", k_exog=3
    )
    mod = ss_mod.build("data", verbose=False)

    coords = {
        ALL_STATE_DIM: ["level", "trend", "beta_1", "beta_2", "beta_3"],
        OBS_STATE_DIM: ["level"],
        TIME_DIM: np.arange(10, dtype="int"),
    }

    with pm.Model(coords=coords):
        exog_data = pm.Data("data_exog", X)
        P0_diag = pm.Exponential("P0_diag", 1, shape=(mod.k_states,))
        P0 = pm.Deterministic("P0", pt.diag(P0_diag))
        initial_trend = pm.Normal("initial_trend", shape=(2,))
        sigma_trend = pm.Exponential("sigma_trend", 1, shape=(2,))
        beta_exog = pm.Normal("beta_exog", shape=(3,))

        mod._insert_random_variables()
        mod._insert_data_variables()
        matrices = mod.unpack_statespace()

        # pylint: disable=unpacking-non-sequence
        latent_states, obs_states = LinearGaussianStateSpace(
            "states",
            *matrices,
            steps=9,
            sequence_names=["d", "Z"],
            dims=[TIME_DIM, ALL_STATE_DIM, OBS_STATE_DIM],
        )
        # pylint: enable=unpacking-non-sequence
        idata = pm.sample_prior_predictive(draws=10)

    assert idata.prior.coords["time"].shape == (10,)
    assert all(
        [dim in idata.prior.states_latent.coords.keys() for dim in [TIME_DIM, ALL_STATE_DIM]]
    )
    assert all(
        [dim in idata.prior.states_observed.coords.keys() for dim in [TIME_DIM, OBS_STATE_DIM]]
    )
    assert not np.any(np.isnan(idata.prior[output_name].values))


def test_lgss_signature():
    # Base case
    x0 = pt.tensor("x0", shape=(None,))
    P0 = pt.tensor("P0", shape=(None, None))
    c = pt.tensor("c", shape=(None,))
    d = pt.tensor("d", shape=(None,))
    T = pt.tensor("T", shape=(None, None))
    Z = pt.tensor("Z", shape=(None, None))
    R = pt.tensor("R", shape=(None, None))
    H = pt.tensor("H", shape=(None, None))
    Q = pt.tensor("Q", shape=(None, None))

    lgss = _LinearGaussianStateSpace.dist(x0, P0, c, d, T, Z, R, H, Q, steps=100)
    assert (
        lgss.owner.op.extended_signature
        == "(s),(s,s),(s),(p),(s,s),(p,s),(s,r),(p,p),(r,r),[rng]->[rng],(t,n)"
    )
    assert lgss.owner.op.ndim_supp == 2
    assert lgss.owner.op.ndims_params == [1, 2, 1, 1, 2, 2, 2, 2, 2]

    # Case with time-varying matrices
    T = pt.tensor("T", shape=(None, None, None))
    lgss = _LinearGaussianStateSpace.dist(
        x0, P0, c, d, T, Z, R, H, Q, steps=100, sequence_names=["T"]
    )

    assert (
        lgss.owner.op.extended_signature
        == "(s),(s,s),(s),(p),(t,s,s),(p,s),(s,r),(p,p),(r,r),[rng]->[rng],(t,n)"
    )
    assert lgss.owner.op.ndim_supp == 2
    assert lgss.owner.op.ndims_params == [1, 2, 1, 1, 3, 2, 2, 2, 2]
