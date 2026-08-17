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
    SimulationSmoother,
    _forward_simulate_latent_and_obs,
    _LinearGaussianStateSpace,
)
from pymc_extras.statespace.filters.kalman_filter import StandardFilter
from pymc_extras.statespace.filters.kalman_smoother import KalmanSmoother
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
    nile_test_test_helper,
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
        initial_trend = pm.Normal("initial_level_trend", shape=(2,))
        sigma_trend = pm.Exponential("sigma_level_trend", 1, shape=(2,))

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
        initial_trend = pm.Normal("initial_level_trend", shape=(2,))
        sigma_trend = pm.Exponential("sigma_level_trend", 1, shape=(2,))
        sigma_me = pm.Exponential("sigma_error", 1)

    return mod


@pytest.fixture(scope="session")
def ss_mod_me():
    ss_mod = structural.LevelTrend(order=2)
    ss_mod += structural.MeasurementError(name="error")
    ss_mod = ss_mod.build("data", verbose=False)

    return ss_mod


@pytest.fixture(scope="session")
def ss_mod_no_me():
    ss_mod = structural.LevelTrend(order=2)
    ss_mod = ss_mod.build("data", verbose=False)

    return ss_mod


@pytest.mark.parametrize("kfilter", filter_names)
def test_loglike_vectors_agree(kfilter, pymc_model):
    # TODO: This test might be flakey, I've gotten random failures
    ss_mod = structural.LevelTrend(order=2).build("data", verbose=False, filter_type=kfilter)
    with pymc_model:
        matrices = ss_mod._insert_random_variables()

        kalman_filter, _ = ss_mod.make_filters()
        filter_outputs = kalman_filter.build_graph(pymc_model["data"], *matrices)
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
        matrices = ss_mod_me._insert_random_variables()

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
        matrices = ss_mod_me._insert_random_variables()

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
    ss_mod = structural.LevelTrend() + structural.Regression(
        name="exog", state_names=["exog_0", "exog_1", "exog_2"]
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
        initial_trend = pm.Normal("initial_level_trend", shape=(2,))
        sigma_trend = pm.Exponential("sigma_level_trend", 1, shape=(2,))
        beta_exog = pm.Normal("beta_exog", shape=(3,))

        matrices = mod._insert_random_variables()
        matrices = mod._insert_data_variables(matrices)

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


@pytest.mark.parametrize("append_x0", [True, False], ids=["with_x0", "without_x0"])
def test_forward_simulation_reads_one_matrix_row_per_timestep(append_x0):
    """Every simulated timestep consumes its own row of a time-varying matrix.

    Simulates a noise-free model whose state counts upward from zero and whose
    observation intercept encodes its own time index, so each returned observation names
    the row it read: ``y_t == 100 * row + alpha_t``.
    """
    steps = 4
    scalar_zero = np.zeros((1, 1), dtype=floatX)
    scalar_one = np.ones((1, 1), dtype=floatX)
    d_time_varying = (np.arange(steps + 1, dtype=floatX) * 100).reshape(-1, 1)

    alpha, y, _ = _forward_simulate_latent_and_obs(
        pt.as_tensor_variable(np.zeros(1, dtype=floatX)),
        pt.as_tensor_variable(scalar_zero),
        pt.as_tensor_variable(np.ones(1, dtype=floatX)),
        pt.as_tensor_variable(d_time_varying),
        pt.as_tensor_variable(scalar_one),
        pt.as_tensor_variable(scalar_one),
        pt.as_tensor_variable(scalar_one),
        pt.as_tensor_variable(scalar_zero),
        pt.as_tensor_variable(scalar_zero),
        steps=steps,
        rng=pytensor.shared(np.random.default_rng(0)),
        sequence_names=("d",),
        append_x0=append_x0,
    )
    alpha_val, y_val = (v.ravel() for v in pytensor.function([], [alpha, y])())

    # The initial state is alpha_0, so timestep t holds the value t and reads row t.
    expected_timesteps = np.arange(0 if append_x0 else 1, steps + 1, dtype=floatX)

    assert_allclose(alpha_val, expected_timesteps)
    assert_allclose(y_val, expected_timesteps * 100 + expected_timesteps)


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


def _analytic_joint_posterior(a0, P0, c, d, T_mat, Z_mat, R_mat, H_mat, Q_mat, y):
    """Closed-form posterior of a stacked LGSSM, used as ground truth.

    Stacks ``alpha_1, ..., alpha_T`` into one big vector and writes the joint
    prior + likelihood as ``MvN(mean_prior, cov_prior)`` plus a Gaussian
    observation model, then conditions on ``y`` analytically. Uses the
    Durbin-Koopman convention where ``(a0, P0)`` is the predicted distribution
    of ``alpha_1``, matching the Kalman filter implementation.
    """
    T_steps, k_endog = y.shape
    k_states = a0.shape[0]

    mean = np.zeros((T_steps, k_states))
    cov = np.zeros((T_steps, T_steps, k_states, k_states))
    mean[0] = a0
    cov[0, 0] = P0
    for t in range(1, T_steps):
        mean[t] = c + T_mat @ mean[t - 1]
        cov[t, t] = T_mat @ cov[t - 1, t - 1] @ T_mat.T + R_mat @ Q_mat @ R_mat.T
        for s in range(t):
            cov[t, s] = T_mat @ cov[t - 1, s]
            cov[s, t] = cov[t, s].T

    prior_mean = mean.reshape(-1)
    prior_cov = cov.transpose(0, 2, 1, 3).reshape(T_steps * k_states, T_steps * k_states)

    Z_block = np.zeros((T_steps * k_endog, T_steps * k_states))
    H_block = np.zeros((T_steps * k_endog, T_steps * k_endog))
    D = np.tile(d, T_steps)
    for t in range(T_steps):
        Z_block[t * k_endog : (t + 1) * k_endog, t * k_states : (t + 1) * k_states] = Z_mat
        H_block[t * k_endog : (t + 1) * k_endog, t * k_endog : (t + 1) * k_endog] = H_mat

    cross = prior_cov @ Z_block.T
    obs_cov = Z_block @ cross + H_block
    gain = cross @ np.linalg.inv(obs_cov)
    post_mean = prior_mean + gain @ (y.reshape(-1) - (D + Z_block @ prior_mean))
    post_cov = prior_cov - gain @ Z_block @ prior_cov

    return post_mean.reshape(T_steps, k_states), post_cov


@pytest.fixture
def small_lgssm():
    """Tiny 2-state, 2-obs LGSSM with non-zero c, d and stable T."""
    k_states, k_endog, n_steps = 2, 2, 15
    a0 = np.array([0.5, -0.2], dtype=floatX)
    P0 = np.eye(k_states, dtype=floatX) * 0.1
    c = np.array([0.05, -0.03], dtype=floatX)
    d = np.array([0.1, 0.02], dtype=floatX)
    T_mat = np.array([[0.9, 0.1], [0.0, 0.8]], dtype=floatX)
    Z_mat = np.array([[1.0, 0.0], [0.5, 1.0]], dtype=floatX)
    R_mat = np.eye(k_states, dtype=floatX)
    H_mat = np.eye(k_endog, dtype=floatX) * 0.2
    Q_mat = np.eye(k_states, dtype=floatX) * 0.3

    rng_ = np.random.default_rng(42)
    y = np.zeros((n_steps, k_endog))
    a_prev = a0.copy()
    for t in range(n_steps):
        a_prev = c + T_mat @ a_prev + R_mat @ rng_.multivariate_normal(np.zeros(k_states), Q_mat)
        y[t] = d + Z_mat @ a_prev + rng_.multivariate_normal(np.zeros(k_endog), H_mat)

    return {
        "a0": a0,
        "P0": P0,
        "c": c,
        "d": d,
        "T": T_mat,
        "Z": Z_mat,
        "R": R_mat,
        "H": H_mat,
        "Q": Q_mat,
        "y": y,
        "n_steps": n_steps,
    }


def _build_simulation_smoother_func(params, seed):
    """Compile a callable that returns ``(a_smooth, sample)`` per call."""
    a0 = pt.as_tensor_variable(params["a0"])
    P0 = pt.as_tensor_variable(params["P0"])
    c = pt.as_tensor_variable(params["c"])
    d = pt.as_tensor_variable(params["d"])
    T_mat = pt.as_tensor_variable(params["T"])
    Z_mat = pt.as_tensor_variable(params["Z"])
    R_mat = pt.as_tensor_variable(params["R"])
    H_mat = pt.as_tensor_variable(params["H"])
    Q_mat = pt.as_tensor_variable(params["Q"])
    y = pt.as_tensor_variable(np.asarray(params["y"], dtype=floatX))

    filt = StandardFilter().build_graph(y, a0, P0, c, d, T_mat, Z_mat, R_mat, H_mat, Q_mat)
    a_smooth, _ = KalmanSmoother().build_graph(T_mat, R_mat, Q_mat, filt[0], filt[3])

    rng_var = pytensor.shared(np.random.default_rng(seed), name="rng")
    sample = SimulationSmoother.dist(
        a_smooth,
        a0,
        P0,
        c,
        d,
        T_mat,
        Z_mat,
        R_mat,
        H_mat,
        Q_mat,
        kalman_filter=StandardFilter(),
        kalman_smoother=KalmanSmoother(),
        rng=rng_var,
    )
    return pm.compile([], [a_smooth, sample], on_unused_input="ignore")


def test_simulation_smoother_signature(small_lgssm):
    """Construction sanity: extended_signature and shape match the spec."""
    params = small_lgssm
    a_smooth = pt.zeros((params["n_steps"], 2))
    sample = SimulationSmoother.dist(
        a_smooth,
        pt.as_tensor_variable(params["a0"]),
        pt.as_tensor_variable(params["P0"]),
        pt.as_tensor_variable(params["c"]),
        pt.as_tensor_variable(params["d"]),
        pt.as_tensor_variable(params["T"]),
        pt.as_tensor_variable(params["Z"]),
        pt.as_tensor_variable(params["R"]),
        pt.as_tensor_variable(params["H"]),
        pt.as_tensor_variable(params["Q"]),
        kalman_filter=StandardFilter(),
        kalman_smoother=KalmanSmoother(),
    )
    assert sample.type.shape == (params["n_steps"], 2)
    assert sample.owner.op.ndim_supp == 2
    assert (
        sample.owner.op.extended_signature
        == "(t,s),(s),(s,s),(s),(p),(s,s),(p,s),(s,r),(p,p),(r,r),[rng]->[rng],(t,s)"
    )

    # Time-varying case: the declared matrix gains a leading time axis.
    d_time_varying = pt.tensor("d", shape=(None, None))
    sample = SimulationSmoother.dist(
        a_smooth,
        pt.as_tensor_variable(params["a0"]),
        pt.as_tensor_variable(params["P0"]),
        pt.as_tensor_variable(params["c"]),
        d_time_varying,
        pt.as_tensor_variable(params["T"]),
        pt.as_tensor_variable(params["Z"]),
        pt.as_tensor_variable(params["R"]),
        pt.as_tensor_variable(params["H"]),
        pt.as_tensor_variable(params["Q"]),
        kalman_filter=StandardFilter(time_varying_names=["obs_intercept"]),
        kalman_smoother=KalmanSmoother(),
        sequence_names=("d",),
    )
    assert (
        sample.owner.op.extended_signature
        == "(t,s),(s),(s,s),(s),(t,p),(s,s),(p,s),(s,r),(p,p),(r,r),[rng]->[rng],(t,s)"
    )


def test_simulation_smoother_joint_covariance(small_lgssm):
    """Empirical joint mean and covariance over draws match the analytic posterior."""
    params = small_lgssm
    post_mean, post_cov = _analytic_joint_posterior(
        params["a0"],
        params["P0"],
        params["c"],
        params["d"],
        params["T"],
        params["Z"],
        params["R"],
        params["H"],
        params["Q"],
        params["y"],
    )
    f = _build_simulation_smoother_func(params, seed=12345)
    n_draws = 5_000
    samples = np.stack([f()[1] for _ in range(n_draws)])
    emp_cov = np.cov(samples.reshape(n_draws, -1).T)
    assert_allclose(samples.mean(0), post_mean, atol=0.05)
    assert_allclose(emp_cov, post_cov, atol=0.04)


def test_simulation_smoother_unbiased_under_large_d(small_lgssm):
    """Draws stay centered on the smoothed mean when the observation intercept is large.

    The Durbin-Koopman identity is invariant to ``d``, so a ``d`` term applied
    inconsistently across the simulated trajectory shows up as a mean shift here while
    leaving the joint covariance intact.
    """
    params = {**small_lgssm, "d": np.array([50.0, -30.0], dtype=floatX)}
    f = _build_simulation_smoother_func(params, seed=99)

    a_smooth = f()[0]
    draws = np.stack([f()[1] for _ in range(2_000)])

    assert_allclose(draws.mean(0), a_smooth, atol=0.1)


def test_simulation_smoother_with_time_varying_matrix(small_lgssm):
    """Draws stay centered on the smoothed mean when a matrix varies over time.

    Exercises the ``sequence_names`` path, where the forward simulation, the filter and
    the smoother must all index the same timestep of ``d``.
    """
    params = small_lgssm
    n_steps, k_endog = params["n_steps"], params["d"].shape[0]

    # A d that swings over time, so a mis-indexed row shifts the smoothed mean.
    d_time_varying = np.linspace(-5.0, 5.0, n_steps * k_endog, dtype=floatX).reshape(
        n_steps, k_endog
    )

    tensors = {
        k: pt.as_tensor_variable(params[k]) for k in ("a0", "P0", "c", "T", "Z", "R", "H", "Q")
    }
    d = pt.as_tensor_variable(d_time_varying)
    y = pt.as_tensor_variable(np.asarray(params["y"], dtype=floatX))

    filt = StandardFilter(time_varying_names=["obs_intercept"]).build_graph(
        y,
        tensors["a0"],
        tensors["P0"],
        tensors["c"],
        d,
        tensors["T"],
        tensors["Z"],
        tensors["R"],
        tensors["H"],
        tensors["Q"],
    )
    a_smooth, _ = KalmanSmoother().build_graph(
        tensors["T"], tensors["R"], tensors["Q"], filt[0], filt[3]
    )

    sample = SimulationSmoother.dist(
        a_smooth,
        tensors["a0"],
        tensors["P0"],
        tensors["c"],
        d,
        tensors["T"],
        tensors["Z"],
        tensors["R"],
        tensors["H"],
        tensors["Q"],
        kalman_filter=StandardFilter(time_varying_names=["obs_intercept"]),
        kalman_smoother=KalmanSmoother(),
        sequence_names=("d",),
        rng=pytensor.shared(np.random.default_rng(7), name="rng"),
    )
    f = pm.compile([], [a_smooth, sample], on_unused_input="ignore")

    smoothed_mean = f()[0]
    draws = np.stack([f()[1] for _ in range(2_000)])

    assert_allclose(draws.mean(0), smoothed_mean, atol=0.1)


def test_simulation_smoother_against_statsmodels(rng):
    """End-to-end check against statsmodels' simulation_smoother on the Nile model.

    Builds a Local Linear Trend in statsmodels and the same matrices in pymc-extras,
    draws from both simulation smoothers, and asserts empirical mean and joint
    covariance agree to MC tolerance.
    """
    sm_res, [data, _a0, _diffuse_P0, c, d, T_mat, Z_mat, R_mat, H_mat, Q_mat] = (
        nile_test_test_helper(rng)
    )
    # Override the helper's diffuse P0=1e6*I with a tight prior so MC noise
    # doesn't dominate the comparison.
    a0 = np.zeros(2, dtype=floatX)
    P0 = np.eye(2, dtype=floatX) * 0.5
    sm_res.model.initialize_known(initial_state=a0, initial_state_cov=P0)
    sm_res = sm_res.model.smooth(sm_res.params)

    params = {
        "a0": a0,
        "P0": P0,
        "c": c,
        "d": d,
        "T": T_mat,
        "Z": Z_mat,
        "R": R_mat,
        "H": H_mat,
        "Q": Q_mat,
        "y": data,
    }
    f = _build_simulation_smoother_func(params, seed=42)

    n_draws = 2_000
    ours = np.stack([f()[1] for _ in range(n_draws)])

    sim = sm_res.model.simulation_smoother(random_state=1234)
    sm_samples = np.empty_like(ours)
    for i in range(n_draws):
        sim.simulate()
        sm_samples[i] = sim.simulated_state.T

    cov_ours = np.cov(ours.reshape(n_draws, -1).T)
    cov_sm = np.cov(sm_samples.reshape(n_draws, -1).T)
    assert_allclose(cov_ours, cov_sm, atol=0.05)
    assert_allclose(ours.mean(0), sm_samples.mean(0), atol=0.1)
