import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose, assert_array_less
from pytensor.tensor.linalg import solve_discrete_lyapunov
from scipy.linalg import solve_discrete_lyapunov as sp_solve_discrete_lyapunov
from scipy.stats import multivariate_normal
from statsmodels.tsa.statespace.mlemodel import MLEModel

from pymc_extras.statespace import structural
from pymc_extras.statespace.filters.distributions import (
    LinearGaussianStateSpace,
    SequenceMvNormal,
    SimulationSmoother,
    StationaryVAR,
    _forward_simulate_latent_and_obs,
    _LinearGaussianStateSpace,
    _predictive_moments,
)
from pymc_extras.statespace.filters.kalman_filter import StandardFilter
from pymc_extras.statespace.filters.kalman_smoother import RTSSmoother
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


def _statsmodels_smoother(params):
    """Smooth ``params["y"]`` with statsmodels on the same nine matrices."""
    model = MLEModel(
        params["y"],
        k_states=params["T"].shape[0],
        k_posdef=params["Q"].shape[0],
        initialization="known",
        initial_state=params["a0"],
        initial_state_cov=params["P0"],
    )
    model["state_intercept"] = params["c"]
    model["obs_intercept"] = params["d"]
    model["transition"] = params["T"]
    model["design"] = params["Z"]
    model["selection"] = params["R"]
    model["obs_cov"] = params["H"]
    model["state_cov"] = params["Q"]

    return model.ssm.smooth()


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


@pytest.fixture
def nile_lgssm(rng):
    """Local linear trend on the Nile data with a tight initial state, so the smoothed
    posterior is well conditioned."""
    _, [y, _, _, c, d, T_mat, Z_mat, R_mat, H_mat, Q_mat] = nile_test_test_helper(rng)

    return {
        "a0": np.zeros(2, dtype=floatX),
        "P0": np.eye(2, dtype=floatX) * 0.5,
        "c": c,
        "d": d,
        "T": T_mat,
        "Z": Z_mat,
        "R": R_mat,
        "H": H_mat,
        "Q": Q_mat,
        "y": y,
    }


class TestSimulationSmoother:
    """Draws from one compiled simulation smoother, fed with the matrices of each test."""

    INPUT_NAMES = ("y", "a0", "P0", "c", "d", "T", "Z", "R", "H", "Q")

    @classmethod
    def setup_class(cls):
        y = pt.tensor("y", dtype=floatX, shape=(None, None))
        a0 = pt.tensor("a0", dtype=floatX, shape=(None,))
        P0 = pt.tensor("P0", dtype=floatX, shape=(None, None))
        c = pt.tensor("c", dtype=floatX, shape=(None,))
        d = pt.tensor("d", dtype=floatX, shape=(None,))
        T_mat = pt.tensor("T", dtype=floatX, shape=(None, None))
        Z_mat = pt.tensor("Z", dtype=floatX, shape=(None, None))
        R_mat = pt.tensor("R", dtype=floatX, shape=(None, None))
        H_mat = pt.tensor("H", dtype=floatX, shape=(None, None))
        Q_mat = pt.tensor("Q", dtype=floatX, shape=(None, None))
        matrices = (a0, P0, c, d, T_mat, Z_mat, R_mat, H_mat, Q_mat)

        filt = StandardFilter().build_graph(y, *matrices)
        a_smooth, _ = RTSSmoother().build_graph(y, matrices, filt)

        cls.rng = pytensor.shared(np.random.default_rng(0), name="rng")
        sample = SimulationSmoother.dist(
            y,
            *matrices,
            kalman_filter=StandardFilter(),
            kalman_smoother=RTSSmoother(),
            rng=cls.rng,
        )
        cls.simulate = pm.compile([y, *matrices], [a_smooth, sample], on_unused_input="ignore")

    def draws(self, params, seed, n_draws):
        """Return the smoothed mean and ``n_draws`` simulation-smoother draws for ``params``."""
        self.rng.set_value(np.random.default_rng(seed))
        args = [np.asarray(params[name], dtype=floatX) for name in self.INPUT_NAMES]

        a_smooth, _ = self.simulate(*args)
        return a_smooth, np.stack([self.simulate(*args)[1] for _ in range(n_draws)])

    def test_draws_are_affine_in_the_data(self, small_lgssm, rng):
        """Two draws from the same rng state differ by the zero-intercept smoother of the data
        difference, which pins down the Durbin-Koopman composition exactly.

        The forward simulation of ``alpha_plus`` and ``y_plus`` depends only on the rng state, so
        resetting it before each call cancels the noise from the difference of the draws.
        """
        params = small_lgssm
        y_shift = rng.normal(size=params["y"].shape).astype(floatX)
        shifted = {**params, "y": params["y"] + y_shift}

        _, [draw] = self.draws(params, seed=7, n_draws=1)
        _, [shifted_draw] = self.draws(shifted, seed=7, n_draws=1)

        zero_intercepts = {
            **params,
            "y": y_shift,
            "a0": np.zeros_like(params["a0"]),
            "c": np.zeros_like(params["c"]),
            "d": np.zeros_like(params["d"]),
        }
        expected_difference = _statsmodels_smoother(zero_intercepts).smoothed_state.T

        assert_allclose(shifted_draw - draw, expected_difference, atol=ATOL, rtol=RTOL)

    @pytest.mark.parametrize(
        "fixture_name, d",
        [
            ("small_lgssm", None),
            ("small_lgssm", np.array([50.0, -30.0])),
            ("nile_lgssm", None),
        ],
        ids=["small", "small_large_d", "nile"],
    )
    def test_draws_match_statsmodels_posterior(self, fixture_name, d, request):
        """The smoothed mean matches statsmodels exactly, and the sample mean, per-step
        covariances, and lag-one autocovariances of the draws match it to Monte Carlo error.

        The smoothed states form a Gaussian Markov chain, so those moments pin down the whole
        joint posterior. The large ``d`` case catches an observation intercept applied
        inconsistently across the simulated trajectory, which shows up as a mean shift.
        """
        params = request.getfixturevalue(fixture_name)
        if d is not None:
            params = {**params, "d": d.astype(floatX)}
        reference = _statsmodels_smoother(params)

        n_draws = 5_000
        a_smooth, draws = self.draws(params, seed=42, n_draws=n_draws)
        assert_allclose(a_smooth, reference.smoothed_state.T, atol=ATOL, rtol=RTOL)

        centered = draws - reference.smoothed_state.T
        sample_cov = np.einsum("nti,ntj->ijt", centered, centered) / n_draws
        sample_autocov = np.einsum("nti,ntj->ijt", centered[:, 1:], centered[:, :-1]) / n_draws
        mc_tolerance = 5 * np.sqrt(2 / n_draws) * reference.smoothed_state_cov.max()

        assert_allclose(draws.mean(0), reference.smoothed_state.T, atol=mc_tolerance)
        assert_allclose(sample_cov, reference.smoothed_state_cov, atol=mc_tolerance)
        assert_allclose(
            sample_autocov, reference.smoothed_state_autocov[..., :-1], atol=mc_tolerance
        )


def test_simulation_smoother_signature(small_lgssm):
    """Construction sanity: extended_signature and shape match the spec."""
    params = small_lgssm
    data = pt.zeros((params["n_steps"], 1))
    sample = SimulationSmoother.dist(
        data,
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
        kalman_smoother=RTSSmoother(),
    )
    assert sample.type.shape == (params["n_steps"], 2)
    assert sample.owner.op.ndim_supp == 2
    assert (
        sample.owner.op.extended_signature
        == "(t,p),(s),(s,s),(s),(p),(s,s),(p,s),(s,r),(p,p),(r,r),[rng]->[rng],(t,s)"
    )

    # Time-varying case: the declared matrix gains a leading time axis.
    d_time_varying = pt.tensor("d", shape=(None, None))
    sample = SimulationSmoother.dist(
        data,
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
        kalman_smoother=RTSSmoother(),
        sequence_names=("d",),
    )
    assert (
        sample.owner.op.extended_signature
        == "(t,p),(s),(s,s),(s),(t,p),(s,s),(p,s),(s,r),(p,p),(r,r),[rng]->[rng],(t,s)"
    )


def test_simulation_smoother_with_time_varying_matrix(small_lgssm):
    """Shifting the data and a time-varying ``d`` by the same per-step sequence leaves a draw
    from the same rng state unchanged.

    Exercises the ``sequence_names`` path, where the forward simulation must index the same
    timestep of ``d`` as the data. A row applied one step off leaves a residual equal to the
    difference between consecutive shifts.
    """
    params = small_lgssm
    n_steps, k_endog = params["n_steps"], params["d"].shape[0]
    matrices = [
        pt.as_tensor_variable(params[name]) for name in ("a0", "P0", "c", "T", "Z", "R", "H", "Q")
    ]
    y = pt.tensor("y", dtype=floatX, shape=(n_steps, k_endog))
    d = pt.tensor("d", dtype=floatX, shape=(n_steps, k_endog))
    rng = pytensor.shared(np.random.default_rng(7), name="rng")

    sample = SimulationSmoother.dist(
        y,
        *matrices[:3],
        d,
        *matrices[3:],
        kalman_filter=StandardFilter(time_varying_names=["obs_intercept"]),
        kalman_smoother=RTSSmoother(),
        sequence_names=("d",),
        rng=rng,
    )
    draw = pm.compile([y, d], sample)

    # Both sequences swing over time, so a mis-indexed row cannot cancel.
    d_time_varying = np.linspace(-5.0, 5.0, n_steps * k_endog, dtype=floatX).reshape(
        n_steps, k_endog
    )
    shift = np.sin(np.arange(n_steps * k_endog, dtype=floatX)).reshape(n_steps, k_endog)
    y_value = np.asarray(params["y"], dtype=floatX)

    rng.set_value(np.random.default_rng(7))
    unshifted = draw(y_value, d_time_varying)
    rng.set_value(np.random.default_rng(7))
    shifted = draw(y_value + shift, d_time_varying + shift)

    assert_allclose(shifted, unshifted, atol=ATOL, rtol=RTOL)


def _var_parameters(rng, k_endog, order, k_exog):
    """Draw a stationary ``A``, a ``B``, and a positive-definite ``Q``.

    ``A`` is shrunk by a fixed factor rather than by ``target / spectral_radius``: scaling ``A``
    by ``c`` does not scale the companion eigenvalues by ``c`` once ``order > 1``, so the
    proportional step converges to the target from above without ever crossing it.
    """
    A = rng.normal(scale=0.4, size=(k_endog, k_endog * order))
    companion = np.zeros((k_endog * order, k_endog * order))
    if order > 1:
        companion[k_endog:, : k_endog * (order - 1)] = np.eye(k_endog * (order - 1))

    while True:
        companion[:k_endog] = A
        if np.abs(np.linalg.eigvals(companion)).max() < 0.9:
            break
        A *= 0.9

    factor = rng.normal(size=(k_endog, k_endog))
    Q = factor @ factor.T + k_endog * np.eye(k_endog)

    return A.astype(floatX), rng.normal(size=(k_endog, k_exog)).astype(floatX), Q.astype(floatX)


def _reference_filter(coefficients, state_cov, endog, exog, exog_coefficients):
    """A Kalman filter over the companion form, built independently of the distribution."""
    n_timesteps, k_endog = endog.shape
    k_states = coefficients.type.shape[1]

    transition = pt.concatenate(
        [coefficients, pt.pad(pt.eye(k_states - k_endog), [(0, 0), (0, k_endog)])], axis=0
    )
    design = pt.concatenate([pt.eye(k_endog), pt.zeros((k_endog, k_states - k_endog))], axis=1)
    selection = pt.concatenate([pt.eye(k_endog), pt.zeros((k_states - k_endog, k_endog))], axis=0)

    return StandardFilter(time_varying_names=["obs_intercept"], cov_jitter=0.0).build_graph(
        pt.specify_shape(pt.as_tensor_variable(endog), (n_timesteps, k_endog)),
        pt.zeros((k_states,)),
        solve_discrete_lyapunov(
            transition,
            pt.linalg.matrix_dot(selection, state_cov, selection.T),
            method="bilinear",
        ),
        pt.zeros((k_states,)),
        pt.as_tensor_variable(exog) @ exog_coefficients.T,
        transition,
        design,
        selection,
        pt.zeros((k_endog, k_endog)),
        state_cov,
    )


def _var_logp_pair(k_endog, order, k_exog, n_timesteps=80):
    """The distribution's log-density and a Kalman filter over the same model.

    The reference is assembled from concatenations rather than from the distribution's own
    companion-form helper: a shared constructor would let both sides be wrong together.
    """
    rng = np.random.default_rng(sum(map(ord, f"statvar{k_endog}{order}{k_exog}")))
    A_true, B_true, Q_true = _var_parameters(rng, k_endog, order, k_exog)
    exog = rng.normal(size=(n_timesteps, k_exog)).astype(floatX)

    endog = np.zeros((n_timesteps, k_endog))
    noise = rng.multivariate_normal(np.zeros(k_endog), Q_true, size=n_timesteps)
    for t in range(n_timesteps):
        lags = [endog[t - lag] if t >= lag else np.zeros(k_endog) for lag in range(1, order + 1)]
        endog[t] = A_true @ np.concatenate(lags) + B_true @ exog[t] + noise[t]
    endog = endog.astype(floatX)

    A = pt.tensor("A", dtype=floatX, shape=(k_endog, k_endog * order))
    B = pt.tensor("B", dtype=floatX, shape=(k_endog, k_exog))
    Q = pt.tensor("Q", dtype=floatX, shape=(k_endog, k_endog))
    k_states = k_endog * order

    fast = pm.logp(
        StationaryVAR.dist(
            A,
            Q,
            pt.as_tensor_variable(endog),
            exog=pt.as_tensor_variable(exog),
            exog_coefficients=B,
        ),
        pt.as_tensor_variable(endog),
    )

    *_, ll = _reference_filter(A, Q, endog, exog, B)

    return fast, ll.sum(), [A, B, Q]


@pytest.mark.parametrize(
    "k_endog, order, k_exog",
    [(1, 1, 0), (3, 2, 2)],
    ids=["k1_p1_m0", "k3_p2_m2"],
)
def test_stationary_var_matches_kalman_filter(k_endog, order, k_exog):
    rng = np.random.default_rng(sum(map(ord, f"draws{k_endog}{order}{k_exog}")))
    fast, kalman, inputs = _var_logp_pair(k_endog, order, k_exog)
    fn = pytensor.function(
        inputs,
        [fast, kalman, *pt.grad(fast, inputs), *pt.grad(kalman, inputs)],
        on_unused_input="ignore",
    )

    for _ in range(3):
        found_logp, expected_logp, dA, dB, dQ, edA, edB, edQ = fn(
            *_var_parameters(rng, k_endog, order, k_exog)
        )
        assert_allclose(found_logp, expected_logp, atol=ATOL, rtol=RTOL)
        assert_allclose(dA, edA, atol=ATOL, rtol=RTOL)
        assert_allclose(dB, edB, atol=ATOL, rtol=RTOL)
        # Only the symmetric part of a gradient wrt a symmetric matrix is meaningful.
        assert_allclose(dQ + dQ.T, edQ + edQ.T, atol=ATOL, rtol=RTOL)


def test_stationary_var_signature():
    A = pt.tensor("A", dtype=floatX, shape=(2, 6))
    Q = pt.tensor("Q", dtype=floatX, shape=(2, 2))
    endog = pt.tensor("endog", dtype=floatX, shape=(100, 2))

    dist = StationaryVAR.dist(A, Q, endog)

    assert dist.type.shape == (100, 2)
    assert dist.owner.op.extended_signature == "(k,l),(k,m),(k,k),(t,m),(t,k),[rng]->[rng],(t,k)"
    assert dist.owner.op.ndim_supp == 2


def test_stationary_var_predictive_moments_match_the_filter(rng):
    """
    Draws come from the one-step-ahead predictive distributions, which are the filter's.

    The first ``order`` rows are the ones worth checking. Their moments come from the Cholesky
    factor of the stationary covariance rather than from the regression.
    """
    k_endog, order, n_timesteps = 2, 2, 40
    coefficients, _, state_cov = _var_parameters(rng, k_endog, order, 0)
    endog = rng.normal(size=(n_timesteps, k_endog)).astype(floatX)

    means, covariances = _predictive_moments(
        pt.as_tensor_variable(coefficients),
        pt.zeros((k_endog, 0)),
        pt.as_tensor_variable(state_cov),
        pt.zeros((n_timesteps, 0)),
        pt.as_tensor_variable(endog),
        order=order,
        k_endog=k_endog,
    )
    _, _, filter_means, _, _, filter_covariances, _ = _reference_filter(
        pt.as_tensor_variable(coefficients),
        pt.as_tensor_variable(state_cov),
        endog,
        np.zeros((n_timesteps, 0), dtype=floatX),
        pt.zeros((k_endog, 0)),
    )

    found_means, found_covariances, expected_means, expected_covariances = pytensor.function(
        [], [means, covariances, filter_means, filter_covariances]
    )()

    assert_allclose(found_means, expected_means, atol=ATOL, rtol=RTOL)
    assert_allclose(found_covariances, expected_covariances, atol=ATOL, rtol=RTOL)


def test_stationary_var_draws_come_from_the_predictive_moments(rng):
    """
    The sampling path uses the same moments the density does.

    ``_predictive_moments`` can be right while ``rv_op`` wires it up wrong, and a test that calls
    the helper directly would not notice. A draw must equal a same-seed ``MvNormal`` draw from
    the moments, since that is the only randomness in ``rv_op``.
    """
    k_endog, order, n_timesteps = 2, 2, 12
    coefficients, _, state_cov = _var_parameters(rng, k_endog, order, 0)
    endog = rng.normal(size=(n_timesteps, k_endog)).astype(floatX)

    means, covariances = _predictive_moments(
        pt.as_tensor_variable(coefficients),
        pt.zeros((k_endog, 0)),
        pt.as_tensor_variable(state_cov),
        pt.zeros((n_timesteps, 0)),
        pt.as_tensor_variable(endog),
        order=order,
        k_endog=k_endog,
    )

    draw = pm.draw(StationaryVAR.dist(coefficients, state_cov, endog), random_seed=13)
    expected = pm.draw(pm.MvNormal.dist(mu=means, cov=covariances, method="svd"), random_seed=13)

    assert_allclose(draw, expected, atol=ATOL, rtol=RTOL)


def test_stationary_var_logp_follows_the_observed_data():
    rng = np.random.default_rng(511)
    A, _, Q = _var_parameters(rng, 2, 2, 0)

    with pm.Model() as model:
        pm.Data("data", rng.normal(size=(30, 2)).astype(floatX))
        StationaryVAR("obs", A, Q, model["data"], observed=model["data"])
        before = model.compile_logp()({})
        pm.set_data({"data": rng.normal(size=(30, 2)).astype(floatX)})

        assert not np.isclose(before, model.compile_logp()({}))


@pytest.mark.parametrize(
    "coefficients, kwargs, message",
    [
        (np.eye(2), {"exog": np.zeros((10, 2))}, "both or neither"),
        (pt.matrix("A"), {}, "known statically"),
    ],
    ids=["exog_without_coefficients", "unshaped_coefficients"],
)
def test_stationary_var_rejects_inconsistent_arguments(coefficients, kwargs, message):
    with pytest.raises(ValueError, match=message):
        StationaryVAR.dist(coefficients, np.eye(2), np.zeros((10, 2)), **kwargs)
