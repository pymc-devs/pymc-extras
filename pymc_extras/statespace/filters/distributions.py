import pymc as pm
import pytensor
import pytensor.tensor as pt

from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import Continuous, SymbolicRandomVariable
from pymc.distributions.shape_utils import get_support_shape_1d
from pymc.logprob.abstract import _logprob
from pymc.pytensorf import intX, normalize_rng_param
from pytensor.graph.basic import Node
from pytensor.tensor.random import multivariate_normal

from pymc_extras.statespace.core.assumptions import declare_time_varying, is_time_varying
from pymc_extras.statespace.filters.utilities import (
    PARAM_NAMES,
    split_by_time_axis,
    unpack_scan_step,
)

floatX = pytensor.config.floatX
COV_ZERO_TOL = 0

lgss_shape_message = (
    "The LinearGaussianStateSpace distribution needs shape information to be constructed. "
    "Ensure that all input matrices have shape information specified."
)


# Core-shape axis labels used to build gufunc signatures: state, observed, exogenous
# (shock), time, and the concatenated state+observed axis.
STATES, OBS, EXOG, TIME, STATE_AND_OBS = "s", "p", "r", "t", "n"

STATESPACE_CORE_SHAPES = {
    "x0": (STATES,),
    "P0": (STATES, STATES),
    "c": (STATES,),
    "d": (OBS,),
    "T": (STATES, STATES),
    "Z": (OBS, STATES),
    "R": (STATES, EXOG),
    "H": (OBS, OBS),
    "Q": (EXOG, EXOG),
}


def _build_signature(core_shapes, sequence_names, output_shape):
    """Assemble an extended gufunc signature, prefixing a time axis onto time-varying matrices.

    Parameters
    ----------
    core_shapes : dict mapping str to tuple of str
        Matrix name to its static core-shape axis labels, in signature order.
    sequence_names : iterable of str
        Names of the matrices that vary over time.
    output_shape : tuple of str
        Core-shape axis labels of the single output.

    Returns
    -------
    signature : str
        Extended signature in pymc's ``[rng]``-aware gufunc format.
    """
    matrix_to_shape = dict(core_shapes)
    for matrix in sequence_names:
        matrix_to_shape[matrix] = (TIME, *matrix_to_shape[matrix])

    inputs = ",".join("(" + ",".join(shapes) + ")" for shapes in matrix_to_shape.values())

    return f"{inputs},[rng]->[rng],({','.join(output_shape)})"


def make_signature(sequence_names):
    return _build_signature(
        STATESPACE_CORE_SHAPES, sequence_names, output_shape=(TIME, STATE_AND_OBS)
    )


def _matrix_dummies(matrices, varying):
    """
    Build fresh input variables for an ``OpFromGraph`` over the statespace matrices.

    A dummy carries none of its source's assumptions, so the ones standing in for a
    time-varying matrix are declared again for use inside the op.

    Parameters
    ----------
    matrices : sequence of TensorVariable
        The matrices, in ``PARAM_NAMES`` order.
    varying : list of bool
        Whether each matrix carries a time axis.

    Returns
    -------
    dummies : list of TensorVariable
        Named inputs for the op.
    declared : list of TensorVariable
        The same inputs, declared time-varying wherever their source was, for the inner graph.
    """
    dummies = []
    for name, matrix in zip(PARAM_NAMES, matrices, strict=True):
        dummy = matrix.type()
        dummy.name = name
        dummies.append(dummy)

    declared = [
        declare_time_varying(dummy) if flag else dummy
        for dummy, flag in zip(dummies, varying, strict=True)
    ]
    return dummies, declared


def _forward_simulate_latent_and_obs(
    a0,
    P0,
    c,
    d,
    T,
    Z,
    R,
    H,
    Q,
    *,
    steps,
    rng,
    method="svd",
    append_x0=True,
):
    r"""Forward-sample a latent state and observation trajectory from an LGSSM.

    Constructs a single ``pytensor.scan`` whose inner step emits the observation for the
    state it receives and then transitions to the next state, so both draws consume the
    same timestep of any time-varying matrix:

    .. math::
        y_t &= d_t + Z_t a_t + \eta_t \\
        a_{t+1} &= c_t + T_t a_t + R_t \epsilon_t

    Time-varying matrices therefore carry one row per returned timestep.

    Parameters
    ----------
    a0, P0, c, d, T, Z, R, H, Q : TensorVariable
        State-space matrices in the canonical order. Either core-shape (static) or
        time-varying ``(time, *core)``; declare the latter with
        :func:`~pymc_extras.statespace.core.assumptions.declare_time_varying`.
    steps : TensorVariable or int
        Number of forward simulation steps.
    rng : RandomGenerator
        Pytensor RNG to thread through the scan.
    method : str, optional
        Multivariate-normal sampling method passed to ``pm.MvNormal.dist``.
        Default ``"svd"``.
    append_x0 : bool, optional
        Prepend the initial state to both trajectories, giving them length ``steps + 1``
        rather than ``steps``. Default True.

    Returns
    -------
    alpha : TensorVariable
        Latent trajectory, shape ``(steps[+1], k_states)``.
    y : TensorVariable
        Observation trajectory, shape ``(steps[+1], k_endog)``.
    next_rng : Variable
        RNG state after sampling.
    """
    sequences, non_sequences, seq_names, non_seq_names = split_by_time_axis(
        dict(zip(PARAM_NAMES, [c, d, T, Z, R, H, Q], strict=True))
    )

    def step_fn(*args):
        (rng, a), (c, d, T, Z, R, H, Q) = unpack_scan_step(
            args, seq_names, non_seq_names, PARAM_NAMES
        )

        middle_rng, y_innovation = pm.MvNormal.dist(
            mu=0, cov=H, rng=rng, method=method, return_next_rng=True
        )
        next_rng, a_innovation = pm.MvNormal.dist(
            mu=0, cov=Q, rng=middle_rng, method=method, return_next_rng=True
        )

        # Emit the observation for the incoming state and transition to the next one, so
        # both consume this step's slice of the time-varying matrices.
        y = d + Z @ a + y_innovation
        a_next = c + T @ a + R @ a_innovation

        return next_rng, a_next, y

    rng_for_scan, init_a_ = pm.MvNormal.dist(a0, P0, rng=rng, method=method, return_next_rng=True)

    # One step per returned timestep, plus one whose state is discarded: the trailing
    # transition has no observation to pair with.
    (next_rng, alpha_seq, y_seq) = pytensor.scan(
        step_fn,
        outputs_info=[rng_for_scan, init_a_, None],
        sequences=sequences or None,
        non_sequences=non_sequences,
        n_steps=steps + 1,
        strict=True,
        return_updates=False,
    )

    # scan writes the initial state into slot zero of the tape and returns a view that
    # drops it; the view's input is the tape, which is the trajectory we simulated.
    alpha_tape = alpha_seq.owner.inputs[0]
    if append_x0:
        alpha = alpha_tape[:-1]
        y = y_seq
    else:
        alpha = alpha_tape[1:-1]
        y = y_seq[1:]

    return alpha, y, next_rng


class LinearGaussianStateSpaceRV(SymbolicRandomVariable):
    default_output = 1
    _print_name = ("LinearGuassianStateSpace", "\\operatorname{LinearGuassianStateSpace}")

    def update(self, node: Node):
        return {node.inputs[-1]: node.outputs[0]}


class _LinearGaussianStateSpace(Continuous):
    def __new__(
        cls,
        name,
        a0,
        P0,
        c,
        d,
        T,
        Z,
        R,
        H,
        Q,
        steps=None,
        append_x0=True,
        method="svd",
        **kwargs,
    ):
        # Ignore dims in support shape because they are just passed along to the "observed" and "latent" distributions
        # created by LinearGaussianStateSpace. This "combined" distribution shouldn't ever be directly used.
        steps = get_support_shape_1d(
            support_shape=steps,
            shape=None,
            dims=None,
            observed=kwargs.get("observed", None),
            support_shape_offset=0,
        )

        return super().__new__(
            cls,
            name,
            a0,
            P0,
            c,
            d,
            T,
            Z,
            R,
            H,
            Q,
            steps=steps,
            append_x0=append_x0,
            method=method,
            **kwargs,
        )

    @classmethod
    def dist(
        cls,
        a0,
        P0,
        c,
        d,
        T,
        Z,
        R,
        H,
        Q,
        steps=None,
        append_x0=True,
        method="svd",
        **kwargs,
    ):
        steps = get_support_shape_1d(
            support_shape=steps, shape=kwargs.get("shape", None), support_shape_offset=0
        )

        if steps is None:
            raise ValueError("Must specify steps or shape parameter")

        steps = pt.as_tensor_variable(intX(steps), ndim=0)

        return super().dist(
            [a0, P0, c, d, T, Z, R, H, Q, steps],
            append_x0=append_x0,
            method=method,
            **kwargs,
        )

    @classmethod
    def rv_op(
        cls,
        a0,
        P0,
        c,
        d,
        T,
        Z,
        R,
        H,
        Q,
        steps,
        size=None,
        rng=None,
        append_x0=True,
        method="svd",
    ):
        varying = is_time_varying(c, d, T, Z, R, H, Q)
        sequence_names = [name for name, flag in zip(PARAM_NAMES, varying, strict=True) if flag]

        a0_, P0_ = a0.type(), P0.type()
        dummies, declared = _matrix_dummies((c, d, T, Z, R, H, Q), varying)

        rng = normalize_rng_param(rng)

        alpha, y, ss_rng = _forward_simulate_latent_and_obs(
            a0_,
            P0_,
            *declared,
            steps=steps,
            rng=rng,
            method=method,
            append_x0=append_x0,
        )
        statespace_ = pt.concatenate([alpha, y], axis=-1)
        statespace_ = pt.specify_shape(statespace_, (steps + int(append_x0), None))

        linear_gaussian_ss_op = LinearGaussianStateSpaceRV(
            inputs=[a0_, P0_, *dummies, steps, rng],
            outputs=[ss_rng, statespace_],
            extended_signature=make_signature(sequence_names),
        )

        linear_gaussian_ss = linear_gaussian_ss_op(a0, P0, c, d, T, Z, R, H, Q, steps, rng)
        return linear_gaussian_ss


class LinearGaussianStateSpace(Continuous):
    """
    Linear Gaussian Statespace distribution

    """

    def __new__(
        cls,
        name,
        a0,
        P0,
        c,
        d,
        T,
        Z,
        R,
        H,
        Q,
        *,
        steps,
        k_endog=None,
        append_x0=True,
        method="svd",
        **kwargs,
    ):
        dims = kwargs.pop("dims", None)
        latent_dims = None
        obs_dims = None
        if dims is not None:
            if len(dims) != 3:
                ValueError(
                    "LinearGaussianStateSpace expects 3 dims: time, all_states, and observed_states"
                )
            time_dim, state_dim, obs_dim = dims
            latent_dims = [time_dim, state_dim]
            obs_dims = [time_dim, obs_dim]

        latent_obs_combined = _LinearGaussianStateSpace(
            f"{name}_combined",
            a0,
            P0,
            c,
            d,
            T,
            Z,
            R,
            H,
            Q,
            steps=steps,
            append_x0=append_x0,
            method=method,
            **kwargs,
        )
        latent_obs_combined = pt.specify_shape(latent_obs_combined, (steps + int(append_x0), None))
        if k_endog is None:
            k_endog = cls._get_k_endog(H)
        latent_slice = slice(None, -k_endog)
        obs_slice = slice(-k_endog, None)

        latent_states = latent_obs_combined[..., latent_slice]
        obs_states = latent_obs_combined[..., obs_slice]

        latent_states = pm.Deterministic(f"{name}_latent", latent_states, dims=latent_dims)
        obs_states = pm.Deterministic(f"{name}_observed", obs_states, dims=obs_dims)

        return latent_states, obs_states

    @classmethod
    def dist(cls, a0, P0, c, d, T, Z, R, H, Q, *, steps=None, **kwargs):
        latent_obs_combined = _LinearGaussianStateSpace.dist(
            a0, P0, c, d, T, Z, R, H, Q, steps=steps, **kwargs
        )
        k_states = T.type.shape[0]

        latent_states = latent_obs_combined[..., :k_states]
        obs_states = latent_obs_combined[..., k_states:]

        return latent_states, obs_states

    @classmethod
    def _get_k_states(cls, T):
        k_states = T.type.shape[0]
        if k_states is None:
            raise ValueError(lgss_shape_message)
        return k_states

    @classmethod
    def _get_k_endog(cls, H):
        k_endog = H.type.shape[0]
        if k_endog is None:
            raise ValueError(lgss_shape_message)

        return k_endog


class KalmanFilterRV(SymbolicRandomVariable):
    default_output = 1
    _print_name = ("KalmanFilter", "\\operatorname{KalmanFilter}")
    extended_signature = "(t,s),(t,s,s),(t),[rng]->[rng],(t,s)"

    def update(self, node: Node):
        return {node.inputs[-1]: node.outputs[0]}


class SequenceMvNormal(Continuous):
    @classmethod
    def dist(cls, mus, covs, logp, method="svd", **kwargs):
        mus, covs, logp = map(pt.as_tensor_variable, (mus, covs, logp))
        return super().dist([mus, covs, logp], method=method, **kwargs)

    @classmethod
    def rv_op(cls, mus, covs, logp, method="svd", size=None, rng=None):
        rng = normalize_rng_param(rng)
        logp_ = logp.type()

        mus_, covs_ = mus.type(), covs.type()
        seq_mvn_rng, mvn_seq = multivariate_normal(
            mean=mus_, cov=covs_, rng=rng, method=method, return_next_rng=True
        )

        mvn_seq_op = KalmanFilterRV(
            inputs=[mus_, covs_, logp_, rng], outputs=[seq_mvn_rng, mvn_seq], ndim_supp=2
        )

        mvn_seq = mvn_seq_op(mus, covs, logp, rng)

        return mvn_seq


@_logprob.register(KalmanFilterRV)
def sequence_mvnormal_logp(op, values, mus, covs, logp, rng, **kwargs):
    return check_parameters(
        logp,
        pt.eq(values[0].shape[0], mus.shape[0]),
        pt.eq(covs.shape[0], mus.shape[0]),
        msg="Observed data and parameters must have the same number of timesteps (dimension 0)",
    )


def _simulation_smoother_signature(sequence_names):
    """Extended gufunc signature for :class:`SimulationSmootherRV`.

    Adds a leading ``a_smooth`` input to the state-space core shapes and outputs the
    sampled latent trajectory.
    """
    return _build_signature(
        {"a_smooth": (TIME, STATES), **STATESPACE_CORE_SHAPES},
        sequence_names,
        output_shape=(TIME, STATES),
    )


class SimulationSmootherRV(SymbolicRandomVariable):
    default_output = 1
    _print_name = ("SimulationSmoother", "\\operatorname{SimSmooth}")

    def update(self, node: Node):
        return {node.inputs[-1]: node.outputs[0]}


class SimulationSmoother(Continuous):
    r"""Durbin-Koopman simulation smoother for a linear Gaussian state-space model.

    Draws a joint sample of the full latent trajectory :math:`\alpha_{1:T}` from
    the smoothing posterior :math:`p(\alpha_{1:T} | y_{1:T})` using the algorithm
    of Durbin and Koopman (2002) [1]_:

    1. Forward-simulate :math:`(\alpha^+, y^+)` from the prior at the current
       parameters.
    2. Filter and smooth :math:`y^+` to obtain :math:`\hat\alpha^+`.
    3. Return :math:`\alpha^{\text{sample}} = \alpha^+ - \hat\alpha^+ + \hat\alpha`,
       where :math:`\hat\alpha` is the smoothed mean of the real data.

    Draws have marginal mean ``a_smooth`` and the full joint posterior covariance,
    including cross-time correlations. Sampling each step's marginal independently with
    :class:`SequenceMvNormal` reproduces the former but not the latter.

    Parameters
    ----------
    a_smooth : TensorVariable
        Real-data smoothed state mean, shape ``(T, k_states)``.
    x0, P0, c, d, T, Z, R, H, Q : TensorVariable
        State-space matrices defining the model.
    kalman_filter : BaseFilter
        Filter object exposing ``build_graph``, called once while building the sampling
        graph. A Python-side graph builder, not a random-variable input.
    kalman_smoother : KalmanSmoother
        Smoother object exposing ``build_graph``, used the same way as ``kalman_filter``.
    method : str, optional
        Multivariate-normal sampling method. Default ``"svd"``.

    References
    ----------
    .. [1] Durbin, J. and Koopman, S. J. (2002). A simple and efficient
       simulation smoother for state space time series analysis. Biometrika 89,
       603-616.
    """

    rv_type = SimulationSmootherRV

    @classmethod
    def dist(
        cls,
        a_smooth,
        x0,
        P0,
        c,
        d,
        T,
        Z,
        R,
        H,
        Q,
        *,
        kalman_filter,
        kalman_smoother,
        method="svd",
        **kwargs,
    ):
        return super().dist(
            [a_smooth, x0, P0, c, d, T, Z, R, H, Q],
            kalman_filter=kalman_filter,
            kalman_smoother=kalman_smoother,
            method=method,
            **kwargs,
        )

    @classmethod
    def rv_op(
        cls,
        a_smooth,
        x0,
        P0,
        c,
        d,
        T,
        Z,
        R,
        H,
        Q,
        *,
        kalman_filter,
        kalman_smoother,
        method="svd",
        size=None,
        rng=None,
    ):
        varying = is_time_varying(c, d, T, Z, R, H, Q)
        sequence_names = [name for name, flag in zip(PARAM_NAMES, varying, strict=True) if flag]

        a_smooth_, x0_, P0_ = (x.type() for x in (a_smooth, x0, P0))
        a_smooth_.name = "a_smooth"
        dummies, (c_, d_, T_, Z_, R_, H_, Q_) = _matrix_dummies((c, d, T, Z, R, H, Q), varying)

        rng = normalize_rng_param(rng)

        # Prefer the static type-shape so the inner scan sequence length is a
        # Python int (JAX requires static lengths for ``lax.scan``); fall back to
        # the symbolic shape only if the model didn't pin it.
        T_static = a_smooth_.type.shape[0]
        steps = T_static if T_static is not None else a_smooth_.shape[0]

        # 1. Forward sim of (alpha_plus, y_plus). The Kalman filter uses the
        # Durbin-Koopman convention where (a0, P0) is the prediction for alpha_1
        # (not the distribution of alpha_0). To produce alpha_1..alpha_T we sample
        # init = alpha_1 ~ N(a0, P0), then run T-1 transition steps and prepend
        # the init.
        alpha_plus, y_plus, mid_rng = _forward_simulate_latent_and_obs(
            x0_,
            P0_,
            c_,
            d_,
            T_,
            Z_,
            R_,
            H_,
            Q_,
            steps=steps - 1,
            rng=rng,
            method=method,
            append_x0=True,
        )

        if T_static is not None:
            y_plus = pt.specify_shape(y_plus, (T_static, *y_plus.type.shape[1:]))
            alpha_plus = pt.specify_shape(alpha_plus, (T_static, *alpha_plus.type.shape[1:]))

        # 2. Filter + smooth y_plus under the same theta.
        a_filt_plus, _, _, P_filt_plus, *_ = kalman_filter.build_graph(
            y_plus,
            x0_,
            P0_,
            c_,
            d_,
            T_,
            Z_,
            R_,
            H_,
            Q_,
        )

        a_smooth_plus, _ = kalman_smoother.build_graph(T_, R_, Q_, a_filt_plus, P_filt_plus)

        if T_static is not None:
            a_smooth_plus = pt.specify_shape(
                a_smooth_plus, (T_static, *a_smooth_plus.type.shape[1:])
            )

        # 3. DK identity. The c-term and d-term cancel because alpha_plus and
        # a_smooth_plus are produced under the same parameters.
        alpha_sample = alpha_plus - a_smooth_plus + a_smooth_

        # ``inline=True`` splices the inner scans into the parent fgraph at compile
        # time, so shape inference reaches through them. The JAX backend needs the
        # resulting static ``n_steps`` to dispatch its scan.
        op = SimulationSmootherRV(
            inputs=[a_smooth_, x0_, P0_, *dummies, rng],
            outputs=[mid_rng, alpha_sample],
            extended_signature=_simulation_smoother_signature(sequence_names),
            inline=True,
        )

        return op(a_smooth, x0, P0, c, d, T, Z, R, H, Q, rng)


@_logprob.register(SimulationSmootherRV)
def simulation_smoother_logp(op, values, *inputs, **kwargs):
    # The simulation smoother is only ever sampled (during posterior predictive),
    # never scored. Return a zero matching the output's shape so PyMC's logp
    # introspection succeeds.
    return pt.zeros_like(values[0])
