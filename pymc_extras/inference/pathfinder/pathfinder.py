#   Copyright 2022 The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import collections
import logging
import time
import warnings

from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field, replace
from enum import Enum, auto
from typing import Any, Literal, Self, TypeAlias

import arviz as az
import filelock
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from numpy.typing import NDArray
from packaging import version
from pymc import Model
from pymc.backends.arviz import coords_and_dims_for_inferencedata
from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.initial_point import make_initial_point_fn
from pymc.model import modelcontext
from pymc.model.core import Point
from pymc.progress_bar import CustomProgress, default_progress_theme
from pymc.pytensorf import compile
from pymc.util import (
    RandomSeed,
    _get_seeds_per_chain,
    get_default_varnames,
)
from pytensor.compile.function.types import Function
from pytensor.compile.mode import FAST_COMPILE, Mode
from pytensor.graph import clone_replace, vectorize_graph
from pytensor.tensor import TensorVariable
from rich.console import Console, Group
from rich.padding import Padding
from rich.progress import TextColumn, TimeElapsedColumn
from rich.table import Column, Table
from rich.text import Text
from threadpoolctl import threadpool_limits

from pymc_extras.inference.laplace_approx.idata import add_data_to_inference_data
from pymc_extras.inference.pathfinder.importance_sampling import (
    importance_sampling as _importance_sampling,
)
from pymc_extras.inference.pathfinder.lbfgs import (
    LBFGS,
    LBFGSException,
    LBFGSInitFailed,
    LBFGSStatus,
    _CachedValueGrad,
    _check_lbfgs_curvature_condition,
)

logger = logging.getLogger(__name__)

REGULARISATION_TERM = 1e-8
DEFAULT_LINKER = "cvm_nogc"

SinglePathfinderFn: TypeAlias = Callable[[int], "PathfinderResult"]


def get_jaxified_logp_of_ravel_inputs(model: Model, jacobian: bool = True) -> Callable:
    """
    Get a JAX function that computes the log-probability of a PyMC model with ravelled inputs.

    Parameters
    ----------
    model : Model
        PyMC model to compute log-probability and gradient.
    jacobian : bool, optional
        Whether to include the Jacobian in the log-probability computation, by default True. Setting to False (not recommended) may result in very high values for pareto k.

    Returns
    -------
    Function
        A JAX function that computes the log-probability of a PyMC model with ravelled inputs.
    """

    from pymc.sampling.jax import get_jaxified_graph

    # TODO: JAX: test if we should get jaxified graph of dlogp as well
    new_logprob, new_input = pm.pytensorf.join_nonshared_inputs(
        model.initial_point(), (model.logp(jacobian=jacobian),), model.value_vars, ()
    )

    logp_func_list = get_jaxified_graph([new_input], new_logprob)

    def logp_func(x):
        return logp_func_list(x)[0]

    return logp_func


def get_logp_dlogp_of_ravel_inputs(
    model: Model, jacobian: bool = True, **compile_kwargs
) -> Function:
    """
    Get the log-probability and its gradient for a PyMC model with ravelled inputs.

    Parameters
    ----------
    model : Model
        PyMC model to compute log-probability and gradient.
    jacobian : bool, optional
        Whether to include the Jacobian in the log-probability computation, by default True. Setting to False (not recommended) may result in very high values for pareto k.
    **compile_kwargs : dict
        Additional keyword arguments to pass to the compile function.

    Returns
    -------
    Function
        A compiled PyTensor function that computes the log-probability and its gradient given ravelled inputs.
    """

    (logP, dlogP), inputs = pm.pytensorf.join_nonshared_inputs(
        model.initial_point(),
        [model.logp(jacobian=jacobian), model.dlogp(jacobian=jacobian)],
        model.value_vars,
    )
    logp_dlogp_fn = compile([inputs], (logP, dlogP), **compile_kwargs)
    logp_dlogp_fn.trust_input = True

    return logp_dlogp_fn


def get_batched_logp_of_ravel_inputs(
    model: Model, jacobian: bool = True, **compile_kwargs
) -> Function:
    """Get a batched logP function: (B, N) -> (B,) evaluated in one compiled call.

    Parameters
    ----------
    model : Model
        PyMC model.
    jacobian : bool, optional
        Whether to include the Jacobian, by default True.
    **compile_kwargs : dict
        Additional keyword arguments to pass to compile.

    Returns
    -------
    Function
        Compiled function taking a (B, N) array and returning (B,) log-probabilities.
    """
    (logP,), single_input = pm.pytensorf.join_nonshared_inputs(
        model.initial_point(),
        [model.logp(jacobian=jacobian)],
        model.value_vars,
    )
    batch_input = pt.matrix("batch_input", dtype="float64")  # (B, N)
    # pytensor.map loops over rows of batch_input one at a time; this avoids
    # broadcasting large intermediate tensors (faster than vectorize_graph for
    # models with many parameters) and handles ops like AdvancedSetSubtensor
    # that vectorize_graph cannot process.
    batched_logP, _ = pytensor.map(
        fn=lambda x_i: clone_replace([logP], replace={single_input: x_i})[0],
        sequences=[batch_input],
    )
    batched_fn = compile([batch_input], batched_logP, **compile_kwargs)
    batched_fn.trust_input = True
    return batched_fn


def convert_flat_trace_to_idata(
    samples: NDArray,
    include_transformed: bool = False,
    postprocessing_backend: Literal["cpu", "gpu"] = "cpu",
    inference_backend: Literal["pymc", "blackjax"] = "pymc",
    model: Model | None = None,
    importance_sampling: Literal["psis", "psir", "identity"] | None = "psis",
) -> az.InferenceData:
    """convert flattened samples to arviz InferenceData format.

    Parameters
    ----------
    samples : NDArray
        flattened samples
    include_transformed : bool
        whether to include transformed variables
    postprocessing_backend : str
        backend for postprocessing transformations, either "cpu" or "gpu"
    inference_backend : str
        backend for inference, either "pymc" or "blackjax"
    model : Model | None
        pymc model for variable transformations
    importance_sampling : str
        importance sampling method used, affects input samples shape

    Returns
    -------
    InferenceData
        arviz inference data object
    """

    if importance_sampling is None:
        # samples.ndim == 3 in this case, otherwise ndim == 2
        num_paths, num_pdraws, N = samples.shape
        samples = samples.reshape(-1, N)

    model = modelcontext(model)
    ip = model.initial_point()
    ip_point_map_info = DictToArrayBijection.map(ip).point_map_info
    trace = collections.defaultdict(list)
    for sample in samples:
        raveld_vars = RaveledVars(sample, ip_point_map_info)
        point = DictToArrayBijection.rmap(raveld_vars, ip)
        for p, v in point.items():
            trace[p].append(v.tolist())

    trace = {k: np.asarray(v)[None, ...] for k, v in trace.items()}

    var_names = model.unobserved_value_vars
    vars_to_sample = list(get_default_varnames(var_names, include_transformed=include_transformed))
    logger.info("Transforming variables...")

    if inference_backend == "pymc":
        # vectorize_graph batches over trace dims; output size matches input, no extra intermediates.
        new_shapes = [v.ndim * (None,) for v in trace.values()]
        replace = {
            var: pt.tensor(dtype="float64", shape=new_shapes[i])
            for i, var in enumerate(model.value_vars)
        }

        outputs = vectorize_graph(vars_to_sample, replace=replace)

        fn = pytensor.function(
            inputs=[*list(replace.values())],
            outputs=outputs,
            mode=FAST_COMPILE,
            on_unused_input="ignore",
        )
        fn.trust_input = True
        result = fn(*list(trace.values()))
        if not isinstance(result, list):
            result = [result]

        if importance_sampling is None:
            result = [res.reshape(num_paths, num_pdraws, *res.shape[2:]) for res in result]

    elif inference_backend == "blackjax":
        import jax

        from pymc.sampling.jax import get_jaxified_graph

        jax_fn = get_jaxified_graph(inputs=model.value_vars, outputs=vars_to_sample)
        result = jax.vmap(jax.vmap(jax_fn))(
            *jax.device_put(list(trace.values()), jax.devices(postprocessing_backend)[0])
        )

    trace = {v.name: r for v, r in zip(vars_to_sample, result)}
    coords, dims = coords_and_dims_for_inferencedata(model)
    idata = az.from_dict(trace, dims=dims, coords=coords)

    return idata


def alpha_recover(
    x: TensorVariable, g: TensorVariable
) -> tuple[TensorVariable, TensorVariable, TensorVariable]:
    """compute the diagonal elements of the inverse Hessian at each iterations of L-BFGS and filter updates.

    Parameters
    ----------
    x : TensorVariable
        position array, shape (L+1, N)
    g : TensorVariable
        gradient array, shape (L+1, N)

    Returns
    -------
    alpha : TensorVariable
        diagonal elements of the inverse Hessian at each iteration of L-BFGS, shape (L, N)
    s : TensorVariable
        position differences, shape (L, N)
    z : TensorVariable
        gradient differences, shape (L, N)

    Notes
    -----
    shapes: L=batch_size, N=num_params
    """

    def compute_alpha_l(s_l, z_l, alpha_lm1) -> TensorVariable:
        # alpha_lm1: (N,)
        # s_l: (N,)
        # z_l: (N,)
        a = pt.sum(alpha_lm1 * z_l**2)
        b = pt.sum(z_l * s_l)
        c = pt.sum(s_l**2 / alpha_lm1)
        inv_alpha_l = (
            a / (b * alpha_lm1)
            + z_l ** 2 / b
            - (a * s_l ** 2) / (b * c * alpha_lm1**2)
        )  # fmt:off
        return 1.0 / inv_alpha_l

    Lp1, N = x.shape
    s = pt.diff(x, axis=0)
    z = pt.diff(g, axis=0)
    alpha_l_init = pt.ones(N)

    alpha = pytensor.scan(
        fn=compute_alpha_l,
        outputs_info=alpha_l_init,
        sequences=[s, z],
        n_steps=Lp1 - 1,
        allow_gc=False,
        return_updates=False,
    )

    # alpha: (L, N)
    return alpha, s, z


def alpha_step_numpy(alpha_prev: NDArray, s: NDArray, z: NDArray) -> NDArray:
    """Pure-numpy single-step alpha update. Stays in sync with compute_alpha_l in alpha_recover.

    Parameters
    ----------
    alpha_prev : NDArray
        previous alpha vector, shape (N,)
    s : NDArray
        position diff x[l] - x[l-1], shape (N,)
    z : NDArray
        gradient diff g[l] - g[l-1], shape (N,)

    Returns
    -------
    NDArray
        updated alpha, shape (N,)
    """
    a = np.sum(alpha_prev * z**2)
    b = np.sum(z * s)
    c = np.sum(s**2 / alpha_prev)
    inv_alpha = (
        a / (b * alpha_prev)
        + z**2 / b
        - (a * s**2) / (b * c * alpha_prev**2)
    )  # fmt: off
    return 1.0 / inv_alpha


def _bfgs_sample_pt(
    x: TensorVariable,
    g: TensorVariable,
    alpha: TensorVariable,
    S: TensorVariable,
    Z: TensorVariable,
    u: TensorVariable,
    J: int,
    N: int,
) -> tuple[TensorVariable, TensorVariable]:
    """Symbolic L-BFGS inverse-Hessian sample.

    The dense vs sparse path is selected at graph-construction time from
    the compile-time constants N and J, so only one branch is ever compiled.

    Parameters
    ----------
    x : (N,) position
    g : (N,) gradient
    alpha : (N,) diagonal scaling
    S : (N, J) position-diff ring buffer
    Z : (N, J) gradient-diff ring buffer
    u : (M, N) standard-normal draws — M is a dynamic runtime dimension
    J : int, L-BFGS history size (compile-time constant)
    N : int, number of parameters (compile-time constant)

    Returns
    -------
    phi : (M, N) samples
    logQ : (M,) log-density under the approximation
    """
    J2 = 2 * J

    # Inverse Hessian factors
    E = pt.triu(S.T @ Z) + pt.eye(J) * REGULARISATION_TERM  # (J, J)
    eta = pt.diag(E)  # (J,)
    AZ = alpha[:, None] * Z  # (N, J)
    beta = pt.concatenate([AZ, S], axis=1)  # (N, 2J)
    E_inv = pt.linalg.solve_triangular(E, pt.eye(J), lower=False)  # (J, J)
    block_dd = E_inv.T @ (pt.diag(eta) + Z.T @ AZ) @ E_inv  # (J, J)
    top = pt.concatenate([pt.zeros((J, J)), -E_inv], axis=1)  # (J, 2J)
    bot = pt.concatenate([-E_inv.T, block_dd], axis=1)  # (J, 2J)
    gamma = pt.concatenate([top, bot], axis=0)  # (2J, 2J)

    inv_sqrt_alpha = 1.0 / pt.sqrt(alpha)
    sqrt_alpha = pt.sqrt(alpha)

    if J2 >= N:
        # Dense path: form H_inv explicitly (small-N regime, O(N²) is fine)
        isa_d = pt.diag(inv_sqrt_alpha)  # (N, N)
        sa_d = pt.diag(sqrt_alpha)  # (N, N)
        IdN = pt.eye(N) * (1.0 + REGULARISATION_TERM)
        H_inv = sa_d @ (IdN + isa_d @ beta @ gamma @ beta.T @ isa_d) @ sa_d
        Lchol = pt.linalg.cholesky(H_inv, lower=False)  # (N, N)
        logdet = 2.0 * pt.sum(pt.log(pt.abs(pt.diag(Lchol))))
        mu = x - H_inv @ g
        phi = (mu[:, None] + Lchol @ u.T).T  # (M, N)
    else:
        # Sparse path: economy QR avoids O(N²) matrices (large-N regime)
        Q, R = pt.linalg.qr(beta * inv_sqrt_alpha[:, None], mode="reduced")  # Q:(N,2J), R:(2J,2J)
        I2J = pt.eye(J2) * (1.0 + REGULARISATION_TERM)
        Lchol = pt.linalg.cholesky(I2J + R @ gamma @ R.T, lower=False)  # (2J, 2J)
        logdet = 2.0 * pt.sum(pt.log(pt.abs(pt.diag(Lchol)))) + pt.sum(pt.log(alpha))
        btg = beta.T @ g[:, None]  # (2J, 1)
        mu = x - ((alpha * g)[:, None] + beta @ (gamma @ btg))[:, 0]  # (N,)
        QtU = Q.T @ u.T  # (2J, M)
        phi = (mu[:, None] + sqrt_alpha[:, None] * (Q @ ((Lchol - pt.eye(J2)) @ QtU) + u.T)).T

    logQ = -0.5 * (logdet + pt.sum(u * u, axis=-1) + N * np.log(2.0 * np.pi))
    return phi, logQ


def make_pathfinder_sample_fn(
    model: Model,
    N: int,
    J: int,
    jacobian: bool,
    compile_kwargs: dict,
) -> tuple[Function, Any, Any]:
    """Compile a single PyTensor function covering bfgs sample + batched logP evaluation.

    The ring buffers S and Z are pytensor shared variables — the caller holds the
    backing numpy arrays and writes to them in-place; pytensor reads from that same
    memory without any per-step copy (borrow semantics via set_value).

    The number of draws M is a dynamic runtime dimension — the same compiled
    function handles both ELBO estimation (small M) and final sampling (large M).

    Parameters
    ----------
    model : Model
    N : int, number of unconstrained parameters
    J : int, L-BFGS history size (maxcor)
    jacobian : bool
    compile_kwargs : dict

    Returns
    -------
    fn : Function
        Compiled: (x, g, alpha, u) → (phi, logQ, logP)
        where u is (M, N) and M is a dynamic dimension.
    s_win_shared : SharedVariable  shape (N, J)
    z_win_shared : SharedVariable  shape (N, J)
    """
    (logP_single,), single_input = pm.pytensorf.join_nonshared_inputs(
        model.initial_point(),
        [model.logp(jacobian=jacobian)],
        model.value_vars,
    )

    x_sym = pt.vector("x", dtype="float64")
    g_sym = pt.vector("g", dtype="float64")
    alpha_sym = pt.vector("alpha", dtype="float64")
    u_sym = pt.matrix("u", dtype="float64")  # (M, N) — M is dynamic

    # Shared ring-buffer state: the caller owns backing numpy arrays and writes
    # them in-place; pytensor reads from the same memory with no per-step copy.
    s_win_shared = pytensor.shared(np.zeros((N, J), dtype="float64"), name="s_win")
    z_win_shared = pytensor.shared(np.zeros((N, J), dtype="float64"), name="z_win")

    phi_sym, logQ_sym = _bfgs_sample_pt(
        x_sym, g_sym, alpha_sym, s_win_shared, z_win_shared, u_sym, J, N
    )

    batched_logP_sym = pytensor.map(
        fn=lambda x_i: clone_replace([logP_single], replace={single_input: x_i})[0],
        sequences=[phi_sym],
        return_updates=False,
    )

    fn = pytensor.function(
        [
            pytensor.In(x_sym, borrow=True),
            pytensor.In(g_sym, borrow=True),
            pytensor.In(alpha_sym, borrow=True),
            pytensor.In(u_sym, borrow=True),
        ],
        [phi_sym, logQ_sym, batched_logP_sym],
        **compile_kwargs,
    )
    fn.trust_input = True
    return fn, s_win_shared, z_win_shared


def make_elbo_from_state_fn(
    model: Model,
    N: int,
    J: int,
    jacobian: bool,
    compile_kwargs: dict,
) -> Function:
    """Compiled (x, g, alpha, S, Z, u) → (phi, logQ, logP) for fixture/tests.

    S, Z are explicit inputs (not shared), for recomputing ELBO from saved state.
    """
    (logP_single,), single_input = pm.pytensorf.join_nonshared_inputs(
        model.initial_point(),
        [model.logp(jacobian=jacobian)],
        model.value_vars,
    )
    x_sym = pt.vector("x", dtype="float64")
    g_sym = pt.vector("g", dtype="float64")
    alpha_sym = pt.vector("alpha", dtype="float64")
    S_sym = pt.matrix("S", dtype="float64")
    Z_sym = pt.matrix("Z", dtype="float64")
    u_sym = pt.matrix("u", dtype="float64")
    phi_sym, logQ_sym = _bfgs_sample_pt(x_sym, g_sym, alpha_sym, S_sym, Z_sym, u_sym, J, N)
    batched_logP_sym = pytensor.map(
        fn=lambda x_i: clone_replace([logP_single], replace={single_input: x_i})[0],
        sequences=[phi_sym],
        return_updates=False,
    )
    fn = pytensor.function(
        [
            pytensor.In(x_sym, borrow=True),
            pytensor.In(g_sym, borrow=True),
            pytensor.In(alpha_sym, borrow=True),
            pytensor.In(S_sym, borrow=True),
            pytensor.In(Z_sym, borrow=True),
            pytensor.In(u_sym, borrow=True),
        ],
        [phi_sym, logQ_sym, batched_logP_sym],
        **compile_kwargs,
    )
    fn.trust_input = True
    return fn


class PathStatus(Enum):
    """
    Statuses of a single-path pathfinder.
    """

    SUCCESS = auto()
    ELBO_ARGMAX_AT_ZERO = auto()
    # Statuses that lead to Exceptions:
    INVALID_LOGP = auto()
    INVALID_LOGQ = auto()
    LBFGS_FAILED = auto()
    PATH_FAILED = auto()


FAILED_PATH_STATUS = [
    PathStatus.INVALID_LOGP,
    PathStatus.INVALID_LOGQ,
    PathStatus.LBFGS_FAILED,
    PathStatus.PATH_FAILED,
]


class PathException(Exception):
    """
    raises a PathException if the path failed.
    """

    DEFAULT_MESSAGE = "Path failed."

    def __init__(self, message=None, status: PathStatus = PathStatus.PATH_FAILED) -> None:
        super().__init__(message or self.DEFAULT_MESSAGE)
        self.status = status


class PathInvalidLogP(PathException):
    """
    raises a PathException if all the logP values in a path are not finite.
    """

    DEFAULT_MESSAGE = "Path failed because all the logP values in a path are not finite."

    def __init__(self, message=None) -> None:
        super().__init__(message or self.DEFAULT_MESSAGE, PathStatus.INVALID_LOGP)


class PathInvalidLogQ(PathException):
    """
    raises a PathException if all the logQ values in a path are not finite.
    """

    DEFAULT_MESSAGE = "Path failed because all the logQ values in a path are not finite."

    def __init__(self, message=None) -> None:
        super().__init__(message or self.DEFAULT_MESSAGE, PathStatus.INVALID_LOGQ)


class LBFGSStreamingCallback:
    """Streaming LBFGS callback: computes ELBO at each accepted step, O(J*N + M*N) peak memory.

    Instead of collecting the full (L+1, N) history, it processes each accepted step
    immediately and tracks only the best state seen so far.

    Parameters
    ----------
    value_grad_fn : Callable
        Single-entry cached value/gradient function (wrap with _CachedValueGrad).
    x0 : NDArray
        Initial position, shape (N,).
    sample_logp_fn : Callable
        Compiled PyTensor function (x, g, alpha, u) → (phi, logQ, logP).
        Built by make_pathfinder_sample_fn.
    s_win_shared : SharedVariable
        PyTensor shared variable for the position-diff ring buffer, shape (N, J).
    z_win_shared : SharedVariable
        PyTensor shared variable for the gradient-diff ring buffer, shape (N, J).
    num_elbo_draws : int
        Number of draws per step for ELBO estimation.
    rng : np.random.Generator
        Random number generator for draw generation.
    J : int
        L-BFGS history size (maxcor).
    epsilon : float
        Tolerance for the LBFGS update condition.
    progress_callback : Callable | None
        Optional progress reporting.
    on_step_callback : Callable | None
        If set, called after each accepted step with (x, g, alpha, s_win, z_win, elbo).
        Used by fixture generation to record per-step state.
    """

    def __init__(
        self,
        value_grad_fn: Callable,
        x0: NDArray,
        sample_logp_fn: Callable,
        s_win_shared: Any,
        z_win_shared: Any,
        num_elbo_draws: int,
        rng: np.random.Generator,
        J: int,
        epsilon: float,
        progress_callback: Callable | None = None,
        on_step_callback: Callable | None = None,
    ) -> None:
        self.value_grad_fn = value_grad_fn
        self.sample_logp_fn = sample_logp_fn
        self.num_elbo_draws = num_elbo_draws
        self._rng = rng
        self.J = J
        self.epsilon = epsilon
        self.progress_callback = progress_callback
        self.on_step_callback = on_step_callback

        N = x0.shape[0]
        self._N = N
        _, g0 = value_grad_fn(x0)

        self.x_prev: NDArray = x0.copy()
        self.g_prev: NDArray = np.array(g0, dtype=np.float64)
        self.alpha_prev: NDArray = np.ones(N, dtype=np.float64)

        # Ring buffer: backed by numpy, shared with pytensor via borrow=True.
        # In-place numpy writes (s_win[:, idx] = s) are visible to the compiled
        # function with no per-step copy.
        self.s_win: NDArray = np.zeros((N, J), dtype=np.float64)
        self.z_win: NDArray = np.zeros((N, J), dtype=np.float64)
        s_win_shared.set_value(self.s_win, borrow=True)
        z_win_shared.set_value(self.z_win, borrow=True)
        self.win_idx: int = -1
        self.best_elbo: float = -np.inf
        self.best_state: dict = {}
        self.best_step_idx: int = 0
        self.step_count: int = 0
        self.any_valid: bool = False
        self._start_time: float = time.time()

    def __call__(self, x: NDArray) -> None:
        value, g = self.value_grad_fn(x)

        s = x - self.x_prev
        z = g - self.g_prev

        if not (np.all(np.isfinite(g)) and np.isfinite(value)):
            return
        if not _check_lbfgs_curvature_condition(s, z, self.epsilon):
            return

        alpha = alpha_step_numpy(self.alpha_prev, s, z)

        # Ring-buffer update (numpy, O(N))
        self.win_idx = (self.win_idx + 1) % self.J
        self.s_win[:, self.win_idx] = s
        self.z_win[:, self.win_idx] = z

        # Sample + logP in a single compiled call.
        # s_win/z_win are shared variables already pointing to self.s_win/z_win —
        # the ring-buffer writes above are visible to pytensor with no copy.
        u = self._rng.standard_normal((self.num_elbo_draws, self._N))
        try:
            _, logQ, logP = self.sample_logp_fn(x, g, alpha, u)
            logP = np.asarray(logP)
            finite = np.isfinite(logP)
            if not np.any(finite):
                elbo = -np.inf
            else:
                logP_safe = np.where(finite, logP, -np.inf)
                elbo = float(np.mean(logP_safe - logQ))
                if not np.isfinite(elbo):
                    elbo = -np.inf
        except Exception:
            elbo = -np.inf

        if np.isfinite(elbo):
            self.any_valid = True

        if self.on_step_callback is not None:
            self.on_step_callback(x, g, alpha, self.s_win.copy(), self.z_win.copy(), elbo)

        if elbo > self.best_elbo:
            self.best_elbo = elbo
            self.best_state = {
                "alpha": alpha.copy(),
                "s_win": self.s_win.copy(),
                "z_win": self.z_win.copy(),
                "win_idx": self.win_idx,
                "x": x.copy(),
                "g": g.copy(),
            }
            self.best_step_idx = self.step_count

        self.alpha_prev = alpha
        self.x_prev = x.copy()
        self.g_prev = g.copy()
        self.step_count += 1

        if self.progress_callback is not None:
            best_elbo = self.best_elbo if np.isfinite(self.best_elbo) else None
            current_elbo = elbo if np.isfinite(elbo) else None
            elapsed = time.time() - self._start_time
            steps_per_sec = self.step_count / elapsed if elapsed > 0 else None
            step_size = float(np.linalg.norm(s))
            self.progress_callback(
                {
                    "lbfgs_steps": self.step_count,
                    "best_elbo": best_elbo,
                    "current_elbo": current_elbo,
                    "step_size": step_size,
                    "steps_per_sec": steps_per_sec,
                }
            )


def make_single_pathfinder_fn(
    model,
    num_draws: int,
    maxcor: int | None,
    maxiter: int,
    ftol: float,
    gtol: float,
    maxls: int,
    num_elbo_draws: int,
    jitter: float,
    epsilon: float,
    max_init_retries: int = 10,
    pathfinder_kwargs: dict[str, Any] = {},
    compile_kwargs: dict[str, Any] = {},
) -> SinglePathfinderFn:
    """
    returns a seedable single-path pathfinder function, where it executes a compiled function that performs the local approximation and sampling part of the Pathfinder algorithm.

    Parameters
    ----------
    model : pymc.Model
        The PyMC model to fit the Pathfinder algorithm to.
    num_draws : int
        Number of samples to draw from the single-path approximation.
    maxcor : int | None
        Maximum number of iterations for the L-BFGS optimisation.
    maxiter : int
        Maximum number of iterations for the L-BFGS optimisation.
    ftol : float
        Tolerance for the decrease in the objective function.
    gtol : float
        Tolerance for the norm of the gradient.
    maxls : int
        Maximum number of line search steps for the L-BFGS algorithm.
    num_elbo_draws : int
        Number of draws for the Evidence Lower Bound (ELBO) estimation.
    jitter : float
        Amount of jitter to apply to initial points. Note that Pathfinder may be highly sensitive to the jitter value. It is recommended to increase num_paths when increasing the jitter value.
    epsilon : float
        value used to filter out large changes in the direction of the update gradient at each iteration l in L. Iteration l is only accepted if s·z >= epsilon * ||z||² for each l in L. Matches Zhang et al. (2022) Algorithm 3 with default epsilon=1e-12.
    max_init_retries : int, optional
        Maximum number of re-jitter retries when LBFGSInitFailed is raised (i.e. the initial
        point yields non-finite value/gradient or the first step is rejected). Each retry uses
        a different jitter seed. Defaults to 10.
    pathfinder_kwargs : dict
        Additional keyword arguments for the Pathfinder algorithm.
    compile_kwargs : dict
        Additional keyword arguments for the PyTensor compiler. If not provided, a
        performant default is used.

    Returns
    -------
    single_pathfinder_fn : Callable
        A seedable single-path pathfinder function that accepts ``(random_seed, progress_callback=None)``.
    """

    compile_kwargs = {"mode": Mode(linker=DEFAULT_LINKER), **compile_kwargs}
    jacobian = pathfinder_kwargs.get("jacobian", True)
    logp_dlogp_kwargs = {"jacobian": jacobian, **compile_kwargs}

    logp_dlogp_func = get_logp_dlogp_of_ravel_inputs(model, **logp_dlogp_kwargs)

    def neg_logp_dlogp_func(x):
        logp, dlogp = logp_dlogp_func(x)
        return -logp, -dlogp

    # initial point
    # TODO: remove make_initial_points function when feature request is implemented: https://github.com/pymc-devs/pymc/issues/7555
    ipfn = make_initial_point_fn(model=model)
    ip = Point(ipfn(None), model=model)
    x_base = DictToArrayBijection.map(ip).data
    N = x_base.shape[0]

    sample_logp_func, s_win_shared, z_win_shared = make_pathfinder_sample_fn(
        model, N, maxcor, jacobian=jacobian, compile_kwargs=compile_kwargs
    )

    def _check_lbfgs_status(status):
        if status in {LBFGSStatus.INIT_FAILED, LBFGSStatus.INIT_FAILED_LOW_UPDATE_PCT}:
            raise LBFGSInitFailed(status)
        elif status == LBFGSStatus.LBFGS_FAILED:
            raise LBFGSException()

    def _make_result(psi, logP_psi, logQ_psi, lbfgs_niter, elbo_argmax, lbfgs_status):
        if np.all(~np.isfinite(logQ_psi)):
            raise PathInvalidLogQ()
        path_status = PathStatus.ELBO_ARGMAX_AT_ZERO if elbo_argmax == 0 else PathStatus.SUCCESS
        return PathfinderResult(
            samples=psi,
            logP=logP_psi,
            logQ=logQ_psi,
            lbfgs_niter=lbfgs_niter,
            elbo_argmax=elbo_argmax,
            lbfgs_status=lbfgs_status,
            path_status=path_status,
        )

    def single_pathfinder_fn(
        random_seed: int, progress_callback: Callable | None = None
    ) -> PathfinderResult:
        if progress_callback is not None:
            progress_callback({"status": "running"})

        # Per-path independent copies of compiled functions for process safety.
        local_logp_dlogp = logp_dlogp_func.copy(share_memory=False)

        def local_neg_logp_dlogp_func(x):
            logp, dlogp = local_logp_dlogp(x)
            return -logp, -dlogp

        local_lbfgs = LBFGS(local_neg_logp_dlogp_func, maxcor, maxiter, ftol, gtol, maxls, epsilon)

        local_sample_logp = sample_logp_func.copy(share_memory=False)

        lbfgs_status = LBFGSStatus.LBFGS_FAILED  # default before LBFGS runs
        try:
            # Derive base seeds once from random_seed. init_seed is an int so we can
            # safely offset it per retry attempt (avoids Generator arithmetic issues).
            _base_init, elbo_seed, final_seed = _get_seeds_per_chain(random_seed, 3)

            for attempt in range(max_init_retries + 1):
                try:
                    init_seed = _base_init + attempt  # different jitter per retry
                    rng = np.random.default_rng(init_seed)
                    jitter_value = rng.uniform(-jitter, jitter, size=x_base.shape)
                    x0 = x_base + jitter_value

                    # Fresh NumPy RNG each attempt → reproducible regardless
                    # of how many ELBO steps the previous attempt took.
                    elbo_rng = np.random.default_rng(elbo_seed + attempt)

                    cached_fn = _CachedValueGrad(local_neg_logp_dlogp_func)
                    streaming_cb = LBFGSStreamingCallback(
                        value_grad_fn=cached_fn,
                        x0=x0,
                        sample_logp_fn=local_sample_logp,
                        s_win_shared=s_win_shared,
                        z_win_shared=z_win_shared,
                        num_elbo_draws=num_elbo_draws,
                        rng=elbo_rng,
                        J=maxcor,
                        epsilon=epsilon,
                        progress_callback=progress_callback,
                    )

                    with threadpool_limits(limits=1):
                        lbfgs_niter, lbfgs_status = local_lbfgs.minimize_streaming(streaming_cb, x0)
                    _check_lbfgs_status(lbfgs_status)

                    if not streaming_cb.any_valid:
                        raise PathInvalidLogP()

                    elbo_argmax = streaming_cb.best_step_idx
                    best_state = streaming_cb.best_state

                    final_rng = np.random.default_rng(final_seed)
                    u_final = final_rng.standard_normal((num_draws, N))
                    # Point shared ring buffers at best-state arrays before final draw.
                    s_win_shared.set_value(best_state["s_win"], borrow=True)
                    z_win_shared.set_value(best_state["z_win"], borrow=True)
                    with threadpool_limits(limits=1):
                        phi_final, logQ_psi_flat, logP_psi_flat = local_sample_logp(
                            best_state["x"],
                            best_state["g"],
                            best_state["alpha"],
                            u_final,
                        )
                    phi_final = np.asarray(phi_final)
                    logQ_psi_flat = np.asarray(logQ_psi_flat)
                    logP_psi_flat = np.asarray(logP_psi_flat)
                    # Add batch dim L=1 to match downstream expectations
                    psi = phi_final[None]  # (1, M, N)
                    logP_psi = logP_psi_flat[None]  # (1, M)
                    logQ_psi = logQ_psi_flat[None]  # (1, M)

                    break  # success — exit retry loop

                except LBFGSInitFailed:
                    if attempt < max_init_retries:
                        logger.debug(
                            "LBFGSInitFailed on attempt %d/%d, retrying with different jitter...",
                            attempt + 1,
                            max_init_retries,
                        )
                        if progress_callback is not None:
                            progress_callback({"status": f"retry {attempt + 1}"})
                    else:
                        if progress_callback is not None:
                            progress_callback({"status": "lbfgs_fail"})
                        return PathfinderResult(
                            lbfgs_status=LBFGSStatus.INIT_FAILED,
                            path_status=PathStatus.LBFGS_FAILED,
                        )

            result = _make_result(psi, logP_psi, logQ_psi, lbfgs_niter, elbo_argmax, lbfgs_status)
            if progress_callback is not None:
                status_str = (
                    "elbo@0" if result.path_status == PathStatus.ELBO_ARGMAX_AT_ZERO else "ok"
                )
                progress_callback({"status": status_str, "lbfgs_steps": int(lbfgs_niter)})
            return result

        except LBFGSException as e:
            if progress_callback is not None:
                progress_callback({"status": "lbfgs_fail"})
            return PathfinderResult(
                lbfgs_status=e.status,
                path_status=PathStatus.LBFGS_FAILED,
            )
        except PathException as e:
            if progress_callback is not None:
                progress_callback({"status": "failed"})
            return PathfinderResult(
                lbfgs_status=lbfgs_status,
                path_status=e.status,
            )

    return single_pathfinder_fn


def _calculate_max_workers(num_paths: int | None = None) -> int:
    """
    calculate the default number of workers to use for concurrent pathfinder runs.
    """
    import os

    total_cpus = os.cpu_count() or 1
    if num_paths is not None:
        return min(num_paths, total_cpus)
    # Legacy process-pool heuristic: 30% of CPUs, minimum 2, rounded to even
    processes = max(2, int(total_cpus * 0.3))
    if processes % 2 != 0:
        processes += 1
    return processes


def _thread(
    fn: SinglePathfinderFn, seed: int, progress_callback: Callable | None = None
) -> "PathfinderResult":
    """
    execute pathfinder runs concurrently using threading.

    No compilation lock is needed here: compiled functions are deepcopied per
    path call inside ``single_pathfinder_fn``, so each thread operates on its
    own independent storage with no shared mutable state.
    """
    rng = np.random.default_rng(seed)
    return fn(rng, progress_callback)


class _QueueCallback:
    """Picklable progress callback that relays updates through a multiprocessing.Queue.

    Worker processes cannot call Rich progress functions directly (they live in the
    main process). This class is picklable and sends ``(idx, info)`` tuples to a
    shared queue; a listener thread in the main process forwards them to the real
    per-path callbacks.
    """

    def __init__(self, queue: Any, idx: int) -> None:
        self.queue = queue
        self.idx = idx

    def __call__(self, info: dict) -> None:
        try:
            self.queue.put_nowait((self.idx, info))
        except Exception:
            pass


def _process(
    fn: SinglePathfinderFn, seed: int, progress_callback: Callable | None = None
) -> "PathfinderResult | bytes":
    """
    execute pathfinder runs concurrently using multiprocessing.
    """
    import cloudpickle

    from pytensor.compile.compilelock import lock_ctx

    in_out_pickled = isinstance(fn, bytes)
    # lock_ctx only guards cache access during unpickling, not computation.
    # Use timeout=-1 (wait indefinitely) so workers don't race to timeout when
    # many paths start simultaneously and each unpickling takes a moment.
    with lock_ctx(timeout=-1):
        actual_fn = cloudpickle.loads(fn) if in_out_pickled else fn

    rng = np.random.default_rng(seed)
    result = actual_fn(rng, progress_callback)
    return cloudpickle.dumps(result) if in_out_pickled else result


def _get_mp_context(mp_ctx: str | None = None) -> str | None:
    """code snippet taken from ParallelSampler in pymc/pymc/sampling/parallel.py"""
    import multiprocessing
    import platform

    if mp_ctx is None or isinstance(mp_ctx, str):
        if mp_ctx is None and platform.system() == "Darwin":
            if platform.processor() == "arm":
                mp_ctx = "fork"
                logger.debug(
                    "mp_ctx is set to 'fork' for MacOS with ARM architecture. "
                    + "This might cause unexpected behavior with JAX, which is inherently multithreaded."
                )
            else:
                mp_ctx = "forkserver"

        mp_ctx = multiprocessing.get_context(mp_ctx)
    return mp_ctx


def _execute_concurrently(
    fn: SinglePathfinderFn,
    seeds: list[int],
    concurrent: Literal["process"] | None,
    max_workers: int | None = None,
    progress_callbacks: list[Callable | None] | None = None,
) -> Iterator["PathfinderResult | bytes"]:
    """
    execute pathfinder runs concurrently.
    """
    if concurrent == "thread":  # pragma: no cover — thread mode is not exposed in the public API
        from concurrent.futures import ThreadPoolExecutor, as_completed
    elif concurrent == "process":
        from concurrent.futures import ProcessPoolExecutor, as_completed

        import cloudpickle
    else:
        raise ValueError(f"Invalid concurrent value: {concurrent}")

    executor_cls = ThreadPoolExecutor if concurrent == "thread" else ProcessPoolExecutor

    concurrent_fn = _thread if concurrent == "thread" else _process

    executor_kwargs = {} if concurrent == "thread" else {"mp_context": _get_mp_context()}

    if max_workers is None:
        max_workers = _calculate_max_workers(num_paths=len(seeds))

    fn = fn if concurrent == "thread" else cloudpickle.dumps(fn)

    if concurrent == "thread":
        callbacks: list[Callable | None] = (
            list(progress_callbacks) if progress_callbacks is not None else [None] * len(seeds)
        )
        if len(callbacks) < len(seeds):
            callbacks.extend([None] * (len(seeds) - len(callbacks)))

        with executor_cls(max_workers=max_workers, **executor_kwargs) as executor:
            futures = [
                executor.submit(concurrent_fn, fn, seed, cb) for seed, cb in zip(seeds, callbacks)
            ]
            for f in as_completed(futures):
                yield f.result()
    else:
        # Process mode: Rich callbacks live in the main process and can't be pickled.
        # Use a Manager Queue (proxy-based, always picklable regardless of start method)
        # to relay (idx, info) messages from workers back to the main process, where a
        # listener thread forwards them to the real per-path callbacks.
        import multiprocessing
        import threading

        mp_manager = multiprocessing.Manager()
        try:
            mp_queue = mp_manager.Queue()
            process_callbacks = [_QueueCallback(mp_queue, i) for i in range(len(seeds))]

            def _listener() -> None:
                while True:
                    item = mp_queue.get()
                    if item is None:  # sentinel
                        break
                    idx, info = item
                    if (
                        progress_callbacks
                        and idx < len(progress_callbacks)
                        and progress_callbacks[idx] is not None
                    ):
                        progress_callbacks[idx](info)

            listener = threading.Thread(target=_listener, daemon=True)
            listener.start()
            try:
                with executor_cls(max_workers=max_workers, **executor_kwargs) as executor:
                    futures = [
                        executor.submit(concurrent_fn, fn, seed, cb)
                        for seed, cb in zip(seeds, process_callbacks)
                    ]
                    for f in as_completed(futures):
                        yield cloudpickle.loads(f.result())
            finally:
                mp_queue.put(None)  # stop listener
                listener.join(timeout=5)
        finally:
            mp_manager.shutdown()


def _execute_serially(
    fn: SinglePathfinderFn,
    seeds: list[int],
    progress_callbacks: list[Callable | None] | None = None,
) -> Iterator["PathfinderResult"]:
    """
    execute pathfinder runs serially.
    """
    callbacks = progress_callbacks or [None] * len(seeds)
    for seed, cb in zip(seeds, callbacks):
        rng = np.random.default_rng(seed)
        yield fn(rng, cb)


def make_generator(
    concurrent: Literal["process"] | None,
    fn: SinglePathfinderFn,
    seeds: list[int],
    max_workers: int | None = None,
    progress_callbacks: list[Callable | None] | None = None,
) -> Iterator["PathfinderResult | bytes"]:
    """
    generator for executing pathfinder runs concurrently or serially.
    """
    if concurrent is not None:
        yield from _execute_concurrently(fn, seeds, concurrent, max_workers, progress_callbacks)
    else:
        yield from _execute_serially(fn, seeds, progress_callbacks)


@dataclass(slots=True, frozen=True)
class PathfinderResult:
    """
    container for storing results from a single pathfinder run.

    Attributes
    ----------
        samples: posterior samples (1, M, N)
        logP: log probability of model (1, M)
        logQ: log probability of approximation (1, M)
        lbfgs_niter: number of lbfgs iterations (1,)
        elbo_argmax: elbo values at convergence (1,)
        lbfgs_status: LBFGS status
        path_status: path status

    where:
        M: number of samples
        N: number of parameters
    """

    samples: NDArray | None = None
    logP: NDArray | None = None
    logQ: NDArray | None = None
    lbfgs_niter: NDArray | None = None
    elbo_argmax: NDArray | None = None
    lbfgs_status: LBFGSStatus = LBFGSStatus.LBFGS_FAILED
    path_status: PathStatus = PathStatus.PATH_FAILED


@dataclass(frozen=True)
class PathfinderConfig:
    """configuration parameters for a single pathfinder"""

    num_draws: int  # same as num_draws_per_path
    maxcor: int
    maxiter: int
    ftol: float
    gtol: float
    maxls: int
    jitter: float
    epsilon: float
    num_elbo_draws: int


@dataclass(slots=True, frozen=True)
class MultiPathfinderResult:
    """
    container for aggregating results from multiple paths.

    Attributes
    ----------
        samples: posterior samples (S, M, N)
        logP: log probability of model (S, M)
        logQ: log probability of approximation (S, M)
        lbfgs_niter: number of lbfgs iterations (S,)
        elbo_argmax: elbo values at convergence (S,)
        lbfgs_status: counter for LBFGS status occurrences
        path_status: counter for path status occurrences
        importance_sampling: importance sampling method used
        warnings: list of warnings
        pareto_k
        pathfinder_config: pathfinder configuration
        compile_time
        compute_time
    where:
        S: number of successful paths, where S <= num_paths
        M: number of samples per path
        N: number of parameters
    """

    samples: NDArray | None = None
    logP: NDArray | None = None
    logQ: NDArray | None = None
    lbfgs_niter: NDArray | None = None
    elbo_argmax: NDArray | None = None
    lbfgs_status: Counter = field(default_factory=Counter)
    path_status: Counter = field(default_factory=Counter)
    importance_sampling: str | None = "psis"
    warnings: list[str] = field(default_factory=list)
    pareto_k: float | None = None

    # config
    num_paths: int | None = None
    num_draws: int | None = None
    pathfinder_config: PathfinderConfig | None = None

    # timing
    compile_time: float | None = None
    compute_time: float | None = None

    all_paths_failed: bool = False  # raises ValueError if all paths failed

    @classmethod
    def from_path_results(cls, path_results: list[PathfinderResult]) -> "MultiPathfinderResult":
        """aggregate successful pathfinder results and count the occurrences of each status in PathStatus and LBFGSStatus"""

        NUMERIC_ATTRIBUTES = ["samples", "logP", "logQ", "lbfgs_niter", "elbo_argmax"]

        success_results = []
        mpr = cls()

        for pr in path_results:
            if pr.path_status not in FAILED_PATH_STATUS:
                success_results.append(tuple(getattr(pr, attr) for attr in NUMERIC_ATTRIBUTES))

            mpr.lbfgs_status[pr.lbfgs_status] += 1
            mpr.path_status[pr.path_status] += 1

        warnings = _get_status_warning(mpr)

        if success_results:
            results_arr = [np.asarray(x) for x in zip(*success_results)]
            return cls(
                *[np.concatenate(x) if x.ndim > 1 else x for x in results_arr],
                lbfgs_status=mpr.lbfgs_status,
                path_status=mpr.path_status,
                warnings=warnings,
            )
        else:
            return cls(
                lbfgs_status=mpr.lbfgs_status,
                path_status=mpr.path_status,
                warnings=warnings,
                all_paths_failed=True,  # raises ValueError later
            )

    def with_timing(self, compile_time: float, compute_time: float) -> Self:
        """add timing information"""
        return replace(self, compile_time=compile_time, compute_time=compute_time)

    def with_pathfinder_config(self, config: PathfinderConfig) -> Self:
        """add pathfinder configuration"""
        return replace(self, pathfinder_config=config)

    def with_warnings(self, warnings: list[str]) -> Self:
        """add warnings"""
        return replace(self, warnings=warnings)

    def with_importance_sampling(
        self,
        num_draws: int,
        method: Literal["psis", "psir", "identity"] | None,
        random_seed: int | None = None,
    ) -> Self:
        """perform importance sampling"""
        if not self.all_paths_failed:
            isres = _importance_sampling(
                samples=self.samples,
                logP=self.logP,
                logQ=self.logQ,
                num_draws=num_draws,
                method=method,
                random_seed=random_seed,
            )
            return replace(
                self,
                samples=isres.samples,
                importance_sampling=method,
                warnings=[*self.warnings, *isres.warnings],
                pareto_k=isres.pareto_k,
            )
        else:
            return self

    def create_summary(self) -> Table:
        """create rich table summary of pathfinder results"""
        table = Table(
            title="Pathfinder Results",
            title_style="none",
            title_justify="left",
            show_header=False,
            box=None,
            padding=(0, 2),
            show_edge=False,
        )
        table.add_column("Description")
        table.add_column("Value")

        # model info
        if self.samples is not None:
            table.add_row("")
            table.add_row("No. model parameters", str(self.samples.shape[-1]))

        # config
        if self.pathfinder_config is not None:
            table.add_row("")
            table.add_row("Configuration:")
            table.add_row("num_draws_per_path", str(self.pathfinder_config.num_draws))
            table.add_row("history size (maxcor)", str(self.pathfinder_config.maxcor))
            table.add_row("max iterations", str(self.pathfinder_config.maxiter))
            table.add_row("ftol", f"{self.pathfinder_config.ftol:.2e}")
            table.add_row("gtol", f"{self.pathfinder_config.gtol:.2e}")
            table.add_row("max line search", str(self.pathfinder_config.maxls))
            table.add_row("jitter", f"{self.pathfinder_config.jitter}")
            table.add_row("epsilon", f"{self.pathfinder_config.epsilon:.2e}")
            table.add_row("ELBO draws", str(self.pathfinder_config.num_elbo_draws))

        # lbfgs
        table.add_row("")
        table.add_row("LBFGS Status:")
        for status, count in self.lbfgs_status.items():
            table.add_row(str(status.name), str(count))

        if self.lbfgs_niter is not None:
            table.add_row(
                "L-BFGS iterations",
                f"mean {np.mean(self.lbfgs_niter):.0f} ± std {np.std(self.lbfgs_niter):.0f}",
            )

        # paths
        table.add_row("")
        table.add_row("Path Status:")
        for status, count in self.path_status.items():
            table.add_row(str(status.name), str(count))

        if self.elbo_argmax is not None:
            table.add_row(
                "ELBO argmax",
                f"mean {np.mean(self.elbo_argmax):.0f} ± std {np.std(self.elbo_argmax):.0f}",
            )

        # importance sampling section
        if not self.all_paths_failed:
            table.add_row("")
            table.add_row("Importance Sampling:")
            table.add_row("Method", self.importance_sampling)
            if self.pareto_k is not None:
                table.add_row("Pareto k", f"{self.pareto_k:.2f}")

        if self.compile_time is not None:
            table.add_row("")
            table.add_row("Timing (seconds):")
            table.add_row("Compile", f"{self.compile_time:.2f}")

        if self.compute_time is not None:
            table.add_row("Compute", f"{self.compute_time:.2f}")

        if self.compile_time is not None and self.compute_time is not None:
            table.add_row("Total", f"{self.compile_time + self.compute_time:.2f}")

        return table

    def display_summary(self) -> None:
        """display summary including warnings"""
        console = Console()
        summary = self.create_summary()

        # warning messages
        if self.warnings:
            warning_text = [
                Text(),  # blank line
                Text("Warnings:"),
                *(
                    Padding(
                        Text("- " + warning, no_wrap=False).wrap(console, width=console.width - 6),
                        (0, 0, 0, 2),  # left padding only
                    )
                    for warning in self.warnings
                ),
            ]
            output = Group(summary, *warning_text)
        else:
            output = summary

        console.print(output)


def _get_status_warning(mpr: MultiPathfinderResult) -> list[str]:
    """get list of relevant LBFGSStatus and PathStatus warnings given a MultiPathfinderResult"""
    warnings = []

    lbfgs_status_message = {
        LBFGSStatus.MAX_ITER_REACHED: "MAX_ITER_REACHED: LBFGS maximum number of iterations reached. Consider increasing maxiter if this occurence is high relative to the number of paths.",
        LBFGSStatus.INIT_FAILED: "INIT_FAILED: LBFGS failed to initialize. Consider reparameterizing the model or reducing jitter if this occurence is high relative to the number of paths.",
        LBFGSStatus.NON_FINITE: "NON_FINITE: LBFGS objective function produced inf or nan at the last iteration. Consider reparameterizing the model or adjusting the pathfinder arguments if this occurence is high relative to the number of paths.",
        LBFGSStatus.LOW_UPDATE_PCT: "LOW_UPDATE_PCT: Majority of LBFGS iterations were not accepted due to the either: (1) LBFGS function or gradient values containing too many inf or nan values or (2) gradient changes being significantly large, set by epsilon. Consider reparameterizing the model, adjusting initvals or jitter or other pathfinder arguments if this occurence is high relative to the number of paths.",
        LBFGSStatus.INIT_FAILED_LOW_UPDATE_PCT: "INIT_FAILED_LOW_UPDATE_PCT: LBFGS failed to initialize due to the either: (1) LBFGS function or gradient values containing too many inf or nan values or (2) gradient changes being significantly large, set by epsilon. Consider reparameterizing the model, adjusting initvals or jitter or other pathfinder arguments if this occurence is high relative to the number of paths.",
    }

    path_status_message = {
        PathStatus.ELBO_ARGMAX_AT_ZERO: "ELBO_ARGMAX_AT_ZERO: ELBO argmax at zero refers to the first iteration during LBFGS. A high occurrence suggests the model's default initial point + jitter values are concentrated in high-density regions in the target distribution and may result in poor exploration of the parameter space. Consider increasing jitter if this occurrence is high relative to the number of paths.",
        PathStatus.INVALID_LOGQ: "INVALID_LOGQ: Invalid logQ values occur when a path's logQ values are not finite. The failed path is not included in samples when importance sampling is used. Consider reparameterizing the model or adjusting the pathfinder arguments if this occurence is high relative to the number of paths.",
    }

    for lbfgs_status in mpr.lbfgs_status:
        if lbfgs_status in lbfgs_status_message:
            warnings.append(lbfgs_status_message.get(lbfgs_status))

    for path_status in mpr.path_status:
        if path_status in path_status_message:
            warnings.append(path_status_message.get(path_status))

    return warnings


def multipath_pathfinder(
    model: Model,
    num_paths: int,
    num_draws: int,
    num_draws_per_path: int,
    maxcor: int,
    maxiter: int,
    ftol: float,
    gtol: float,
    maxls: int,
    num_elbo_draws: int,
    jitter: float,
    epsilon: float,
    importance_sampling: Literal["psis", "psir", "identity"] | None,
    progressbar: bool,
    concurrent: Literal["process"] | None = "process",
    random_seed: RandomSeed = None,
    max_init_retries: int = 10,
    pathfinder_kwargs: dict[str, Any] = {},
    compile_kwargs: dict[str, Any] = {},
    display_summary: bool = True,
) -> MultiPathfinderResult:
    """
    Fit the Pathfinder Variational Inference algorithm using multiple paths with PyMC/PyTensor backend.

    Parameters
    ----------
    model : pymc.Model
        The PyMC model to fit the Pathfinder algorithm to.
    num_paths : int
        Number of independent paths to run in the Pathfinder algorithm. (default is 4) It is recommended to increase num_paths when increasing the jitter value.
    num_draws : int, optional
        Total number of samples to draw from the fitted approximation (default is 1000).
    num_draws_per_path : int, optional
        Number of samples to draw per path (default is 1000).
    maxcor : int, optional
        Maximum number of variable metric corrections used to define the limited memory matrix (default is None). If None, maxcor is set to ceil(3 * log(N)) or 5 whichever is greater, where N is the number of model parameters.
    maxiter : int, optional
        Maximum number of iterations for the L-BFGS optimisation (default is 1000).
    ftol : float, optional
        Tolerance for the decrease in the objective function (default is 1e-5).
    gtol : float, optional
        Tolerance for the norm of the gradient (default is 1e-8).
    maxls : int, optional
        Maximum number of line search steps for the L-BFGS algorithm (default is 1000).
    num_elbo_draws : int, optional
        Number of draws for the Evidence Lower Bound (ELBO) estimation (default is 10).
    jitter : float, optional
        Amount of jitter to apply to initial points (default is 2.0). Note that Pathfinder may be highly sensitive to the jitter value. It is recommended to increase num_paths when increasing the jitter value.
    epsilon: float
        value used to filter out large changes in the direction of the update gradient at each iteration l in L. Iteration l is only accepted if s·z >= epsilon * ||z||² for each l in L. Matches Zhang et al. (2022) Algorithm 3 with default epsilon=1e-12.
    importance_sampling : str, None, optional
        Method to apply sampling based on log importance weights (logP - logQ).
        "psis" : Pareto Smoothed Importance Sampling (default)
                Recommended for more stable results.
        "psir" : Pareto Smoothed Importance Resampling
                Less stable than PSIS.
        "identity" : Applies log importance weights directly without resampling.
        None : No importance sampling weights. Returns raw samples of size (num_paths, num_draws_per_path, N) where N is number of model parameters. Other methods return samples of size (num_draws, N).
    progressbar : bool, optional
        Whether to display a progress bar (default is False). Setting this to True will likely increase the computation time.
    random_seed : RandomSeed, optional
        Random seed for reproducibility.
    postprocessing_backend : str, optional
        Backend for postprocessing transformations, either "cpu" or "gpu" (default is "cpu"). This is only relevant if inference_backend is "blackjax".
    inference_backend : str, optional
        Backend for inference, either "pymc" or "blackjax" (default is "pymc").
    concurrent : str, optional
        How to run paths: ``"process"`` (default) spawns separate worker processes for true
        parallelism, matching PyMC's approach for parallel chains.  Set to ``None`` for serial
        execution (useful for debugging).
        ``"thread"`` is intentionally not offered: pytensor compiled functions share intermediate
        op storage across ``Function.copy()`` instances, so concurrent thread calls corrupt each
        other's in-flight state.  Processes have fully independent memory and are always safe.
    max_init_retries : int, optional
        Maximum number of re-jitter retries per path when LBFGSInitFailed is raised (default is 10).
    pathfinder_kwargs
        Additional keyword arguments for the Pathfinder algorithm.
    compile_kwargs
        Additional keyword arguments for the PyTensor compiler. If not provided, a
        performant default is used.

    Returns
    -------
    MultiPathfinderResult
        The result containing samples and other information from the Multi-Path Pathfinder algorithm.
    """

    *path_seeds, choice_seed = _get_seeds_per_chain(random_seed, num_paths + 1)

    pathfinder_config = PathfinderConfig(
        num_draws=num_draws_per_path,
        maxcor=maxcor,
        maxiter=maxiter,
        ftol=ftol,
        gtol=gtol,
        maxls=maxls,
        num_elbo_draws=num_elbo_draws,
        jitter=jitter,
        epsilon=epsilon,
    )

    compile_start = time.time()
    single_pathfinder_fn = make_single_pathfinder_fn(
        model,
        **asdict(pathfinder_config),
        max_init_retries=max_init_retries,
        pathfinder_kwargs=pathfinder_kwargs,
        compile_kwargs=compile_kwargs,
    )
    compile_end = time.time()

    results = []
    compute_start = time.time()
    try:
        # Per-path progress bar (one row per path, updated in real time)
        progress = CustomProgress(
            TextColumn(
                "{task.description}", table_column=Column("Path", min_width=7, no_wrap=True)
            ),
            TextColumn(
                "{task.fields[status]}", table_column=Column("Status", min_width=10, no_wrap=True)
            ),
            TextColumn(
                "{task.fields[lbfgs_steps]}",
                table_column=Column("Steps", min_width=6, no_wrap=True),
            ),
            TextColumn(
                "{task.fields[steps_per_sec]}",
                table_column=Column("Steps/s", min_width=8, no_wrap=True),
            ),
            TextColumn(
                "{task.fields[best_elbo]}",
                table_column=Column("Best ELBO", min_width=12, no_wrap=True),
            ),
            TextColumn(
                "{task.fields[current_elbo]}",
                table_column=Column("Cur ELBO", min_width=12, no_wrap=True),
            ),
            TextColumn(
                "{task.fields[step_size]}",
                table_column=Column("Step size", min_width=10, no_wrap=True),
            ),
            TimeElapsedColumn(table_column=Column("Elapsed", min_width=8, no_wrap=True)),
            include_headers=True,
            console=Console(theme=default_progress_theme),
            disable=not progressbar,
        )

        # Create one task per path and build per-path progress callbacks
        task_ids = []
        path_callbacks: list[Callable | None] = []
        with progress:
            for i in range(num_paths):
                tid = progress.add_task(
                    f"Path {i + 1}",
                    status="queued",
                    lbfgs_steps=0,
                    steps_per_sec="—",
                    best_elbo="—",
                    current_elbo="—",
                    step_size="—",
                    total=None,
                )
                task_ids.append(tid)

                def _make_cb(task_id: int) -> Callable:
                    def cb(info: dict) -> None:
                        fields: dict[str, Any] = {}
                        if "status" in info and info["status"] is not None:
                            fields["status"] = info["status"]
                        if "lbfgs_steps" in info:
                            fields["lbfgs_steps"] = info["lbfgs_steps"]
                        if "best_elbo" in info:
                            val = info["best_elbo"]
                            fields["best_elbo"] = (
                                f"{val:.3f}" if val is not None and np.isfinite(float(val)) else "—"
                            )
                        if "current_elbo" in info:
                            val = info["current_elbo"]
                            fields["current_elbo"] = (
                                f"{val:.3f}" if val is not None and np.isfinite(float(val)) else "—"
                            )
                        if "step_size" in info:
                            val = info["step_size"]
                            fields["step_size"] = (
                                f"{val:.2e}" if val is not None and np.isfinite(float(val)) else "—"
                            )
                        if "steps_per_sec" in info:
                            val = info["steps_per_sec"]
                            fields["steps_per_sec"] = (
                                f"{val:.1f}/s"
                                if val is not None and np.isfinite(float(val))
                                else "—"
                            )
                        if fields:
                            progress.update(task_id, **fields)

                    return cb

                path_callbacks.append(_make_cb(tid))

            # concurrent="process" gives true parallelism via separate worker processes
            # (matching PyMC's approach). concurrent="thread" uses threads but may serialize
            # due to the Python GIL. concurrent=None is serial (useful for debugging).
            generator = make_generator(
                concurrent=concurrent,
                fn=single_pathfinder_fn,
                seeds=path_seeds,
                progress_callbacks=path_callbacks,
            )

            for result in generator:
                try:
                    if isinstance(result, Exception):
                        raise result
                    else:
                        results.append(result)
                except filelock.Timeout:
                    logger.warning("Lock timeout. Retrying...")
                    num_attempts = 0
                    while num_attempts < 10:
                        try:
                            results.append(result)
                            logger.info("Lock acquired. Continuing...")
                            break
                        except filelock.Timeout:
                            num_attempts += 1
                            time.sleep(0.5)
                            logger.warning(f"Lock timeout. Retrying... ({num_attempts}/10)")
                except Exception as e:
                    logger.warning("Unexpected error in a path: %s", str(e))
                    results.append(
                        PathfinderResult(
                            path_status=PathStatus.PATH_FAILED,
                            lbfgs_status=LBFGSStatus.LBFGS_FAILED,
                        )
                    )
    except (KeyboardInterrupt, StopIteration) as e:
        # User is free to abort early — MultiPathfinderResult collects all successful results so far.
        if isinstance(e, StopIteration):
            logger.info(str(e))
    finally:
        compute_end = time.time()
        if results:
            mpr = (
                MultiPathfinderResult.from_path_results(results)
                .with_pathfinder_config(config=pathfinder_config)
                .with_importance_sampling(
                    num_draws=num_draws, method=importance_sampling, random_seed=choice_seed
                )
                .with_timing(
                    compile_time=compile_end - compile_start,
                    compute_time=compute_end - compute_start,
                )
            )
            # Display summary conditionally
            if display_summary:
                mpr.display_summary()
            if mpr.all_paths_failed:
                raise ValueError(
                    "All paths failed. Consider decreasing the jitter or reparameterizing the model."
                )
        else:
            raise ValueError(
                "BUG: Failed to iterate!"
                "Please report this issue at: "
                "https://github.com/pymc-devs/pymc-extras/issues "
                "with your code to reproduce the issue and the following details:\n"
                f"pathfinder_config: \n{pathfinder_config}\n"
                f"compile_kwargs: {compile_kwargs}\n"
                f"pathfinder_kwargs: {pathfinder_kwargs}\n"
                f"num_paths: {num_paths}\n"
                f"num_draws: {num_draws}\n"
            )

    return mpr


def fit_pathfinder(
    model=None,
    num_paths: int = 4,  # I
    num_draws: int = 1000,  # R
    num_draws_per_path: int = 1000,  # M
    maxcor: int | None = None,  # J
    maxiter: int = 1000,  # L^max
    ftol: float = 1e-5,
    gtol: float = 1e-8,
    maxls: int = 1000,
    num_elbo_draws: int = 10,  # K
    jitter: float = 2.0,
    epsilon: float = 1e-12,
    importance_sampling: Literal["psis", "psir", "identity"] | None = "psis",
    progressbar: bool = True,
    concurrent: Literal["process"] | None = None,
    max_init_retries: int = 10,
    random_seed: RandomSeed | None = None,
    postprocessing_backend: Literal["cpu", "gpu"] = "cpu",
    inference_backend: Literal["pymc", "blackjax"] = "pymc",
    pathfinder_kwargs: dict[str, Any] = {},
    compile_kwargs: dict[str, Any] = {},
    initvals: dict[str, Any] | None = None,
    # New pathfinder result integration options
    add_pathfinder_groups: bool = True,
    display_summary: bool | Literal["auto"] = "auto",
    store_diagnostics: bool = False,
    pathfinder_group: str = "pathfinder",
    paths_group: str = "pathfinder_paths",
    diagnostics_group: str = "pathfinder_diagnostics",
    config_group: str = "pathfinder_config",
) -> az.InferenceData:
    """
    Fit the Pathfinder Variational Inference algorithm.

    This function fits the Pathfinder algorithm to a given PyMC model, allowing for multiple paths and draws. It supports both PyMC and BlackJAX backends.

    Parameters
    ----------
    model : pymc.Model
        The PyMC model to fit the Pathfinder algorithm to.
    num_paths : int
        Number of independent paths to run in the Pathfinder algorithm. (default is 4) It is recommended to increase num_paths when increasing the jitter value.
    num_draws : int, optional
        Total number of samples to draw from the fitted approximation (default is 1000).
    num_draws_per_path : int, optional
        Number of samples to draw per path (default is 1000).
    maxcor : int, optional
        Maximum number of variable metric corrections used to define the limited memory matrix (default is None). If None, maxcor is set to ceil(3 * log(N)) or 5 whichever is greater, where N is the number of model parameters.
    maxiter : int, optional
        Maximum number of iterations for the L-BFGS optimisation (default is 1000).
    ftol : float, optional
        Tolerance for the decrease in the objective function (default is 1e-5).
    gtol : float, optional
        Tolerance for the norm of the gradient (default is 1e-8).
    maxls : int, optional
        Maximum number of line search steps for the L-BFGS algorithm (default is 1000).
    num_elbo_draws : int, optional
        Number of draws for the Evidence Lower Bound (ELBO) estimation (default is 10).
    jitter : float, optional
        Amount of jitter to apply to initial points (default is 2.0). Note that Pathfinder may be highly sensitive to the jitter value. It is recommended to increase num_paths when increasing the jitter value.
    epsilon: float
        value used to filter out large changes in the direction of the update gradient at each iteration l in L. Iteration l is only accepted if s·z >= epsilon * ||z||² for each l in L. Matches Zhang et al. (2022) Algorithm 3 with default epsilon=1e-12.
    importance_sampling : str, None, optional
        Method to apply sampling based on log importance weights (logP - logQ).
        Options are:

        - "psis" : Pareto Smoothed Importance Sampling (default). Usually most stable.
        - "psir" : Pareto Smoothed Importance Resampling. Less stable than PSIS.
        - "identity" : Applies log importance weights directly without resampling.
        - None : No importance sampling weights. Returns raw samples of size (num_paths, num_draws_per_path, N) where N is number of model parameters. Other methods return samples of size (num_draws, N).

    progressbar : bool, optional
        Whether to display a progress bar (default is True). Setting this to False will likely reduce the computation time.
    random_seed : RandomSeed, optional
        Random seed for reproducibility.
    postprocessing_backend : str, optional
        Backend for postprocessing transformations, either "cpu" or "gpu" (default is "cpu"). This is only relevant if inference_backend is "blackjax".
    inference_backend : str, optional
        Backend for inference, either "pymc" or "blackjax" (default is "pymc").
    concurrent : str, optional
        How to run paths: ``"process"`` (default) spawns separate worker processes for true
        parallelism, matching PyMC's approach for parallel chains.  Set to ``None`` for serial
        execution (useful for debugging).
        ``"thread"`` is intentionally not offered: pytensor compiled functions share intermediate
        op storage across ``Function.copy()`` instances, so concurrent thread calls corrupt each
        other's in-flight state.  Processes have fully independent memory and are always safe.
    max_init_retries : int, optional
        Maximum number of re-jitter retries per path when the initial point fails (default is 10).
    pathfinder_kwargs
        Additional keyword arguments for the Pathfinder algorithm.
    compile_kwargs
        Additional keyword arguments for the PyTensor compiler. If not provided, a
        performant default is used.
    initvals: dict | None = None
        Initial values for the model parameters, as str:ndarray key-value pairs. Paritial initialization is permitted.
        If None, the model's default initial values are used.
    add_pathfinder_groups : bool, optional
        Whether to add pathfinder results as additional groups to the InferenceData (default is True).
        When True, adds pathfinder and pathfinder_paths groups with optimization diagnostics.
    display_summary : bool or "auto", optional
        Whether to display the pathfinder results summary (default is "auto").
        "auto" preserves current behavior, False suppresses console output.
    store_diagnostics : bool, optional
        Whether to include potentially large diagnostic arrays in the pathfinder groups (default is False).
    pathfinder_group : str, optional
        Name for the main pathfinder results group (default is "pathfinder").
    paths_group : str, optional
        Name for the per-path results group (default is "pathfinder_paths").
    diagnostics_group : str, optional
        Name for the diagnostics group (default is "pathfinder_diagnostics").
    config_group : str, optional
        Name for the configuration group (default is "pathfinder_config").

    Returns
    -------
    :class:`~arviz.InferenceData`
        The inference data containing the results of the Pathfinder algorithm.

    References
    ----------
    Zhang, L., Carpenter, B., Gelman, A., & Vehtari, A. (2022). Pathfinder: Parallel quasi-Newton variational inference. Journal of Machine Learning Research, 23(306), 1-49.
    """

    model = modelcontext(model)

    if initvals is not None:
        model = pm.model.fgraph.clone_model(model)  # Create a clone of the model
        for (
            rv_name,
            ivals,
        ) in initvals.items():  # Set the initial values for the variables in the clone
            model.set_initval(model.named_vars[rv_name], ivals)

    valid_importance_sampling = {"psis", "psir", "identity", None}

    if importance_sampling is not None:
        importance_sampling = importance_sampling.lower()

    if importance_sampling not in valid_importance_sampling:
        raise ValueError(f"Invalid importance sampling method: {importance_sampling}")

    N = DictToArrayBijection.map(model.initial_point()).data.shape[0]

    if maxcor is None:
        # Based on tests, this seems to be a good default value. Higher maxcor values do not necessarily lead to better results and can slow down the algorithm. Also, if results do benefit from a higher maxcor value, the improvement may be diminishing w.r.t. the increase in maxcor.
        maxcor = np.ceil(3 * np.log(N)).astype(np.int32)
        maxcor = max(maxcor, 5)

    # Handle display_summary logic
    should_display_summary = display_summary == "auto" or display_summary is True

    if inference_backend == "pymc":
        mp_result = multipath_pathfinder(
            model,
            num_paths=num_paths,
            num_draws=num_draws,
            num_draws_per_path=num_draws_per_path,
            maxcor=maxcor,
            maxiter=maxiter,
            ftol=ftol,
            gtol=gtol,
            maxls=maxls,
            num_elbo_draws=num_elbo_draws,
            jitter=jitter,
            epsilon=epsilon,
            importance_sampling=importance_sampling,
            progressbar=progressbar,
            concurrent=concurrent,
            max_init_retries=max_init_retries,
            random_seed=random_seed,
            pathfinder_kwargs=pathfinder_kwargs,
            compile_kwargs=compile_kwargs,
            display_summary=should_display_summary,
        )
        pathfinder_samples = mp_result.samples
    elif inference_backend == "blackjax":
        import blackjax
        import jax

        if version.parse(blackjax.__version__).major < 1:
            raise ImportError("fit_pathfinder requires blackjax 1.0 or above")

        jitter_seed, pathfinder_seed, sample_seed = _get_seeds_per_chain(random_seed, 3)
        # TODO: extend initial points with jitter_scale to blackjax
        # TODO: extend blackjax pathfinder to multiple paths
        x0, _ = DictToArrayBijection.map(model.initial_point())
        logp_func = get_jaxified_logp_of_ravel_inputs(model)
        pathfinder_state, pathfinder_info = blackjax.vi.pathfinder.approximate(
            rng_key=jax.random.key(pathfinder_seed),
            logdensity_fn=logp_func,
            initial_position=x0,
            num_samples=num_elbo_draws,
            maxiter=maxiter,
            maxcor=maxcor,
            maxls=maxls,
            ftol=ftol,
            gtol=gtol,
            **pathfinder_kwargs,
        )
        pathfinder_samples, _ = blackjax.vi.pathfinder.sample(
            rng_key=jax.random.key(sample_seed),
            state=pathfinder_state,
            num_samples=num_draws,
        )
    else:
        raise ValueError(f"Invalid inference_backend: {inference_backend}")

    logger.info("Transforming variables...")

    idata = convert_flat_trace_to_idata(
        pathfinder_samples,
        postprocessing_backend=postprocessing_backend,
        inference_backend=inference_backend,
        model=model,
        importance_sampling=importance_sampling,
    )

    idata = add_data_to_inference_data(idata, progressbar, model, compile_kwargs)

    # Add pathfinder results to InferenceData if requested
    if add_pathfinder_groups:
        if inference_backend == "pymc":
            from pymc_extras.inference.pathfinder.idata import add_pathfinder_to_inference_data

            idata = add_pathfinder_to_inference_data(
                idata=idata,
                result=mp_result,
                model=model,
                group=pathfinder_group,
                paths_group=paths_group,
                diagnostics_group=diagnostics_group,
                config_group=config_group,
                store_diagnostics=store_diagnostics,
            )
        else:
            warnings.warn(
                f"Pathfinder diagnostic groups are only supported with the PyMC backend. "
                f"Current backend is '{inference_backend}', which does not support adding "
                "pathfinder diagnostics to InferenceData. The InferenceData will only contain "
                "posterior samples. To add diagnostic groups, use inference_backend='pymc', "
                "or set add_pathfinder_groups=False to suppress this warning.",
                UserWarning,
                stacklevel=2,
            )

    return idata
