from collections.abc import Callable

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from numpy.typing import NDArray
from pymc import Model
from pymc.pytensorf import compile
from pytensor.compile.executor import Function
from pytensor.graph import clone_replace, vectorize_graph
from pytensor.tensor import TensorVariable

REGULARIZATION_TERM = 1e-8


def alpha_step_numpy(alpha_prev: NDArray, s: NDArray, z: NDArray) -> NDArray:
    """Single-step update of the L-BFGS inverse-Hessian diagonal estimate (Zhang et al. 2022).

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
    # Guard: b≈0 or c≈0 makes the update divide by zero; keep alpha_prev to avoid NaN.
    z_sq = float(np.sum(z**2)) + 1e-30
    if abs(b) < 1e-14 * z_sq or c <= 0 or not np.isfinite(c):
        return alpha_prev.copy()
    inv_alpha = (
        a / (b * alpha_prev)
        + z**2 / b
        - (a * s**2) / (b * c * alpha_prev**2)
    )  # fmt: off
    alpha_out = 1.0 / inv_alpha
    if not np.all(np.isfinite(alpha_out)) or np.any(alpha_out <= 0):
        return alpha_prev.copy()
    return alpha_out


def _bfgs_sample_pt(
    x: TensorVariable,
    g: TensorVariable,
    alpha: TensorVariable,
    S: TensorVariable,
    Z: TensorVariable,
    u: TensorVariable,
    J: int,
    N: int,
) -> tuple[TensorVariable, TensorVariable, TensorVariable]:
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
    inv_hessian_diag : (N,) diagonal of sampler covariance used for `phi`
    """
    J2 = 2 * J

    # Inverse Hessian factors
    E = pt.triu(S.T @ Z) + pt.eye(J) * REGULARIZATION_TERM  # (J, J)
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
        IdN = pt.eye(N) * (1.0 + REGULARIZATION_TERM)
        H_inv = sa_d @ (IdN + isa_d @ beta @ gamma @ beta.T @ isa_d) @ sa_d
        Lchol = pt.linalg.cholesky(H_inv, lower=False)  # (N, N)
        logdet = 2.0 * pt.sum(pt.log(pt.abs(pt.diag(Lchol))))
        mu = x - H_inv @ g
        phi = (mu[:, None] + Lchol @ u.T).T  # (M, N)
        inv_hessian_diag = pt.diag(H_inv)
    else:
        # Sparse path: economy QR avoids O(N²) matrices (large-N regime)
        Q, R = pt.linalg.qr(beta * inv_sqrt_alpha[:, None], mode="reduced")  # Q:(N,2J), R:(2J,2J)
        I2J = pt.eye(J2) * (1.0 + REGULARIZATION_TERM)
        Lchol = pt.linalg.cholesky(I2J + R @ gamma @ R.T, lower=False)  # (2J, 2J)
        logdet = 2.0 * pt.sum(pt.log(pt.abs(pt.diag(Lchol)))) + pt.sum(pt.log(alpha))
        btg = beta.T @ g[:, None]  # (2J, 1)
        mu = x - ((alpha * g)[:, None] + beta @ (gamma @ btg))[:, 0]  # (N,)
        QtU = Q.T @ u.T  # (2J, M)
        phi = (mu[:, None] + sqrt_alpha[:, None] * (Q @ ((Lchol - pt.eye(J2)) @ QtU) + u.T)).T
        # diag(Q (L L^T - I) Q^T) = row_norm_sq(Q L) - row_norm_sq(Q)
        ql = Q @ Lchol
        inv_hessian_diag = alpha * (1.0 + pt.sum(ql * ql, axis=1) - pt.sum(Q * Q, axis=1))

    logQ = -0.5 * (logdet + pt.sum(u * u, axis=-1) + N * np.log(2.0 * np.pi))
    return phi, logQ, inv_hessian_diag


def make_pathfinder_sample_fn(
    model: Model,
    N: int,
    J: int,
    jacobian: bool,
    compile_kwargs: dict | None = None,
    vectorize: bool = False,
) -> Function:
    """Compile a single PyTensor function covering bfgs sample + batched logP evaluation.

    S and Z are passed as explicit inputs (not shared variables) so each caller
    can pass its own arrays, avoiding shared mutable state.

    The number of draws M is a dynamic runtime dimension — the same compiled
    function handles both ELBO estimation (small M) and final sampling (large M).

    Parameters
    ----------
    model : Model
    N : int, number of unconstrained parameters
    J : int, L-BFGS history size (maxcor)
    jacobian : bool
    compile_kwargs : dict, optional
    vectorize : bool, optional
        If True, use vectorize_graph instead of pytensor.map. Default False.

    Returns
    -------
    fn : Function
        Compiled: (x, g, alpha, s_win, z_win, u) → (phi, logQ, logP, inv_hessian_diag)
        where s_win, z_win are (N, J), u is (M, N), and M is a dynamic dimension.
    """
    compile_kwargs = compile_kwargs or {}

    (logP_single,), single_input = pm.pytensorf.join_nonshared_inputs(
        model.initial_point(),
        [model.logp(jacobian=jacobian)],
        model.value_vars,
    )

    x_sym = pt.tensor("x", shape=(N,), dtype="float64")
    g_sym = pt.tensor("g", shape=(N,), dtype="float64")
    alpha_sym = pt.tensor("alpha", shape=(N,), dtype="float64")
    s_win_sym = pt.tensor("s_win", shape=(N, J), dtype="float64")
    z_win_sym = pt.tensor("z_win", shape=(N, J), dtype="float64")
    u_sym = pt.matrix("u", dtype="float64")  # (M, N) — M is dynamic

    phi_sym, logQ_sym, inv_hessian_diag_sym = _bfgs_sample_pt(
        x_sym, g_sym, alpha_sym, s_win_sym, z_win_sym, u_sym, J, N
    )

    if vectorize:
        batched_logP_sym = vectorize_graph(logP_single, replace={single_input: phi_sym})
    else:
        batched_logP_sym = pytensor.map(
            fn=lambda x_i: clone_replace([logP_single], replace={single_input: x_i})[0],
            sequences=[phi_sym],
            return_updates=False,
        )

    outputs = [phi_sym, logQ_sym, batched_logP_sym, inv_hessian_diag_sym]

    # Compile via pm.compile (not raw pytensor.function) so the logP graph's CheckParameterValue
    # ops become Switches to -inf rather than raising: a Gaussian draw landing outside the model's
    # support is a zero-density point, scored -inf and handled by the ELBO's finite mask.
    fn = compile(
        [
            pytensor.In(x_sym, borrow=True),
            pytensor.In(g_sym, borrow=True),
            pytensor.In(alpha_sym, borrow=True),
            pytensor.In(s_win_sym, borrow=True),
            pytensor.In(z_win_sym, borrow=True),
            pytensor.In(u_sym, borrow=True),
        ],
        outputs,
        **compile_kwargs,
    )
    fn.trust_input = True
    return fn


def get_jaxified_logp_of_ravel_inputs(model: Model, jacobian: bool = True) -> Callable:
    """Get a JAX function computing the log-probability of a PyMC model with raveled inputs.

    Parameters
    ----------
    model : Model
        PyMC model to compute the log-probability for.
    jacobian : bool, optional
        Whether to include the Jacobian in the log-probability computation. Setting to False
        (not recommended) may produce very high pareto-k values. Default True.

    Returns
    -------
    callable
        A JAX function computing the log-probability with raveled inputs.
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
    """Get the log-probability and its gradient for a PyMC model with raveled inputs.

    Parameters
    ----------
    model : Model
        PyMC model to compute the log-probability and gradient for.
    jacobian : bool, optional
        Whether to include the Jacobian in the log-probability computation. Setting to False
        (not recommended) may produce very high pareto-k values. Default True.
    **compile_kwargs : dict
        Additional keyword arguments passed to the PyTensor ``compile`` function.

    Returns
    -------
    Function
        Compiled function returning the log-probability and its gradient given raveled inputs.
    """

    (logP, dlogP), inputs = pm.pytensorf.join_nonshared_inputs(
        model.initial_point(),
        [model.logp(jacobian=jacobian), model.dlogp(jacobian=jacobian)],
        model.value_vars,
    )
    logp_dlogp_fn = compile([inputs], (logP, dlogP), **compile_kwargs)
    logp_dlogp_fn.trust_input = True

    return logp_dlogp_fn


def get_neg_logp_dlogp_of_ravel_inputs(
    model: Model, jacobian: bool = True, **compile_kwargs
) -> Function:
    """Compiled function (x,) -> (-logP, -dlogP) for L-BFGS minimization."""
    (logP, dlogP), inputs = pm.pytensorf.join_nonshared_inputs(
        model.initial_point(),
        [model.logp(jacobian=jacobian), model.dlogp(jacobian=jacobian)],
        model.value_vars,
    )
    neg_logp_dlogp_fn = compile([inputs], (-logP, -dlogP), **compile_kwargs)
    neg_logp_dlogp_fn.trust_input = True
    return neg_logp_dlogp_fn
