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
import scipy.linalg as sp_linalg

from numpy.typing import NDArray
from packaging import version
from pymc import Model
from pymc.backends.arviz import coords_and_dims_for_inferencedata
from pymc.blocking import DictToArrayBijection, RaveledVars
from pymc.initial_point import make_initial_point_fn
from pymc.model import modelcontext
from pymc.model.core import Point
from pymc.progress_bar import CustomProgress, default_progress_theme
from pymc.pytensorf import (
    compile,
    find_rng_nodes,
    reseed_rngs,
)
from pymc.util import (
    RandomSeed,
    _get_seeds_per_chain,
    get_default_varnames,
)
from pytensor.compile.function.types import Function
from pytensor.compile.mode import FAST_COMPILE, Mode
from pytensor.graph import Apply, Op, clone_replace, vectorize_graph
from pytensor.tensor import TensorConstant, TensorVariable
from rich.console import Console, Group
from rich.padding import Padding
from rich.progress import BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text

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

    # assert np.all(alpha.eval() > 0), "alpha cannot be negative"
    # alpha: (L, N)
    return alpha, s, z


def inverse_hessian_factors_from_SZ(
    alpha: TensorVariable,
    S: TensorVariable,
    Z: TensorVariable,
    J: TensorConstant,
) -> tuple[TensorVariable, TensorVariable]:
    """compute inverse Hessian factors from pre-built chi matrices.

    Parameters
    ----------
    alpha : TensorVariable
        diagonal scaling vector, shape (L, N)
    S : TensorVariable
        sliding window of s-diffs, shape (L, N, J)
    Z : TensorVariable
        sliding window of z-diffs, shape (L, N, J)
    J : TensorConstant
        history size for L-BFGS

    Returns
    -------
    beta : TensorVariable
        low-rank update matrix, shape (L, N, 2J)
    gamma : TensorVariable
        low-rank update matrix, shape (L, 2J, 2J)

    Notes
    -----
    shapes: L=batch_size, N=num_params, J=history_size
    """

    L = alpha.shape[0]

    # E: (L, J, J)
    Ij = pt.eye(J)[None, ...]
    E = pt.triu(pt.matrix_transpose(S) @ Z)
    E += Ij * REGULARISATION_TERM

    # eta: (L, J)
    eta = pt.diagonal(E, axis1=-2, axis2=-1)

    # AZ: (L, N, J) — replaces alpha_diag @ Z (avoids (L, N, N))
    AZ = alpha[..., None] * Z

    # beta: (L, N, 2J)
    beta = pt.concatenate([AZ, S], axis=-1)

    # more performant and numerically precise to use solve than inverse: https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.linalg.inv.html

    # E_inv: (L, J, J)
    E_inv = pt.linalg.solve_triangular(E, Ij, check_finite=False)

    # eta_diag: (L, J, J) — broadcast diagonal, avoids scan(diag)
    eta_diag = pt.eye(J)[None, :, :] * eta[:, None, :]

    # Zt_AZ: (L, J, J) — replaces Z.T @ alpha_diag @ Z
    Zt_AZ = pt.matrix_transpose(Z) @ AZ

    # block_dd: (L, J, J)
    block_dd = pt.matrix_transpose(E_inv) @ (eta_diag + Zt_AZ) @ E_inv

    # (L, J, 2J)
    gamma_top = pt.concatenate([pt.zeros((L, J, J)), -E_inv], axis=-1)

    # (L, J, 2J)
    gamma_bottom = pt.concatenate([-pt.matrix_transpose(E_inv), block_dd], axis=-1)

    # (L, 2J, 2J)
    gamma = pt.concatenate([gamma_top, gamma_bottom], axis=1)

    return beta, gamma


def inverse_hessian_factors(
    alpha: TensorVariable,
    s: TensorVariable,
    z: TensorVariable,
    J: TensorConstant,
) -> tuple[TensorVariable, TensorVariable]:
    """compute the inverse hessian factors for the BFGS approximation.

    Parameters
    ----------
    alpha : TensorVariable
        diagonal scaling matrix, shape (L, N)
    s : TensorVariable
        position differences, shape (L, N)
    z : TensorVariable
        gradient differences, shape (L, N)
    J : TensorConstant
        history size for L-BFGS

    Returns
    -------
    beta : TensorVariable
        low-rank update matrix, shape (L, N, 2J)
    gamma : TensorVariable
        low-rank update matrix, shape (L, 2J, 2J)

    Notes
    -----
    shapes: L=batch_size, N=num_params, J=history_size
    """

    # NOTE: get_chi_matrix_1 is a modified version of get_chi_matrix_2 to closely follow Zhang et al., (2022)
    # NOTE: get_chi_matrix_2 is from blackjax which MAYBE incorrectly implemented
    def get_chi_matrix_1(diff: TensorVariable, J: TensorConstant) -> TensorVariable:
        L, N = diff.shape
        j_last = pt.as_tensor(J - 1)  # since indexing starts at 0

        def chi_update(diff_l, chi_lm1) -> TensorVariable:
            chi_l = pt.roll(chi_lm1, -1, axis=0)
            return pt.set_subtensor(chi_l[j_last], diff_l)

        chi_init = pt.zeros((J, N))
        chi_mat = pytensor.scan(
            fn=chi_update,
            outputs_info=chi_init,
            sequences=[diff],
            allow_gc=False,
            return_updates=False,
        )

        chi_mat = pt.matrix_transpose(chi_mat)

        # (L, N, J)
        return chi_mat

    def get_chi_matrix_2(diff: TensorVariable, J: TensorConstant) -> TensorVariable:
        L = diff.shape[0]

        # diff_padded: (L+J, N)
        pad_width = pt.zeros(shape=(2, 2), dtype="int32")
        pad_width = pt.set_subtensor(pad_width[0, 0], J - 1)
        diff_padded = pt.pad(diff, pad_width, mode="constant")

        index = pt.arange(L)[..., None] + pt.arange(J)[None, ...]
        index = index.reshape((L, J))

        chi_mat = pt.matrix_transpose(diff_padded[index])

        # (L, N, J)
        return chi_mat

    S = get_chi_matrix_2(s, J)
    Z = get_chi_matrix_2(z, J)

    return inverse_hessian_factors_from_SZ(alpha, S, Z, J)


def bfgs_sample_dense(
    x: TensorVariable,
    g: TensorVariable,
    alpha: TensorVariable,
    beta: TensorVariable,
    gamma: TensorVariable,
    inv_sqrt_alpha: TensorVariable,
    sqrt_alpha: TensorVariable,
    u: TensorVariable,
) -> tuple[TensorVariable, TensorVariable]:
    """sample from the BFGS approximation using dense matrix operations.

    Parameters
    ----------
    x : TensorVariable
        position array, shape (L, N)
    g : TensorVariable
        gradient array, shape (L, N)
    alpha : TensorVariable
        diagonal scaling vector, shape (L, N)
    beta : TensorVariable
        low-rank update matrix, shape (L, N, 2J)
    gamma : TensorVariable
        low-rank update matrix, shape (L, 2J, 2J)
    inv_sqrt_alpha : TensorVariable
        1/sqrt(alpha) vector, shape (L, N)
    sqrt_alpha : TensorVariable
        sqrt(alpha) vector, shape (L, N)
    u : TensorVariable
        random normal samples, shape (L, M, N)

    Returns
    -------
    phi : TensorVariable
        samples from the approximation, shape (L, M, N)
    logdet : TensorVariable
        log determinant of covariance, shape (L,)

    Notes
    -----
    shapes: L=batch_size, N=num_params, J=history_size, M=num_samples
    Dense path is used when 2J >= N (small-N regime); (N,N) diagonals are
    acceptable here.
    """

    # Re-expand vectors to diagonal matrices (small-N regime: 2J >= N)
    sqrt_alpha_diag = pytensor.scan(
        lambda a: pt.diag(a), sequences=[sqrt_alpha], return_updates=False
    )
    inv_sqrt_alpha_diag = pytensor.scan(
        lambda a: pt.diag(a), sequences=[inv_sqrt_alpha], return_updates=False
    )

    N = x.shape[-1]
    IdN = pt.eye(N)[None, ...]
    IdN += IdN * REGULARISATION_TERM

    # inverse Hessian
    H_inv = (
        sqrt_alpha_diag
        @ (
            IdN
            + inv_sqrt_alpha_diag @ beta @ gamma @ pt.matrix_transpose(beta) @ inv_sqrt_alpha_diag
        )
        @ sqrt_alpha_diag
    )

    Lchol = pt.linalg.cholesky(H_inv, lower=False, check_finite=False, on_error="nan")

    logdet = 2.0 * pt.sum(pt.log(pt.abs(pt.diagonal(Lchol, axis1=-2, axis2=-1))), axis=-1)

    # mu = x - pt.einsum("ijk,ik->ij", H_inv, g) # causes error: Multiple destroyers of g

    batched_dot = pt.vectorize(pt.dot, signature="(ijk),(ilk)->(ij)")
    mu = x - batched_dot(H_inv, pt.matrix_transpose(g[..., None]))

    phi = pt.matrix_transpose(
        # (L, N, 1)
        mu[..., None]
        # (L, N, M)
        + Lchol @ pt.matrix_transpose(u)
    )  # fmt: off

    return phi, logdet


def bfgs_sample_sparse(
    x: TensorVariable,
    g: TensorVariable,
    alpha: TensorVariable,
    beta: TensorVariable,
    gamma: TensorVariable,
    inv_sqrt_alpha: TensorVariable,
    sqrt_alpha: TensorVariable,
    u: TensorVariable,
) -> tuple[TensorVariable, TensorVariable]:
    """sample from the BFGS approximation using sparse matrix operations.

    Parameters
    ----------
    x : TensorVariable
        position array, shape (L, N)
    g : TensorVariable
        gradient array, shape (L, N)
    alpha : TensorVariable
        diagonal scaling vector, shape (L, N)
    beta : TensorVariable
        low-rank update matrix, shape (L, N, 2J)
    gamma : TensorVariable
        low-rank update matrix, shape (L, 2J, 2J)
    inv_sqrt_alpha : TensorVariable
        1/sqrt(alpha) vector, shape (L, N)
    sqrt_alpha : TensorVariable
        sqrt(alpha) vector, shape (L, N)
    u : TensorVariable
        random normal samples, shape (L, M, N)

    Returns
    -------
    phi : TensorVariable
        samples from the approximation, shape (L, M, N)
    logdet : TensorVariable
        log determinant of covariance, shape (L,)

    Notes
    -----
    shapes: L=batch_size, N=num_params, J=history_size, M=num_samples
    """

    # qr_input: (L, N, 2J) — A3.2: broadcast instead of (L,N,N) matmul
    qr_input = beta * inv_sqrt_alpha[..., None]
    # pt.linalg.qr is a Blockwise op: call directly on (L, N, 2J) to avoid
    # pytensor.scan overhead (scan has ~100ms Python startup cost even for L=1).
    # mode='reduced' gives Q:(L,N,2J), R:(L,2J,2J) instead of full Q:(L,N,N)
    # which would allocate O(N^2) = 208MB for N=5101, causing ~900ms per call.
    # Economy and full QR are mathematically equivalent for this computation.
    Q, R = pt.linalg.qr(qr_input, mode="reduced")

    IdN = pt.eye(R.shape[1])[None, ...]
    IdN += IdN * REGULARISATION_TERM

    Lchol_input = IdN + R @ gamma @ pt.matrix_transpose(R)

    # TODO: make robust Lchol calcs more robust, ie. try exceptions, increase REGULARISATION_TERM if non-finite exists
    Lchol = pt.linalg.cholesky(Lchol_input, lower=False, check_finite=False, on_error="nan")

    logdet = 2.0 * pt.sum(pt.log(pt.abs(pt.diagonal(Lchol, axis1=-2, axis2=-1))), axis=-1)
    logdet += pt.sum(pt.log(alpha), axis=-1)

    # A3.3: compute H_inv @ g without forming (L, N, N) H_inv
    # NOTE: sign is "x -" to match Stan (differs from Zhang et al., 2022); same for dense.
    g_col = g[..., None]  # (L, N, 1)
    ag = (alpha * g)[..., None]  # (L, N, 1)
    btg = pt.matrix_transpose(beta) @ g_col  # (L, 2J, 1)
    corr = beta @ (gamma @ btg)  # (L, N, 1)
    Hinv_g = ag + corr  # (L, N, 1)
    mu = x - Hinv_g[..., 0]  # (L, N)

    phi = pt.matrix_transpose(
        # (L, N, 1)
        mu[..., None]
        # A3.4: (L, N, 1) * (L, N, M) — broadcast instead of (L, N, N) matmul
        + sqrt_alpha[..., None]
        * (
            # (L, N, 2J), (L, 2J, 2J) -> (L, N, 2J)
            (Q @ (Lchol - IdN))
            # (L, 2J, N), (L, N, M) -> (L, 2J, M)
            @ (pt.matrix_transpose(Q) @ pt.matrix_transpose(u))
            # (L, N, M)
            + pt.matrix_transpose(u)
        )
    )  # fmt: off

    return phi, logdet


def bfgs_sample(
    num_samples: TensorConstant,
    x: TensorVariable,  # position
    g: TensorVariable,  # grad
    alpha: TensorVariable,
    beta: TensorVariable,
    gamma: TensorVariable,
    index: TensorVariable | None = None,
) -> tuple[TensorVariable, TensorVariable]:
    """sample from the BFGS approximation using the inverse hessian factors.

    Parameters
    ----------
    num_samples : TensorConstant
        number of samples to draw
    x : TensorVariable
        position array, shape (L, N)
    g : TensorVariable
        gradient array, shape (L, N)
    alpha : TensorVariable
        diagonal scaling matrix, shape (L, N)
    beta : TensorVariable
        low-rank update matrix, shape (L, N, 2J)
    gamma : TensorVariable
        low-rank update matrix, shape (L, 2J, 2J)
    index : TensorVariable | None
        optional index for selecting a single path

    Returns
    -------
    if index is None:
        phi: samples from local approximations over L (L, M, N)
        logQ_phi: log density of samples of phi (L, M)
    else:
        psi: samples from local approximations where ELBO is maximized (1, M, N)
        logQ_psi: log density of samples of psi (1, M)

    Notes
    -----
    shapes: L=batch_size, N=num_params, J=history_size, M=num_samples
    """

    if index is not None:
        x = x[index][None, ...]
        g = g[index][None, ...]
        alpha = alpha[index][None, ...]
        beta = beta[index][None, ...]
        gamma = gamma[index][None, ...]

    L, N, JJ = beta.shape

    # A3.1: compute vectors instead of (L, N, N) diagonal matrices
    inv_sqrt_alpha = pt.sqrt(1.0 / alpha)  # (L, N)
    sqrt_alpha = pt.sqrt(alpha)  # (L, N)

    u = pt.random.normal(size=(L, num_samples, N))

    sample_inputs = (
        x,
        g,
        alpha,
        beta,
        gamma,
        inv_sqrt_alpha,
        sqrt_alpha,
        u,
    )

    phi, logdet = pytensor.ifelse(
        JJ >= N,
        bfgs_sample_dense(*sample_inputs),
        bfgs_sample_sparse(*sample_inputs),
    )

    logQ_phi = -0.5 * (
        logdet[..., None]
        + pt.sum(u * u, axis=-1)
        + N * pt.log(2.0 * pt.pi)
    )  # fmt: off

    mask = pt.isnan(logQ_phi) | pt.isinf(logQ_phi)
    logQ_phi = pt.set_subtensor(logQ_phi[mask], pt.inf)
    return phi, logQ_phi


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


def _bfgs_sample_numpy(
    x: NDArray,
    g: NDArray,
    alpha: NDArray,
    s_win: NDArray,
    z_win: NDArray,
    M: int,
    rng: np.random.Generator,
) -> tuple[NDArray, NDArray]:
    """Draw M samples from the L-BFGS inverse-Hessian approximation in pure NumPy.

    Mirrors the PyTensor ``bfgs_sample`` logic (both dense and sparse paths) but
    operates entirely outside of PyTensor's compiled graph, eliminating CVM loop,
    scan, and Blockwise overhead.  The caller is responsible for passing an
    independent ``rng`` per path for reproducibility.

    Parameters
    ----------
    x : (N,) position at this LBFGS step
    g : (N,) gradient at this LBFGS step
    alpha : (N,) diagonal scaling vector
    s_win : (N, J) sliding window of position diffs
    z_win : (N, J) sliding window of gradient diffs
    M : number of samples to draw
    rng : NumPy random Generator (advanced in-place)

    Returns
    -------
    phi : (M, N) samples
    logQ : (M,) log-density under the approximation
    """
    N = x.shape[0]
    J = s_win.shape[1]
    S, Z = s_win, z_win

    # ---- inverse_hessian_factors_from_SZ (NumPy port) ----
    E = np.triu(S.T @ Z) + np.eye(J) * REGULARISATION_TERM  # (J, J)
    eta = np.diag(E)  # (J,)
    AZ = alpha[:, None] * Z  # (N, J)
    beta = np.concatenate([AZ, S], axis=1)  # (N, 2J)
    E_inv = sp_linalg.solve_triangular(E, np.eye(J), check_finite=False)  # (J, J)
    block_dd = E_inv.T @ (np.diag(eta) + Z.T @ AZ) @ E_inv  # (J, J)
    gamma = np.block([[np.zeros((J, J)), -E_inv], [-E_inv.T, block_dd]])  # (2J, 2J)

    inv_sqrt_alpha = np.sqrt(1.0 / alpha)
    sqrt_alpha = np.sqrt(alpha)
    J2 = 2 * J

    if J2 >= N:
        # Dense path (small-N regime): form H_inv explicitly, O(N²) is fine here
        isa_d = np.diag(inv_sqrt_alpha)  # (N, N)
        sa_d = np.diag(sqrt_alpha)  # (N, N)
        IdN = np.eye(N) * (1.0 + REGULARISATION_TERM)
        H_inv = sa_d @ (IdN + isa_d @ beta @ gamma @ beta.T @ isa_d) @ sa_d
        Lchol = sp_linalg.cholesky(H_inv, lower=False, check_finite=False)
        logdet = 2.0 * np.sum(np.log(np.abs(np.diag(Lchol))))
        mu = x - H_inv @ g
        u = rng.standard_normal((M, N))
        phi = (mu[:, None] + Lchol @ u.T).T  # (M, N)
    else:
        # Sparse path (large-N regime): economy QR avoids O(N²) matrices
        # overwrite_a=True avoids an extra copy; check_finite=False skips NaN scan.
        Q, R = sp_linalg.qr(
            beta * inv_sqrt_alpha[:, None], mode="economic", overwrite_a=True, check_finite=False
        )  # Q:(N,2J), R:(2J,2J)
        I2J = np.eye(J2) * (1.0 + REGULARISATION_TERM)
        Lchol = sp_linalg.cholesky(I2J + R @ gamma @ R.T, lower=False, check_finite=False)
        logdet = 2.0 * np.sum(np.log(np.abs(np.diag(Lchol)))) + np.sum(np.log(alpha))
        btg = beta.T @ g[:, None]  # (2J, 1)
        mu = x - ((alpha * g)[:, None] + beta @ (gamma @ btg))[:, 0]  # (N,)
        u = rng.standard_normal((M, N))
        QtU = Q.T @ u.T  # (2J, M)
        phi = (mu[:, None] + sqrt_alpha[:, None] * (Q @ ((Lchol - np.eye(J2)) @ QtU) + u.T)).T

    logQ = -0.5 * (logdet + np.sum(u * u, axis=-1) + N * np.log(2.0 * np.pi))
    return phi, logQ


class LogLike(Op):
    """
    Op that computes log-densities using a batched logp function.

    The stored ``logp_func`` must accept a 2-D array of shape ``(B, N)`` and
    return a 1-D array of shape ``(B,)``.  A single compiled call is made for
    the entire batch, replacing the previous per-sample ``np.apply_along_axis``
    loop and its associated Python-dispatch overhead.
    """

    __props__ = ("logp_func",)

    def __init__(self, logp_func: Callable):
        self.logp_func = logp_func
        super().__init__()

    def make_node(self, inputs):
        inputs = pt.as_tensor(inputs)
        outputs = pt.tensor(dtype="float64", shape=(None, None))
        return Apply(self, [inputs], [outputs])

    def perform(self, node: Apply, inputs, outputs) -> None:
        phi = inputs[0]  # (L, M, N)
        batch = phi.reshape(-1, phi.shape[-1])  # (L*M, N)
        logP_flat = np.asarray(self.logp_func(batch))  # (L*M,) — one compiled call
        logP = logP_flat.reshape(phi.shape[:-1])  # (L, M)
        mask = np.isnan(logP) | np.isinf(logP)
        if np.all(mask):
            raise PathInvalidLogP()
        outputs[0][0] = np.where(mask, -np.inf, logP)


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


def make_pathfinder_body(
    logp_func: Callable,
    num_draws: int,
    maxcor: int,
    num_elbo_draws: int,
    **compile_kwargs: Any,
) -> Function:
    """
    computes the inner components of the Pathfinder algorithm (post-LBFGS) using PyTensor variables and returns a compiled pytensor.function.

    Parameters
    ----------
    logp_func : Callable
        The target density function.
    num_draws : int
        Number of samples to draw from the single-path approximation.
    maxcor : int
        The maximum number of iterations for the L-BFGS algorithm.
    num_elbo_draws : int
        The number of draws for the Evidence Lower Bound (ELBO) estimation.
    compile_kwargs : dict
        Additional keyword arguments for the PyTensor compiler.

    Returns
    -------
    pathfinder_body_fn : Function
        A compiled pytensor.function that performs the inner components of the Pathfinder algorithm (post-LBFGS).

        pathfinder_body_fn inputs:
            x_full: (L+1, N),
            g_full: (L+1, N)
        pathfinder_body_fn outputs:
            psi: (1, M, N),
            logP_psi: (1, M),
            logQ_psi: (1, M),
            elbo_argmax: (1,)
    """

    # x_full, g_full: (L+1, N)
    x_full = pt.matrix("x", dtype="float64")
    g_full = pt.matrix("g", dtype="float64")

    num_draws = pt.constant(num_draws, "num_draws", dtype="int32")
    num_elbo_draws = pt.constant(num_elbo_draws, "num_elbo_draws", dtype="int32")
    maxcor = pt.constant(maxcor, "maxcor", dtype="int32")

    alpha, s, z = alpha_recover(x_full, g_full)
    beta, gamma = inverse_hessian_factors(alpha, s, z, J=maxcor)

    # ignore initial point - x, g: (L, N)
    x = x_full[1:]
    g = g_full[1:]

    phi, logQ_phi = bfgs_sample(
        num_samples=num_elbo_draws, x=x, g=g, alpha=alpha, beta=beta, gamma=gamma
    )

    loglike = LogLike(logp_func)
    logP_phi = loglike(phi)
    elbo = pt.mean(logP_phi - logQ_phi, axis=-1)
    elbo_argmax = pt.argmax(elbo, axis=0)

    # TODO: move the raise PathInvalidLogQ from single_pathfinder_fn to here to avoid computing logP_psi if logQ_psi is invalid. Possible setup: logQ_phi = PathCheck()(logQ_phi, ~pt.all(mask)), where PathCheck uses pytensor raise.

    # sample from the single-path approximation
    psi, logQ_psi = bfgs_sample(
        num_samples=num_draws,
        x=x,
        g=g,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        index=elbo_argmax,
    )
    logP_psi = loglike(psi)

    # return psi, logP_psi, logQ_psi, elbo_argmax

    pathfinder_body_fn = compile(
        [x_full, g_full],
        [psi, logP_psi, logQ_psi, elbo_argmax],
        **compile_kwargs,
    )
    pathfinder_body_fn.trust_input = True
    return pathfinder_body_fn


def make_elbo_fn(
    logp_func: Callable,
    maxcor: int,
    num_elbo_draws: int,
    **compile_kwargs: Any,
) -> Function:
    """
    Compile a function returning per-step ELBO values from LBFGS history.

    Parameters
    ----------
    logp_func : Callable
        The target log-density function.
    maxcor : int
        L-BFGS history size (number of variable metric corrections).
    num_elbo_draws : int
        Number of Monte Carlo draws for ELBO estimation per step.
    compile_kwargs : dict
        Additional keyword arguments for the PyTensor compiler.

    Returns
    -------
    elbo_fn : Function
        Compiled function: inputs (x_full, g_full) of shape (L+1, N),
        output elbo of shape (L,).
    """
    x_full = pt.matrix("x", dtype="float64")
    g_full = pt.matrix("g", dtype="float64")

    maxcor = pt.constant(maxcor, "maxcor", dtype="int32")
    num_elbo_draws = pt.constant(num_elbo_draws, "num_elbo_draws", dtype="int32")

    alpha, s, z = alpha_recover(x_full, g_full)
    beta, gamma = inverse_hessian_factors(alpha, s, z, J=maxcor)

    x = x_full[1:]
    g = g_full[1:]

    phi, logQ_phi = bfgs_sample(
        num_samples=num_elbo_draws, x=x, g=g, alpha=alpha, beta=beta, gamma=gamma
    )

    loglike = LogLike(logp_func)
    logP_phi = loglike(phi)
    elbo = pt.mean(logP_phi - logQ_phi, axis=-1)

    elbo_fn = compile([x_full, g_full], [elbo], **compile_kwargs)
    elbo_fn.trust_input = True
    return elbo_fn


def make_step_elbo_fn(
    logp_func: Callable,
    maxcor: int,
    num_elbo_draws: int,
    **compile_kwargs: Any,
) -> Function:
    """Compile a single-step ELBO function for use inside the streaming LBFGS callback.

    Parameters
    ----------
    logp_func : Callable
        The target log-density function.
    maxcor : int
        L-BFGS history size (J).
    num_elbo_draws : int
        Number of Monte Carlo draws for ELBO estimation.
    compile_kwargs : dict
        Additional keyword arguments for the PyTensor compiler.

    Returns
    -------
    Function
        Compiled function: inputs (alpha_l, s_win, z_win, x_l, g_l), output scalar ELBO.
        alpha_l: (N,), s_win: (N, J), z_win: (N, J), x_l: (N,), g_l: (N,)
    """
    J = pt.constant(maxcor, "maxcor", dtype="int32")
    M = pt.constant(num_elbo_draws, "num_elbo_draws", dtype="int32")

    alpha_l = pt.vector("alpha_l", dtype="float64")  # (N,)
    s_win = pt.matrix("s_win", dtype="float64")  # (N, J)
    z_win = pt.matrix("z_win", dtype="float64")  # (N, J)
    x_l = pt.vector("x_l", dtype="float64")  # (N,)
    g_l = pt.vector("g_l", dtype="float64")  # (N,)

    # Add batch dim: (1, N, J) — no transpose needed since s_win is already (N, J)
    S_l = s_win[None, :, :]
    Z_l = z_win[None, :, :]
    beta_l, gamma_l = inverse_hessian_factors_from_SZ(alpha_l[None], S_l, Z_l, J)

    phi, logQ = bfgs_sample(M, x_l[None], g_l[None], alpha_l[None], beta_l, gamma_l)
    logP = LogLike(logp_func)(phi)
    elbo = pt.mean(logP - logQ)

    fn = compile([alpha_l, s_win, z_win, x_l, g_l], [elbo], **compile_kwargs)
    fn.trust_input = True
    return fn


def make_step_sample_fn(
    logp_func: Callable,
    num_draws: int,
    maxcor: int,
    **compile_kwargs: Any,
) -> Function:
    """Compile a single-step sample function for drawing from the best local approximation.

    Parameters
    ----------
    logp_func : Callable
        The target log-density function.
    num_draws : int
        Number of samples to draw.
    maxcor : int
        L-BFGS history size (J).
    compile_kwargs : dict
        Additional keyword arguments for the PyTensor compiler.

    Returns
    -------
    Function
        Compiled function: inputs (alpha_l, s_win, z_win, x_l, g_l),
        outputs (psi, logP_psi, logQ_psi) of shapes (1, M, N), (1, M), (1, M).
    """
    J = pt.constant(maxcor, "maxcor", dtype="int32")
    M = pt.constant(num_draws, "num_draws", dtype="int32")

    alpha_l = pt.vector("alpha_l", dtype="float64")
    s_win = pt.matrix("s_win", dtype="float64")
    z_win = pt.matrix("z_win", dtype="float64")
    x_l = pt.vector("x_l", dtype="float64")
    g_l = pt.vector("g_l", dtype="float64")

    S_l = s_win[None, :, :]
    Z_l = z_win[None, :, :]
    beta_l, gamma_l = inverse_hessian_factors_from_SZ(alpha_l[None], S_l, Z_l, J)

    psi, logQ_psi = bfgs_sample(M, x_l[None], g_l[None], alpha_l[None], beta_l, gamma_l)
    logP_psi = LogLike(logp_func)(psi)

    fn = compile([alpha_l, s_win, z_win, x_l, g_l], [psi, logP_psi, logQ_psi], **compile_kwargs)
    fn.trust_input = True
    return fn


class LBFGSStreamingCallback:
    """Streaming LBFGS callback: computes ELBO at each accepted step, O(J*N + M*N) peak memory.

    Replaces LBFGSHistoryManager when streaming=True. Instead of collecting the full
    (L+1, N) history, it processes each accepted step immediately and tracks only the
    best state seen so far.

    Parameters
    ----------
    value_grad_fn : Callable
        Single-entry cached value/gradient function (wrap with _CachedValueGrad).
    x0 : NDArray
        Initial position, shape (N,).
    step_elbo_fn : Callable
        Compiled ELBO function from make_step_elbo_fn.
    J : int
        L-BFGS history size (maxcor).
    epsilon : float
        Tolerance for the LBFGS update condition (same as in LBFGSHistoryManager).
    """

    def __init__(
        self,
        value_grad_fn: Callable,
        x0: NDArray,
        step_elbo_fn: Callable,
        J: int,
        epsilon: float,
    ) -> None:
        self.value_grad_fn = value_grad_fn
        self.step_elbo_fn = step_elbo_fn
        self.J = J
        self.epsilon = epsilon

        N = x0.shape[0]
        _, g0 = value_grad_fn(x0)

        self.x_prev: NDArray = x0.copy()
        self.g_prev: NDArray = np.array(g0, dtype=np.float64)
        self.alpha_prev: NDArray = np.ones(N, dtype=np.float64)
        self.s_win: NDArray = np.zeros((N, J), dtype=np.float64)  # (N, J) ring buffer
        self.z_win: NDArray = np.zeros((N, J), dtype=np.float64)
        self.win_idx: int = -1
        self.best_elbo: float = -np.inf
        self.best_state: dict = {}
        self.best_step_idx: int = 0
        self.step_count: int = 0
        self.any_valid: bool = False

    def __call__(self, x: NDArray) -> None:
        # Step 1: get (value, grad) — free if _CachedValueGrad wrapper is used
        value, g = self.value_grad_fn(x)

        # Step 2: compute diffs before entry check (shared computation)
        s = x - self.x_prev
        z = g - self.g_prev

        # Step 3: entry condition (same logic as LBFGSHistoryManager for non-initial steps)
        if not (np.all(np.isfinite(g)) and np.isfinite(value)):
            return
        if not _check_lbfgs_curvature_condition(s, z, self.epsilon):
            return

        # Step 4: alpha update (pure numpy, O(N))
        alpha = alpha_step_numpy(self.alpha_prev, s, z)

        # Step 5: ring-buffer column write — O(N), no allocation
        self.win_idx = (self.win_idx + 1) % self.J
        self.s_win[:, self.win_idx] = s
        self.z_win[:, self.win_idx] = z

        # Step 6: ELBO computation — catch PathInvalidLogP (B9)
        try:
            (elbo,) = self.step_elbo_fn(alpha, self.s_win, self.z_win, x, g)
            elbo = float(elbo)
        except PathInvalidLogP:
            elbo = -np.inf

        # Step 7: validity tracking
        if np.isfinite(elbo):
            self.any_valid = True

        # Step 8: update best state
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

        # Step 9: advance state
        self.alpha_prev = alpha
        self.x_prev = x.copy()
        self.g_prev = g.copy()
        self.step_count += 1


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
    streaming: bool = True,
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
        value used to filter out large changes in the direction of the update gradient at each iteration l in L. Iteration l is only accepted if delta_theta[l] * delta_grad[l] > epsilon * L2_norm(delta_grad[l]) for each l in L.
    streaming : bool, optional
        Whether to use streaming (per-step) LBFGS processing for O(J*N + M*N) peak memory,
        by default True. Set to False to fall back to full-history batch processing.
        Streaming processes each accepted LBFGS step immediately; for large maxiter this
        incurs Python dispatch overhead (one compiled call per step) vs the non-streaming
        vectorised graph over all L steps at once.
    pathfinder_kwargs : dict
        Additional keyword arguments for the Pathfinder algorithm.
    compile_kwargs : dict
        Additional keyword arguments for the PyTensor compiler. If not provided, the default linker is "cvm_nogc".

    Returns
    -------
    single_pathfinder_fn : Callable
        A seedable single-path pathfinder function.
    """

    compile_kwargs = {"mode": Mode(linker=DEFAULT_LINKER), **compile_kwargs}
    logp_dlogp_kwargs = {"jacobian": pathfinder_kwargs.get("jacobian", True), **compile_kwargs}

    logp_dlogp_func = get_logp_dlogp_of_ravel_inputs(model, **logp_dlogp_kwargs)
    batched_logp_func = get_batched_logp_of_ravel_inputs(model, **logp_dlogp_kwargs)

    def neg_logp_dlogp_func(x):
        logp, dlogp = logp_dlogp_func(x)
        return -logp, -dlogp

    # initial point
    # TODO: remove make_initial_points function when feature request is implemented: https://github.com/pymc-devs/pymc/issues/7555
    ipfn = make_initial_point_fn(model=model)
    ip = Point(ipfn(None), model=model)
    x_base = DictToArrayBijection.map(ip).data

    # lbfgs
    LBFGS(neg_logp_dlogp_func, maxcor, maxiter, ftol, gtol, maxls, epsilon)

    if streaming:
        # Streaming ELBO/sampling is done in pure NumPy — no PyTensor compilation
        # needed.  _bfgs_sample_numpy handles both dense and sparse paths and
        # delegates logP evaluation to the already-compiled batched_logp_func.
        pass
    else:
        # Compile non-streaming pathfinder body
        pathfinder_body_fn = make_pathfinder_body(
            batched_logp_func, num_draws, maxcor, num_elbo_draws, **compile_kwargs
        )
        find_rng_nodes(pathfinder_body_fn.maker.fgraph.outputs)

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

    def single_pathfinder_fn(random_seed: int) -> PathfinderResult:
        # Per-path independent copies of compiled functions.
        # PyTensor Function objects share input/output storage on the same
        # instance: concurrent calls from different threads corrupt each other.
        # fn.copy(share_memory=False) gives independent storage that reuses
        # the already-compiled C shared library — no recompilation cost.
        local_logp_dlogp = logp_dlogp_func.copy(share_memory=False)

        def local_neg_logp_dlogp_func(x):
            logp, dlogp = local_logp_dlogp(x)
            return -logp, -dlogp

        local_lbfgs = LBFGS(local_neg_logp_dlogp_func, maxcor, maxiter, ftol, gtol, maxls, epsilon)

        if streaming:
            local_batched_logp = batched_logp_func.copy(share_memory=False)
        else:
            local_body_fn = pathfinder_body_fn.copy(share_memory=False)
            path_rngs = find_rng_nodes(local_body_fn.maker.fgraph.outputs)

        lbfgs_status = LBFGSStatus.LBFGS_FAILED  # default before LBFGS runs
        try:
            init_seed, elbo_seed, final_seed = _get_seeds_per_chain(random_seed, 3)
            rng = np.random.default_rng(init_seed)
            jitter_value = rng.uniform(-jitter, jitter, size=x_base.shape)
            x0 = x_base + jitter_value

            if streaming:
                elbo_rng = np.random.default_rng(elbo_seed)

                def _numpy_step_elbo(alpha, s_win, z_win, x, g):
                    try:
                        phi, logQ = _bfgs_sample_numpy(
                            x, g, alpha, s_win, z_win, num_elbo_draws, elbo_rng
                        )
                    except np.linalg.LinAlgError:
                        raise PathInvalidLogP()
                    logP = np.asarray(local_batched_logp(phi))
                    finite = np.isfinite(logP)
                    if not np.any(finite):
                        raise PathInvalidLogP()
                    logP = np.where(finite, logP, -np.inf)
                    return (float(np.mean(logP - logQ)),)

                cached_fn = _CachedValueGrad(local_neg_logp_dlogp_func)
                streaming_cb = LBFGSStreamingCallback(
                    value_grad_fn=cached_fn,
                    x0=x0,
                    step_elbo_fn=_numpy_step_elbo,
                    J=maxcor,
                    epsilon=epsilon,
                )

                lbfgs_niter, lbfgs_status = local_lbfgs.minimize_streaming(streaming_cb, x0)
                _check_lbfgs_status(lbfgs_status)

                if not streaming_cb.any_valid:
                    raise PathInvalidLogP()

                elbo_argmax = streaming_cb.best_step_idx
                best_state = streaming_cb.best_state

                final_rng = np.random.default_rng(final_seed)
                try:
                    phi_final, logQ_psi_flat = _bfgs_sample_numpy(
                        best_state["x"],
                        best_state["g"],
                        best_state["alpha"],
                        best_state["s_win"],
                        best_state["z_win"],
                        num_draws,
                        final_rng,
                    )
                except np.linalg.LinAlgError:
                    raise PathInvalidLogP()
                logP_psi_flat = np.asarray(local_batched_logp(phi_final))
                psi = phi_final[None]  # (1, M, N)
                logP_psi = logP_psi_flat[None]  # (1, M)
                logQ_psi = logQ_psi_flat[None]  # (1, M)

            else:
                x, g, lbfgs_niter, lbfgs_status = local_lbfgs.minimize(x0)
                _check_lbfgs_status(lbfgs_status)

                reseed_rngs(path_rngs, [elbo_seed, final_seed])
                psi, logP_psi, logQ_psi, elbo_argmax = local_body_fn(x, g)

            return _make_result(psi, logP_psi, logQ_psi, lbfgs_niter, elbo_argmax, lbfgs_status)

        except LBFGSException as e:
            return PathfinderResult(
                lbfgs_status=e.status,
                path_status=PathStatus.LBFGS_FAILED,
            )
        except PathException as e:
            return PathfinderResult(
                lbfgs_status=lbfgs_status,
                path_status=e.status,
            )

    return single_pathfinder_fn


def _calculate_max_workers() -> int:
    """
    calculate the default number of workers to use for concurrent pathfinder runs.
    """

    # from limited testing, setting values higher than 0.3 makes multiprocessing a lot slower.
    import multiprocessing

    total_cpus = multiprocessing.cpu_count() or 1
    processes = max(2, int(total_cpus * 0.3))
    if processes % 2 != 0:
        processes += 1
    return processes


def _thread(fn: SinglePathfinderFn, seed: int) -> "PathfinderResult":
    """
    execute pathfinder runs concurrently using threading.
    """

    # kernel crashes without lock_ctx
    from pytensor.compile.compilelock import lock_ctx

    with lock_ctx():
        rng = np.random.default_rng(seed)
        result = fn(rng)
    return result


def _process(fn: SinglePathfinderFn, seed: int) -> "PathfinderResult | bytes":
    """
    execute pathfinder runs concurrently using multiprocessing.
    """
    import cloudpickle

    from pytensor.compile.compilelock import lock_ctx

    with lock_ctx():
        in_out_pickled = isinstance(fn, bytes)
        fn = cloudpickle.loads(fn)
        rng = np.random.default_rng(seed)
        result = fn(rng) if not in_out_pickled else cloudpickle.dumps(fn(rng))
    return result


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
    concurrent: Literal["thread", "process"] | None,
    max_workers: int | None = None,
) -> Iterator["PathfinderResult | bytes"]:
    """
    execute pathfinder runs concurrently.
    """
    if concurrent == "thread":
        from concurrent.futures import ThreadPoolExecutor, as_completed
    elif concurrent == "process":
        from concurrent.futures import ProcessPoolExecutor, as_completed

        import cloudpickle
    else:
        raise ValueError(f"Invalid concurrent value: {concurrent}")

    executor_cls = ThreadPoolExecutor if concurrent == "thread" else ProcessPoolExecutor

    concurrent_fn = _thread if concurrent == "thread" else _process

    executor_kwargs = {} if concurrent == "thread" else {"mp_context": _get_mp_context()}

    max_workers = max_workers or (None if concurrent == "thread" else _calculate_max_workers())

    fn = fn if concurrent == "thread" else cloudpickle.dumps(fn)

    with executor_cls(max_workers=max_workers, **executor_kwargs) as executor:
        futures = [executor.submit(concurrent_fn, fn, seed) for seed in seeds]
        for f in as_completed(futures):
            yield (f.result() if concurrent == "thread" else cloudpickle.loads(f.result()))


def _execute_serially(fn: SinglePathfinderFn, seeds: list[int]) -> Iterator["PathfinderResult"]:
    """
    execute pathfinder runs serially.
    """
    for seed in seeds:
        rng = np.random.default_rng(seed)
        yield fn(rng)


def make_generator(
    concurrent: Literal["thread", "process"] | None,
    fn: SinglePathfinderFn,
    seeds: list[int],
    max_workers: int | None = None,
) -> Iterator["PathfinderResult | bytes"]:
    """
    generator for executing pathfinder runs concurrently or serially.
    """
    if concurrent is not None:
        yield from _execute_concurrently(fn, seeds, concurrent, max_workers)
    else:
        yield from _execute_serially(fn, seeds)


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

        # if not success_results:
        #     raise ValueError(
        #         "All paths failed. Consider decreasing the jitter or reparameterizing the model."
        #     )

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
    concurrent: Literal["thread", "process"] | None,
    random_seed: RandomSeed,
    streaming: bool = True,
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
        value used to filter out large changes in the direction of the update gradient at each iteration l in L. Iteration l is only accepted if delta_theta[l] * delta_grad[l] > epsilon * L2_norm(delta_grad[l]) for each l in L. (default is 1e-8).
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
        Whether to run paths concurrently, either "thread" or "process" or None (default is None). Setting concurrent to None runs paths serially and is generally faster with smaller models because of the overhead that comes with concurrency.
    pathfinder_kwargs
        Additional keyword arguments for the Pathfinder algorithm.
    compile_kwargs
        Additional keyword arguments for the PyTensor compiler. If not provided, the default linker is "cvm_nogc".

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
        streaming=streaming,
        pathfinder_kwargs=pathfinder_kwargs,
        compile_kwargs=compile_kwargs,
    )
    compile_end = time.time()

    # NOTE: from limited tests, no concurrency is faster than thread, and thread is faster than process. But I suspect this also depends on the model size and maxcor setting.
    generator = make_generator(
        concurrent=concurrent,
        fn=single_pathfinder_fn,
        seeds=path_seeds,
    )

    results = []
    compute_start = time.time()
    try:
        desc = f"Paths Complete: {{path_idx}}/{num_paths}"
        progress = CustomProgress(
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeRemainingColumn(),
            TextColumn("/"),
            TimeElapsedColumn(),
            console=Console(theme=default_progress_theme),
            disable=not progressbar,
        )
        with progress:
            task = progress.add_task(desc.format(path_idx=0), completed=0, total=num_paths)
            for path_idx, result in enumerate(generator, start=1):
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
                finally:
                    # TODO: display LBFGS and Path Status in real time
                    progress.update(
                        task,
                        description=desc.format(path_idx=path_idx),
                        completed=path_idx,
                    )
            # Ensure the progress bar visually reaches 100% and shows 'Completed'
            progress.update(task, completed=num_paths, description="Completed")
    except (KeyboardInterrupt, StopIteration) as e:
        # if exception is raised here, MultiPathfinderResult will collect all the successful results and report the results. User is free to abort the process earlier and the results will still be collected and return az.InferenceData.
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
    epsilon: float = 1e-8,
    importance_sampling: Literal["psis", "psir", "identity"] | None = "psis",
    progressbar: bool = True,
    concurrent: Literal["thread", "process"] | None = None,
    streaming: bool = True,
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
        value used to filter out large changes in the direction of the update gradient at each iteration l in L. Iteration l is only accepted if delta_theta[l] * delta_grad[l] > epsilon * L2_norm(delta_grad[l]) for each l in L. (default is 1e-8).
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
        Whether to run paths concurrently, either "thread" or "process" or None (default is None). Setting concurrent to None runs paths serially and is generally faster with smaller models because of the overhead that comes with concurrency.
    pathfinder_kwargs
        Additional keyword arguments for the Pathfinder algorithm.
    compile_kwargs
        Additional keyword arguments for the PyTensor compiler. If not provided, the default linker is "cvm_nogc".
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
            streaming=streaming,
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
