import logging

from types import EllipsisType
from typing import TYPE_CHECKING, Literal

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pymc.util import RandomState

from pymc_extras.statespace.core import dummy_graph
from pymc_extras.statespace.core.fit_recovery import (
    coords_from_idata,
    dims_from_idata,
    verify_group,
)
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_DIM,
    SHOCK_DIM,
    STRUCTURAL_SHOCK_DIM,
    TIME_DIM,
)

if TYPE_CHECKING:
    from pymc_extras.statespace.core.statespace import PyMCStateSpace

_log = logging.getLogger("pymc.experimental.statespace")

DEFAULT_IRF_STEPS = 40


def _shock_permutation(
    shock_names: list[str], shock_order: list[str | EllipsisType] | None
) -> np.ndarray:
    """
    Map a user-supplied recursive ordering onto positions in the model's shock dimension.

    Parameters
    ----------
    shock_names : list of str
        Shock names in the order the model stores them.
    shock_order : list of str or ellipsis, optional
        The recursive ordering to impose. Entries name shocks; a single ``...`` stands for every
        shock not named elsewhere, kept in the model's own order. Without an ``...``, every shock
        must be named. Defaults to ``shock_names`` itself.

    Returns
    -------
    ndarray of int
        Indices ``perm`` such that ``[shock_names[i] for i in perm]`` is the resolved ordering.
    """
    if shock_order is None:
        return np.arange(len(shock_names), dtype="int")

    named = [name for name in shock_order if name is not Ellipsis]
    n_ellipsis = len(shock_order) - len(named)

    if n_ellipsis > 1:
        raise ValueError(
            f"shock_order may contain at most one `...`, but {n_ellipsis} were given. One `...` "
            "already stands for every shock not named elsewhere."
        )

    unknown = [name for name in named if name not in shock_names]
    if unknown:
        raise ValueError(
            f"shock_order names shocks the model does not have: {unknown}. "
            f"Model shocks are {shock_names}."
        )

    duplicated = sorted({name for name in named if named.count(name) > 1})
    if duplicated:
        raise ValueError(f"shock_order names the same shock more than once: {duplicated}.")

    rest = [name for name in shock_names if name not in named]

    if n_ellipsis:
        fill = shock_order.index(Ellipsis)
        resolved = named[:fill] + rest + named[fill:]
    elif rest:
        raise ValueError(
            "shock_order must name every shock in the model, or use `...` to stand for the "
            f"rest. Missing: {rest}."
        )
    else:
        resolved = named

    return np.array([shock_names.index(name) for name in resolved], dtype="int")


def _orthogonal_impulse_matrix(Q: pt.TensorVariable, perm: np.ndarray) -> pt.TensorVariable:
    r"""
    Build the impact matrix of a recursive (Cholesky) identification scheme.

    Factor the shock covariance as :math:`Q = B B^\top` with :math:`B` triangular *in the ordering
    given by* ``perm``, so that the reduced-form shocks satisfy :math:`u = B \varepsilon` for
    orthonormal structural shocks :math:`\varepsilon`. Column :math:`j` of :math:`B` is the
    contemporaneous impact of a one-standard-deviation innovation to the :math:`j`-th shock in the
    chosen ordering.

    Parameters
    ----------
    Q : TensorVariable
        Shock covariance matrix, with rows and columns in the model's own shock order.
    perm : ndarray of int
        Recursive ordering, as returned by :func:`_shock_permutation`.

    Returns
    -------
    TensorVariable
        Impact matrix with rows in the model's shock order and columns in ``perm`` order.
    """
    Q_ordered = Q[perm][:, perm]
    L = pt.linalg.cholesky(Q_ordered)

    # Undo the row permutation so rows line up with the model's shock order again; columns stay in
    # the user's ordering, since they index structural shocks rather than model shocks.
    inverse_perm = np.argsort(perm)
    return L[inverse_perm]


def impulse_response_function(
    ss_mod: "PyMCStateSpace",
    idata,
    n_steps: int | None = None,
    use_posterior_cov: bool = True,
    shock_size: float | np.ndarray | None = None,
    shock_cov: np.ndarray | None = None,
    shock_trajectory: np.ndarray | None = None,
    orthogonalize_shocks: bool = False,
    shock_order: list[str | EllipsisType] | None = None,
    random_seed: RandomState | None = None,
    mvn_method: Literal["cholesky", "eigh", "svd"] = "svd",
    group: str = "posterior",
    **kwargs,
):
    verify_group(group)
    options = [shock_size, shock_cov, shock_trajectory]
    n_options = sum(x is not None for x in options)
    Q = None  # No covariance matrix needed if a trajectory is provided. Will be overwritten later if needed.

    compile_kwargs = kwargs.pop("compile_kwargs", {})
    compile_kwargs.setdefault("mode", ss_mod.mode)

    if n_options > 1:
        raise ValueError("Specify exactly 0 or 1 of shock_size, shock_cov, or shock_trajectory")
    elif n_options == 1:
        # If the user passed an alternative parameterization for the shocks of the IRF, don't use the posterior
        use_posterior_cov = False

    if orthogonalize_shocks and (shock_size is not None or shock_trajectory is not None):
        raise ValueError(
            "orthogonalize_shocks=True identifies shocks from a covariance matrix, so it cannot be "
            "combined with shock_size or shock_trajectory. Pass shock_cov, or use the posterior "
            "covariance."
        )
    if shock_order is not None and not orthogonalize_shocks:
        raise ValueError("shock_order is only meaningful when orthogonalize_shocks=True.")

    if shock_trajectory is not None:
        # Validate the shock trajectory
        trajectory_steps, k = shock_trajectory.shape

        if k != ss_mod.k_posdef:
            raise ValueError(
                "If shock_trajectory is provided, there must be a trajectory provided for each shock. "
                f"Model has {ss_mod.k_posdef} shocks, but shock_trajectory has only {k} columns"
            )
        if n_steps is not None and n_steps != trajectory_steps:
            _log.warning(
                "Both n_steps and shock_trajectory were provided but do not agree. Length of "
                "shock_trajectory will take priority, and n_steps will be ignored."
            )
        n_steps = trajectory_steps
        shock_trajectory = pt.as_tensor_variable(shock_trajectory)

    elif n_steps is None:
        n_steps = DEFAULT_IRF_STEPS

    fit_coords = coords_from_idata(ss_mod, idata, "observed_data")
    fit_dims = dims_from_idata(ss_mod, idata, group)
    simulation_coords = fit_coords.copy()
    simulation_coords[TIME_DIM] = np.arange(n_steps, dtype="int")

    if orthogonalize_shocks:
        shock_names = list(fit_coords[SHOCK_DIM])
        perm = _shock_permutation(shock_names, shock_order)
        simulation_coords[STRUCTURAL_SHOCK_DIM] = [shock_names[i] for i in perm]
        irf_dims = [STRUCTURAL_SHOCK_DIM, TIME_DIM, ALL_STATE_DIM]
    else:
        irf_dims = [TIME_DIM, ALL_STATE_DIM]

    with pm.Model(coords=simulation_coords):
        dummy_graph.build_dummy_graph(ss_mod, coords=fit_coords, dims=fit_dims)
        matrices = ss_mod._insert_random_variables()

        matrices = ss_mod._insert_constant_timestep(matrices, step=n_steps)
        P0, _, c, d, T, Z, R, H, post_Q = matrices

        if use_posterior_cov:
            Q = post_Q
        elif shock_cov is not None:
            Q = pt.as_tensor_variable(shock_cov)

        # The scan carries one column per impulse, so orthogonalized IRFs trace every structural
        # shock through its own copy of the system in a single pass.
        n_impulse = ss_mod.k_posdef if orthogonalize_shocks else 1
        empty_trajectory = pt.zeros((n_steps, ss_mod.k_posdef, n_impulse))

        if orthogonalize_shocks:
            if Q is None:
                raise ValueError(
                    "orthogonalize_shocks=True needs a shock covariance matrix to factor. Pass "
                    "shock_cov, or leave use_posterior_cov=True to take it from the posterior."
                )

            # Column j of the impact matrix is the impulse that isolates structural shock j.
            impulse = _orthogonal_impulse_matrix(Q, perm)
            shock_trajectory = pt.set_subtensor(empty_trajectory[0], impulse)

        elif shock_trajectory is None:
            if Q is not None:
                init_shock = pm.MvNormal(
                    "initial_shock", mu=0, cov=Q, dims=[SHOCK_DIM], method=mvn_method
                )
            else:
                init_shock = pm.Deterministic(
                    "initial_shock",
                    pt.as_tensor_variable(np.atleast_1d(shock_size)),
                    dims=[SHOCK_DIM],
                )
            shock_trajectory = pt.set_subtensor(empty_trajectory[0, :, 0], init_shock)

        else:
            shock_trajectory = pt.as_tensor_variable(shock_trajectory)[..., None]

        x0 = pt.zeros((ss_mod.k_states, n_impulse))
        time_varying_T = "transition" in ss_mod.ssm.time_varying_names

        def irf_step(*args):
            if time_varying_T:
                shock, T, x, c, R = args
            else:
                shock, x, c, T, R = args

            next_x = pt.expand_dims(c, -1) + T @ x + R @ shock
            return next_x

        sequences = [shock_trajectory, T] if time_varying_T else [shock_trajectory]
        non_sequences = [c, R] if time_varying_T else [c, T, R]

        irf = pytensor.scan(
            irf_step,
            sequences=sequences,
            outputs_info=[x0],
            non_sequences=non_sequences,
            n_steps=n_steps,
            strict=True,
            return_updates=False,
        )

        # Scan stacks over time, giving (time, state, impulse).
        irf = irf.transpose(2, 0, 1) if orthogonalize_shocks else irf[..., 0]

        pm.Deterministic("irf", irf, dims=irf_dims)

        irf_idata = pm.sample_posterior_predictive(
            idata[group],
            var_names=["irf"],
            random_seed=random_seed,
            compile_kwargs=compile_kwargs,
            **kwargs,
        )

        return irf_idata["posterior_predictive"]
