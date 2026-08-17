import logging

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
from pymc_extras.statespace.utils.constants import ALL_STATE_DIM, SHOCK_DIM, TIME_DIM

if TYPE_CHECKING:
    from pymc_extras.statespace.core.statespace import PyMCStateSpace

_log = logging.getLogger("pymc.experimental.statespace")


def impulse_response_function(
    ss_mod: "PyMCStateSpace",
    idata,
    n_steps: int = 40,
    use_posterior_cov: bool = True,
    shock_size: float | np.ndarray | None = None,
    shock_cov: np.ndarray | None = None,
    shock_trajectory: np.ndarray | None = None,
    orthogonalize_shocks: bool = False,
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

    if shock_trajectory is not None:
        # Validate the shock trajectory
        n, k = shock_trajectory.shape
        steps = n

        if k != ss_mod.k_posdef:
            raise ValueError(
                "If shock_trajectory is provided, there must be a trajectory provided for each shock. "
                f"Model has {ss_mod.k_posdef} shocks, but shock_trajectory has only {k} columns"
            )
        if steps is not None and steps != n:
            _log.warning(
                "Both steps and shock_trajectory were provided but do not agree. Length of "
                "shock_trajectory will take priority, and steps will be ignored."
            )
        n_steps = n  # Overwrite steps with the length of the shock trajectory
        shock_trajectory = pt.as_tensor_variable(shock_trajectory)

    fit_coords = coords_from_idata(ss_mod, idata, "observed_data")
    simulation_coords = fit_coords.copy()
    simulation_coords[TIME_DIM] = np.arange(n_steps, dtype="int")

    with pm.Model(coords=simulation_coords):
        dummy_graph.build_dummy_graph(
            ss_mod, coords=fit_coords, dims=dims_from_idata(ss_mod, idata, group)
        )
        matrices = ss_mod._insert_random_variables()

        matrices = ss_mod._insert_constant_timestep(matrices, step=n_steps)
        P0, _, c, d, T, Z, R, H, post_Q = matrices
        x0 = pm.Deterministic("x0_new", pt.zeros(ss_mod.k_states), dims=[ALL_STATE_DIM])

        if use_posterior_cov:
            Q = post_Q
            if orthogonalize_shocks:
                Q = pt.linalg.cholesky(Q) / pt.diag(Q)
        elif shock_cov is not None:
            Q = pt.as_tensor_variable(shock_cov)
            if orthogonalize_shocks:
                Q = pt.linalg.cholesky(Q) / pt.diag(Q)

        if shock_trajectory is None:
            shock_trajectory = pt.zeros((n_steps, ss_mod.k_posdef))
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
            shock_trajectory = pt.set_subtensor(shock_trajectory[0], init_shock)

        else:
            shock_trajectory = pt.as_tensor_variable(shock_trajectory)

        time_varying_T = "transition" in ss_mod.ssm.time_varying_names

        def irf_step(*args):
            if time_varying_T:
                shock, T, x, c, R = args
            else:
                shock, x, c, T, R = args

            next_x = c + T @ x + R @ shock
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

        pm.Deterministic("irf", irf, dims=[TIME_DIM, ALL_STATE_DIM])

        irf_idata = pm.sample_posterior_predictive(
            idata[group],
            var_names=["irf"],
            random_seed=random_seed,
            compile_kwargs=compile_kwargs,
            **kwargs,
        )

        return irf_idata["posterior_predictive"]
