from typing import TYPE_CHECKING, Literal

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pymc.model.transform.optimization import freeze_dims_and_data
from pymc.util import RandomState
from xarray import DataTree

from pymc_extras.statespace.core import dummy_graph
from pymc_extras.statespace.core.fit_recovery import (
    coords_from_idata,
    data_from_idata,
    dims_from_idata,
    exog_from_idata,
    verify_group,
)
from pymc_extras.statespace.filters.distributions import (
    LinearGaussianStateSpace,
    SequenceMvNormal,
    SimulationSmoother,
)
from pymc_extras.statespace.filters.utilities import stabilize
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_DIM,
    FILTER_OUTPUT_DIMS,
    FILTER_OUTPUT_TYPES,
    LONG_MATRIX_NAMES,
    MATRIX_DIMS,
    MATRIX_NAMES,
    OBS_STATE_DIM,
    SHORT_NAME_TO_LONG,
    TIME_DIM,
)
from pymc_extras.statespace.utils.data_tools import register_data_with_pymc

if TYPE_CHECKING:
    from pymc_extras.statespace.core.statespace import PyMCStateSpace


def _sample_conditional(
    ss_mod: "PyMCStateSpace",
    idata: DataTree,
    group: str,
    random_seed: RandomState | None = None,
    data: pt.TensorLike | None = None,
    mvn_method: Literal["cholesky", "eigh", "svd"] = "svd",
    **kwargs,
):
    """
    Common functionality shared between `sample_conditional_prior` and `sample_conditional_posterior`. See those
    methods for details.

    Parameters
    ----------
    idata : DataTree
        A DataTree object containing the posterior distribution over model parameters.

    group : str
        DataTree group from which to draw samples. Should be one of "prior" or "posterior".

    random_seed : int, RandomState or Generator, optional
        Seed for the random number generator.

    data: pt.TensorLike, optional
        Observed data on which to condition the model. If not provided, the function will use the data that was
        provided when the model was built.

    mvn_method: str, default "svd"
        Method used to invert the covariance matrix when calculating the pdf of a multivariate normal
        (or when generating samples). One of "cholesky", "eigh", or "svd". "cholesky" is fastest, but least robust
        to ill-conditioned matrices, while "svd" is slow but extremely robust.

        In general, if your model has measurement error, "cholesky" will be safe to use. Otherwise, "svd" is
        recommended. "eigh" can also be tried if sampling with "svd" is very slow, but it is not as robust as "svd".

    kwargs:
        Additional keyword arguments are passed to pymc.sample_posterior_predictive

    Returns
    -------
    DataTree
        A DataTree object containing sampled trajectories from the requested conditional distribution,
        with data variables "filtered_{group}", "predicted_{group}", and "smoothed_{group}".
    """
    verify_group(group)
    group_idata = getattr(idata, group)

    compile_kwargs = kwargs.pop("compile_kwargs", {})
    compile_kwargs.setdefault("mode", ss_mod.mode)

    fit_coords = coords_from_idata(ss_mod, idata, "observed_data")
    fit_dims = dims_from_idata(ss_mod, idata, group)
    if data is None:
        data = data_from_idata(idata, "constant_data")

    with pm.Model(coords=fit_coords) as forward_model:
        matrices, grouped_outputs = dummy_graph.kalman_filter_outputs_from_dummy_graph(
            ss_mod,
            data,
            coords=fit_coords,
            dims=fit_dims,
            exog=exog_from_idata(ss_mod, idata, "constant_data"),
        )
        # The returned matrices keep the n_timesteps placeholder; pin it to the span the filter
        # actually ran over so the observed-state graphs line up with grouped_outputs.
        n_timesteps = grouped_outputs[0][0].shape[0]
        x0, P0, c, d, T, Z, R, H, Q = ss_mod._insert_constant_timestep(matrices, n_timesteps)

        state_dims = (
            (TIME_DIM, ALL_STATE_DIM)
            if all([dim in fit_coords for dim in [TIME_DIM, ALL_STATE_DIM]])
            else (None, None)
        )
        obs_dims = (
            (TIME_DIM, OBS_STATE_DIM)
            if all([dim in fit_coords for dim in [TIME_DIM, OBS_STATE_DIM]])
            else (None, None)
        )

        for name, (mu, cov) in zip(FILTER_OUTPUT_TYPES, grouped_outputs, strict=True):
            if name == "smoothed":
                # The simulation smoother draws the whole latent path jointly, so the
                # states carry their cross-time posterior covariance.
                kalman_filter, kalman_smoother = ss_mod.make_filters()
                latent_states = SimulationSmoother(
                    f"{name}_{group}",
                    a_smooth=mu,
                    x0=x0,
                    P0=P0,
                    c=c,
                    d=d,
                    T=T,
                    Z=Z,
                    R=R,
                    H=H,
                    Q=Q,
                    kalman_filter=kalman_filter,
                    kalman_smoother=kalman_smoother,
                    dims=state_dims,
                    method=mvn_method,
                )
                # Conditional on a joint draw of the latent path, the observation noise
                # is iid, so H alone is the correct per-step covariance. Adding the
                # zeros materializes the time axis, which keeps SequenceMvNormal's
                # ``(n),(n,n)->(n)`` gufunc from collapsing to a length-1 scan when H is
                # rank-3 but time-broadcastable.
                obs_mu = d + (Z @ latent_states[..., None]).squeeze(-1)
                obs_cov = pt.zeros((obs_mu.shape[0], 1, 1), dtype=H.dtype) + H
                obs_logp = pt.zeros_like(obs_mu)
            else:
                SequenceMvNormal(
                    f"{name}_{group}",
                    mus=mu,
                    covs=cov,
                    logp=pt.zeros_like(mu),
                    dims=state_dims,
                    method=mvn_method,
                )

                obs_mu = d + (Z @ mu[..., None]).squeeze(-1)
                obs_cov = Z @ cov @ pt.swapaxes(Z, -2, -1) + H
                obs_logp = pt.zeros_like(obs_mu)

            SequenceMvNormal(
                f"{name}_{group}_observed",
                mus=obs_mu,
                covs=obs_cov,
                logp=obs_logp,
                dims=obs_dims,
                method=mvn_method,
            )

    # TODO: Remove this after pm.Flat initial values are fixed
    forward_model.rvs_to_initial_values = {
        rv: None for rv in forward_model.rvs_to_initial_values.keys()
    }

    frozen_model = freeze_dims_and_data(forward_model)
    with frozen_model:
        idata_conditional = pm.sample_posterior_predictive(
            group_idata,
            var_names=[
                f"{name}_{group}{suffix}"
                for name in FILTER_OUTPUT_TYPES
                for suffix in ["", "_observed"]
            ],
            random_seed=random_seed,
            compile_kwargs=compile_kwargs,
            **kwargs,
        )

    return idata_conditional.posterior_predictive


def _sample_unconditional(
    ss_mod: "PyMCStateSpace",
    idata: DataTree,
    group: str,
    steps: int | None = None,
    use_data_time_dim: bool = False,
    random_seed: RandomState | None = None,
    mvn_method: Literal["cholesky", "eigh", "svd"] = "svd",
    **kwargs,
):
    """
    Draw unconditional sample trajectories according to state space dynamics, using random samples from the
    a provided trace. The state space update equations are:

        X[t+1] = T @ X[t] + R @ eta[t], eta ~ N(0, Q)
        Y[t] = Z @ X[t] + nu[t], nu ~ N(0, H)
        x[0] ~ N(a0, P0)

    Parameters
    ----------
    idata : DataTree
        A DataTree object with a posterior group containing samples from the
        posterior distribution over model parameters.

    steps : Optional[int], default=None
        The number of time steps to sample for the unconditional trajectories. If not provided (None),
        the function will sample trajectories for the entire available time dimension in the posterior.
        Otherwise, it will generate trajectories for the specified number of steps.

    use_data_time_dim : bool, default=False
        If True, the function uses the time dimension present in the provided `idata` object to sample
        unconditional trajectories. If False, a custom time dimension is created based on the number of steps
        specified, or if steps is None, it uses the entire available time dimension in the posterior.

    random_seed : int, RandomState or Generator, optional
        Seed for the random number generator.

    mvn_method: str, default "svd"
        Method used to invert the covariance matrix when calculating the pdf of a multivariate normal
        (or when generating samples). One of "cholesky", "eigh", or "svd". "cholesky" is fastest, but least robust
        to ill-conditioned matrices, while "svd" is slow but extremely robust.

        In general, if your model has measurement error, "cholesky" will be safe to use. Otherwise, "svd" is
        recommended. "eigh" can also be tried if sampling with "svd" is very slow, but it is not as robust as "svd".

    kwargs:
        Additional keyword arguments are passed to pymc.sample_posterior_predictive

    Returns
    -------
    DataTree
        An Arviz InfereceData with two groups, posterior_latent and posterior_observed

        - posterior_latent represents the latent state trajectories `X[t]`, which follows the dynamics:
          `x[t+1] = T @ x[t] + R @ eta[t]`, where `eta ~ N(0, Q)`.

        - posterior_observed represents the observed state trajectories `Y[t]`, which is obtained from
          the latent state trajectories: `y[t] = Z @ x[t] + nu[t]`, where `nu ~ N(0, H)`.
    """
    verify_group(group)

    compile_kwargs = kwargs.pop("compile_kwargs", {})
    compile_kwargs.setdefault("mode", ss_mod.mode)

    group_idata = getattr(idata, group)
    trajectory_dims = None
    fit_coords = coords_from_idata(ss_mod, idata, "observed_data")
    fit_dims = dims_from_idata(ss_mod, idata, group)
    temp_coords = fit_coords.copy()

    if use_data_time_dim:
        # A trajectory of n steps spans n + 1 timesteps, the initial state plus one per step.
        data_steps = len(temp_coords[TIME_DIM]) - 1
        if steps is not None and steps != data_steps:
            raise ValueError(
                f"steps={steps} does not match the time dimension in idata, which spans "
                f"{data_steps} steps. Pass steps={data_steps}, or steps=None to take it from "
                "the data."
            )
        steps = data_steps
    elif steps is not None:
        temp_coords[TIME_DIM] = np.arange(1 + steps, dtype="int")
    else:
        steps = len(temp_coords[TIME_DIM]) - 1

    if all([dim in fit_coords for dim in [TIME_DIM, ALL_STATE_DIM, OBS_STATE_DIM]]):
        trajectory_dims = [TIME_DIM, ALL_STATE_DIM, OBS_STATE_DIM]

    with pm.Model(coords=temp_coords if trajectory_dims is not None else None) as forward_model:
        dummy_graph.build_dummy_graph(ss_mod, coords=fit_coords, dims=fit_dims)
        matrices = ss_mod._insert_random_variables()

        for entry in exog_from_idata(ss_mod, idata, "constant_data").values():
            pm.Data(**entry)

        matrices = ss_mod._insert_data_variables(matrices)
        # The unconditional trajectory spans ``steps + 1`` timesteps, and time-varying
        # matrices carry one row per timestep.
        matrices = ss_mod._insert_constant_timestep(matrices, step=steps + 1)
        x0, P0, c, d, T, Z, R, H, Q = matrices

        if not ss_mod.measurement_error:
            H_jittered = pm.Deterministic("H_jittered", stabilize(H))
            matrices = [x0, P0, c, d, T, Z, R, H_jittered, Q]

        LinearGaussianStateSpace(
            group,
            *matrices,
            steps=steps,
            dims=trajectory_dims,
            method=mvn_method,
            k_endog=ss_mod.k_endog,
        )

    # TODO: Remove this after pm.Flat has its initial_value fixed
    forward_model.rvs_to_initial_values = {
        rv: None for rv in forward_model.rvs_to_initial_values.keys()
    }
    frozen_model = freeze_dims_and_data(forward_model)

    with frozen_model:
        idata_unconditional = pm.sample_posterior_predictive(
            group_idata,
            var_names=[f"{group}_latent", f"{group}_observed"],
            random_seed=random_seed,
            compile_kwargs=compile_kwargs,
            **kwargs,
        )

    return idata_unconditional.posterior_predictive


def sample_conditional_prior(
    ss_mod: "PyMCStateSpace",
    idata: DataTree,
    random_seed: RandomState | None = None,
    mvn_method: Literal["cholesky", "eigh", "svd"] = "svd",
    **kwargs,
) -> DataTree:
    return _sample_conditional(
        ss_mod,
        idata=idata,
        group="prior",
        random_seed=random_seed,
        mvn_method=mvn_method,
        **kwargs,
    )


def sample_conditional_posterior(
    ss_mod: "PyMCStateSpace",
    idata: DataTree,
    random_seed: RandomState | None = None,
    mvn_method: Literal["cholesky", "eigh", "svd"] = "svd",
    **kwargs,
):
    return _sample_conditional(
        ss_mod,
        idata=idata,
        group="posterior",
        random_seed=random_seed,
        mvn_method=mvn_method,
        **kwargs,
    )


def sample_unconditional_prior(
    ss_mod: "PyMCStateSpace",
    idata: DataTree,
    steps: int | None = None,
    use_data_time_dim: bool = False,
    random_seed: RandomState | None = None,
    mvn_method: Literal["cholesky", "eigh", "svd"] = "svd",
    **kwargs,
) -> DataTree:
    return _sample_unconditional(
        ss_mod,
        idata=idata,
        group="prior",
        steps=steps,
        use_data_time_dim=use_data_time_dim,
        random_seed=random_seed,
        mvn_method=mvn_method,
        **kwargs,
    )


def sample_unconditional_posterior(
    ss_mod: "PyMCStateSpace",
    idata: DataTree,
    steps: int | None = None,
    use_data_time_dim: bool = False,
    random_seed: RandomState | None = None,
    mvn_method: Literal["cholesky", "eigh", "svd"] = "svd",
    **kwargs,
) -> DataTree:
    return _sample_unconditional(
        ss_mod,
        idata=idata,
        group="posterior",
        steps=steps,
        use_data_time_dim=use_data_time_dim,
        random_seed=random_seed,
        mvn_method=mvn_method,
        **kwargs,
    )


def sample_statespace_matrices(
    ss_mod: "PyMCStateSpace",
    idata,
    matrix_names: str | list[str] | None = None,
    group: str = "posterior",
    **kwargs,
):
    verify_group(group)

    compile_kwargs = kwargs.pop("compile_kwargs", {})
    compile_kwargs.setdefault("mode", ss_mod.mode)

    if isinstance(matrix_names, str):
        matrix_names = [matrix_names]

    if matrix_names is None:
        # Not the short names: x0 and P0 collide with parameters most models declare.
        matrix_names = LONG_MATRIX_NAMES
    else:
        unknown_matrix_names = set(matrix_names).difference(MATRIX_NAMES, LONG_MATRIX_NAMES)
        if unknown_matrix_names:
            raise ValueError(f"{sorted(unknown_matrix_names)} not a valid statespace matrix name!")

    fit_coords = coords_from_idata(ss_mod, idata, "observed_data")
    fit_dims = dims_from_idata(ss_mod, idata, group)

    with pm.Model(coords=fit_coords) as forward_model:
        dummy_graph.build_dummy_graph(ss_mod, coords=fit_coords, dims=fit_dims)
        matrices = ss_mod._insert_random_variables()

        for entry in exog_from_idata(ss_mod, idata, "constant_data").values():
            pm.Data(**entry)

        matrices = ss_mod._insert_data_variables(matrices)
        for short_name, matrix in zip(MATRIX_NAMES, matrices, strict=True):
            long_name = SHORT_NAME_TO_LONG[short_name]
            if (long_name in matrix_names) or (short_name in matrix_names):
                name = long_name if long_name in matrix_names else short_name
                matrix_dims = MATRIX_DIMS[short_name]
                if matrix.ndim == len(matrix_dims) + 1:
                    # A time-varying matrix carries a leading time axis its static dims do not name.
                    matrix_dims = (TIME_DIM, *matrix_dims)
                known_dims = [x if x in fit_coords else None for x in matrix_dims]
                pm.Deterministic(name, matrix, dims=known_dims)

    # TODO: Remove this after pm.Flat has its initial_value fixed
    forward_model.rvs_to_initial_values = {
        rv: None for rv in forward_model.rvs_to_initial_values.keys()
    }
    frozen_model = freeze_dims_and_data(forward_model)
    with frozen_model:
        matrix_idata = pm.sample_posterior_predictive(
            idata if group == "posterior" else idata["prior"],
            var_names=matrix_names,
            extend_inferencedata=False,
            compile_kwargs=compile_kwargs,
            **kwargs,
        )

    return matrix_idata


def sample_filter_outputs(
    ss_mod: "PyMCStateSpace",
    idata,
    filter_output_names: str | list[str] | None = None,
    group: str = "posterior",
    **kwargs,
):
    if isinstance(filter_output_names, str):
        filter_output_names = [filter_output_names]

    if filter_output_names is None:
        filter_output_names = list(FILTER_OUTPUT_DIMS.keys())
    else:
        unknown_filter_output_names = set(filter_output_names).difference(FILTER_OUTPUT_DIMS)
        if unknown_filter_output_names:
            raise ValueError(
                f"{sorted(unknown_filter_output_names)} not a valid filter output name!"
            )
        filter_output_names = [x for x in FILTER_OUTPUT_DIMS.keys() if x in filter_output_names]

    compile_kwargs = kwargs.pop("compile_kwargs", {})
    compile_kwargs.setdefault("mode", ss_mod.mode)

    with pm.Model(coords=ss_mod.coords) as m:
        dummy_graph.build_dummy_graph(
            ss_mod,
            coords=coords_from_idata(ss_mod, idata, "observed_data"),
            dims=dims_from_idata(ss_mod, idata, group),
        )
        matrices = ss_mod._insert_random_variables()

        for entry in exog_from_idata(ss_mod, idata, "constant_data").values():
            pm.Data(**entry)

        matrices = ss_mod._insert_data_variables(matrices)

        x0, P0, c, d, T, Z, R, H, Q = matrices
        data = data_from_idata(idata, "constant_data")

        obs_coords = m.coords.get(OBS_STATE_DIM, None)

        data, nan_mask = register_data_with_pymc(
            data,
            n_obs=ss_mod.ssm.k_endog,
            obs_coords=obs_coords,
        )

        kalman_filter, kalman_smoother = ss_mod.make_filters()
        filter_outputs = kalman_filter.build_graph(data, x0, P0, c, d, T, Z, R, H, Q)

        smoother_outputs = kalman_smoother.build_graph(
            T, R, Q, filter_outputs[0], filter_outputs[3]
        )

        filter_outputs = filter_outputs[:-1] + list(smoother_outputs)
        for output in filter_outputs:
            if output.name in filter_output_names:
                dims = FILTER_OUTPUT_DIMS[output.name]
                pm.Deterministic(output.name, output, dims=dims)

    with freeze_dims_and_data(m):
        return pm.sample_posterior_predictive(
            idata if group == "posterior" else idata["prior"],
            var_names=filter_output_names,
            compile_kwargs=compile_kwargs,
            **kwargs,
        )
