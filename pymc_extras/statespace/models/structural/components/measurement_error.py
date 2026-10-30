import numpy as np

from pytensor import tensor as pt
from pytensor.raise_op import Assert

from pymc_extras.statespace.core.properties import (
    Coord,
    Data,
    Parameter,
)
from pymc_extras.statespace.models.structural.core import Component
from pymc_extras.statespace.utils.constants import OBSERVED_DATA_NAME, TIME_DIM


class MeasurementError(Component):
    r"""
    Measurement error component for structural time series models.

    This component adds observation noise to the model by introducing a variance parameter
    that affects the observation covariance matrix H. Unlike other components, it has no
    hidden states and should only be used in combination with other components.

    Parameters
    ----------
    name : str, optional
        Name of the measurement error component. Default is "MeasurementError".
    observed_state_names : list[str] | None, optional
        Names of the observed variables. If None, defaults to ["data"].
    share_states: bool, default False
        Whether latent states are shared across the observed states. If True, there will be only one set of latent
        states, which are observed by all observed states. If False, each observed state has its own set of
        latent states. This argument has no effect if `k_endog` is 1.

    variance_data : str | None, optional
        Name of a ``pm.Data`` container holding known observation variances (not
        standard deviations). When supplied, no ``sigma_{name}`` prior is required.
        Use shape ``(time,)`` for one observed state or ``share_states=True``;
        otherwise use ``(time, n_observed_states)`` in ``observed_state_names``
        order. Values must be finite and nonnegative. Runtime assertions remain
        active after ``pm.set_data`` under the standard and numba backends;
        JAX drops Assert operations and does not enforce these checks.
        Forecasts require future variances in ``scenario``; the last observed
        variance is never automatically extrapolated. Default is None, which
        preserves the estimated-variance behavior.

    Notes
    -----
    The measurement error component models observation noise as:

    .. math::

        y_t = \text{signal}_t + \varepsilon_t, \quad \varepsilon_t \sim N(0, \sigma^2)

    Where :math:`\text{signal}_t` is the true signal from other components and
    :math:`\sigma^2` is the measurement error variance.

    This component:
        - Has no hidden states (k_states = 0)
        - Has no innovations (k_posdef = 0)
        - Adds a single parameter: sigma_{name}
        - Modifies the observation covariance matrix H

    Examples
    --------
    **Basic usage with trend component:**

    .. code:: python

        from pymc_extras.statespace import structural as st
        import pymc as pm
        import pytensor.tensor as pt

        trend = st.LevelTrend(order=2, innovations_order=1)
        error = st.MeasurementError()

        ss_mod = (trend + error).build()

        # Use with PyMC
        with pm.Model(coords=ss_mod.coords) as model:
            P0 = pm.Deterministic('P0', pt.eye(ss_mod.k_states) * 10, dims=ss_mod.param_dims['P0'])
            initial_trend = pm.Normal('initial_trend', sigma=10, dims=ss_mod.param_dims['initial_trend'])
            sigma_obs = pm.Exponential('sigma_obs', 1, dims=ss_mod.param_dims['sigma_obs'])

            ss_mod.build_statespace_graph(data)
            idata = pm.sample()

    **Known, time-varying sampling variance:**

    .. code:: python

        measurement = st.MeasurementError(
            name="sampling_error", variance_data="sampling_variance"
        )
        ss_mod = (st.LevelTrend(order=1) + measurement).build()
        with pm.Model(coords=ss_mod.coords):
            pm.Data("sampling_variance", sampling_variance, dims="time")
            # Define the remaining priors listed by ss_mod, then build the graph.
            ss_mod.build_statespace_graph(data)

    Known and estimated measurement-error components can be added: their
    variances sum on each observed state's diagonal.

    **Multivariate measurement error:**

    .. code:: python

        # For multiple observed variables
        # This creates separate measurement error variances for each variable
        # sigma_obs_error will have shape (3,) for the three variables
        error = st.MeasurementError(
            name="obs_error",
            observed_state_names=["gdp", "unemployment", "inflation"]
        )

    **Complete model example:**

    .. code:: python

        trend = st.LevelTrend(order=2, innovations_order=1)
        seasonal = st.TimeSeasonality(season_length=12, innovations=True)
        error = st.MeasurementError()

        model = (trend + seasonal + error).build()

        # The model now includes:
        # - Trend parameters: level_trend, sigma_trend
        # - Seasonal parameters: seasonal_coefs, sigma_seasonal
        # - Measurement error parameter: sigma_obs

    See Also
    --------
    Component : Base class for all structural components.
    StructuralTimeSeries : Complete model class.
    """

    def __init__(
        self,
        name: str = "MeasurementError",
        observed_state_names: list[str] | None = None,
        share_states: bool = False,
        variance_data: str | None = None,
    ):
        if variance_data is not None and (
            not isinstance(variance_data, str)
            or not variance_data.strip()
            or variance_data == OBSERVED_DATA_NAME
        ):
            raise ValueError("variance_data must be a non-empty name other than 'data'.")
        self.variance_data = variance_data

        if observed_state_names is None:
            observed_state_names = ["data"]

        self.share_states = share_states

        k_endog = len(observed_state_names)
        k_states = 0
        k_posdef = 0

        super().__init__(
            name,
            k_endog,
            k_states,
            k_posdef,
            measurement_error=True,
            combine_hidden_states=False,
            base_observed_state_names=observed_state_names,
            share_states=share_states,
        )

    def set_data_info(self) -> Data | tuple[Data, ...] | None:
        if self.variance_data is None:
            return None
        shared = self.share_states or self.k_endog == 1
        return Data(
            name=self.variance_data,
            shape=(None,) if shared else (None, self.k_endog),
            dims=(TIME_DIM,) if shared else (TIME_DIM, f"endog_{self.name}"),
            is_exogenous=True,
        )

    def set_parameters(self) -> Parameter | tuple[Parameter, ...] | None:
        if self.variance_data is not None:
            return None
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog

        sigma_param = Parameter(
            name=f"sigma_{self.name}",
            shape=(k_endog_effective,) if k_endog_effective > 1 else (),
            dims=(f"endog_{self.name}",) if k_endog_effective > 1 else None,
            constraints="Positive",
        )

        return (sigma_param,)

    def set_coords(self) -> Coord | tuple[Coord, ...] | None:
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog
        observed_state_names = self.base_observed_state_names

        coords_container = []

        if k_endog_effective > 1:
            endog_coord = Coord(dimension=f"endog_{self.name}", labels=observed_state_names)
            coords_container.append(endog_coord)

        return tuple(coords_container) if coords_container else None

    def make_symbolic_graph(self) -> None:
        if self.variance_data is not None:
            shared = self.share_states or self.k_endog == 1
            variance = self.make_and_register_data(
                self.variance_data, shape=(None,) if shared else (None, self.k_endog)
            )
            valid = pt.all(~pt.isnan(variance) & ~pt.isinf(variance) & (variance >= 0))
            variance = Assert(
                f"variance_data '{self.variance_data}' must contain finite, nonnegative variances."
            )(variance, valid)
            diagonal = variance[:, None] if shared else variance
            self.ssm["obs_cov"] = diagonal[..., None] * pt.eye(self.k_endog)
            self.ssm.declare_time_varying("obs_cov")
            return
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog

        sigma_shape = () if k_endog_effective == 1 else (k_endog_effective,)
        error_sigma = self.make_and_register_variable(f"sigma_{self.name}", shape=sigma_shape)

        diag_idx = np.diag_indices(self.k_endog)
        idx = np.s_["obs_cov", diag_idx[0], diag_idx[1]]
        self.ssm[idx] = error_sigma**2
