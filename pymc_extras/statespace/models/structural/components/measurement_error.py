import numpy as np

from pymc_extras.statespace.models.structural.core import Component


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

        trend = st.LevelTrendComponent(order=2, innovations_order=1)
        error = st.MeasurementError()

        ss_mod = (trend + error).build()

        # Use with PyMC
        with pm.Model(coords=ss_mod.coords) as model:
            P0 = pm.Deterministic('P0', pt.eye(ss_mod.k_states) * 10, dims=ss_mod.param_dims['P0'])
            initial_trend = pm.Normal('initial_trend', sigma=10, dims=ss_mod.param_dims['initial_trend'])
            sigma_obs = pm.Exponential('sigma_obs', 1, dims=ss_mod.param_dims['sigma_obs'])

            ss_mod.build_statespace_graph(data)
            idata = pm.sample()

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

        trend = st.LevelTrendComponent(order=2, innovations_order=1)
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
    ):
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
            observed_state_names=observed_state_names,
            share_states=share_states,
        )

    def populate_component_properties(self):
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog

        self.param_names = [f"sigma_{self.name}"]
        self.param_dims = {}
        self.coords = {}

        if k_endog_effective > 1:
            self.param_dims[f"sigma_{self.name}"] = (f"endog_{self.name}",)
            self.coords[f"endog_{self.name}"] = self.observed_state_names

        self.param_info = {
            f"sigma_{self.name}": {
                "shape": (k_endog_effective,) if k_endog_effective > 1 else (),
                "constraints": "Positive",
                "dims": (f"endog_{self.name}",) if k_endog_effective > 1 else None,
            }
        }

    def make_symbolic_graph(self) -> None:
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog

        sigma_shape = () if k_endog_effective == 1 else (k_endog_effective,)
        error_sigma = self.make_and_register_variable(f"sigma_{self.name}", shape=sigma_shape)

        diag_idx = np.diag_indices(self.k_endog)
        idx = np.s_["obs_cov", diag_idx[0], diag_idx[1]]
        self.ssm[idx] = error_sigma**2
