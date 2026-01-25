import numpy as np

from pytensor import tensor as pt

from pymc_extras.statespace.core.properties import (
    Coord,
    Data,
    Parameter,
    Shock,
    State,
)
from pymc_extras.statespace.models.structural.core import Component
from pymc_extras.statespace.models.utilities import validate_names
from pymc_extras.statespace.utils.constants import TIME_DIM


class Regression(Component):
    r"""
    Regression component for exogenous variables in a structural time series model

    Parameters
    ----------
    name : str | None, default "regression"
        A name for this regression component. Used to label dimensions and coordinates.

    state_names : list[str] | None, default None
        List of strings for regression coefficient labels. If provided, must be of length
        k_exog. If None and k_exog is provided, coefficients will be named
        "{name}_1, {name}_2, ...".

    observed_state_names : list[str] | None, default None
        List of strings for observed state labels. If None, defaults to ["data"].

    innovations : bool, default False
        Whether to include stochastic innovations in the regression coefficients,
        allowing them to vary over time. If True, coefficients follow a random walk.

    share_states: bool, default False
        Whether latent states are shared across the observed states. If True, there will be only one set of latent
        states, which are observed by all observed states. If False, each observed state has its own set of
        latent states.

    Notes
    -----
    This component implements regression with exogenous variables in a structural time series
    model. The regression component can be expressed as:

    .. math::
        y_t = \beta_t^T x_t + \epsilon_t

    Where :math:`y_t` is the dependent variable, :math:`x_t` is the vector of exogenous
    variables, :math:`\beta_t` is the vector of regression coefficients, and :math:`\epsilon_t`
    is the error term.

    When ``innovations=False`` (default), the coefficients are constant over time:
    :math:`\beta_t = \beta_0` for all t.

    When ``innovations=True``, the coefficients follow a random walk:
    :math:`\beta_{t+1} = \beta_t + \eta_t`, where :math:`\eta_t \sim N(0, \Sigma_\beta)`.

    The component supports both univariate and multivariate regression. In the multivariate
    case, separate coefficients are estimated for each endogenous variable (i.e time series).

    Examples
    --------
    Simple regression with constant coefficients:

    .. code:: python

        from pymc_extras.statespace import structural as st
        import pymc as pm
        import pytensor.tensor as pt

        trend = st.LevelTrendComponent(order=1, innovations_order=1)
        regression = st.Regression(k_exog=2, state_names=['intercept', 'slope'])
        ss_mod = (trend + regression).build()

        with pm.Model(coords=ss_mod.coords) as model:
            # Prior for regression coefficients
            betas = pm.Normal('betas', dims=ss_mod.param_dims['beta_regression'])

            # Prior for trend innovations
            sigma_trend = pm.Exponential('sigma_trend', 1)

            ss_mod.build_statespace_graph(data)
            idata = pm.sample()

    Multivariate regression with time-varying coefficients:
    - There are 2 exogenous variables (price and income effects)
    - There are 2 endogenous variables (sales and revenue)
    - The regression coefficients are allowed to vary over time (`innovations=True`)

    .. code:: python

        regression = st.Regression(
            k_exog=2,
            state_names=['price_effect', 'income_effect'],
            observed_state_names=['sales', 'revenue'],
            innovations=True
        )

        with pm.Model(coords=ss_mod.coords) as model:
            betas = pm.Normal('betas', dims=ss_mod.param_dims['beta_regression'])

            # Innovation variance for time-varying coefficients
            sigma_beta = pm.Exponential('sigma_beta', 1, dims=ss_mod.param_dims['sigma_beta_regression'])

            ss_mod.build_statespace_graph(data)
            idata = pm.sample()
    """

    def __init__(
        self,
        name: str | None = "regression",
        state_names: list[str] | None = None,
        observed_state_names: list[str] | None = None,
        innovations=False,
        share_states: bool = False,
    ):
        self.share_states = share_states

        if observed_state_names is None:
            observed_state_names = ["data"]

        self.innovations = innovations
        validate_names(state_names, var_name="state_names", optional=False)
        k_exog = len(state_names)

        k_states = k_exog
        k_endog = len(observed_state_names)
        k_posdef = k_exog

        super().__init__(
            name=name,
            k_endog=k_endog,
            k_states=k_states * k_endog if not share_states else k_states,
            k_posdef=k_posdef * k_endog if not share_states else k_posdef,
            base_state_names=state_names,
            share_states=share_states,
            base_observed_state_names=observed_state_names,
            measurement_error=False,
            combine_hidden_states=False,
            obs_state_idxs=np.ones(k_states),
        )

    def set_states(self) -> State | tuple[State, ...] | None:
        base_names = self.base_state_names
        observed_state_names = self.base_observed_state_names

        if self.share_states:
            state_names = [f"{name}[{self.name}_shared]" for name in base_names]
        else:
            state_names = [
                f"{name}[{obs_name}]" for obs_name in observed_state_names for name in base_names
            ]

        hidden_states = [State(name=name, observed=False, shared=True) for name in state_names]
        observed_states = [
            State(name=name, observed=True, shared=False) for name in observed_state_names
        ]
        return *hidden_states, *observed_states

    def set_parameters(self) -> Parameter | tuple[Parameter, ...] | None:
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog
        k_states = self.k_states // k_endog_effective

        beta_parameter = Parameter(
            name=f"beta_{self.name}",
            shape=(k_endog_effective, k_states) if k_endog_effective > 1 else (k_states,),
            dims=(
                (f"endog_{self.name}", f"state_{self.name}")
                if k_endog_effective > 1
                else (f"state_{self.name}",)
            ),
            constraints=None,
        )

        params_container = [beta_parameter]

        if self.innovations:
            sigma_parameter = Parameter(
                name=f"sigma_beta_{self.name}",
                shape=(k_states,),
                dims=(f"state_{self.name}",),
                constraints="Positive",
            )

            params_container.append(sigma_parameter)

        return tuple(params_container)

    def set_data_info(self) -> Data | tuple[Data, ...] | None:
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog
        k_states = self.k_states // k_endog_effective

        data_prop = Data(
            name=f"data_{self.name}",
            shape=(None, k_states),
            dims=(TIME_DIM, f"state_{self.name}"),
            is_exogenous=True,
        )
        return (data_prop,)

    def set_shocks(self) -> Shock | tuple[Shock, ...] | None:
        base_names = self.base_state_names

        if self.share_states:
            shock_names = [f"{state_name}_shared" for state_name in base_names]
        else:
            shock_names = base_names

        return tuple(Shock(name=name) for name in shock_names)

    def set_coords(self) -> tuple[Coord, ...] | None:
        regression_state_coord = Coord(
            dimension=f"state_{self.name}", labels=tuple(self.base_state_names)
        )
        endogenous_state_coord = Coord(
            dimension=f"endog_{self.name}", labels=self.observed_state_names
        )

        return regression_state_coord, endogenous_state_coord

    def make_symbolic_graph(self) -> None:
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog

        k_states = self.k_states // k_endog_effective

        betas = self.make_and_register_variable(
            f"beta_{self.name}", shape=(k_endog, k_states) if k_endog_effective > 1 else (k_states,)
        )
        regression_data = self.make_and_register_data(f"data_{self.name}", shape=(None, k_states))

        self.ssm["initial_state", :] = betas.ravel()
        self.ssm["transition", :, :] = pt.eye(self.k_states)
        self.ssm["selection", :, :] = pt.eye(self.k_states)

        if self.share_states:
            self.ssm["design"] = pt.specify_shape(
                pt.join(1, *[pt.expand_dims(regression_data, 1) for _ in range(k_endog)]),
                (None, k_endog, self.k_states),
            )
        else:
            Z = pt.linalg.block_diag(*[pt.expand_dims(regression_data, 1) for _ in range(k_endog)])
            self.ssm["design"] = pt.specify_shape(
                Z, (None, k_endog, regression_data.type.shape[1] * k_endog)
            )

        if self.innovations:
            sigma_beta = self.make_and_register_variable(
                f"sigma_beta_{self.name}",
                (k_states,) if k_endog_effective == 1 else (k_endog, k_states),
            )
            row_idx, col_idx = np.diag_indices(self.k_states)
            self.ssm["state_cov", row_idx, col_idx] = sigma_beta.ravel() ** 2
