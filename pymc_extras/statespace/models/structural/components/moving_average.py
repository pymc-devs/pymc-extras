from collections.abc import Sequence

import numpy as np
import pytensor.tensor as pt

from pytensor import config

from pymc_extras.statespace.core.properties import (
    Coord,
    Parameter,
    Shock,
    State,
)
from pymc_extras.statespace.models.structural.core import Component
from pymc_extras.statespace.models.structural.utils import order_to_mask


class MovingAverage(Component):
    r"""
    Moving average timeseries component

    Parameters
    ----------
    order: int or sequence of int

        If int, the number of lagged innovations to include in the model.
        If a sequence, an array-like of zeros and ones indicating which lags to include in the model.

    name: str, default "moving_average"
        A name for this moving average component. Used to label dimensions and coordinates.

    observed_state_names: list[str] | None, default None
        List of strings for observed state labels. If None, defaults to ["data"].

    share_states: bool, default False
        Whether latent states are shared across the observed states. If True, there will be only one set of latent
        states, which are observed by all observed states. If False, each observed state has its own set of
        latent states. This argument has no effect if `k_endog` is 1.

    coefficients: array-like, optional
        Fixed values for the moving average coefficients. If given, the coefficients are not estimated and no
        ``params_{name}`` parameter is added to the model. Must have one entry per included lag. With several
        observed states and ``share_states=False``, pass either one set of coefficients, used for every observed
        state, or an array of shape ``(n_observed_states, n_lags)``.

    Notes
    -----
    A moving average component models a process driven by its own current and past innovations:

    .. math::
        x_t = \epsilon_t + \sum_{i=1}^q \theta_i \epsilon_{t-i}, \quad \epsilon_t \sim N(0, \sigma^2)

    Where ``q``, the number of lagged innovations, is the order of the process. By default, all lags up to ``q`` are
    included in the model. To disable lags, pass a list of zeros and ones to the ``order`` argument. For example,
    ``order=[1, 0, 1]`` would become:

    .. math::
        x_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_3 \epsilon_{t-3}

    The coefficient :math:`\theta_2` has been constrained to zero.

    The process is written in state space form with ``q + 1`` hidden states, :math:`L_0` to :math:`L_q`. The first
    state, :math:`L_0`, is :math:`x_t` and is the contribution of this component to the observed data.

    No invertibility constraint is placed on the coefficients. Each non-invertible set of coefficients has an
    invertible counterpart, with a rescaled sigma, that gives the same autocovariances, so the two are not identified
    unless the priors rule one out.

    .. warning:: This class is meant to be used as a component in a structural time series model. For modeling of
              stationary processes with ARIMA, use ``statespace.BayesianSARIMAX``.

    Examples
    --------
    Model a timeseries as a local level plus an MA(2) process:

    .. code:: python

        from pymc_extras.statespace import structural as st
        import pymc as pm
        import pytensor.tensor as pt

        trend = st.LevelTrend(order=1, innovations_order=0)
        ma = st.MovingAverage(order=2)
        ss_mod = (trend + ma).build()

        with pm.Model(coords=ss_mod.coords) as model:
            P0 = pm.Deterministic('P0', pt.eye(ss_mod.k_states) * 10, dims=ss_mod.param_dims['P0'])
            initial_trend = pm.Normal('initial_level_trend', sigma=10, dims=ss_mod.param_dims['initial_level_trend'])
            params_ma = pm.Normal('params_moving_average', dims=ss_mod.param_dims['params_moving_average'])
            sigma_ma = pm.Exponential('sigma_moving_average', 1)

            ss_mod.build_statespace_graph(data)
            idata = pm.sample(nuts_sampler='numpyro')

    Fix the coefficients instead of estimating them. For example, an equal-weight moving average over the last 20
    innovations:

    .. code:: python

        import numpy as np
        from pymc_extras.statespace import structural as st

        ma = st.MovingAverage(order=20, coefficients=np.ones(20))

    """

    def __init__(
        self,
        order: int = 1,
        name: str = "moving_average",
        observed_state_names: list[str] | None = None,
        share_states: bool = False,
        coefficients: np.ndarray | Sequence[float] | None = None,
    ):
        if observed_state_names is None:
            observed_state_names = ["data"]

        k_endog = len(observed_state_names)
        k_endog_effective = k_posdef = 1 if share_states else k_endog

        order = order_to_mask(order)
        ma_lags = np.flatnonzero(order).ravel().astype(int) + 1
        k_states = len(order) + 1

        if coefficients is not None:
            coefficients = np.asarray(coefficients, dtype=config.floatX)
            if coefficients.shape not in [
                (len(ma_lags),),
                (k_endog_effective, len(ma_lags)),
            ]:
                raise ValueError(
                    f"coefficients must have shape ({len(ma_lags)},) or "
                    f"({k_endog_effective}, {len(ma_lags)}), one entry per included lag, "
                    f"got {coefficients.shape}"
                )

        self.share_states = share_states
        self.order = order
        self.ma_lags = ma_lags
        self.coefficients = coefficients

        state_names = [f"L{i}_{name}" for i in range(k_states)]

        super().__init__(
            name=name,
            k_endog=k_endog,
            k_states=k_states * k_endog_effective,
            k_posdef=k_posdef,
            measurement_error=False,
            combine_hidden_states=True,
            base_observed_state_names=observed_state_names,
            base_state_names=state_names,
            obs_state_idxs=np.tile(np.r_[[1.0], np.zeros(k_states - 1)], k_endog_effective),
            share_states=share_states,
        )

    def set_states(self) -> State | tuple[State, ...] | None:
        base_names = self.base_state_names
        observed_state_names = self.base_observed_state_names

        if self.share_states:
            state_names = [f"{name}[shared]" for name in base_names]
        else:
            state_names = [
                f"{name}[{state_name}]"
                for state_name in observed_state_names
                for name in base_names
            ]

        hidden_states = [State(name=name, observed=False, shared=True) for name in state_names]
        observed_states = [
            State(name=name, observed=True, shared=False) for name in observed_state_names
        ]
        return *hidden_states, *observed_states

    def set_parameters(self) -> Parameter | tuple[Parameter, ...] | None:
        k_endog_effective = 1 if self.share_states else self.k_endog
        k_nonzero = len(self.ma_lags)

        sigma_param = Parameter(
            name=f"sigma_{self.name}",
            shape=(k_endog_effective,) if k_endog_effective > 1 else (),
            dims=(f"endog_{self.name}",) if k_endog_effective > 1 else None,
            constraints="Positive",
        )

        if self.coefficients is not None:
            return (sigma_param,)

        ma_param = Parameter(
            name=f"params_{self.name}",
            shape=(k_endog_effective, k_nonzero) if k_endog_effective > 1 else (k_nonzero,),
            dims=(f"lag_{self.name}",)
            if k_endog_effective == 1
            else (
                f"endog_{self.name}",
                f"lag_{self.name}",
            ),
            constraints=None,
        )

        return ma_param, sigma_param

    def set_shocks(self) -> Shock | tuple[Shock, ...] | None:
        if self.share_states:
            shock_names = [f"{self.name}[shared]"]
        else:
            shock_names = [f"{self.name}[{obs_name}]" for obs_name in self.observed_state_names]

        return tuple(Shock(name=name) for name in shock_names)

    def set_coords(self) -> Coord | tuple[Coord, ...] | None:
        k_endog_effective = 1 if self.share_states else self.k_endog

        coord_container = []

        if self.coefficients is None:
            coord_container.append(
                Coord(dimension=f"lag_{self.name}", labels=self.ma_lags.tolist())
            )

        if k_endog_effective > 1:
            coord_container.append(
                Coord(dimension=f"endog_{self.name}", labels=self.observed_state_names)
            )

        return tuple(coord_container)

    def make_symbolic_graph(self) -> None:
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog

        k_states = self.k_states // k_endog_effective
        k_nonzero = len(self.ma_lags)

        if self.coefficients is None:
            ma_params = self.make_and_register_variable(
                f"params_{self.name}",
                shape=(k_nonzero,) if k_endog_effective == 1 else (k_endog_effective, k_nonzero),
            )
        else:
            ma_params = pt.as_tensor_variable(self.coefficients)

        sigma_ma = self.make_and_register_variable(
            f"sigma_{self.name}", shape=() if k_endog_effective == 1 else (k_endog_effective,)
        )

        # One row of coefficients per set of latent states
        ma_params = pt.broadcast_to(ma_params, (k_endog_effective, k_nonzero))

        T = pt.eye(k_states, k=1)
        self.ssm["transition"] = pt.linalg.block_diag(*[T for _ in range(k_endog_effective)])

        Rs = [
            pt.zeros((k_states, 1))[0, 0].set(1.0)[self.ma_lags, 0].set(ma_params[i])
            for i in range(k_endog_effective)
        ]
        self.ssm["selection"] = pt.linalg.block_diag(*Rs)

        Zs = [pt.zeros((1, k_states))[0, 0].set(1.0) for _ in range(k_endog)]

        if self.share_states:
            Z = pt.join(0, *Zs)
        else:
            Z = pt.linalg.block_diag(*Zs)
        self.ssm["design"] = Z

        self.ssm["state_cov"] = pt.diag(pt.atleast_1d(sigma_ma**2))
