import warnings

import numpy as np
import pytensor.tensor as pt

from pymc_extras.statespace.core.properties import Coord, Parameter, Shock, State
from pymc_extras.statespace.models.structural.core import Component
from pymc_extras.statespace.models.structural.utils import order_to_mask
from pymc_extras.statespace.utils.constants import POSITION_DERIVATIVE_NAMES


class LevelTrend(Component):
    r"""
    Level and trend component of a structural time series model

    Parameters
    ----------
    order : int
        Number of time derivatives of the trend to include in the model. For example, when order=3, the trend will
        be of the form ``y = a + b * t + c * t ** 2``, where the coefficients ``a, b, c`` come from the initial
        state values.

    innovations_order : int or sequence of int, optional
        The number of stochastic innovations to include in the model. By default, ``innovations_order = order``

    name : str, default "level_trend"
        A name for this level-trend component. Used to label dimensions and coordinates.

    observed_state_names : list[str] | None, default None
        List of strings for observed state labels. If None, defaults to ["data"].

    share_states: bool, default False
        Whether latent states are shared across the observed states. If True, there will be only one set of latent
        states, which are observed by all observed states. If False, each observed state has its own set of
        latent states. This argument has no effect if `k_endog` is 1.

    Notes
    -----
    This class implements the level and trend components of the general structural time series model. In the most
    general form, the level and trend is described by a system of two time-varying equations.

    .. math::
        \begin{align}
            \mu_{t+1} &= \mu_t + \nu_t + \zeta_t \\
            \nu_{t+1} &= \nu_t + \xi_t
            \zeta_t &\sim N(0, \sigma_\zeta) \\
            \xi_t &\sim N(0, \sigma_\xi)
        \end{align}

    Where :math:`\mu_{t+1}` is the mean of the timeseries at time t, and :math:`\nu_t` is the drift or the slope of
    the process. When both innovations :math:`\zeta_t` and :math:`\xi_t` are included in the model, it is known as a
    *local linear trend* model. This system of two equations, corresponding to ``order=2``, can be expanded or
    contracted by adding or removing equations. ``order=3`` would add an acceleration term to the sytsem:

    .. math::
        \begin{align}
            \mu_{t+1} &= \mu_t + \nu_t + \zeta_t \\
            \nu_{t+1} &= \nu_t + \eta_t + \xi_t \\
            \eta_{t+1} &= \eta_{t-1} + \omega_t \\
            \zeta_t &\sim N(0, \sigma_\zeta) \\
            \xi_t &\sim N(0, \sigma_\xi) \\
            \omega_t &\sim N(0, \sigma_\omega)
        \end{align}

    After setting all innovation terms to zero and defining initial states :math:`\mu_0, \nu_0, \eta_0`, these equations
    can be collapsed to:

    .. math::
        \mu_t = \mu_0 + \nu_0 \cdot t + \eta_0 \cdot t^2

    Which clarifies how the order and initial states influence the model. In particular, the initial states are the
    coefficients on the intercept, slope, acceleration, and so on.

    In this light, allowing for innovations can be understood as allowing these coefficients to vary over time. Each
    component can be individually selected for time variation by passing a list to the ``innovations_order`` argument.
    For example, a constant intercept with time varying trend and acceleration is specified as ``order=3,
    innovations_order=[0, 1, 1]``.

    By choosing the ``order`` and ``innovations_order``, a large variety of models can be obtained. Notable
    models include:

    * Constant intercept, ``order=1, innovations_order=0``

    .. math::
        \mu_t = \mu

    * Constant linear slope, ``order=2, innovations_order=0``

    .. math::
        \mu_t = \mu_{t-1} + \nu

    * Gaussian Random Walk, ``order=1, innovations_order=1``

    .. math::
        \mu_t = \mu_{t-1} + \zeta_t

    * Gaussian Random Walk with Drift, ``order=2, innovations_order=1``

    .. math::
        \mu_t = \mu_{t-1} + \nu + \zeta_t

    * Smooth Trend, ``order=2, innovations_order=[0, 1]``

    .. math::
        \begin{align}
            \mu_t &= \mu_{t-1} + \nu_{t-1} \\
            \nu_t &= \nu_{t-1} + \xi_t
        \end{align}

    * Local Level, ``order=2, innovations_order=2``

    [1] notes that the smooth trend model produces more gradually changing slopes than the full local linear trend
    model, and is equivalent to an "integrated trend model".

    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
        Time Series Analysis by State Space Methods: Second Edition.
        Oxford University Press.

    """

    def __init__(
        self,
        order: int | list[int] = 2,
        innovations_order: int | list[int] | None = None,
        name: str = "level_trend",
        observed_state_names: list[str] | None = None,
        share_states: bool = False,
    ):
        self.share_states = share_states

        if innovations_order is None:
            innovations_order = order

        if observed_state_names is None:
            observed_state_names = ["data"]
        k_endog = len(observed_state_names)

        self._order_mask = order_to_mask(order)
        max_state = np.flatnonzero(self._order_mask)[-1].item() + 1

        # If the user passes excess zeros, raise an error. The alternative is to prune them, but this would cause
        # the shape of the state to be different to what the user expects.
        if len(self._order_mask) > max_state:
            raise ValueError(
                f"order={order} is invalid. The highest derivative should not be set to zero. If you want a "
                f"lower order model, explicitly omit the zeros."
            )
        k_states = max_state
        state_names = POSITION_DERIVATIVE_NAMES[:max_state]

        if isinstance(innovations_order, int):
            n = innovations_order
            innovations_order = order_to_mask(k_states)
            if n > 0:
                innovations_order[n:] = False
            else:
                innovations_order[:] = False
        else:
            innovations_order = order_to_mask(innovations_order)

        self.innovations_order = innovations_order[:max_state]
        k_posdef = int(sum(innovations_order))

        super().__init__(
            name,
            k_endog=k_endog,
            k_states=k_states * k_endog if not share_states else k_states,
            k_posdef=k_posdef * k_endog if not share_states else k_posdef,
            base_state_names=state_names,
            base_observed_state_names=observed_state_names,
            measurement_error=False,
            combine_hidden_states=False,
            obs_state_idxs=np.tile(
                np.array([1.0] + [0.0] * (k_states - 1)),
                k_endog if not share_states else 1,
            ),
            share_states=share_states,
        )

    def set_states(self) -> State | tuple[State, ...] | None:
        observed_state_names = self.base_observed_state_names

        if self.share_states:
            state_names = [f"{name}[{self.name}_shared]" for name in self.base_state_names]
        else:
            state_names = [
                f"{name}[{obs_name}]"
                for obs_name in observed_state_names
                for name in self.base_state_names
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
        k_posdef = self.k_posdef // k_endog_effective

        initial_param = Parameter(
            name=f"initial_{self.name}",
            shape=((k_endog_effective, k_states) if k_endog_effective > 1 else (k_states,)),
            dims=(
                (f"endog_{self.name}", f"state_{self.name}")
                if k_endog_effective > 1
                else (f"state_{self.name}",)
            ),
            constraints=None,
        )

        if self.k_posdef > 0:
            sigma_param = Parameter(
                name=f"sigma_{self.name}",
                shape=((k_posdef,) if k_endog_effective == 1 else (k_endog_effective, k_posdef)),
                dims=(
                    (f"shock_{self.name}",)
                    if k_endog_effective == 1
                    else (f"endog_{self.name}", f"shock_{self.name}")
                ),
                constraints="Positive",
            )
            return initial_param, sigma_param
        else:
            return (initial_param,)

    def set_shocks(self) -> Shock | tuple[Shock, ...] | None:
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog
        k_states = self.k_states // k_endog_effective
        name_slice = POSITION_DERIVATIVE_NAMES[:k_states]

        if self.k_posdef > 0:
            base_shock_names = [
                name for name, mask in zip(name_slice, self.innovations_order) if mask
            ]

            if self.share_states:
                shock_names = [f"{name}[{self.name}_shared]" for name in base_shock_names]
            else:
                shock_names = [
                    f"{name}[{obs_name}]"
                    for obs_name in self.observed_state_names
                    for name in base_shock_names
                ]

            return tuple(Shock(name=name) for name in shock_names)
        return None

    def set_coords(self) -> Coord | tuple[Coord, ...] | None:
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog
        k_states = self.k_states // k_endog_effective
        name_slice = POSITION_DERIVATIVE_NAMES[:k_states]

        base_names = [name for name, mask in zip(name_slice, self._order_mask) if mask]

        base_shock_names = [name for name, mask in zip(name_slice, self.innovations_order) if mask]

        state_coord = Coord(dimension=f"state_{self.name}", labels=tuple(base_names))

        coords_container = [state_coord]

        if k_endog > 1:
            endog_coord = Coord(
                dimension=f"endog_{self.name}",
                labels=self.observed_state_names,
            )
            coords_container.append(endog_coord)

        if self.k_posdef > 0:
            shock_coord = Coord(dimension=f"shock_{self.name}", labels=tuple(base_shock_names))
            coords_container.append(shock_coord)

        return tuple(coords_container)

    def make_symbolic_graph(self) -> None:
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog

        k_states = self.k_states // k_endog_effective
        k_posdef = self.k_posdef // k_endog_effective

        initial_trend = self.make_and_register_variable(
            f"initial_{self.name}",
            shape=(k_states,) if k_endog_effective == 1 else (k_endog, k_states),
        )
        self.ssm["initial_state", :] = initial_trend.ravel()

        triu_idx = pt.triu_indices(k_states)
        T = pt.zeros((k_states, k_states))[triu_idx[0], triu_idx[1]].set(1)

        self.ssm["transition", :, :] = pt.specify_shape(
            pt.linalg.block_diag(*[T for _ in range(k_endog_effective)]),
            (self.k_states, self.k_states),
        )

        R = np.eye(k_states)
        R = R[:, self.innovations_order]

        self.ssm["selection", :, :] = pt.specify_shape(
            pt.linalg.block_diag(*[R for _ in range(k_endog_effective)]),
            (self.k_states, self.k_posdef),
        )

        Z = np.array([1.0] + [0.0] * (k_states - 1)).reshape((1, -1))

        if self.share_states:
            self.ssm["design", :, :] = pt.specify_shape(
                pt.join(0, *[Z for _ in range(k_endog)]), (self.k_endog, self.k_states)
            )
        else:
            self.ssm["design", :, :] = pt.specify_shape(
                pt.linalg.block_diag(*[Z for _ in range(k_endog)]),
                (self.k_endog, self.k_states),
            )

        if k_posdef > 0:
            sigma_trend = self.make_and_register_variable(
                f"sigma_{self.name}",
                shape=(k_posdef,) if k_endog_effective == 1 else (k_endog, k_posdef),
            )
            diag_idx = np.diag_indices(k_posdef * k_endog_effective)
            idx = np.s_["state_cov", diag_idx[0], diag_idx[1]]
            self.ssm[idx] = (sigma_trend**2).ravel()


def __getattr__(name: str):
    if name == "LevelTrendComponent":
        warnings.warn(
            "LevelTrendComponent is deprecated and will be removed in a future release. "
            "Use LevelTrend instead.",
            FutureWarning,
            stacklevel=2,
        )
        return LevelTrend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
