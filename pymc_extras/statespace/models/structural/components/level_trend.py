import numpy as np
import pytensor.tensor as pt

from pymc_extras.statespace.models.structural.core import Component
from pymc_extras.statespace.models.structural.utils import order_to_mask
from pymc_extras.statespace.utils.constants import POSITION_DERIVATIVE_NAMES


class LevelTrendComponent(Component):
    r"""
    Level and trend component of a structural time series model

    Parameters
    ----------
    __________
    order : int

        Number of time derivatives of the trend to include in the model. For example, when order=3, the trend will
        be of the form ``y = a + b * t + c * t ** 2``, where the coefficients ``a, b, c`` come from the initial
        state values.

    innovations_order : int or sequence of int, optional

        The number of stochastic innovations to include in the model. By default, ``innovations_order = order``

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
    ):
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
            k_states=k_states * k_endog,
            k_posdef=k_posdef * k_endog,
            observed_state_names=observed_state_names,
            measurement_error=False,
            combine_hidden_states=False,
            obs_state_idxs=np.tile(np.array([1.0] + [0.0] * (k_states - 1)), k_endog),
        )

    def populate_component_properties(self):
        k_endog = self.k_endog
        k_states = self.k_states // k_endog
        k_posdef = self.k_posdef // k_endog

        name_slice = POSITION_DERIVATIVE_NAMES[:k_states]
        self.param_names = [f"{self.name}_initial"]
        base_names = [name for name, mask in zip(name_slice, self._order_mask) if mask]
        self.state_names = [
            f"{name}[{obs_name}]" for obs_name in self.observed_state_names for name in base_names
        ]
        self.param_dims = {f"{self.name}_initial": (f"{self.name}_state",)}
        self.coords = {f"{self.name}_state": base_names}

        if k_endog > 1:
            self.param_dims[f"{self.name}_state"] = (
                f"{self.name}_endog",
                f"{self.name}_state",
            )
            self.param_dims = {f"{self.name}_initial": (f"{self.name}_endog", f"{self.name}_state")}
            self.coords[f"{self.name}_endog"] = self.observed_state_names

        shape = (k_endog, k_states) if k_endog > 1 else (k_states,)
        self.param_info = {f"{self.name}_initial": {"shape": shape, "constraints": None}}

        if self.k_posdef > 0:
            self.param_names += [f"{self.name}_sigma"]

            shock_base_names = [
                name for name, mask in zip(name_slice, self.innovations_order) if mask
            ]
            self.shock_names = [
                f"{name}[{obs_name}]"
                for obs_name in self.observed_state_names
                for name in shock_base_names
            ]

            self.param_dims[f"{self.name}_sigma"] = (
                (f"{self.name}_shock",)
                if k_endog == 1
                else (f"{self.name}_endog", f"{self.name}_shock")
            )
            self.coords[f"{self.name}_shock"] = self.shock_names
            self.param_info[f"{self.name}_sigma"] = {
                "shape": (k_posdef,) if k_endog == 1 else (k_endog, k_posdef),
                "constraints": "Positive",
            }

        for name in self.param_names:
            self.param_info[name]["dims"] = self.param_dims[name]

    def make_symbolic_graph(self) -> None:
        k_endog = self.k_endog
        k_states = self.k_states // k_endog
        k_posdef = self.k_posdef // k_endog

        initial_trend = self.make_and_register_variable(
            f"{self.name}_initial",
            shape=(k_states,) if k_endog == 1 else (k_endog, k_states),
        )
        self.ssm["initial_state", :] = initial_trend.ravel()

        triu_idx = pt.triu_indices(k_states)
        T = pt.zeros((k_states, k_states))[triu_idx[0], triu_idx[1]].set(1)

        self.ssm["transition", :, :] = pt.specify_shape(
            pt.linalg.block_diag(*[T for _ in range(k_endog)]), (self.k_states, self.k_states)
        )

        R = np.eye(k_states)
        R = R[:, self.innovations_order]

        self.ssm["selection", :, :] = pt.specify_shape(
            pt.linalg.block_diag(*[R for _ in range(k_endog)]), (self.k_states, self.k_posdef)
        )

        Z = np.array([1.0] + [0.0] * (k_states - 1)).reshape((1, -1))

        self.ssm["design", :, :] = pt.specify_shape(
            pt.linalg.block_diag(*[Z for _ in range(k_endog)]), (self.k_endog, self.k_states)
        )

        if k_posdef > 0:
            sigma_trend = self.make_and_register_variable(
                f"{self.name}_sigma",
                shape=(k_posdef,) if k_endog == 1 else (k_endog, k_posdef),
            )
            diag_idx = np.diag_indices(k_posdef * k_endog)
            idx = np.s_["state_cov", diag_idx[0], diag_idx[1]]
            self.ssm[idx] = (sigma_trend**2).ravel()
