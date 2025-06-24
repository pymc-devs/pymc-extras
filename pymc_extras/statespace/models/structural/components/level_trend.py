import numpy as np

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
        name: str = "LevelTrend",
        observed_state_names: list[str] | None = None,
    ):
        if innovations_order is None:
            innovations_order = order

        if observed_state_names is None:
            observed_state_names = ["data"]

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
            k_endog=len(observed_state_names),
            k_states=k_states,
            k_posdef=k_posdef,
            observed_state_names=observed_state_names,
            measurement_error=False,
            combine_hidden_states=False,
            obs_state_idxs=np.array([1.0] + [0.0] * (k_states - 1)),
        )

    def populate_component_properties(self):
        name_slice = POSITION_DERIVATIVE_NAMES[: self.k_states]
        self.param_names = ["initial_trend"]
        self.state_names = [name for name, mask in zip(name_slice, self._order_mask) if mask]
        self.param_dims = {"initial_trend": ("trend_state",)}
        self.coords = {"trend_state": self.state_names}
        self.param_info = {"initial_trend": {"shape": (self.k_states,), "constraints": None}}

        if self.k_posdef > 0:
            self.param_names += ["sigma_trend"]
            self.shock_names = [
                name for name, mask in zip(name_slice, self.innovations_order) if mask
            ]
            self.param_dims["sigma_trend"] = ("trend_shock",)
            self.coords["trend_shock"] = self.shock_names
            self.param_info["sigma_trend"] = {"shape": (self.k_posdef,), "constraints": "Positive"}

        for name in self.param_names:
            self.param_info[name]["dims"] = self.param_dims[name]

    def make_symbolic_graph(self) -> None:
        initial_trend = self.make_and_register_variable("initial_trend", shape=(self.k_states,))
        self.ssm["initial_state", :] = initial_trend
        triu_idx = np.triu_indices(self.k_states)
        self.ssm[np.s_["transition", triu_idx[0], triu_idx[1]]] = 1

        R = np.eye(self.k_states)
        R = R[:, self.innovations_order]
        self.ssm["selection", :, :] = R

        self.ssm["design", 0, :] = np.array([1.0] + [0.0] * (self.k_states - 1))

        if self.k_posdef > 0:
            sigma_trend = self.make_and_register_variable("sigma_trend", shape=(self.k_posdef,))
            diag_idx = np.diag_indices(self.k_posdef)
            idx = np.s_["state_cov", diag_idx[0], diag_idx[1]]
            self.ssm[idx] = sigma_trend**2
