import numpy as np

from pytensor import tensor as pt
from scipy import linalg

from pymc_extras.statespace.models.structural.core import Component
from pymc_extras.statespace.models.structural.utils import _frequency_transition_block


class CycleComponent(Component):
    r"""
    A component for modeling longer-term cyclical effects

    Supports both univariate and multivariate time series. For multivariate time series,
    each endogenous variable gets its own independent cycle component with separate
    cosine/sine states and optional variable-specific innovation variances.

    Parameters
    ----------
    name: str
        Name of the component. Used in generated coordinates and state names. If None, a descriptive name will be
        used.

    cycle_length: int, optional
        The length of the cycle, in the calendar units of your data. For example, if your data is monthly, and you
        want to model a 12-month cycle, use ``cycle_length=12``. You cannot specify both ``cycle_length`` and
        ``estimate_cycle_length``.

    estimate_cycle_length: bool, default False
        Whether to estimate the cycle length. If True, an additional parameter, ``cycle_length`` will be added to the
        model. You cannot specify both ``cycle_length`` and ``estimate_cycle_length``.

    dampen: bool, default False
        Whether to dampen the cycle by multiplying by a dampening factor :math:`\rho` at every timestep. If true,
        an additional parameter, ``dampening_factor`` will be added to the model.

    innovations: bool, default True
        Whether to include stochastic innovations in the strength of the seasonal effect. If True, an additional
        parameter, ``sigma_{name}`` will be added to the model.
        For multivariate time series, this is a vector (variable-specific innovation variances).

    observed_state_names: list[str], optional
        Names of the observed state variables. For univariate time series, defaults to ``["data"]``.
        For multivariate time series, specify a list of names for each endogenous variable.

    Notes
    -----
    The cycle component is very similar in implementation to the frequency domain seasonal component, expect that it
    is restricted to n=1. The cycle component can be expressed:

    .. math::
        \begin{align}
            \gamma_t &= \rho \gamma_{t-1} \cos \lambda + \rho \gamma_{t-1}^\star \sin \lambda + \omega_{t} \\
            \gamma_{t}^\star &= -\rho \gamma_{t-1} \sin \lambda + \rho \gamma_{t-1}^\star \cos \lambda + \omega_{t}^\star \\
            \lambda &= \frac{2\pi}{s}
        \end{align}

    Where :math:`s` is the ``cycle_length``. [1] recommend that this component be used for longer term cyclical
    effects, such as business cycles, and that the seasonal component be used for shorter term effects, such as
    weekly or monthly seasonality.

    Unlike a FrequencySeasonality component, the length of a CycleComponent can be estimated.

    **Multivariate Support:**
    For multivariate time series with k endogenous variables, the component creates:
    - 2k states (cosine and sine components for each variable)
    - Block diagonal transition and selection matrices
    - Variable-specific innovation variances (optional)
    - Proper parameter shapes: (k, 2) for initial states, (k,) for innovation variances

    Examples
    --------
    **Univariate Example:**
    Estimate a business cycle with length between 6 and 12 years:

    .. code:: python

        from pymc_extras.statespace import structural as st
        import pymc as pm
        import pytensor.tensor as pt
        import pandas as pd
        import numpy as np

        data = np.random.normal(size=(100, 1))

        # Build the structural model
        grw = st.LevelTrendComponent(order=1, innovations_order=1)
        cycle = st.CycleComponent('business_cycle', estimate_cycle_length=True, dampen=False)
        ss_mod = (grw + cycle).build()

        # Estimate with PyMC
        with pm.Model(coords=ss_mod.coords) as model:
            P0 = pm.Deterministic('P0', pt.eye(ss_mod.k_states), dims=ss_mod.param_dims['P0'])
            intitial_trend = pm.Normal('initial_trend', dims=ss_mod.param_dims['initial_trend'])
            sigma_trend = pm.HalfNormal('sigma_trend', dims=ss_mod.param_dims['sigma_trend'])

            cycle_strength = pm.Normal("business_cycle", dims=ss_mod.param_dims["business_cycle"])
            cycle_length = pm.Uniform('business_cycle_length', lower=6, upper=12)

            sigma_cycle = pm.HalfNormal('sigma_business_cycle', sigma=1)

            ss_mod.build_statespace_graph(data)
            idata = pm.sample()

    **Multivariate Example:**
    Model cycles for multiple economic indicators with variable-specific innovation variances:

    .. code:: python

        # Multivariate cycle component
        cycle = st.CycleComponent(
            name='business_cycle',
            cycle_length=12,
            estimate_cycle_length=False,
            innovations=True,
            dampen=True,
            observed_state_names=['gdp', 'unemployment', 'inflation']
        )

        # Build the model
        ss_mod = cycle.build()

        # In PyMC model:
        with pm.Model(coords=ss_mod.coords) as model:
            P0 = pm.Deterministic("P0", pt.eye(ss_mod.k_states), dims=ss_mod.param_dims["P0"])
            # Initial states: shape (3, 2) for 3 variables, 2 states each
            cycle_init = pm.Normal('business_cycle', dims=('business_cycle_endog', 'business_cycle_state'))

            # Dampening factor: scalar (shared across variables)
            dampening = pm.Uniform('business_cycle_dampening_factor', lower=0.8, upper=1.0)

            # Innovation variances: shape (3,) for variable-specific variances
            sigma_cycle = pm.HalfNormal('sigma_business_cycle', dims=('business_cycle_endog',))

            ss_mod.build_statespace_graph(data)
            idata = pm.sample()

    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
        Time Series Analysis by State Space Methods: Second Edition.
        Oxford University Press.
    """

    def __init__(
        self,
        name: str | None = None,
        cycle_length: int | None = None,
        estimate_cycle_length: bool = False,
        dampen: bool = False,
        innovations: bool = True,
        observed_state_names: list[str] | None = None,
    ):
        if observed_state_names is None:
            observed_state_names = ["data"]

        if cycle_length is None and not estimate_cycle_length:
            raise ValueError("Must specify cycle_length if estimate_cycle_length is False")
        if cycle_length is not None and estimate_cycle_length:
            raise ValueError("Cannot specify cycle_length if estimate_cycle_length is True")
        if name is None:
            cycle = int(cycle_length) if cycle_length is not None else "Estimate"
            name = f"Cycle[s={cycle}, dampen={dampen}, innovations={innovations}]"

        self.estimate_cycle_length = estimate_cycle_length
        self.cycle_length = cycle_length
        self.innovations = innovations
        self.dampen = dampen
        self.n_coefs = 1

        k_endog = len(observed_state_names)

        k_states = 2 * k_endog
        k_posdef = 2 * k_endog

        obs_state_idx = np.zeros(k_states)
        obs_state_idx[slice(0, k_states, 2)] = 1

        super().__init__(
            name=name,
            k_endog=k_endog,
            k_states=k_states,
            k_posdef=k_posdef,
            measurement_error=False,
            combine_hidden_states=True,
            obs_state_idxs=obs_state_idx,
            observed_state_names=observed_state_names,
        )

    def make_symbolic_graph(self) -> None:
        if self.k_endog == 1:
            self.ssm["design", 0, slice(0, self.k_states, 2)] = 1
            self.ssm["selection", :, :] = np.eye(self.k_states)
            init_state = self.make_and_register_variable(f"{self.name}", shape=(self.k_states,))

        else:
            Z = np.array([1.0, 0.0]).reshape((1, -1))
            design_matrix = linalg.block_diag(*[Z for _ in range(self.k_endog)])
            self.ssm["design", :, :] = pt.as_tensor_variable(design_matrix)

            R = np.eye(2)  # 2x2 identity for each cycle component
            selection_matrix = linalg.block_diag(*[R for _ in range(self.k_endog)])
            self.ssm["selection", :, :] = pt.as_tensor_variable(selection_matrix)

            init_state = self.make_and_register_variable(f"{self.name}", shape=(self.k_endog, 2))

        self.ssm["initial_state", :] = init_state.ravel()

        if self.estimate_cycle_length:
            lamb = self.make_and_register_variable(f"{self.name}_length", shape=())
        else:
            lamb = self.cycle_length

        if self.dampen:
            rho = self.make_and_register_variable(f"{self.name}_dampening_factor", shape=())
        else:
            rho = 1

        T = rho * _frequency_transition_block(lamb, j=1)
        if self.k_endog == 1:
            self.ssm["transition", :, :] = T
        else:
            # can't make the linalg.block_diag logic work here
            # doing it manually for now
            for i in range(self.k_endog):
                start_idx = i * 2
                end_idx = (i + 1) * 2
                self.ssm["transition", start_idx:end_idx, start_idx:end_idx] = T

        if self.innovations:
            if self.k_endog == 1:
                sigma_cycle = self.make_and_register_variable(f"sigma_{self.name}", shape=())
                self.ssm["state_cov", :, :] = pt.eye(self.k_posdef) * sigma_cycle**2
            else:
                sigma_cycle = self.make_and_register_variable(
                    f"sigma_{self.name}", shape=(self.k_endog,)
                )
                # can't make the linalg.block_diag logic work here
                # doing it manually for now
                for i in range(self.k_endog):
                    start_idx = i * 2
                    end_idx = (i + 1) * 2
                    Q_block = pt.eye(2) * sigma_cycle[i] ** 2
                    self.ssm["state_cov", start_idx:end_idx, start_idx:end_idx] = Q_block

    def populate_component_properties(self):
        self.state_names = [f"{self.name}_{f}" for f in ["Cos", "Sin"]]
        self.param_names = [f"{self.name}"]

        if self.k_endog == 1:
            self.param_dims = {self.name: (f"{self.name}_state",)}
            self.coords = {f"{self.name}_state": self.state_names}
            self.param_info = {
                f"{self.name}": {
                    "shape": (2,),
                    "constraints": None,
                    "dims": (f"{self.name}_state",),
                }
            }
        else:
            self.param_dims = {self.name: (f"{self.name}_endog", f"{self.name}_state")}
            self.coords = {
                f"{self.name}_state": self.state_names,
                f"{self.name}_endog": self.observed_state_names,
            }
            self.param_info = {
                f"{self.name}": {
                    "shape": (self.k_endog, 2),
                    "constraints": None,
                    "dims": (f"{self.name}_endog", f"{self.name}_state"),
                }
            }

        if self.estimate_cycle_length:
            self.param_names += [f"{self.name}_length"]
            self.param_info[f"{self.name}_length"] = {
                "shape": (),
                "constraints": "Positive, non-zero",
                "dims": None,
            }

        if self.dampen:
            self.param_names += [f"{self.name}_dampening_factor"]
            self.param_info[f"{self.name}_dampening_factor"] = {
                "shape": (),
                "constraints": "0 < x ≤ 1",
                "dims": None,
            }

        if self.innovations:
            self.param_names += [f"sigma_{self.name}"]
            if self.k_endog == 1:
                self.param_info[f"sigma_{self.name}"] = {
                    "shape": (),
                    "constraints": "Positive",
                    "dims": None,
                }
            else:
                self.param_dims[f"sigma_{self.name}"] = (f"{self.name}_endog",)
                self.param_info[f"sigma_{self.name}"] = {
                    "shape": (self.k_endog,),
                    "constraints": "Positive",
                    "dims": (f"{self.name}_endog",),
                }
            self.shock_names = self.state_names.copy()
