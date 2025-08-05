import numpy as np

from pytensor import tensor as pt
from pytensor.tensor.slinalg import block_diag

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

    share_states: bool, default False
        Whether latent states are shared across the observed states. If True, there will be only one set of latent
        states, which are observed by all observed states. If False, each observed state has its own set of
        latent states. This argument has no effect if `k_endog` is 1.

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
        cycle = st.CycleComponent(
            "business_cycle", cycle_length=12, estimate_cycle_length=False, innovations=True, dampen=True
        )
        ss_mod = (grw + cycle).build()

        # Estimate with PyMC
        with pm.Model(coords=ss_mod.coords) as model:
            P0 = pm.Deterministic('P0', pt.eye(ss_mod.k_states), dims=ss_mod.param_dims['P0'])

            initial_level_trend = pm.Normal('initial_level_trend', dims=ss_mod.param_dims['initial_level_trend'])
            sigma_level_trend = pm.HalfNormal('sigma_level_trend', dims=ss_mod.param_dims['sigma_level_trend'])

            business_cycle = pm.Normal("business_cycle", dims=ss_mod.param_dims["business_cycle"])
            dampening = pm.Beta("dampening_factor_business_cycle", 2, 2)
            sigma_cycle = pm.HalfNormal("sigma_business_cycle", sigma=1)

            ss_mod.build_statespace_graph(data)
            idata = pm.sample(
                nuts_sampler="nutpie", nuts_sampler_kwargs={"backend": "JAX", "gradient_backend": "JAX"}
            )

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
        ss_mod = cycle.build()

        with pm.Model(coords=ss_mod.coords) as model:
            P0 = pm.Deterministic("P0", pt.eye(ss_mod.k_states), dims=ss_mod.param_dims["P0"])
            # Initial states: shape (3, 2) for 3 variables, 2 states each
            business_cycle = pm.Normal('business_cycle', dims=ss_mod.param_dims["business_cycle"])

            # Dampening factor: scalar (shared across variables)
            dampening = pm.Beta("dampening_factor_business_cycle", 2, 2)

            # Innovation variances: shape (3,) for variable-specific variances
            sigma_cycle = pm.HalfNormal(
                "sigma_business_cycle", dims=ss_mod.param_dims["sigma_business_cycle"]
            )

            ss_mod.build_statespace_graph(data)
            idata = pm.sample(
                nuts_sampler="nutpie", nuts_sampler_kwargs={"backend": "JAX", "gradient_backend": "JAX"}
            )

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
        share_states: bool = False,
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

        self.share_states = share_states
        self.estimate_cycle_length = estimate_cycle_length
        self.cycle_length = cycle_length
        self.innovations = innovations
        self.dampen = dampen
        self.n_coefs = 1

        k_endog = len(observed_state_names)

        k_states = 2 if share_states else 2 * k_endog
        k_posdef = 2 if share_states else 2 * k_endog

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
            share_states=share_states,
        )

    def make_symbolic_graph(self) -> None:
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog

        Z = np.array([1.0, 0.0]).reshape((1, -1))
        design_matrix = block_diag(*[Z for _ in range(k_endog_effective)])
        self.ssm["design", :, :] = pt.as_tensor_variable(design_matrix)

        # selection matrix R defines structure of innovations (always identity for cycle components)
        # when innovations=False, state cov Q=0, hence R @ Q @ R.T = 0
        R = np.eye(2)  # 2x2 identity for each cycle component
        selection_matrix = block_diag(*[R for _ in range(k_endog_effective)])
        self.ssm["selection", :, :] = pt.as_tensor_variable(selection_matrix)

        init_state = self.make_and_register_variable(
            f"params_{self.name}",
            shape=(k_endog_effective, 2) if k_endog_effective > 1 else (self.k_states,),
        )
        self.ssm["initial_state", :] = init_state.ravel()

        if self.estimate_cycle_length:
            lamb = self.make_and_register_variable(f"length_{self.name}", shape=())
        else:
            lamb = self.cycle_length

        if self.dampen:
            rho = self.make_and_register_variable(f"dampening_factor_{self.name}", shape=())
        else:
            rho = 1

        T = rho * _frequency_transition_block(lamb, j=1)
        transition = block_diag(*[T for _ in range(k_endog_effective)])
        self.ssm["transition"] = pt.specify_shape(transition, (self.k_states, self.k_states))

        if self.innovations:
            if k_endog_effective == 1:
                sigma_cycle = self.make_and_register_variable(f"sigma_{self.name}", shape=())
                self.ssm["state_cov", :, :] = pt.eye(self.k_posdef) * sigma_cycle**2
            else:
                sigma_cycle = self.make_and_register_variable(
                    f"sigma_{self.name}", shape=(k_endog_effective,)
                )
                state_cov = block_diag(
                    *[pt.eye(2) * sigma_cycle[i] ** 2 for i in range(k_endog_effective)]
                )
                self.ssm["state_cov"] = pt.specify_shape(state_cov, (self.k_states, self.k_states))
        else:
            # explicitly set state cov to 0 when no innovations
            self.ssm["state_cov", :, :] = pt.zeros((self.k_posdef, self.k_posdef))

    def populate_component_properties(self):
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog

        base_names = [f"{f}_{self.name}" for f in ["Cos", "Sin"]]

        if self.share_states:
            self.state_names = [f"{name}[shared]" for name in base_names]
        else:
            self.state_names = [
                f"{name}[{var_name}]" if k_endog_effective > 1 else name
                for var_name in self.observed_state_names
                for name in base_names
            ]

        self.param_names = [f"params_{self.name}"]

        if k_endog_effective == 1:
            self.param_dims = {f"params_{self.name}": (f"state_{self.name}",)}
            self.coords = {f"state_{self.name}": base_names}
            self.param_info = {
                f"params_{self.name}": {
                    "shape": (2,),
                    "constraints": None,
                    "dims": (f"state_{self.name}",),
                }
            }
        else:
            self.param_dims = {f"params_{self.name}": (f"endog_{self.name}", f"state_{self.name}")}
            self.coords = {
                f"state_{self.name}": [f"Cos_{self.name}", f"Sin_{self.name}"],
                f"endog_{self.name}": self.observed_state_names,
            }
            self.param_info = {
                f"params_{self.name}": {
                    "shape": (k_endog_effective, 2),
                    "constraints": None,
                    "dims": (f"endog_{self.name}", f"state_{self.name}"),
                }
            }

        if self.estimate_cycle_length:
            self.param_names += [f"length_{self.name}"]
            self.param_info[f"length_{self.name}"] = {
                "shape": () if k_endog_effective == 1 else (k_endog_effective,),
                "constraints": "Positive, non-zero",
                "dims": None if k_endog_effective == 1 else (f"endog_{self.name}",),
            }

        if self.dampen:
            self.param_names += [f"dampening_factor_{self.name}"]
            self.param_info[f"dampening_factor_{self.name}"] = {
                "shape": () if k_endog_effective == 1 else (k_endog_effective,),
                "constraints": "0 < x ≤ 1",
                "dims": None if k_endog_effective == 1 else (f"endog_{self.name}",),
            }

        if self.innovations:
            self.param_names += [f"sigma_{self.name}"]
            if k_endog_effective == 1:
                self.param_info[f"sigma_{self.name}"] = {
                    "shape": (),
                    "constraints": "Positive",
                    "dims": None,
                }
            else:
                self.param_dims[f"sigma_{self.name}"] = (f"endog_{self.name}",)
                self.param_info[f"sigma_{self.name}"] = {
                    "shape": (k_endog_effective,),
                    "constraints": "Positive",
                    "dims": (f"endog_{self.name}",),
                }
            self.shock_names = self.state_names.copy()
