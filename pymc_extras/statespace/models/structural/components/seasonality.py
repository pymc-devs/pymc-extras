import numpy as np

from pytensor import tensor as pt

from pymc_extras.statespace.models.structural.core import Component
from pymc_extras.statespace.models.structural.utils import _frequency_transition_block


class TimeSeasonality(Component):
    r"""
    Seasonal component, modeled in the time domain

    Parameters
    ----------
    season_length: int
        The number of periods in a single seasonal cycle, e.g. 12 for monthly data with annual seasonal pattern, 7 for
        daily data with weekly seasonal pattern, etc.

    innovations: bool, default True
        Whether to include stochastic innovations in the strength of the seasonal effect

    name: str, default None
        A name for this seasonal component. Used to label dimensions and coordinates. Useful when multiple seasonal
        components are included in the same model. Default is ``f"Seasonal[s={season_length}]"``

    state_names: list of str, default None
        List of strings for seasonal effect labels. If provided, it must be of length ``season_length``. An example
        would be ``state_names = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']`` when data is daily with a weekly
        seasonal pattern (``season_length = 7``).

        If None, states will be numbered ``[State_0, ..., State_s]``

    remove_first_state: bool, default True
        If True, the first state will be removed from the model. This is done because there are only n-1 degrees of
        freedom in the seasonal component, and one state is not identified. If False, the first state will be
        included in the model, but it will not be identified -- you will need to handle this in the priors (e.g. with
        ZeroSumNormal).

    Notes
    -----
    A seasonal effect is any pattern that repeats every fixed interval. Although there are many possible ways to
    model seasonal effects, the implementation used here is the one described by [1] as the "canonical" time domain
    representation. The seasonal component can be expressed:

    .. math::
        \gamma_t = -\sum_{i=1}^{s-1} \gamma_{t-i} + \omega_t, \quad \omega_t \sim N(0, \sigma_\gamma)

    Where :math:`s` is the ``seasonal_length`` parameter and :math:`\omega_t` is the (optional) stochastic innovation.
    To give interpretation to the :math:`\gamma` terms, it is helpful to work  through the algebra for a simple
    example. Let :math:`s=4`, and omit the shock term. Define initial conditions :math:`\gamma_0, \gamma_{-1},
    \gamma_{-2}`. The value of the seasonal component for the first 5 timesteps will be:

    .. math::
        \begin{align}
            \gamma_1 &= -\gamma_0 - \gamma_{-1} - \gamma_{-2} \\
             \gamma_2 &= -\gamma_1 - \gamma_0 - \gamma_{-1} \\
                       &= -(-\gamma_0 - \gamma_{-1} - \gamma_{-2}) - \gamma_0 - \gamma_{-1}  \\
                       &= (\gamma_0 - \gamma_0 )+ (\gamma_{-1} - \gamma_{-1}) + \gamma_{-2} \\
                       &= \gamma_{-2} \\
              \gamma_3 &= -\gamma_2 - \gamma_1 - \gamma_0  \\
                       &= -\gamma_{-2} - (-\gamma_0 - \gamma_{-1} - \gamma_{-2}) - \gamma_0 \\
                       &=  (\gamma_{-2} - \gamma_{-2}) + \gamma_{-1} + (\gamma_0 - \gamma_0) \\
                       &= \gamma_{-1} \\
              \gamma_4 &= -\gamma_3 - \gamma_2 - \gamma_1 \\
                       &= -\gamma_{-1} - \gamma_{-2} -(-\gamma_0 - \gamma_{-1} - \gamma_{-2}) \\
                       &= (\gamma_{-2} - \gamma_{-2}) + (\gamma_{-1} - \gamma_{-1}) + \gamma_0 \\
                       &= \gamma_0 \\
              \gamma_5 &= -\gamma_4 - \gamma_3 - \gamma_2 \\
                       &= -\gamma_0 - \gamma_{-1} - \gamma_{-2} \\
                       &= \gamma_1
        \end{align}

    This exercise shows that, given a list ``initial_conditions`` of length ``s-1``, the effects of this model will be:

        - Period 1: ``-sum(initial_conditions)``
        - Period 2: ``initial_conditions[-1]``
        - Period 3: ``initial_conditions[-2]``
        - ...
        - Period s: ``initial_conditions[0]``
        - Period s+1: ``-sum(initial_condition)``

    And so on. So for interpretation, the ``season_length - 1`` initial states are, when reversed, the coefficients
    associated with ``state_names[1:]``.

    .. warning::
        Although the ``state_names`` argument expects a list of length ``season_length``, only ``state_names[1:]``
        will be saved as model dimensions, since the 1st coefficient is not identified (it is defined as
        :math:`-\sum_{i=1}^{s} \gamma_{t-i}`).

    Examples
    --------
    Estimate monthly with a model with a gaussian random walk trend and monthly seasonality:

    .. code:: python

        from pymc_extras.statespace import structural as st
        import pymc as pm
        import pytensor.tensor as pt
        import pandas as pd

        # Get month names
        state_names = pd.date_range('1900-01-01', '1900-12-31', freq='MS').month_name().tolist()

        # Build the structural model
        grw = st.LevelTrendComponent(order=1, innovations_order=1)
        annual_season = st.TimeSeasonality(season_length=12, name='annual', state_names=state_names, innovations=False)
        ss_mod = (grw + annual_season).build()

        # Estimate with PyMC
        with pm.Model(coords=ss_mod.coords) as model:
            P0 = pm.Deterministic('P0', pt.eye(ss_mod.k_states) * 10, dims=ss_mod.param_dims['P0'])
            intitial_trend = pm.Deterministic('initial_trend', pt.zeros(1), dims=ss_mod.param_dims['initial_trend'])
            annual_coefs = pm.Normal('annual_coefs', sigma=1e-2, dims=ss_mod.param_dims['annual_coefs'])
            trend_sigmas = pm.HalfNormal('trend_sigmas', sigma=1e-6, dims=ss_mod.param_dims['trend_sigmas'])
            ss_mod.build_statespace_graph(data)
            idata = pm.sample(nuts_sampler='numpyro')

    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
        Time Series Analysis by State Space Methods: Second Edition.
        Oxford University Press.
    """

    def __init__(
        self,
        season_length: int,
        innovations: bool = True,
        name: str | None = None,
        state_names: list | None = None,
        remove_first_state: bool = True,
        observed_state_names: list[str] | None = None,
    ):
        if observed_state_names is None:
            observed_state_names = ["data"]

        if name is None:
            name = f"Seasonal[s={season_length}]"
        if state_names is None:
            state_names = [f"{name}_{i}" for i in range(season_length)]
        else:
            if len(state_names) != season_length:
                raise ValueError(
                    f"state_names must be a list of length season_length, got {len(state_names)}"
                )
            state_names = state_names.copy()

        self.innovations = innovations
        self.remove_first_state = remove_first_state

        if self.remove_first_state:
            # In traditional models, the first state isn't identified, so we can help out the user by automatically
            # discarding it.
            # TODO: Can this be stashed and reconstructed automatically somehow?
            state_names.pop(0)

        self.provided_state_names = state_names

        k_states = season_length - int(self.remove_first_state)
        k_endog = len(observed_state_names)
        k_posdef = int(innovations)

        super().__init__(
            name=name,
            k_endog=k_endog,
            k_states=k_states * k_endog,
            k_posdef=k_posdef * k_endog,
            observed_state_names=observed_state_names,
            measurement_error=False,
            combine_hidden_states=True,
            obs_state_idxs=np.tile(np.array([1.0] + [0.0] * (k_states - 1)), k_endog),
        )

    def populate_component_properties(self):
        k_states = self.k_states // self.k_endog
        k_endog = self.k_endog

        self.state_names = [
            f"{state_name}[{endog_name}]"
            for endog_name in self.observed_state_names
            for state_name in self.provided_state_names
        ]
        self.param_names = [f"{self.name}_coefs"]

        self.param_info = {
            f"{self.name}_coefs": {
                "shape": (k_states,) if k_endog == 1 else (k_endog, k_states),
                "constraints": None,
                "dims": (f"{self.name}_state",)
                if k_endog == 1
                else (f"{self.name}_endog", f"{self.name}_state"),
            }
        }
        self.param_dims = {f"{self.name}_coefs": (f"{self.name}_state",)}
        self.coords = {f"{self.name}_state": self.state_names}

        if self.innovations:
            self.param_names += [f"sigma_{self.name}"]
            self.param_info[f"sigma_{self.name}"] = {
                "shape": (),
                "constraints": "Positive",
                "dims": None,
            }
            self.shock_names = [f"{self.name}[{name}]" for name in self.observed_state_names]

    def make_symbolic_graph(self) -> None:
        k_states = self.k_states // self.k_endog
        k_posdef = self.k_posdef // self.k_endog
        k_endog = self.k_endog

        if self.remove_first_state:
            # In this case, parameters are normalized to sum to zero, so the current state is the negative sum of
            # all previous states.
            T = np.eye(k_states, k=-1)
            T[0, :] = -1
        else:
            # In this case we assume the user to be responsible for ensuring the states sum to zero, so T is just a
            # circulant matrix that cycles between the states.
            T = np.eye(k_states, k=1)
            T[-1, 0] = 1

        self.ssm["transition", :, :] = pt.linalg.block_diag(*[T for _ in range(k_endog)])

        Z = pt.zeros((1, k_states))[0, 0].set(1)
        self.ssm["design", :, :] = pt.linalg.block_diag(*[Z for _ in range(k_endog)])

        initial_states = self.make_and_register_variable(
            f"{self.name}_coefs", shape=(k_states,) if k_endog == 1 else (k_endog, k_states)
        )
        self.ssm["initial_state", :] = initial_states.ravel()

        if self.innovations:
            R = pt.zeros((k_states, k_posdef))[0, 0].set(1.0)
            self.ssm["selection", :, :] = pt.join(0, *[R for _ in range(k_endog)])
            season_sigma = self.make_and_register_variable(
                f"sigma_{self.name}", shape=() if k_endog == 1 else (k_endog,)
            )
            cov_idx = ("state_cov", *np.diag_indices(k_posdef * k_endog))
            self.ssm[cov_idx] = season_sigma**2


class FrequencySeasonality(Component):
    r"""
    Seasonal component, modeled in the frequency domain

    Parameters
    ----------
    season_length: float
        The number of periods in a single seasonal cycle, e.g. 12 for monthly data with annual seasonal pattern, 7 for
        daily data with weekly seasonal pattern, etc. Non-integer seasonal_length is also permitted, for example
        365.2422 days in a (solar) year.

    n: int
        Number of fourier features to include in the seasonal component. Default is ``season_length // 2``, which
        is the maximum possible. A smaller number can be used for a more wave-like seasonal pattern.

    name: str, default None
        A name for this seasonal component. Used to label dimensions and coordinates. Useful when multiple seasonal
        components are included in the same model. Default is ``f"Seasonal[s={season_length}, n={n}]"``

    innovations: bool, default True
        Whether to include stochastic innovations in the strength of the seasonal effect

    Notes
    -----
    A seasonal effect is any pattern that repeats every fixed interval. Although there are many possible ways to
    model seasonal effects, the implementation used here is the one described by [1] as the "canonical" frequency domain
    representation. The seasonal component can be expressed:

    .. math::
        \begin{align}
            \gamma_t &= \sum_{j=1}^{2n} \gamma_{j,t} \\
            \gamma_{j, t+1} &= \gamma_{j,t} \cos \lambda_j + \gamma_{j,t}^\star \sin \lambda_j + \omega_{j, t} \\
            \gamma_{j, t}^\star &= -\gamma_{j,t} \sin \lambda_j + \gamma_{j,t}^\star \cos \lambda_j + \omega_{j,t}^\star
            \lambda_j &= \frac{2\pi j}{s}
        \end{align}

    Where :math:`s` is the ``seasonal_length``.

    Unlike a ``TimeSeasonality`` component, a ``FrequencySeasonality`` component does not require integer season
    length. In addition, for long seasonal periods, it is possible to obtain a more compact state space representation
    by choosing ``n << s // 2``. Using ``TimeSeasonality``, an annual seasonal pattern in daily data requires 364
    states, whereas ``FrequencySeasonality`` always requires ``2 * n`` states, regardless of the ``seasonal_length``.
    The price of this compactness is less representational power. At ``n = 1``, the seasonal pattern will be a pure
    sine wave. At ``n = s // 2``, any arbitrary pattern can be represented.

    One cost of the added flexibility of ``FrequencySeasonality`` is reduced interpretability. States of this model are
    coefficients :math:`\gamma_1, \gamma^\star_1, \gamma_2, \gamma_2^\star ..., \gamma_n, \gamma^\star_n` associated
    with different frequencies in the fourier representation of the seasonal pattern. As a result, it is not possible
    to isolate and identify a "Monday" effect, for instance.
    """

    def __init__(
        self,
        season_length,
        n=None,
        name=None,
        innovations=True,
        observed_state_names: list[str] | None = None,
    ):
        if observed_state_names is None:
            observed_state_names = ["data"]

        k_endog = len(observed_state_names)

        if n is None:
            n = int(season_length / 2)
        if name is None:
            name = f"Frequency[s={season_length}, n={n}]"

        k_states = n * 2
        self.n = n
        self.season_length = season_length
        self.innovations = innovations

        # If the model is completely saturated (n = s // 2), the last state will not be identified, so it shouldn't
        # get a parameter assigned to it and should just be fixed to zero.
        # Test this way (rather than n == s // 2) to catch cases when n is non-integer.
        self.last_state_not_identified = self.season_length / self.n == 2.0
        self.n_coefs = k_states - int(self.last_state_not_identified)

        obs_state_idx = np.zeros(k_states)
        obs_state_idx[slice(0, k_states, 2)] = 1
        obs_state_idx = np.tile(obs_state_idx, k_endog)

        super().__init__(
            name=name,
            k_endog=k_endog,
            k_states=k_states * k_endog,
            k_posdef=k_states * int(self.innovations) * k_endog,
            observed_state_names=observed_state_names,
            measurement_error=False,
            combine_hidden_states=True,
            obs_state_idxs=obs_state_idx,
        )

    def make_symbolic_graph(self) -> None:
        k_endog = self.k_endog
        k_states = self.k_states // k_endog
        k_posdef = self.k_posdef // k_endog
        n_coefs = self.n_coefs

        Z = pt.zeros((1, k_states))[0, slice(0, k_states, 2)].set(1.0)

        self.ssm["design", :, :] = pt.linalg.block_diag(*[Z for _ in range(k_endog)])

        init_state = self.make_and_register_variable(
            f"{self.name}", shape=(n_coefs,) if k_endog == 1 else (k_endog, n_coefs)
        )

        init_state_idx = np.concatenate(
            [
                np.arange(k_states * i, (i + 1) * k_states, dtype=int)[:n_coefs]
                for i in range(k_endog)
            ],
            axis=0,
        )

        self.ssm["initial_state", init_state_idx] = init_state.ravel()

        T_mats = [_frequency_transition_block(self.season_length, j + 1) for j in range(self.n)]
        T = pt.linalg.block_diag(*T_mats)
        self.ssm["transition", :, :] = pt.linalg.block_diag(*[T for _ in range(k_endog)])

        if self.innovations:
            sigma_season = self.make_and_register_variable(
                f"sigma_{self.name}", shape=() if k_endog == 1 else (k_endog,)
            )
            self.ssm["selection", :, :] = pt.eye(self.k_states)
            self.ssm["state_cov", :, :] = pt.eye(self.k_posdef) * pt.repeat(
                sigma_season**2, k_posdef
            )

    def populate_component_properties(self):
        k_endog = self.k_endog
        n_coefs = self.n_coefs
        k_states = self.k_states // k_endog

        self.state_names = [
            f"{self.name}_{f}_{i}[{obs_state_name}]"
            for obs_state_name in self.observed_state_names
            for i in range(self.n)
            for f in ["Cos", "Sin"]
        ]
        self.param_names = [f"{self.name}"]

        self.param_dims = {self.name: (f"{self.name}_state",)}
        self.param_info = {
            f"{self.name}": {
                "shape": (n_coefs,) if k_endog == 1 else (k_endog, n_coefs),
                "constraints": None,
                "dims": (f"{self.name}_state",)
                if k_endog == 1
                else (f"{self.name}_endog", f"{self.name}_state"),
            }
        }

        # Regardless of whether the fourier basis are saturated, there will always be one symbolic state per basis.
        # That's why the self.states is just a simple loop over everything. But when saturated, one of those states
        # doesn't have an associated **parameter**, so the coords need to be adjusted to reflect this.
        init_state_idx = np.concatenate(
            [
                np.arange(k_states * i, (i + 1) * k_states, dtype=int)[:n_coefs]
                for i in range(k_endog)
            ],
            axis=0,
        )
        self.coords = {f"{self.name}_state": [self.state_names[i] for i in init_state_idx]}

        if self.innovations:
            self.shock_names = self.state_names.copy()
            self.param_names += [f"sigma_{self.name}"]
            self.param_info[f"sigma_{self.name}"] = {
                "shape": () if k_endog == 1 else (k_endog, n_coefs),
                "constraints": "Positive",
                "dims": None if k_endog == 1 else (f"{self.name}_endog",),
            }
