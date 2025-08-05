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
        daily data with weekly seasonal pattern, etc. It must be greater than one.

    duration: int, default 1
        Number of time steps for each seasonal period.
        This determines how long each seasonal period is held constant before moving to the next.

    innovations: bool, default True
        Whether to include stochastic innovations in the strength of the seasonal effect

    name: str, default None
        A name for this seasonal component. Used to label dimensions and coordinates. Useful when multiple seasonal
        components are included in the same model. Default is ``f"Seasonal[s={season_length}, d={duration}]"``

    state_names: list of str, default None
        List of strings for seasonal effect labels. If provided, it must be of length ``season_length`` times ``duration``.
        An example would be ``state_names = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']`` when data is daily with a weekly
        seasonal pattern (``season_length = 7``).

        If None and ``duration = 1``, states will be named as ``[State_0, ..., State_s-1]`` (here s is ``season_length``).
        If None and ``duration > 1``, states will be named as ``[State_0_0, ..., State_s-1_d-1]`` (here d is ``duration``).

    remove_first_state: bool, default True
        If True, the first state will be removed from the model. This is done because there are only ``season_length-1`` degrees of
        freedom in the seasonal component, and one state is not identified. If False, the first state will be
        included in the model, but it will not be identified -- you will need to handle this in the priors (e.g. with
        ZeroSumNormal).

    observed_state_names: list[str] | None, default None
        List of strings for observed state labels. If None, defaults to ["data"].

    share_states: bool, default False
        Whether latent states are shared across the observed states. If True, there will be only one set of latent
        states, which are observed by all observed states. If False, each observed state has its own set of
        latent states. This argument has no effect if `k_endog` is 1.

    Notes
    -----
    A seasonal effect is any pattern that repeats at fixed intervals. There are several ways to model such effects;
    here, we present two models that are straightforward extensions of those described in [1].

    **First model** (``remove_first_state=True``)

    In this model, the state vector is defined as:

    .. math::
        \alpha_t :=(\gamma_t, \ldots, \gamma_{t-d(s-1)+1}), \quad t \ge 0.

    This vector has length :math:`d(s-1)`, where:

    - :math:`s` is the ``seasonal_length`` parameter, and
    - :math:`d` is the ``duration`` parameter.

    The components of the initial vector :math:`\alpha_{0}` are given by

    .. math::
        \gamma_{-l} := \tilde{\gamma}_{k_l}, \quad \text{where} \quad k_l := \left\lfloor \frac{l}{d} \right\rfloor \bmod s \quad \text{and} \quad l=0,\ldots, d(s-1)-1.

    Here, the values

    .. math::
        \tilde{\gamma}_{0}, \ldots, \tilde{\gamma}_{s-2},

    represent the initial seasonal states. The transition matrix of this model is the :math:`d(s-1) \times d(s-1)` matrix

    .. math::
        \begin{bmatrix}
            -\mathbf{1}_d & -\mathbf{1}_d & \cdots & -\mathbf{1}_d & -\mathbf{1}_d \\
            \mathbf{1}_d & \mathbf{0}_d & \cdots & \mathbf{0}_d & \mathbf{0}_d \\
            \mathbf{0}_d & \mathbf{1}_d & \cdots & \mathbf{0}_d & \mathbf{0}_d \\
            \vdots & \vdots & \ddots & \vdots \\
            \mathbf{0}_d & \mathbf{0}_d & \cdots & \mathbf{1}_d & \mathbf{0}_d
        \end{bmatrix}

    where :math:`\mathbf{1}_d` and  :math:`\mathbf{0}_d` denote the :math:`d \times d` identity and null matrices, respectively.

    **Second model** (``remove_first_state=False``)

    In contrast, the state vector in the second model is defined as:

    .. math::
        \alpha_t=(\gamma_t, \ldots, \gamma_{t-ds+1}), \quad t \ge 0.

    This vector has length :math:`ds`. The components of the initial state vector :math:`\alpha_{0}` are defined similarly:

    .. math::
        \gamma_{-l} := \tilde{\gamma}_{k_l}, \quad \text{where} \quad k_l := \left\lfloor \frac{l}{d} \right\rfloor \bmod s \quad \text{and} \quad l=0,\ldots, ds-1.

    In this case, the initial seasonal states :math:`\tilde{\gamma}_{0}, \ldots, \tilde{\gamma}_{s-1}` are required to satisfy the following condition:

    .. math::
        \sum_{i=0}^{s-1} \tilde{\gamma}_{i} = 0.

    The transition matrix of this model is the following :math:`ds \times ds` circulant matrix:

    .. math::
        \begin{bmatrix}
            0 & 1 & 0 & \cdots & 0 \\
            0 & 0 & 1 & \cdots & 0 \\
            \vdots & \vdots & \ddots & \ddots & \vdots \\
            0 & 0 & \cdots & 0 & 1 \\
            1 & 0 & \cdots & 0 & 0
        \end{bmatrix}

    To give interpretation to the :math:`\gamma` terms, it is helpful to work through the algebra for a simple
    example. Let :math:`s=4`, :math:`d=1`, ``remove_first_state=True``, and omit the shock term. Then, we have
    :math:`\gamma_{-i} = \tilde{\gamma}_{-i}`, for :math:`i=-2,\ldots, 0` and the value of the seasonal component
    for the first 5 timesteps will be:

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

    In the next example, we set :math:`s=2`, :math:`d=2`, ``remove_first_state=True``, and omit the shock term.
    By definition, the initial vector :math:`\alpha_{0}` is

    .. math::
        \alpha_0=(\tilde{\gamma}_{0}, \tilde{\gamma}_{0}, \tilde{\gamma}_{-1}, \tilde{\gamma}_{-1})

    and the transition matrix is

    .. math::
        \begin{bmatrix}
            -1 &  0 & -1 &  0 \\
             0 & -1 &  0 & -1 \\
             1 &  0 &  0 &  0 \\
             0 &  1 &  0 &  0 \\
        \end{bmatrix}

    It is easy to verify that:

    .. math::
        \begin{align}
            \gamma_1 &= -\tilde{\gamma}_0 - \tilde{\gamma}_{-1}\\
            \gamma_2 &= -(-\tilde{\gamma}_0 - \tilde{\gamma}_{-1})-\tilde{\gamma}_0\\
                     &= \tilde{\gamma}_{-1}\\
            \gamma_3 &= -\tilde{\gamma}_{-1} +(\tilde{\gamma}_0 + \tilde{\gamma}_{-1})\\
                     &= \tilde{\gamma}_{0}\\
            \gamma_4 &= -\tilde{\gamma}_0 - \tilde{\gamma}_{-1}.\\
        \end{align}

    .. warning::
        Although the ``state_names`` argument expects a list of length ``season_length`` times ``duration``,
        only ``state_names[duration:]`` will be saved as model dimensions, since the first coefficient is not identified
        (it is defined as :math:`-\sum_{i=1}^{s-1} \tilde{\gamma}_{-i}`).

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
        annual_season = st.TimeSeasonality(
            season_length=12, name="annual", state_names=state_names, innovations=False
        )
        ss_mod = (grw + annual_season).build()

        with pm.Model(coords=ss_mod.coords) as model:
            P0 = pm.Deterministic('P0', pt.eye(ss_mod.k_states) * 10, dims=ss_mod.param_dims['P0'])

            initial_level_trend = pm.Deterministic(
                "initial_level_trend", pt.zeros(1), dims=ss_mod.param_dims["initial_level_trend"]
            )
            sigma_level_trend = pm.HalfNormal(
                "sigma_level_trend", sigma=1e-6, dims=ss_mod.param_dims["sigma_level_trend"]
            )
            params_annual = pm.Normal("params_annual", sigma=1e-2, dims=ss_mod.param_dims["params_annual"])

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
        season_length: int,
        duration: int = 1,
        innovations: bool = True,
        name: str | None = None,
        state_names: list | None = None,
        remove_first_state: bool = True,
        observed_state_names: list[str] | None = None,
        share_states: bool = False,
    ):
        if observed_state_names is None:
            observed_state_names = ["data"]

        if season_length <= 1 or not isinstance(season_length, int):
            raise ValueError(
                f"season_length must be an integer greater than 1, got {season_length}"
            )
        if duration <= 0 or not isinstance(duration, int):
            raise ValueError(f"duration must be a positive integer, got {duration}")
        if name is None:
            name = f"Seasonal[s={season_length}, d={duration}]"
        if state_names is None:
            if duration > 1:
                state_names = [
                    f"{name}_{i}_{j}" for i in range(season_length) for j in range(duration)
                ]
            else:
                state_names = [f"{name}_{i}" for i in range(season_length)]
        else:
            if len(state_names) != season_length * duration:
                raise ValueError(
                    f"state_names must be a list of length season_length*duration, got {len(state_names)}"
                )
            state_names = state_names.copy()

        self.share_states = share_states
        self.innovations = innovations
        self.duration = duration
        self.remove_first_state = remove_first_state
        self.season_length = season_length

        if self.remove_first_state:
            # In traditional models, the first state isn't identified, so we can help out the user by automatically
            # discarding it.
            # TODO: Can this be stashed and reconstructed automatically somehow?
            state_names = state_names[duration:]

        self.provided_state_names = state_names

        k_states = (season_length - int(self.remove_first_state)) * duration
        k_endog = len(observed_state_names)
        k_posdef = int(innovations)

        super().__init__(
            name=name,
            k_endog=k_endog,
            k_states=k_states if share_states else k_states * k_endog,
            k_posdef=k_posdef if share_states else k_posdef * k_endog,
            observed_state_names=observed_state_names,
            measurement_error=False,
            combine_hidden_states=True,
            obs_state_idxs=np.tile(
                np.array([1.0] + [0.0] * (k_states - 1)), 1 if share_states else k_endog
            ),
            share_states=share_states,
        )

    def populate_component_properties(self):
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog

        k_states = self.k_states // k_endog_effective

        if self.share_states:
            self.state_names = [
                f"{state_name}[{self.name}_shared]" for state_name in self.provided_state_names
            ]
        else:
            self.state_names = [
                f"{state_name}[{endog_name}]"
                for endog_name in self.observed_state_names
                for state_name in self.provided_state_names
            ]

        self.param_names = [f"params_{self.name}"]

        self.param_info = {
            f"params_{self.name}": {
                "shape": (k_states,) if k_endog == 1 else (k_endog, k_states),
                "constraints": None,
                "dims": (f"state_{self.name}",)
                if k_endog_effective == 1
                else (f"endog_{self.name}", f"state_{self.name}"),
            }
        }

        self.param_dims = {
            f"params_{self.name}": (f"state_{self.name}",)
            if k_endog_effective == 1
            else (f"endog_{self.name}", f"state_{self.name}")
        }

        self.coords = (
            {f"state_{self.name}": self.provided_state_names}
            if k_endog_effective == 1
            else {
                f"endog_{self.name}": self.observed_state_names,
                f"state_{self.name}": self.provided_state_names,
            }
        )

        if self.innovations:
            self.param_names += [f"sigma_{self.name}"]
            self.param_info[f"sigma_{self.name}"] = {
                "shape": () if k_endog_effective == 1 else (k_endog,),
                "constraints": "Positive",
                "dims": None if k_endog_effective == 1 else (f"endog_{self.name}",),
            }
            if self.share_states:
                self.shock_names = [f"{self.name}[shared]"]
            else:
                self.shock_names = [f"{self.name}[{name}]" for name in self.observed_state_names]

            if k_endog > 1:
                self.param_dims[f"sigma_{self.name}"] = (f"endog_{self.name}",)

    def make_symbolic_graph(self) -> None:
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog
        k_states = self.k_states // k_endog_effective
        duration = self.duration

        k_unique_states = k_states // duration
        k_posdef = self.k_posdef // k_endog_effective

        if self.remove_first_state:
            # In this case, parameters are normalized to sum to zero, so the current state is the negative sum of
            # all previous states.
            zero_d = pt.zeros((self.duration, self.duration))
            id_d = pt.eye(self.duration)

            row_blocks = []

            # First row: all -1_d blocks
            first_row = [-id_d for _ in range(self.season_length - 1)]
            row_blocks.append(pt.concatenate(first_row, axis=1))

            # Rows 2 to season_length-1: shifted identity blocks
            for i in range(self.season_length - 2):
                row = []
                for j in range(self.season_length - 1):
                    if j == i:
                        row.append(id_d)
                    else:
                        row.append(zero_d)
                row_blocks.append(pt.concatenate(row, axis=1))

            # Stack blocks
            T = pt.concatenate(row_blocks, axis=0)
        else:
            # In this case we assume the user to be responsible for ensuring the states sum to zero, so T is just a
            # circulant matrix that cycles between the states.
            T = pt.eye(k_states, k=1)
            T = pt.set_subtensor(T[-1, 0], 1)

        self.ssm["transition", :, :] = pt.linalg.block_diag(*[T for _ in range(k_endog_effective)])

        Z = pt.zeros((1, k_states))[0, 0].set(1)
        self.ssm["design", :, :] = pt.linalg.block_diag(*[Z for _ in range(k_endog_effective)])

        initial_states = self.make_and_register_variable(
            f"params_{self.name}",
            shape=(k_unique_states,)
            if k_endog_effective == 1
            else (k_endog_effective, k_unique_states),
        )
        if k_endog_effective == 1:
            self.ssm["initial_state", :] = pt.extra_ops.repeat(initial_states, duration, axis=0)
        else:
            self.ssm["initial_state", :] = pt.extra_ops.repeat(
                initial_states, duration, axis=1
            ).ravel()

        if self.innovations:
            R = pt.zeros((k_states, k_posdef))[0, 0].set(1.0)
            self.ssm["selection", :, :] = pt.join(0, *[R for _ in range(k_endog_effective)])
            season_sigma = self.make_and_register_variable(
                f"sigma_{self.name}", shape=() if k_endog_effective == 1 else (k_endog_effective,)
            )
            cov_idx = ("state_cov", *np.diag_indices(k_posdef * k_endog_effective))
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

    observed_state_names: list[str] | None, default None
        List of strings for observed state labels. If None, defaults to ["data"].

    share_states: bool, default False
        Whether latent states are shared across the observed states. If True, there will be only one set of latent
        states, which are observed by all observed states. If False, each observed state has its own set of
        latent states. This argument has no effect if `k_endog` is 1.

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
        season_length: int,
        n: int | None = None,
        name: str | None = None,
        innovations: bool = True,
        observed_state_names: list[str] | None = None,
        share_states: bool = False,
    ):
        if observed_state_names is None:
            observed_state_names = ["data"]

        self.share_states = share_states
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
        self.last_state_not_identified = (self.season_length / self.n) == 2.0
        self.n_coefs = k_states - int(self.last_state_not_identified)

        obs_state_idx = np.zeros(k_states)
        obs_state_idx[slice(0, k_states, 2)] = 1
        obs_state_idx = np.tile(obs_state_idx, 1 if share_states else k_endog)

        super().__init__(
            name=name,
            k_endog=k_endog,
            k_states=k_states if share_states else k_states * k_endog,
            k_posdef=k_states * int(self.innovations)
            if share_states
            else k_states * int(self.innovations) * k_endog,
            share_states=share_states,
            observed_state_names=observed_state_names,
            measurement_error=False,
            combine_hidden_states=True,
            obs_state_idxs=obs_state_idx,
        )

    def make_symbolic_graph(self) -> None:
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog

        k_states = self.k_states // k_endog_effective
        k_posdef = self.k_posdef // k_endog_effective
        n_coefs = self.n_coefs

        Z = pt.zeros((1, k_states))[0, slice(0, k_states, 2)].set(1.0)

        self.ssm["design", :, :] = pt.linalg.block_diag(*[Z for _ in range(k_endog_effective)])

        init_state = self.make_and_register_variable(
            f"params_{self.name}", shape=(n_coefs,) if k_endog == 1 else (k_endog, n_coefs)
        )

        init_state_idx = np.concatenate(
            [
                np.arange(k_states * i, (i + 1) * k_states, dtype=int)[:n_coefs]
                for i in range(k_endog_effective)
            ],
            axis=0,
        )

        self.ssm["initial_state", init_state_idx] = init_state.ravel()

        T_mats = [_frequency_transition_block(self.season_length, j + 1) for j in range(self.n)]
        T = pt.linalg.block_diag(*T_mats)
        self.ssm["transition", :, :] = pt.linalg.block_diag(*[T for _ in range(k_endog_effective)])

        if self.innovations:
            sigma_season = self.make_and_register_variable(
                f"sigma_{self.name}", shape=() if k_endog_effective == 1 else (k_endog_effective,)
            )
            self.ssm["selection", :, :] = pt.eye(self.k_states)
            self.ssm["state_cov", :, :] = pt.eye(self.k_posdef) * pt.repeat(
                sigma_season**2, k_posdef
            )

    def populate_component_properties(self):
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog
        n_coefs = self.n_coefs

        base_names = [f"{f}_{i}_{self.name}" for i in range(self.n) for f in ["Cos", "Sin"]]

        if self.share_states:
            self.state_names = [f"{name}[shared]" for name in base_names]
        else:
            self.state_names = [
                f"{name}[{obs_state_name}]"
                for obs_state_name in self.observed_state_names
                for name in base_names
            ]

        # Trim state names if the model is saturated
        param_state_names = base_names[:n_coefs]

        self.param_names = [f"params_{self.name}"]
        self.param_dims = {
            f"params_{self.name}": (f"state_{self.name}",)
            if k_endog_effective == 1
            else (f"endog_{self.name}", f"state_{self.name}")
        }
        self.param_info = {
            f"params_{self.name}": {
                "shape": (n_coefs,) if k_endog_effective == 1 else (k_endog_effective, n_coefs),
                "constraints": None,
                "dims": (f"state_{self.name}",)
                if k_endog_effective == 1
                else (f"endog_{self.name}", f"state_{self.name}"),
            }
        }

        self.coords = (
            {f"state_{self.name}": param_state_names}
            if k_endog == 1
            else {
                f"endog_{self.name}": self.observed_state_names,
                f"state_{self.name}": param_state_names,
            }
        )

        if self.innovations:
            self.param_names += [f"sigma_{self.name}"]
            self.shock_names = self.state_names.copy()
            self.param_info[f"sigma_{self.name}"] = {
                "shape": () if k_endog_effective == 1 else (k_endog_effective, n_coefs),
                "constraints": "Positive",
                "dims": None if k_endog_effective == 1 else (f"endog_{self.name}",),
            }
            if k_endog_effective > 1:
                self.param_dims[f"sigma_{self.name}"] = (f"endog_{self.name}",)
