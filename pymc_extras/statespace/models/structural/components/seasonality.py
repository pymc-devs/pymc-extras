from collections.abc import Sequence

import numpy as np

from pytensor import tensor as pt
from pytensor.tensor import TensorVariable

from pymc_extras.statespace.core.properties import (
    Coord,
    Parameter,
    Shock,
    State,
)
from pymc_extras.statespace.models.structural.core import Component
from pymc_extras.statespace.models.structural.utils import _frequency_transition_block

__all__ = ["TimeSeasonality", "FrequencySeasonality"]


class TimeSeasonality(Component):
    """
    Create a TimeSeasonality component for a state space model.
    """

    def __init__(
        self,
        season_length: int,
        duration: int = 1,
        innovations: bool = True,
        name: str | None = None,
        state_names: Sequence[str] | None = None,
        remove_first_state: bool = True,
        observed_state_names: Sequence[str] | None = None,
        share_states: bool = False,
        start_state: str | int | None = None,
        use_time_varying: bool = True,
    ):
        r"""
        Deterministic seasonal pattern with optional stochastic drift.

        Many time series exhibit regular patterns tied to the calendar: sales spike in December, electricity demand peaks
        on weekday evenings, ice cream consumption rises in summer. This component captures such effects by estimating a
        distinct effect for each period within a seasonal cycle, subject to the constraint that effects sum to zero over a
        complete cycle. This ensures the seasonality captures deviations from the level,  not the level itself.

        Parameters
        ----------
        season_length : int
            Number of periods in one complete seasonal cycle. Must be at least 2.

        duration : int, default 1
            Number of observations each seasonal effect spans. The default (1) means each observation gets its own seasonal
            effect. Set ``duration > 1`` when your data frequency is finer than your seasonal pattern—for example, daily
            observations with monthly seasonality (``season_length=12``, ``duration≈30``).

        innovations : bool, default True
            If True, seasonal effects evolve stochastically over time, allowing the seasonal pattern to change gradually.
            If False, the pattern is deterministic (constant across all cycles).

        name : str, optional
            Label for this component, used in parameter names and coordinates. Defaults to ``"Seasonal[s={season_length},
            d={duration}]"``.

        state_names : sequence of str, optional
            Labels for each seasonal period. Length must equal ``season_length``. These appear in output coordinates,
            making results interpretable. For example, a weekly season might use the names of the days of the week.

            Defaults to ``["{name}_0", "{name}_1", ...]``.

        remove_first_state : bool, default True
            Controls how the sum-to-zero constraint is enforced.

            - **True** (recommended): Estimates ``s-1`` free parameters; the first seasonal effect is computed as the
            negative sum of the others. This is the Durbin-Koopman [1]_ formulation.
            - **False**: Estimates all ``s`` parameters. You must enforce the constraint yourself, typically via a
            ``ZeroSumNormal`` prior. Use this when you want symmetric treatment of all seasons.

            .. warning::
                With ``remove_first_state=True``, the first element of ``state_names`` does not
                appear in the parameter coordinates (since it's not a free parameter).

        observed_state_names : sequence of str, optional
            Labels for observed series. Defaults to ``["data"]`` for univariate models.

        share_states : bool, default False
            For multivariate models: if True, all series share the same seasonal pattern; if False,
            each series has independent seasonal effects. Ignored if ``k_endog=1``.

        start_state : str or int, optional
            Which seasonal period corresponds to the first observation (t=0). Specify as either a name from
            ``state_names`` or an integer index. Use this when your sample doesn't start at the beginning of a cycle—
            for instance, if you have weekly seasonality but your data begins on a Wednesday, set ``start_state="Wed"``
            or ``start_state=3``.

            The index refers to positions in the original ``state_names`` (before any removal).

        use_time_varying : bool, default True
            If True and duration > 1, the transition matrix will be time-varying to correctly handle the shifting
            seasonal effects. If False, a single very large and sparse transition matrix will be used. Ignored if
            duration = 1. The time-varying approach is suggested for now to keep the state space small.

        Notes
        -----
        **The Model**

        The observation at time :math:`t` is influenced by a seasonal effect :math:`\gamma_t`:

        .. math::

            y_t = \ldots + \gamma_t + \varepsilon_t

        where the seasonal effect cycles through :math:`s` values, repeating every :math:`s` observations (or every
        :math:`s \times d` observations if ``duration > 1``).

        To ensure identifiability—separating seasonality from the overall level—we impose:

        .. math::

            \sum_{j=0}^{s-1} \gamma_j = 0

        **Enforcing the Constraint: Two Approaches**

        1. **Durbin-Koopman formulation** (``remove_first_state=True``)

           Parameterize only :math:`\gamma_1, \ldots, \gamma_{s-1}` as free parameters, then define :math:`\gamma_0 =
           -\sum_{j=1}^{s-1} \gamma_j`. The state vector tracks these :math:`s-1` values, and the transition matrix
           rotates through the cycle while computing the implied :math:`\gamma_0` automatically. The state transition
           follows:

           .. math::

               T_\gamma = \begin{bmatrix}
                   -1 & -1 & \cdots & -1 \\
                   1 & 0 & \cdots & 0 \\
                   0 & 1 & \ddots & \vdots \\
                   \vdots & & \ddots & 0
               \end{bmatrix}

           This formulation is statistically efficient (minimal state dimension) and guarantees the constraint by
           construction.

        2. **Unconstrained formulation** (``remove_first_state=False``)

           All :math:`s` seasonal effects are free parameters. The state simply cycles via a permutation matrix. The
           sum-to-zero constraint must be imposed through the prior (e.g., ``pm.ZeroSumNormal``). This formulation
           treats all states symmetrically and can be more intuitive when you want to directly interpret each seasonal
           effect, but it has a slightly larger state dimension.

        **Duration: Handling Mismatched Frequencies**

        When ``duration > 1``, each seasonal effect is held constant for :math:`d` consecutive observations before
        transitioning to the next. This produces a step-function pattern and is useful when data frequency exceeds
        seasonal frequency (e.g., when observations are daily, but the seasonal pattern repeats monthly).

        The total cycle length becomes :math:`s \times d` observations.

        **Stochastic Seasonality**

        With ``innovations=True``, seasonal effects evolve over time:

        .. math::

            \gamma_{j,t+1} = \gamma_{j,t} + \omega_{j,t}, \quad \omega_{j,t} \sim N(0, \sigma^2_\gamma)

        This allows the seasonal pattern to adapt—capturing phenomena like shifting holiday shopping patterns or
        changing commuter behavior. The latent season effect evolves with a Gaussian random walk. The smoothness of
        evolution is controlled by the prior on ``sigma_{name}``.

        Examples
        --------
        Weekly seasonality for daily data:

        >>> mod = st.TimeSeasonality(
        ...     season_length=7,
        ...     state_names=['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'],
        ...     start_state='Mon',  # Data starts on Monday
        ...     name='day_of_week',
        ... )

        Monthly seasonality for daily data (each month held constant for ~30 days):

        >>> mod = st.TimeSeasonality(
        ...     season_length=12,
        ...     duration=30,
        ...     name='month',
        ... )

        See Also
        --------
        FrequencySeasonality :
            Alternative parameterization using Fourier basis functions. More compact for long seasonal periods but less
            interpretable (effects do not map to specific calendar periods). Can handle non-integer season lengths.

        References
        ----------
        .. [1] Durbin, J., & Koopman, S. J. (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford
               University Press. Section 3.2.
        """
        if observed_state_names is None:
            observed_state_names = ["data"]

        if not isinstance(season_length, int) or season_length <= 1:
            raise ValueError(
                f"season_length must be an integer greater than 1, got {season_length}"
            )
        if not isinstance(duration, int) or duration <= 0:
            raise ValueError(f"duration must be a positive integer, got {duration}")
        if name is None:
            name = f"Seasonal[s={season_length}, d={duration}]"

        # The user only provides unique names. If duration > 1, the states will be repeated with suffixes _0, _1, ...,
        # _{duration-1} to create unique state names for each position in the cycle.
        if state_names is None:
            state_names = [f"{name}_{i}" for i in range(season_length)]
        else:
            if len(state_names) != season_length:
                raise ValueError(
                    f"state_names must be a list of length season_length={season_length}, got {len(state_names)}"
                )
            state_names = list(state_names)

        # Validate and convert start_state to an index
        if start_state is not None:
            if isinstance(start_state, str):
                if start_state not in state_names:
                    raise ValueError(
                        f"start_state '{start_state}' not found in state_names. "
                        f"Available names: {state_names}"
                    )
                start_idx = state_names.index(start_state)
            elif isinstance(start_state, int):
                if not (0 <= start_state < season_length):
                    raise ValueError(
                        f"start_state index must be in [0, {season_length}), got {start_state}"
                    )
                start_idx = start_state
            else:
                raise ValueError(
                    f"start_state must be a string (state name) or int (index), got {type(start_state)}"
                )
        else:
            start_idx = 0

        self.start_idx = start_idx
        self.share_states = share_states
        self.innovations = innovations
        self.duration = duration
        self.remove_first_state = remove_first_state
        self.season_length = season_length
        self.use_time_varying = use_time_varying

        if self.remove_first_state:
            # TODO: Can we somehow use a transformation to preserve all of the user's states?
            state_names = state_names[1:]

        self.provided_state_names = state_names

        # When using time-varying transition matrices with duration > 1, we don't need
        # to expand the state dimension. The time-varying T handles the duration logic.
        use_tv = use_time_varying and duration > 1
        if use_tv:
            k_states = season_length - int(remove_first_state)
        else:
            k_states = duration * (season_length - int(remove_first_state))

        k_endog = len(observed_state_names)
        k_posdef = int(innovations)

        super().__init__(
            name=name,
            k_endog=k_endog,
            k_states=k_states if share_states else k_states * k_endog,
            k_posdef=k_posdef if share_states else k_posdef * k_endog,
            base_observed_state_names=observed_state_names,
            measurement_error=False,
            combine_hidden_states=True,
            obs_state_idxs=np.tile(
                np.array([1.0] + [0.0] * (k_states - 1)), 1 if share_states else k_endog
            ),
            share_states=share_states,
        )

    @property
    def n_seasons(self) -> int:
        """Number of unique seasonal parameters (season_length - 1 if remove_first_state, else season_length)."""
        return self.season_length - int(self.remove_first_state)

    @property
    def _uses_time_varying_transition(self) -> bool:
        """Whether this component uses time-varying transition matrices."""
        return self.use_time_varying and self.duration > 1

    def set_states(self) -> State | tuple[State, ...] | None:
        observed_state_names = self.base_observed_state_names

        # Expand state names for duration > 1, but NOT when using time-varying T
        # (time-varying T keeps the state compact)
        if self.duration > 1 and not self._uses_time_varying_transition:
            expanded_state_names = [
                f"{state_name}_{i}"
                for state_name in self.provided_state_names
                for i in range(self.duration)
            ]
        else:
            expanded_state_names = self.provided_state_names

        if self.share_states:
            state_names = [
                f"{state_name}[{self.name}_shared]" for state_name in expanded_state_names
            ]
        else:
            state_names = [
                f"{state_name}[{endog_name}]"
                for endog_name in observed_state_names
                for state_name in expanded_state_names
            ]

        hidden_states = [State(name=name, observed=False, shared=True) for name in state_names]
        observed_states = [
            State(name=name, observed=True, shared=False) for name in observed_state_names
        ]
        return *hidden_states, *observed_states

    def set_parameters(self) -> Parameter | tuple[Parameter, ...] | None:
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog
        k_unique_params = self.n_seasons

        seasonal_param = Parameter(
            name=f"params_{self.name}",
            shape=(k_unique_params,)
            if k_endog_effective == 1
            else (k_endog_effective, k_unique_params),
            dims=(f"state_{self.name}",)
            if k_endog_effective == 1
            else (f"endog_{self.name}", f"state_{self.name}"),
            constraints=None,
        )

        params_container = [seasonal_param]

        if self.innovations:
            sigma_param = Parameter(
                name=f"sigma_{self.name}",
                shape=() if k_endog_effective == 1 else (k_endog,),
                dims=None if k_endog_effective == 1 else (f"endog_{self.name}",),
                constraints="Positive",
            )
            params_container.append(sigma_param)

        return tuple(params_container)

    def set_shocks(self) -> Shock | tuple[Shock, ...] | None:
        observed_state_names = self.observed_state_names
        if self.innovations:
            if self.share_states:
                shock_names = [f"{self.name}[shared]"]
            else:
                shock_names = [f"{self.name}[{name}]" for name in observed_state_names]

            return tuple(Shock(name=name) for name in shock_names)
        return None

    def set_coords(self) -> Coord | tuple[Coord, ...] | None:
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog
        observed_state_names = self.observed_state_names

        state_coord = Coord(dimension=f"state_{self.name}", labels=tuple(self.provided_state_names))
        coords_container = [state_coord]

        if k_endog_effective > 1:
            endog_coord = Coord(dimension=f"endog_{self.name}", labels=observed_state_names)
            coords_container.append(endog_coord)

        return tuple(coords_container)

    def _k_endog_effective(self) -> int:
        return 1 if self.share_states else self.k_endog

    def _k_states_per_endog(self) -> int:
        return self.k_states // self._k_endog_effective()

    def _k_posdef_per_endog(self) -> int:
        return self.k_posdef // self._k_endog_effective()

    def _build_dk_seasonal_rotation(self) -> TensorVariable:
        """Build the (s-1) x (s-1) Durbin-Koopman seasonal transition matrix."""
        n = self.season_length - 1
        T_gamma = pt.zeros((n, n))
        T_gamma = pt.set_subtensor(T_gamma[0, :], -1.0)
        if n > 1:
            T_gamma = pt.set_subtensor(T_gamma[1:, :-1], pt.eye(n - 1))
        return T_gamma

    def _build_circulant_rotation(self) -> TensorVariable:
        """Build simple circulant permutation matrix of size season_length."""
        n = self.season_length
        T = pt.eye(n, k=1)
        return pt.set_subtensor(T[-1, 0], 1)

    def _build_static_transition(self) -> TensorVariable:
        """Build static transition matrix (2D) for duration >= 1."""
        k_states = self._k_states_per_endog()

        if not self.remove_first_state:
            T_rotate = self._build_circulant_rotation()

            if self.duration == 1:
                return T_rotate

            # Duration > 1: block structure with circulant rotation at wrap
            n = self.season_length
            I_n = pt.eye(n)
            T = pt.zeros((k_states, k_states))

            for k in range(self.duration - 1):
                row_slice = slice(k * n, (k + 1) * n)
                col_slice = slice((k + 1) * n, (k + 2) * n)
                T = pt.set_subtensor(T[row_slice, col_slice], I_n)

            last_row_slice = slice((self.duration - 1) * n, self.duration * n)
            T = pt.set_subtensor(T[last_row_slice, :n], T_rotate)
            return T

        if self.duration == 1:
            return self._build_dk_seasonal_rotation()

        # Duration > 1: block structure with D&K rotation at wrap
        n = self.season_length - 1
        T_gamma = self._build_dk_seasonal_rotation()
        I_n = pt.eye(n)
        T = pt.zeros((k_states, k_states))

        for k in range(self.duration - 1):
            row_slice = slice(k * n, (k + 1) * n)
            col_slice = slice((k + 1) * n, (k + 2) * n)
            T = pt.set_subtensor(T[row_slice, col_slice], I_n)

        last_row_slice = slice((self.duration - 1) * n, self.duration * n)
        T = pt.set_subtensor(T[last_row_slice, :n], T_gamma)
        return T

    def _build_time_varying_transition(self) -> TensorVariable:
        """Build time-varying transition matrix (3D) for duration > 1 with time-varying mode."""
        if self.remove_first_state:
            n = self.season_length - 1
            T_rotate = self._build_dk_seasonal_rotation()
        else:
            n = self.season_length
            T_rotate = self._build_circulant_rotation()

        T_hold = pt.eye(n)

        time_idx = pt.arange(self.n_timesteps)
        is_rotation_step = pt.eq(time_idx % self.duration, self.duration - 1)

        return pt.where(
            is_rotation_step[:, None, None],
            pt.broadcast_to(T_rotate, (self.n_timesteps, n, n)),
            pt.broadcast_to(T_hold, (self.n_timesteps, n, n)),
        )

    def _build_transition_matrix(self) -> TensorVariable:
        """Build the full transition matrix, handling multivariate via block_diag."""
        k_endog_effective = self._k_endog_effective()

        if self._uses_time_varying_transition:
            T_single = self._build_time_varying_transition()
            if k_endog_effective == 1:
                return T_single
            # For multivariate: build block diagonal for each time step
            # T_single is (n_timesteps, n, n), we need (n_timesteps, k_states, k_states)
            blocks = [T_single for _ in range(k_endog_effective)]
            # Stack along a new axis then reshape to block diagonal per timestep
            return pt.linalg.block_diag(*blocks)
        else:
            T_single = self._build_static_transition()
            return pt.linalg.block_diag(*[T_single for _ in range(k_endog_effective)])

    def _build_design_matrix(self) -> TensorVariable:
        """Build the design matrix Z that extracts the first state."""
        k_states = self._k_states_per_endog()
        k_endog_effective = self._k_endog_effective()
        Z = pt.zeros((1, k_states))[0, 0].set(1)
        return pt.linalg.block_diag(*[Z for _ in range(k_endog_effective)])

    def _build_initial_state_dk(self, initial_params: TensorVariable) -> TensorVariable:
        """Build initial state for Durbin-Koopman formulation (remove_first_state=True)."""
        k_endog_effective = self._k_endog_effective()
        k_unique_params = self.season_length - 1
        use_tv = self._uses_time_varying_transition

        if k_endog_effective == 1:
            gamma_0 = -pt.sum(initial_params)
            if k_unique_params > 1:
                reordered = pt.concatenate([[gamma_0], initial_params[1:][::-1]])
            else:
                reordered = pt.atleast_1d(gamma_0)
            # Only tile when NOT using time-varying transition
            if use_tv:
                return reordered
            else:
                return pt.tile(reordered, self.duration)
        else:
            gamma_0 = -pt.sum(initial_params, axis=1, keepdims=True)
            if k_unique_params > 1:
                reordered = pt.concatenate([gamma_0, initial_params[:, 1:][:, ::-1]], axis=1)
            else:
                reordered = gamma_0
            if use_tv:
                return reordered.ravel()
            else:
                return pt.tile(reordered, (1, self.duration)).ravel()

    def _build_initial_state_simple(self, initial_params: TensorVariable) -> TensorVariable:
        """Build initial state for simple formulation (remove_first_state=False)."""
        k_endog_effective = self._k_endog_effective()
        use_tv = self._uses_time_varying_transition

        if k_endog_effective == 1:
            if use_tv:
                return initial_params
            else:
                return pt.extra_ops.repeat(initial_params, self.duration, axis=0)
        else:
            if use_tv:
                return initial_params.ravel()
            else:
                return pt.extra_ops.repeat(initial_params, self.duration, axis=1).ravel()

    def _apply_start_state_shift(
        self, initial_state: TensorVariable, T: TensorVariable
    ) -> TensorVariable:
        """Shift initial state to account for start_state offset."""
        if self.start_idx == 0:
            return initial_state

        k_endog_effective = self._k_endog_effective()

        if self._uses_time_varying_transition:
            # Time-varying case: shift by start_idx rotations
            # Each rotation is one application of T_rotate, which happens every 'duration' steps
            if self.remove_first_state:
                T_rotate = self._build_dk_seasonal_rotation()
            else:
                T_rotate = self._build_circulant_rotation()
            if k_endog_effective == 1:
                return pt.linalg.matrix_power(T_rotate, self.start_idx) @ initial_state
            else:
                T_full = pt.linalg.block_diag(*[T_rotate for _ in range(k_endog_effective)])
                return pt.linalg.matrix_power(T_full, self.start_idx) @ initial_state
        else:
            # Static case: shift by start_idx * duration applications of T
            shift_steps = self.start_idx * self.duration
            if k_endog_effective == 1:
                return pt.linalg.matrix_power(T, shift_steps) @ initial_state
            else:
                T_full = pt.linalg.block_diag(*[T for _ in range(k_endog_effective)])
                return pt.linalg.matrix_power(T_full, shift_steps) @ initial_state

    def _build_selection_and_state_cov(self) -> None:
        """Build selection matrix R and state covariance Q for innovations."""
        if not self.innovations:
            return

        k_endog_effective = self._k_endog_effective()
        k_states = self._k_states_per_endog()
        k_posdef = self._k_posdef_per_endog()

        R = pt.zeros((k_states, k_posdef))[0, 0].set(1.0)
        self.ssm["selection", :, :] = pt.join(0, *[R for _ in range(k_endog_effective)])

        sigma = self.make_and_register_variable(
            f"sigma_{self.name}",
            shape=() if k_endog_effective == 1 else (k_endog_effective,),
        )
        cov_idx = ("state_cov", *np.diag_indices(k_posdef * k_endog_effective))
        self.ssm[cov_idx] = sigma**2

    def make_symbolic_graph(self) -> None:
        k_endog_effective = self._k_endog_effective()
        k_unique_params = self.n_seasons

        # Transition matrix
        T = self._build_transition_matrix()
        if T.ndim == 3:
            self.ssm["transition"] = T
        else:
            self.ssm["transition", :, :] = T

        # Design matrix
        self.ssm["design", :, :] = self._build_design_matrix()

        # Initial state parameters
        initial_params = self.make_and_register_variable(
            f"params_{self.name}",
            shape=(k_unique_params,)
            if k_endog_effective == 1
            else (k_endog_effective, k_unique_params),
        )

        # Build initial state
        if self.remove_first_state:
            initial_state = self._build_initial_state_dk(initial_params)
        else:
            initial_state = self._build_initial_state_simple(initial_params)

        # Apply start_state shift (handles time-varying vs static internally)
        T_for_shift = self._build_static_transition()
        initial_state = self._apply_start_state_shift(initial_state, T_for_shift)

        self.ssm["initial_state", :] = initial_state

        # Selection and state covariance
        self._build_selection_and_state_cov()


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
        season_length: int | float,
        n: int | None = None,
        name: str | None = None,
        innovations: bool = True,
        observed_state_names: Sequence[str] | None = None,
        share_states: bool = False,
    ):
        if observed_state_names is None:
            observed_state_names = ["data"]

        if not isinstance(season_length, int | float) or season_length <= 0:
            raise ValueError(f"season_length must be a positive number, got {season_length}")

        self.share_states = share_states
        k_endog = len(observed_state_names)

        if n is None:
            n = int(season_length / 2)
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"n must be a positive integer, got {n}")
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
            base_observed_state_names=observed_state_names,
            measurement_error=False,
            combine_hidden_states=True,
            obs_state_idxs=obs_state_idx,
        )

    def set_states(self) -> State | tuple[State, ...] | None:
        observed_state_names = self.base_observed_state_names
        base_names = [f"{f}_{i}_{self.name}" for i in range(self.n) for f in ["Cos", "Sin"]]

        if self.share_states:
            state_names = [f"{name}[shared]" for name in base_names]
        else:
            state_names = [
                f"{name}[{obs_state_name}]"
                for obs_state_name in self.base_observed_state_names
                for name in base_names
            ]

        hidden_states = [State(name=name, observed=False, shared=True) for name in state_names]
        observed_states = [
            State(name=name, observed=True, shared=False) for name in observed_state_names
        ]
        return *hidden_states, *observed_states

    def set_parameters(self) -> Parameter | tuple[Parameter, ...] | None:
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog
        n_coefs = self.n_coefs

        freq_param = Parameter(
            name=f"params_{self.name}",
            shape=(n_coefs,) if k_endog_effective == 1 else (k_endog_effective, n_coefs),
            dims=(f"state_{self.name}",)
            if k_endog_effective == 1
            else (f"endog_{self.name}", f"state_{self.name}"),
            constraints=None,
        )

        params_container = [freq_param]

        if self.innovations:
            sigma_param = Parameter(
                name=f"sigma_{self.name}",
                shape=() if k_endog_effective == 1 else (k_endog_effective, n_coefs),
                dims=None if k_endog_effective == 1 else (f"endog_{self.name}",),
                constraints="Positive",
            )

            params_container.append(sigma_param)

        return tuple(params_container)

    def set_shocks(self) -> Shock | tuple[Shock, ...] | None:
        if self.innovations:
            return tuple(Shock(name=name) for name in self.state_names)
        return None

    def set_coords(self) -> Coord | tuple[Coord, ...] | None:
        k_endog = self.k_endog
        n_coefs = self.n_coefs
        observed_state_names = self.observed_state_names

        base_names = [f"{f}_{i}_{self.name}" for i in range(self.n) for f in ["Cos", "Sin"]]

        # Trim state names if the model is saturated
        param_state_names = base_names[:n_coefs]

        state_coords = Coord(dimension=f"state_{self.name}", labels=tuple(param_state_names))

        coord_container = [state_coords]

        if k_endog > 1:
            endog_coords = Coord(dimension=f"endog_{self.name}", labels=observed_state_names)
            coord_container.append(endog_coords)

        return tuple(coord_container)

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
