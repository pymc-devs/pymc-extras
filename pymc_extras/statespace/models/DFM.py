from collections.abc import Sequence

import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.compile.mode import Mode

from pymc_extras.statespace.core.properties import (
    Coord,
    Data,
    Parameter,
    Shock,
    State,
)
from pymc_extras.statespace.core.statespace import PyMCStateSpace
from pymc_extras.statespace.models.utilities import validate_names
from pymc_extras.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    AR_PARAM_DIM,
    ERROR_AR_PARAM_DIM,
    EXOG_COEF_STATE_DIM,
    EXOG_STATE_DIM,
    FACTOR_DIM,
    JITTER_DEFAULT,
    MISSING_FILL,
    NON_EXOG_STATE_DIM,
    OBS_STATE_AUX_DIM,
    OBS_STATE_DIM,
    TIME_DIM,
)

floatX = pytensor.config.floatX


def _make_var_companion_matrix(ar_coeffs, k_series: int, p: int):
    r"""
    Build the VAR(p) companion matrix for a block of jointly modeled series.

    Parameters
    ----------
    ar_coeffs : TensorVariable
        Autoregressive coefficients of shape ``(k_series, p * k_series)``, the horizontal
        concatenation :math:`[A_1 | A_2 | \cdots | A_p]`.
    k_series : int
        Number of series in the block.
    p : int
        Lag order.

    Returns
    -------
    companion : TensorVariable
        Companion matrix of shape ``(k_series * p, k_series * p)``, ordered lag-major so the first
        ``k_series`` rows are the current values of each series.
    """
    size = k_series * p
    companion = pt.zeros((size, size), dtype=floatX)
    companion = companion[:k_series].set(ar_coeffs)

    if p > 1:
        companion = companion[k_series:, : k_series * (p - 1)].set(
            pt.eye(k_series * (p - 1), dtype=floatX)
        )

    return companion


def _make_independent_ar_companion_matrix(ar_coeffs, k_series: int, p: int):
    r"""
    Build the companion matrix for ``k_series`` independent AR(p) processes.

    Each series evolves on its own coefficients with no cross-series terms, but the states are
    interleaved lag-major, matching :func:`_make_var_companion_matrix`.

    Parameters
    ----------
    ar_coeffs : TensorVariable
        Autoregressive coefficients of shape ``(k_series, p)``, one row per series.
    k_series : int
        Number of independent series.
    p : int
        Lag order.

    Returns
    -------
    companion : TensorVariable
        Companion matrix of shape ``(k_series * p, k_series * p)``.
    """
    size = k_series * p
    companion = pt.zeros((size, size), dtype=floatX)

    # Row j of the first block row carries series j's own coefficients, each placed in the column
    # holding that lag's copy of series j. Flattened lag-major, those are ar_coeffs.T.ravel().
    rows = np.tile(np.arange(k_series), p)
    cols = np.arange(size)
    companion = companion[rows, cols].set(ar_coeffs.T.ravel())

    if p > 1:
        companion = companion[k_series:, : k_series * (p - 1)].set(
            pt.eye(k_series * (p - 1), dtype=floatX)
        )

    return companion


class BayesianDynamicFactor(PyMCStateSpace):
    r"""
    Dynamic Factor Model

    Notes
    -----
    The Dynamic Factor Model (DFM) is a multivariate state-space model that represents a
    high-dimensional time series as being driven by a smaller set of unobserved dynamic factors.

    Given observed series :math:`\{y_t\}_{t=0}^T`, where

    .. math::
        y_t = \begin{bmatrix} y_{1,t} & y_{2,t} & \cdots & y_{k_{\text{endog}},t} \end{bmatrix}^T,

    the DFM treats each series as a linear combination of a few latent factors and, optionally,
    autoregressive errors. Writing :math:`k` for the number of factors (``k_factors``), :math:`p`
    for the order of the factor process (``factor_order``), and :math:`q` for the order of the error
    process (``error_order``), the model in reduced form is

    .. math::
        y_t &= \Lambda f_t + B x_t + u_t + \eta_t \\
        f_t &= A_1 f_{t-1} + \cdots + A_p f_{t-p} + \varepsilon_{f,t} \\
        u_t &= C_1 u_{t-1} + \cdots + C_q u_{t-q} + \varepsilon_{u,t}

    where :math:`f_t` is the vector of latent dynamic factors of size :math:`k`, :math:`x_t` is an
    optional vector of exogenous variables, :math:`u_t` is a vector of autoregressive observation
    errors following a VAR(q) if ``error_var=True`` and independent AR(q) processes otherwise, and
    :math:`\eta_t \sim \mathcal{N}(0, H_t)` is an optional measurement error, included when
    ``measurement_error=True``. The innovations satisfy
    :math:`\varepsilon_{f,t} \sim \mathcal{N}(0, I)` and
    :math:`\varepsilon_{u,t} \sim \mathcal{N}(0, \Sigma_u)`. Factor innovations are standardized to
    an identity covariance in order to identify the factors.

    Internally the model stacks all current and lagged latent factors and, when present, the AR
    observation errors into a single state vector of dimension
    :math:`k_{\text{states}} = k \cdot p + k_{\text{endog}} \cdot q`, where
    :math:`k_{\text{endog}}` is the number of observed series. States are ordered lag-major: the
    current value of every factor comes first, then the first lag of every factor, and so on, with
    the error states laid out the same way.

    .. math::
        s_t = \begin{bmatrix}
            f_t(1) & \cdots & f_t(k) &
            f_{t-1}(1) & \cdots & f_{t-1}(k) & \cdots &
            u_t(1) & \cdots & u_t(k_{\text{endog}}) & \cdots
        \end{bmatrix}^T
        \in \mathbb{R}^{k_{\text{states}}}

    The transition equation is :math:`s_{t+1} = T s_t + R \epsilon_t`, where :math:`T` is
    block-diagonal in the factor, error, and exogenous components. Each block is a companion matrix:
    its first block row holds the autoregressive coefficients and its subdiagonal holds an identity
    that shifts each lag forward. :math:`\epsilon_t` collects the independent shocks,

    .. math::
        \epsilon_t = \begin{bmatrix} \epsilon_{f,t} \\ \epsilon_{u,t} \end{bmatrix}
        \in \mathbb{R}^{k + k_{\text{endog}}},

    and :math:`R` selects which states each shock enters.

    The observation equation is :math:`y_t = Z s_t + \eta_t`, with design matrix

    .. math::
        Z = \begin{bmatrix} \Lambda & 0 & I & 0 \end{bmatrix}
        \in \mathbb{R}^{k_{\text{endog}} \times k_{\text{states}}}

    where :math:`\Lambda` holds the factor loadings, the identity block picks out the current error
    states when :math:`q > 0`, and the zero blocks cover the lagged states, which do not enter the
    observation equation directly.

    When exogenous variables :math:`x_t` are present, the implementation follows
    :mod:`pymc_extras.statespace.models.structural.components.regression`: the state vector is
    extended with the regression coefficients and :math:`Z` becomes three-dimensional, with the
    leading axis indexing time. Unlike statsmodels, which places exogenous variables only in the
    observation equation, they are fully integrated into the state-space representation here, which
    is what allows time-varying coefficients via ``exog_innovations``.

    .. warning::

        Identification can be an issue, particularly when many observed series load onto only a few
        latent factors. These models are identified only up to a sign flip in the factor loadings.
        Proper prior specification is crucial for good estimation and inference.

    Examples
    --------
    Estimate a dynamic factor model with one latent factor following an AR(2), and AR(1) errors:

    .. code:: python

        import pymc as pm
        import pytensor.tensor as pt

        import pymc_extras.statespace as pmss

        # data is a wide DataFrame of observed series indexed by time
        dfm_mod = pmss.BayesianDynamicFactor(
            k_factors=1,
            factor_order=2,
            endog_names=data.columns,
            error_order=1,
            error_var=False,
            error_cov_type="diagonal",
            measurement_error=True,
        )

        with pm.Model(coords=dfm_mod.coords) as pymc_mod:
            x0 = pm.Normal("x0", dims=["state"])

            P0_diag = pm.HalfNormal("P0_diag", dims=["state"])
            P0 = pm.Deterministic("P0", pt.diag(P0_diag), dims=["state", "state_aux"])

            factor_loadings = pm.Normal("factor_loadings", dims=["observed_state", "factor"])
            factor_ar = pm.Normal("factor_ar", dims=["factor", "lag_ar"])
            error_ar = pm.Normal("error_ar", dims=["observed_state", "error_lag_ar"])

            error_sigma = pm.HalfNormal("error_sigma", dims=["observed_state"])
            sigma_obs = pm.HalfNormal("sigma_obs", dims=["observed_state"])

            dfm_mod.build_statespace_graph(data=data)
            idata = pm.sample()

    """

    def __init__(
        self,
        k_factors: int,
        factor_order: int,
        endog_names: Sequence[str] | None = None,
        exog_state_names: Sequence[str] | None = None,
        shared_exog_states: bool = False,
        exog_innovations: bool = False,
        error_order: int = 0,
        error_var: bool = False,
        error_cov_type: str = "diagonal",
        filter_type: str = "standard",
        measurement_error: bool = False,
        verbose: bool = True,
        mode: str | Mode | None = None,
        cov_jitter: float = JITTER_DEFAULT,
        missing_fill_value: float = MISSING_FILL,
    ):
        r"""
        Create a Bayesian Dynamic Factor Model.

        Parameters
        ----------
        k_factors : int
            Number of latent factors.
        factor_order : int
            Order of the VAR process for the latent factors. If 0, the factors have no
            autoregressive dynamics and are modeled as white noise, :math:`f_t = \varepsilon_{f,t}`.
            The state vector still carries one state per factor, but ``factor_ar`` does not exist.
        endog_names : sequence of str
            Names of the observed time series.
        exog_state_names : sequence of str, optional
            Names of the exogenous variables. Default None, for a model with no exogenous
            regressors.
        shared_exog_states : bool, optional
            Whether the exogenous latent states are shared across the observed states. If True there
            is a single set of exogenous states seen by every observed state; if False each observed
            state gets its own set. Default False.
        exog_innovations : bool, optional
            Whether to allow time-varying regression coefficients. If True, the coefficients follow
            a random walk. Default False.
        error_order : int, optional
            Order of the AR process for the observation error component. Default 0, corresponding to
            white noise errors.
        error_var : bool, optional
            If True, the errors are modeled jointly as a VAR process; otherwise each error is an
            independent AR process. Default False.
        error_cov_type : {'scalar', 'diagonal', 'unstructured'}, optional
            Structure of the covariance matrix of the observation errors. Default 'diagonal'.
        filter_type : str, optional
            The type of Kalman Filter to use. Options are "standard", "univariate", and "cholesky".
            See the docs for kalman filters for more details. Default "standard".
        measurement_error : bool, optional
            If True, a measurement error term is added to the model. Default False.
        verbose : bool, optional
            If True, a message will be logged to the terminal explaining the variable names,
            dimensions, and supports. Default True.
        mode : str or Mode, optional
            Pytensor compile mode, used in auxiliary sampling methods such as
            ``sample_conditional_posterior`` and ``forecast``. The mode does **not** effect calls to
            ``pm.sample``. Regardless of whether a mode is specified, it can always be overwritten
            via the ``compile_kwargs`` argument to all sampling methods. Default None.
        cov_jitter : float, optional
            Jitter added to the diagonal of every covariance matrix at each filtering step, for
            numerical stability. Post-estimation graphs are built with this same value. Default
            1e-8, or 1e-6 if ``pytensor.config.floatX`` is float32.
        missing_fill_value : float, optional
            Sentinel used to mask missing observations. Set this only if your data legitimately
            contains the default sentinel. Post-estimation graphs are built with this same value.
            Default -9999.0.
        """
        validate_names(endog_names, var_name="endog_names", optional=False)
        validate_names(exog_state_names, var_name="exog_state_names", optional=True)

        self.endog_names = tuple(endog_names)
        self.exog_state_names = tuple(exog_state_names) if exog_state_names is not None else ()

        self.k_endog = k_endog = len(self.endog_names)
        self.k_exog = len(self.exog_state_names)
        self.has_exog = self.k_exog > 0

        self.k_factors = k_factors
        self.factor_order = factor_order
        self.error_order = error_order
        self.error_var = error_var
        self.error_cov_type = error_cov_type
        self.shared_exog_states = shared_exog_states
        self.exog_innovations = exog_innovations

        self.k_exog_states = self.k_exog if shared_exog_states else self.k_exog * k_endog

        # A factor_order of 0 still gets one state per factor, it just has no dynamics.
        k_factor_states = max(factor_order, 1) * k_factors
        k_error_states = k_endog * error_order

        k_states = k_factor_states + k_error_states + self.k_exog_states
        k_posdef = k_factors + (k_endog if error_order > 0 else 0) + self.k_exog_states

        super().__init__(
            k_endog=k_endog,
            k_states=k_states,
            k_posdef=k_posdef,
            filter_type=filter_type,
            verbose=verbose,
            measurement_error=measurement_error,
            mode=mode,
            cov_jitter=cov_jitter,
            missing_fill_value=missing_fill_value,
        )

    def set_parameters(self) -> Parameter | tuple[Parameter, ...] | None:
        parameters = []

        k_endog = self.k_endog
        k_states = self.k_states

        parameters.append(
            Parameter(
                name="x0",
                shape=(k_states - self.k_exog_states,),
                dims=(NON_EXOG_STATE_DIM if self.has_exog else ALL_STATE_DIM,),
                constraints=None,
            )
        )

        parameters.append(
            Parameter(
                name="P0",
                shape=(k_states, k_states),
                dims=(ALL_STATE_DIM, ALL_STATE_AUX_DIM),
                constraints="Positive Semi-definite",
            )
        )

        parameters.append(
            Parameter(
                name="factor_loadings",
                shape=(k_endog, self.k_factors),
                dims=(OBS_STATE_DIM, FACTOR_DIM),
                constraints=None,
            )
        )

        if self.factor_order > 0:
            parameters.append(
                Parameter(
                    name="factor_ar",
                    shape=(self.k_factors, self.factor_order * self.k_factors),
                    dims=(FACTOR_DIM, AR_PARAM_DIM),
                    constraints=None,
                )
            )

        if self.error_order > 0:
            error_ar_shape = (
                (k_endog, self.error_order * k_endog)
                if self.error_var
                else (k_endog, self.error_order)
            )
            parameters.append(
                Parameter(
                    name="error_ar",
                    shape=error_ar_shape,
                    dims=(OBS_STATE_DIM, ERROR_AR_PARAM_DIM),
                    constraints=None,
                )
            )

        if self.error_cov_type == "scalar":
            parameters.append(
                Parameter(
                    name="error_sigma",
                    shape=(),
                    dims=(),
                    constraints="Positive",
                )
            )
        elif self.error_cov_type == "diagonal":
            parameters.append(
                Parameter(
                    name="error_sigma",
                    shape=(k_endog,),
                    dims=(OBS_STATE_DIM,),
                    constraints="Positive",
                )
            )
        elif self.error_cov_type == "unstructured":
            parameters.append(
                Parameter(
                    name="error_cov",
                    shape=(k_endog, k_endog),
                    dims=(OBS_STATE_DIM, OBS_STATE_AUX_DIM),
                    constraints="Positive Semi-definite",
                )
            )

        if self.measurement_error:
            parameters.append(
                Parameter(
                    name="sigma_obs",
                    shape=(k_endog,),
                    dims=(OBS_STATE_DIM,),
                    constraints="Positive",
                )
            )

        if self.has_exog:
            parameters.append(
                Parameter(
                    name="beta",
                    shape=(self.k_exog_states,),
                    dims=(EXOG_COEF_STATE_DIM,),
                    constraints=None,
                )
            )

            if self.exog_innovations:
                parameters.append(
                    Parameter(
                        name="beta_sigma",
                        shape=(self.k_exog_states,),
                        dims=(EXOG_COEF_STATE_DIM,),
                        constraints="Positive",
                    )
                )

        return tuple(parameters)

    def set_states(self) -> State | tuple[State, ...] | None:
        # States are laid out lag-major to match the companion matrices built in
        # make_symbolic_graph: every series at lag 0, then every series at lag 1, and so on.
        names = [
            f"L{lag}.factor_{i}"
            for lag in range(max(self.factor_order, 1))
            for i in range(1, self.k_factors + 1)
        ]

        names.extend(
            f"L{lag}.error_{i}"
            for lag in range(self.error_order)
            for i in range(1, self.k_endog + 1)
        )

        if self.has_exog:
            if self.shared_exog_states:
                names.extend(f"beta_{exog_name}[shared]" for exog_name in self.exog_state_names)
            else:
                names.extend(
                    f"beta_{exog_name}[{endog_name}]"
                    for endog_name in self.endog_names
                    for exog_name in self.exog_state_names
                )

        hidden_states = [State(name=name, observed=False, shared=False) for name in names]
        observed_states = [
            State(name=name, observed=True, shared=False) for name in self.endog_names
        ]

        return *hidden_states, *observed_states

    def set_shocks(self) -> Shock | tuple[Shock, ...] | None:
        shock_names = [f"factor_shock_{i}" for i in range(1, self.k_factors + 1)]

        if self.error_order > 0:
            shock_names.extend(f"error_shock_{i}" for i in range(1, self.k_endog + 1))

        if self.has_exog:
            # Each exogenous shock drives one coefficient state, so it carries that state's name.
            if self.shared_exog_states:
                shock_names.extend(
                    f"exog_shock_{exog_name}[shared]" for exog_name in self.exog_state_names
                )
            else:
                shock_names.extend(
                    f"exog_shock_{exog_name}[{endog_name}]"
                    for endog_name in self.endog_names
                    for exog_name in self.exog_state_names
                )

        return tuple(Shock(name=name) for name in shock_names)

    def set_data_info(self) -> Data | tuple[Data, ...] | None:
        data = []

        if self.has_exog:
            data.append(
                Data(
                    name="exog_data",
                    shape=(None, self.k_exog),
                    dims=(TIME_DIM, EXOG_STATE_DIM),
                    is_exogenous=True,
                )
            )

        return tuple(data)

    def set_coords(self) -> Coord | tuple[Coord, ...] | None:
        coords = list(self.default_coords())

        factor_labels = tuple(f"factor_{i + 1}" for i in range(self.k_factors))
        coords.append(Coord(dimension=FACTOR_DIM, labels=factor_labels))

        if self.factor_order > 0:
            ar_labels = tuple(range(1, (self.factor_order * self.k_factors) + 1))
            coords.append(Coord(dimension=AR_PARAM_DIM, labels=ar_labels))

        if self.error_order > 0:
            if self.error_var:
                error_ar_labels = tuple(range(1, (self.error_order * self.k_endog) + 1))
            else:
                error_ar_labels = tuple(range(1, self.error_order + 1))
            coords.append(Coord(dimension=ERROR_AR_PARAM_DIM, labels=error_ar_labels))

        if self.has_exog:
            k_non_exog_states = self.k_states - self.k_exog_states
            coords.append(Coord(dimension=EXOG_STATE_DIM, labels=self.exog_state_names))
            coords.append(
                Coord(dimension=EXOG_COEF_STATE_DIM, labels=self.state_names[k_non_exog_states:])
            )
            coords.append(
                Coord(dimension=NON_EXOG_STATE_DIM, labels=self.state_names[:k_non_exog_states])
            )

        return tuple(coords)

    def make_symbolic_graph(self):
        if self.has_exog:
            # Regression coefficients are states, so their prior is the tail of the initial state.
            initial_factor_states = self.make_and_register_variable(
                "x0", shape=(self.k_states - self.k_exog_states,), dtype=floatX
            )
            initial_betas = self.make_and_register_variable(
                "beta", shape=(self.k_exog_states,), dtype=floatX
            )
            x0 = pt.concatenate([initial_factor_states, initial_betas], axis=0)
        else:
            x0 = self.make_and_register_variable("x0", shape=(self.k_states,), dtype=floatX)

        self.ssm["initial_state", :] = x0

        P0 = self.make_and_register_variable(
            "P0", shape=(self.k_states, self.k_states), dtype=floatX
        )
        self.ssm["initial_state_cov", :, :] = P0

        self.ssm["design"] = self._build_design_matrix()
        if self.has_exog:
            self.ssm.declare_time_varying("design")

        self.ssm["transition", :, :] = pt.linalg.block_diag(*self._build_transition_blocks())

        self._build_selection_matrix()

        error_cov = self._build_error_cov()
        self._build_state_cov(error_cov)
        self._build_obs_cov(error_cov)

    def _build_design_matrix(self):
        r"""
        Assemble the design matrix :math:`Z`.

        The factor and error components give the block row :math:`[\Lambda | 0 | I | 0]`, where
        :math:`\Lambda` are the factor loadings, :math:`I` picks out the current error states, and
        the zero blocks cover lagged states. With exogenous regressors the result gains a leading
        time axis and the exogenous block is concatenated on the right.
        """
        factor_loadings = self.make_and_register_variable(
            "factor_loadings", shape=(self.k_endog, self.k_factors), dtype=floatX
        )
        matrix_parts = [factor_loadings]

        if self.factor_order > 1:
            matrix_parts.append(
                pt.zeros((self.k_endog, self.k_factors * (self.factor_order - 1)), dtype=floatX)
            )

        if self.error_order > 0:
            matrix_parts.append(pt.eye(self.k_endog, dtype=floatX))
            matrix_parts.append(
                pt.zeros((self.k_endog, self.k_endog * (self.error_order - 1)), dtype=floatX)
            )

        if len(matrix_parts) == 1:
            # Copy so the design matrix is its own node rather than an alias of the registered
            # factor_loadings variable, which the matrix substitution machinery relies on.
            design_matrix = factor_loadings.copy()
        else:
            design_matrix = pt.concatenate(matrix_parts, axis=1)
        design_matrix.name = "design"

        if not self.has_exog:
            return design_matrix

        exog_data = self.make_and_register_data("exog_data", shape=(None, self.k_exog))
        if self.shared_exog_states:
            # One set of coefficients seen by every observed series: (time, k_endog, k_exog).
            Z_exog = pt.join(1, *[pt.expand_dims(exog_data, 1) for _ in range(self.k_endog)])
        else:
            # Each observed series gets its own block of coefficients, laid out endog-major:
            # (time, k_endog, k_exog * k_endog).
            Z_exog = pt.linalg.block_diag(
                *[pt.expand_dims(exog_data, 1) for _ in range(self.k_endog)]
            )

        design_matrix_time = pt.tile(design_matrix, (Z_exog.shape[0], 1, 1))
        design_matrix = pt.concatenate([design_matrix_time, Z_exog], axis=2)
        design_matrix.name = "design"

        return design_matrix

    def _build_transition_blocks(self) -> list[pt.TensorVariable]:
        r"""
        Build the per-component blocks of the transition matrix :math:`T`.

        Returns
        -------
        blocks : list of TensorVariable
            Companion matrices for the factor and error components and, when exogenous regressors
            are present, an identity block for the coefficient states. Assembling these with
            :func:`pytensor.tensor.linalg.block_diag` gives :math:`T`.
        """
        blocks = []

        if self.factor_order > 0:
            factor_ar = self.make_and_register_variable(
                "factor_ar",
                shape=(self.k_factors, self.factor_order * self.k_factors),
                dtype=floatX,
            )
            blocks.append(_make_var_companion_matrix(factor_ar, self.k_factors, self.factor_order))
        else:
            blocks.append(pt.zeros((self.k_factors, self.k_factors), dtype=floatX))

        if self.error_order > 0:
            if self.error_var:
                error_ar = self.make_and_register_variable(
                    "error_ar", shape=(self.k_endog, self.error_order * self.k_endog), dtype=floatX
                )
                blocks.append(_make_var_companion_matrix(error_ar, self.k_endog, self.error_order))
            else:
                error_ar = self.make_and_register_variable(
                    "error_ar", shape=(self.k_endog, self.error_order), dtype=floatX
                )
                blocks.append(
                    _make_independent_ar_companion_matrix(error_ar, self.k_endog, self.error_order)
                )

        # Exogenous coefficients are constant, or a random walk when exog_innovations is set. Either
        # way the transition is the identity; only the shock loading differs.
        if self.has_exog:
            blocks.append(pt.eye(self.k_exog_states, dtype=floatX))

        return blocks

    def _build_selection_matrix(self) -> None:
        """Wire each shock into the state it enters, in the order the shocks are declared."""
        for i in range(self.k_factors):
            self.ssm["selection", i, i] = 1.0

        if self.error_order > 0:
            for i in range(self.k_endog):
                row = max(self.factor_order, 1) * self.k_factors + i
                col = self.k_factors + i
                self.ssm["selection", row, col] = 1.0

        if self.has_exog and self.exog_innovations:
            col_start = self.k_factors + (self.k_endog if self.error_order > 0 else 0)
            self.ssm[
                "selection",
                self.k_states - self.k_exog_states : self.k_states,
                col_start : col_start + self.k_exog_states,
            ] = pt.eye(self.k_exog_states, dtype=floatX)

    def _build_error_cov(self):
        """Build the covariance of the idiosyncratic observation errors."""
        if self.error_cov_type == "scalar":
            error_sigma = self.make_and_register_variable("error_sigma", shape=(), dtype=floatX)
            return pt.eye(self.k_endog, dtype=floatX) * error_sigma**2
        elif self.error_cov_type == "diagonal":
            error_sigma = self.make_and_register_variable(
                "error_sigma", shape=(self.k_endog,), dtype=floatX
            )
            return pt.diag(error_sigma**2)
        elif self.error_cov_type == "unstructured":
            return self.make_and_register_variable(
                "error_cov", shape=(self.k_endog, self.k_endog), dtype=floatX
            )

        raise ValueError(
            "error_cov_type must be one of 'scalar', 'diagonal', or 'unstructured', got "
            f"{self.error_cov_type!r}"
        )

    def _build_state_cov(self, error_cov) -> None:
        r"""Build the state covariance :math:`Q` from the blocks the model actually has."""
        # Factor innovations are standardized to identity in order to identify the factors.
        blocks = [pt.eye(self.k_factors, dtype=floatX)]

        if self.error_order > 0:
            blocks.append(error_cov)

        if self.has_exog:
            if self.exog_innovations:
                beta_sigma = self.make_and_register_variable(
                    "beta_sigma", shape=(self.k_exog_states,), dtype=floatX
                )
                blocks.append(pt.diag(beta_sigma))
            else:
                blocks.append(pt.zeros((self.k_exog_states, self.k_exog_states), dtype=floatX))

        self.ssm["state_cov", :, :] = pt.linalg.block_diag(*blocks)

    def _build_obs_cov(self, error_cov) -> None:
        r"""Build the observation covariance :math:`H`."""
        obs_cov = pt.zeros((self.k_endog, self.k_endog), dtype=floatX)

        # Without error states the idiosyncratic error is not in the state vector, so it lands here.
        if self.error_order == 0:
            obs_cov = obs_cov + error_cov

        if self.measurement_error:
            sigma_obs = self.make_and_register_variable(
                "sigma_obs", shape=(self.k_endog,), dtype=floatX
            )
            obs_cov = obs_cov + pt.diag(sigma_obs**2)

        self.ssm["obs_cov", :, :] = obs_cov
