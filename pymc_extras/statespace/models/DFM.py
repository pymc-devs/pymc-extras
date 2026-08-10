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
    Dynamic Factor Models

    Notes
    -----
    The Dynamic Factor Model (DFM) is a multivariate state-space model used to represent high-dimensional time series
    as being driven by a smaller set of unobserved dynamic factors.

    Given a set of observed time series :math:`\{y_t\}_{t=0}^T`, where

    .. math::
        y_t = \begin{bmatrix} y_{1,t} & y_{2,t} & \cdots & y_{k_{\text{endog}},t} \end{bmatrix}^T,

    the DFM assumes that each series is a linear combination of a few latent factors and (optionally) autoregressive errors.

    Let:
    - :math:`k` be the number of dynamic factors (k_factors),
    - :math:`p` be the order of the latent factor process (factor_order),
    - :math:`q` be the order of the observation error process (error_order).

    The model equations are in reduced form is:

    .. math::
        y_t &= \Lambda f_t + B x_t + u_t + \eta_t \\
        f_t &= A_1 f_{t-1} + \cdots + A_p f_{t-p} + \varepsilon_{f,t} \\
        u_t &= C_1 u_{t-1} + \cdots + C_q u_{t-q} + \varepsilon_{u,t}

    Where:
    - :math:`f_t` is the vector of latent dynamic factors (size :math:`k`),
    - :math:`x_t` is an optional vector of exogenous variables
    - :math:`u_t` is a vector of autoregressive observation errors (if `error_var=True` with a VAR(q) structure, else treated as independent AR processes),
    - :math:`\eta_t \sim \mathcal{N}(0, H_t)` is an optional measurement error (if `measurement_error=True`),
    - :math:`\varepsilon_{f,t} \sim \mathcal{N}(0, I)` and :math:`\varepsilon_{u,t} \sim \mathcal{N}(0, \Sigma_u)` are independent noise terms.
        To identify the factors, the innovations to the factor process are standardized with identity covariance.

    Internally, the model is represented in state-space form by stacking all current and lagged latent factors and (if present)
    AR observation errors into a single state vector of dimension:  :math:: k_{\text{states}} = k \cdot p + k_{\text{endog}} \cdot q,
    where :math:`k_{\text{endog}}` is the number of observed time series.

    The state vector is defined as:

    .. math::
        s_t = \begin{bmatrix}
            f_t(1) \\
            \vdots \\
            f_t(k) \\
            f_{t-p+1}(1) \\
            \vdots \\
            f_{t-p+1}(k) \\
            u_t(1) \\
            \vdots \\
            u_t(k_{\text{endog}}) \\
            \vdots \\
            u_{t-q+1}(1) \\
            \vdots \\
            u_{t-q+1}(k_{\text{endog}})
        \end{bmatrix}
        \in \mathbb{R}^{k_{\text{states}}}

    The transition equation is given by:

    .. math::
        s_{t+1} = T s_t + R \epsilon_t

    Where:
    - :math:`T` is the state transition matrix, composed of:
        - VAR coefficients :math:`A_1, \dots, A_{p*k_factors}` for the factors,
        - (if enabled) AR coefficients :math:`C_1, \dots, C_q` for the observation errors.
        .. math::
            T = \begin{bmatrix}
            A_{1,1}  &   A_{1,2}  &   \cdots  &   A_{1,p}  &   0       &   0       &   \cdots  &   0 \\
            A_{2,1}  &   A_{2,2}  &   \cdots  &   A_{2,p}  &   0       &   0       &   \cdots  &   0 \\
            1       &   0       &   \cdots  &   0          &   0       &   0       &   \cdots  &   0 \\
            0       &   1       &   \cdots  &   0          &   0       &   0       &   \cdots  &   0 \\
            \vdots  &   \vdots  &   \ddots  &   \vdots     &   \vdots  &   \vdots  &   \ddots  &   \vdots \\
            \hline
            0       &   0       &   \cdots  &   0       &   C_{1,1}  &  \cdots  &    C_{1,2} &   C_{1,q} \\
            0       &   0       &   \cdots  &   0       &   1       &   0       &   \cdots  &   0 \\
            0       &   0       &   \cdots  &   0       &   0       &   1       &   \cdots  &   0 \\
            \vdots  &   \vdots  &           &   \vdots  &   \vdots  &   \vdots  &   \ddots  &   \vdots
            \end{bmatrix}
            \in \mathbb{R}^{k_{\text{states}} \times k_{\text{states}}}

    - :math:`\epsilon_t` contains the independent shocks (innovations) and has dimension :math:`k + k_{\text{endog}}` if AR errors are included.
        .. math::
            \epsilon_t = \begin{bmatrix}
            \epsilon_{f,t} \\
            \epsilon_{u,t}
            \end{bmatrix}
            \in \mathbb{R}^{k +  k_{\text{endog}}}

    - :math:`R` is a selection matrix mapping shocks to state transitions.
        .. math::
            R = \begin{bmatrix}
            1       &   0       &   \cdots  &   0       &   0       &   0       &   \cdots  &   0 \\
            0       &   1       &   \cdots  &   0       &   0       &   0       &   \cdots  &   0 \\
            \vdots  &   \vdots  &   \ddots  &   \vdots  &   \vdots  &   \vdots  &   \ddots  &   \vdots \\
            0       &   0       &   \cdots  &   1       &   0       &   0       &   \cdots  &   0 \\
            0       &   0       &   \cdots  &   0       &   1       &   0       &   \cdots  &   0 \\
            0       &   0       &   \cdots  &   0       &   0       &   1       &   \cdots  &   0 \\
            \vdots  &   \vdots  &   \ddots  &   \vdots  &   \vdots  &   \vdots  &   \ddots  &   \vdots \\
            \end{bmatrix}
            \in \mathbb{R}^{k_{\text{states}} \times (k + k_{\text{endog}})}

    The observation equation is given by:

    .. math::

        y_t = Z s_t + \eta_t

    where

    - :math:`y_t` is the vector of observed variables at time :math:`t`

    - :math:`Z` is the design matrix of the state space representation
        .. math::
            Z = \begin{bmatrix}
            \lambda_{1,1}       &   \lambda_{1,k}   &   \vdots    &   1   &   0   &   \cdots  &   0   &   0   &   \cdots  &   0 \\
            \lambda_{2,1}       &   \lambda_{2,k}   &   \vdots    &   0   &   1   &   \cdots   &   0   &   \cdots  &   0 \\
            \vdots              &   \vdots          &   \vdots  &   \vdots  &   \ddots  &   \vdots  &   \vdots  &   \ddots  &   \vdots \\
            \lambda_{k_{\text{endog}},1}  &   \cdots  &   \lambda_{k_{\text{endog}},k}  &   0   &   0   &   \cdots  &   1   &   0   &   \cdots  &   0 \\
            \end{bmatrix}
            \in \mathbb{R}^{k_{\text{endog}} \times k_{\text{states}}}

    - :math:`\eta_t` is the vector of observation errors at time :math:`t`

    When exogenous variables :math:`x_t` are present, the implementation follows `pymc_extras/statespace/models/structural/components/regression.py`.
    In this case, the state vector is extended to include the beta parameters, and the design matrix is modified accordingly,
    becoming 3-dimensional to handle time-varying exogenous regressors.
    This approach provides greater flexibility, controlled by the boolean flags `shared_exog_state` and `exog_innovations`.
    Unlike Statsmodels, where exogenous variables are included only in the observation equation, here they are fully integrated into the state-space
    representation.

    .. warning::

        Identification can be an issue, particularly when many observed series load onto only a few latent factors.
        These models are only identified up to a sign flip in the factor loadings. Proper prior specification is crucial
        for good estimation and inference.

    Examples
    --------
    The following code snippet estimates a dynamic factor model with 1 latent factors,
    a AR(2) structure on the factor and a AR(1) structure on the errors:

    .. code:: python

        import pymc_extras.statespace as pmss
        import pymc as pm

        # Create DFM Statespace Model
        dfm_mod = pmss.BayesianDynamicFactor(
                k_factors=1,                # Number of latent dynamic factors
                factor_order=2,             # Number of lags for the latent factor process
                endog_names=data.columns,   # Names of the observed time series (endogenous variables) (we could also use k_endog = len(data.columns))
                error_order=1,              # Order of the autoregressive process for the observation noise (i.e., AR(q) error, here q=1)
                error_var=False,            # If False, models errors as separate AR processes
                error_cov_type="diagonal",  # Structure of the observation error covariance matrix: uncorrelated noise across series
                measurement_error=True,     # Whether to include a measurement error term in the model
                verbose=True
            )

        # Unpack coords
        coords = dfm_mod.coords


        with pm.Model(coords=coords) as pymc_mod:
            # Priors for the initial state mean and covariance
            x0 = pm.Normal("x0", dims=["state_dim"])
            P0 = pm.HalfNormal("P0", dims=["state_dim", "state_dim"])

            # Factor loadings: shape (k_endog, k_factors)
            factor_loadings = pm.Normal("factor_loadings", sigma=1, dims=["k_endog", "k_factors"])

            # AR coefficients for factor dynamics: shape (k_factors, factor_order)
            factor_ar = pm.Normal("factor_ar", sigma=1, dims=["k_factors", "k_factors" * "factor_order"])

            # AR coefficients for observation noise: shape (k_endog, error_order)
            error_ar = pm.Normal("error_ar", sigma=1, dims=["k_endog", "error_order"])

            # Std devs for observation noise: shape (k_endog,)
            error_sigma = pm.HalfNormal("error_sigma", dims=["k_endog"])

            # Observation noise covariance matrix
            obs_sigma = pm.HalfNormal("sigma_obs", dims=["k_endog"])

            # Build the symbolic graph and attach it to the model
            dfm_mod.build_statespace_graph(data=data)

            # Sampling
            idata = pm.sample(
                draws=500,
                chains=2,
                nuts_sampler="nutpie",
                nuts_sampler_kwargs={"backend": "jax", "gradient_backend": "jax"},
            )

    """

    def __init__(
        self,
        k_factors: int,
        factor_order: int,
        endog_names: Sequence[str] | None = None,
        exog_names: Sequence[str] | None = None,
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
        """
        Create a Bayesian Dynamic Factor Model.

        Parameters
        ----------
        k_factors : int
            Number of latent factors.

        factor_order : int
            Order of the VAR process for the latent factors. If set to 0, the factors have no autoregressive dynamics
            and are modeled as a white noise process, i.e., :math:`f_t = \varepsilon_{f,t}`.
            Therefore, the state vector will include one state per factor and "factor_ar" will not exist.

        endog_names : Sequence of str, optional
            Names of the observed time series.

        exog_names : Sequence of str, optional
            Names of the exogenous variables.

        shared_exog_states: bool, optional
            Whether exogenous latent states are shared across the observed states. If True, there will be only one set of exogenous latent
            states, which are observed by all observed states. If False, each observed state has its own set of exogenous latent states.

        exog_innovations : bool, optional
            Whether to allow time-varying regression coefficients. If True, coefficients follow a random walk.

        error_order : int, optional
            Order of the AR process for the observation error component.
            Default is 0, corresponding to white noise errors.

        error_var : bool, optional
            If True, errors are modeled jointly via a VAR process;
            otherwise, each error is modeled separately.

        error_cov_type : {'scalar', 'diagonal', 'unstructured'}, optional
            Structure of the covariance matrix of the observation errors.

        filter_type : str, optional
            The type of Kalman Filter to use. Options are "standard", "univariate", and "cholesky".
            See the docs for kalman filters for more details. Default "standard".

        measurement_error: bool, default True
            If true, a measurement error term is added to the model.

        verbose: bool, default True
            If true, a message will be logged to the terminal explaining the variable names, dimensions, and supports.

        mode : str or Mode, optional
            Pytensor compile mode, used in auxiliary sampling methods such as
            ``sample_conditional_posterior`` and ``forecast``. The mode does **not** effect calls to
            ``pm.sample``. Regardless of whether a mode is specified, it can always be overwritten
            via the ``compile_kwargs`` argument to all sampling methods. Default None.


        cov_jitter: float, optional
            Jitter added to the diagonal of every covariance matrix at each filtering step, for numerical
            stability. Post-estimation graphs are built with this same value. Default 1e-8, or 1e-6 if
            ``pytensor.config.floatX`` is float32.

        missing_fill_value: float, optional
            Sentinel used to mask missing observations. Set this only if your data legitimately contains the
            default sentinel. Post-estimation graphs are built with this same value. Default -9999.0.
        """

        validate_names(endog_names, var_name="endog_names", optional=False)
        k_endog = len(endog_names)
        self.endog_names = tuple(endog_names)
        self.k_endog = k_endog
        self.k_factors = k_factors
        self.factor_order = factor_order
        self.error_order = error_order
        self.error_var = error_var
        self.error_cov_type = error_cov_type

        if exog_names is not None:
            self.shared_exog_states = shared_exog_states
            self.exog_innovations = exog_innovations
            validate_names(
                exog_names, var_name="exog_names", optional=True
            )  # Not sure if this adds anything
            k_exog = len(exog_names)
            self.k_exog = k_exog
            self.exog_names = exog_names
        else:
            self.k_exog = 0

        self.k_exog_states = self.k_exog * self.k_endog if not shared_exog_states else self.k_exog
        self.exog_flag = self.k_exog > 0

        # Determine the dimension for the latent factor states.
        # For static factors, one use k_factors.
        # For dynamic factors with lags, the state include current factors and past lags.
        # If factor_order is 0, we treat the factor as static (no dynamics),
        # but it is still included in the state vector with one state per factor. Factor_ar paramter will not exist in this case.
        k_factor_states = max(self.factor_order, 1) * k_factors

        # Determine the dimension for the error component.
        # If error_order > 0 then we add additional states for error dynamics, otherwise white noise error.
        k_error_states = k_endog * error_order if error_order > 0 else 0

        # Total state dimension
        k_states = k_factor_states + k_error_states + self.k_exog_states

        # Number of independent shocks.
        # Typically, the latent factors introduce k_factors shocks.
        # If error_order > 0 and errors are modeled jointly or separately, add appropriate count.
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

        # x0 - initial state
        parameters.append(
            Parameter(
                name="x0",
                shape=(k_states - self.k_exog_states,),
                dims=(NON_EXOG_STATE_DIM if self.exog_flag else ALL_STATE_DIM,),
                constraints=None,
            )
        )

        # P0 - initial covariance
        parameters.append(
            Parameter(
                name="P0",
                shape=(k_states, k_states),
                dims=(ALL_STATE_DIM, ALL_STATE_AUX_DIM),
                constraints="Positive Semi-definite",
            )
        )

        # factor_loadings
        parameters.append(
            Parameter(
                name="factor_loadings",
                shape=(k_endog, self.k_factors),
                dims=(OBS_STATE_DIM, FACTOR_DIM),
                constraints=None,
            )
        )

        # factor_ar - only if factor_order > 0
        if self.factor_order > 0:
            parameters.append(
                Parameter(
                    name="factor_ar",
                    shape=(self.k_factors, self.factor_order * self.k_factors),
                    dims=(FACTOR_DIM, AR_PARAM_DIM),
                    constraints=None,
                )
            )

        # error_ar - only if error_order > 0
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

        # error_sigma or error_cov depending on error_cov_type
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

        # sigma_obs - only if measurement_error
        if self.measurement_error:
            parameters.append(
                Parameter(
                    name="sigma_obs",
                    shape=(k_endog,),
                    dims=(OBS_STATE_DIM,),
                    constraints="Positive",
                )
            )

        # beta - only if exog_flag
        if self.exog_flag:
            parameters.append(
                Parameter(
                    name="beta",
                    shape=(self.k_exog_states,),
                    dims=(EXOG_COEF_STATE_DIM,),
                    constraints=None,
                )
            )

            # beta_sigma - only if exog_innovations
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
            for i in range(self.k_factors)
        ]

        names.extend(
            f"L{lag}.error_{i}" for lag in range(self.error_order) for i in range(self.k_endog)
        )

        if self.exog_flag:
            if self.shared_exog_states:
                names.extend([f"beta_{exog_name}[shared]" for exog_name in self.exog_names])
            else:
                names.extend(
                    f"beta_{exog_name}[{endog_name}]"
                    for endog_name in self.endog_names
                    for exog_name in self.exog_names
                )

        hidden_states = [State(name=name, observed=False, shared=False) for name in names]
        observed_states = [
            State(name=name, observed=True, shared=False) for name in self.endog_names
        ]

        return *hidden_states, *observed_states

    def set_shocks(self) -> Shock | tuple[Shock, ...] | None:
        shock_names = [f"factor_shock_{i}" for i in range(self.k_factors)]

        if self.error_order > 0:
            shock_names.extend(f"error_shock_{i}" for i in range(self.k_endog))

        if self.exog_flag:
            if self.shared_exog_states:
                shock_names.extend(f"exog_shock_{i}.shared" for i in range(self.k_exog))
            else:
                # Ordered to match the exogenous states in set_states, which are endog-major.
                shock_names.extend(
                    f"exog_shock_{i}.endog_{j}"
                    for j in range(self.k_endog)
                    for i in range(self.k_exog)
                )

        return tuple(Shock(name=name) for name in shock_names)

    def set_data_info(self) -> Data | tuple[Data, ...] | None:
        data = []

        if self.exog_flag:
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
        k_endog = self.k_endog
        coords = list(self.default_coords())

        # Factor coords
        factor_labels = tuple(f"factor_{i + 1}" for i in range(self.k_factors))
        coords.append(Coord(dimension=FACTOR_DIM, labels=factor_labels))

        # AR param coords for factors
        if self.factor_order > 0:
            ar_labels = tuple(range(1, (self.factor_order * self.k_factors) + 1))
            coords.append(Coord(dimension=AR_PARAM_DIM, labels=ar_labels))

        # AR param coords for errors
        if self.error_order > 0:
            if self.error_var:
                error_ar_labels = tuple(range(1, (self.error_order * k_endog) + 1))
            else:
                error_ar_labels = tuple(range(1, self.error_order + 1))
            coords.append(Coord(dimension=ERROR_AR_PARAM_DIM, labels=error_ar_labels))

        # Exogenous coords
        if self.exog_flag:
            k_non_exog_states = self.k_states - self.k_exog_states
            coords.append(Coord(dimension=EXOG_STATE_DIM, labels=tuple(self.exog_names)))
            coords.append(
                Coord(dimension=EXOG_COEF_STATE_DIM, labels=self.state_names[k_non_exog_states:])
            )
            coords.append(
                Coord(dimension=NON_EXOG_STATE_DIM, labels=self.state_names[:k_non_exog_states])
            )

        return tuple(coords)

    def make_symbolic_graph(self):
        if self.exog_flag:
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
        if self.exog_flag:
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

        if not self.exog_flag:
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
        if self.exog_flag:
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

        if self.exog_flag and self.exog_innovations:
            col_start = self.k_factors + (self.k_endog if self.error_order > 0 else 0)
            self.ssm[
                "selection",
                self.k_states - self.k_exog_states : self.k_states,
                col_start : col_start + self.k_exog_states,
            ] = pt.eye(self.k_exog_states, dtype=floatX)

    def _build_state_cov(self, error_cov) -> None:
        r"""Build the state covariance :math:`Q` from the blocks the model actually has."""
        # Factor innovations are standardized to identity in order to identify the factors.
        blocks = [pt.eye(self.k_factors, dtype=floatX)]

        if self.error_order > 0:
            blocks.append(error_cov)

        if self.exog_flag:
            if self.exog_innovations:
                beta_sigma = self.make_and_register_variable(
                    "beta_sigma", shape=(self.k_exog_states,), dtype=floatX
                )
                blocks.append(pt.diag(beta_sigma))
            else:
                blocks.append(pt.zeros((self.k_exog_states, self.k_exog_states), dtype=floatX))

        self.ssm["state_cov", :, :] = pt.linalg.block_diag(*blocks)

    def _build_error_cov(self):
        """Build the covariance of the idiosyncratic observation errors."""
        if self.error_cov_type == "scalar":
            error_sigma = self.make_and_register_variable("error_sigma", shape=(), dtype=floatX)
            return pt.eye(self.k_endog, dtype=floatX) * error_sigma
        elif self.error_cov_type == "diagonal":
            error_sigma = self.make_and_register_variable(
                "error_sigma", shape=(self.k_endog,), dtype=floatX
            )
            return pt.diag(error_sigma)
        elif self.error_cov_type == "unstructured":
            return self.make_and_register_variable(
                "error_cov", shape=(self.k_endog, self.k_endog), dtype=floatX
            )

        raise ValueError(
            "error_cov_type must be one of 'scalar', 'diagonal', or 'unstructured', got "
            f"{self.error_cov_type!r}"
        )

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
            obs_cov = obs_cov + pt.diag(sigma_obs)

        self.ssm["obs_cov", :, :] = obs_cov
