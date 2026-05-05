import copy

import numpy as np
import pytensor
import pytensor.tensor as pt

floatX = pytensor.config.floatX
KeyLike = tuple[str | int, ...] | str

# core_ndim: number of trailing dims that define the matrix's "core" shape (1 for vectors,
# 2 for matrices). Anything stored to the left of those dims is interpreted as a time and/or
# batch dim by downstream consumers; the representation does not try to distinguish them.
# time_varying: False rules out leading dims entirely (used for x0 and P0).
_MATRIX_SPECS: dict[str, tuple[int, bool]] = {
    "design": (2, True),
    "obs_intercept": (1, True),
    "obs_cov": (2, True),
    "transition": (2, True),
    "state_intercept": (1, True),
    "selection": (2, True),
    "state_cov": (2, True),
    "initial_state": (1, False),
    "initial_state_cov": (2, False),
}


class PytensorRepresentation:
    r"""
    Container for the nine state-space matrices of a linear Gaussian model.

    Each matrix is stored in its *natural* shape: ``core_shape`` if static, or
    ``(time, *core_shape)`` if time-varying. Optional leading batch dims are accepted and
    propagate untouched -- use ``vectorize_graph`` if a downstream consumer needs to lift
    over them.

    Parameters
    ----------
    k_endog : int
        Number of observed states.
    k_states : int
        Number of hidden states.
    k_posdef : int, optional
        Rank of the selection matrix :math:`R`; also the number of stochastic shocks. Default
        ``k_states``.
    design : array_like, optional
        Initial value for the design matrix :math:`Z`.
    obs_intercept : array_like, optional
        Initial value for :math:`d`.
    obs_cov : array_like, optional
        Initial value for the observation noise covariance :math:`H`.
    transition : array_like, optional
        Initial value for the transition matrix :math:`T`.
    state_intercept : array_like, optional
        Initial value for :math:`c`.
    selection : array_like, optional
        Initial value for the selection matrix :math:`R`.
    state_cov : array_like, optional
        Initial value for the state innovation covariance :math:`Q`.
    initial_state : array_like, optional
        Initial value for :math:`\bar x_0`.
    initial_state_cov : array_like, optional
        Initial value for :math:`P_0`.

    Notes
    -----
    A linear state-space system has

    .. math::
        \begin{align}
            x_t &= T_t x_{t-1} + c_t + R_t \varepsilon_t \\
            y_t &= Z_t x_t + d_t + \eta_t \\
            \varepsilon_t &\sim N(0, Q_t) \\
            \eta_t &\sim N(0, H_t) \\
            x_0 &\sim N(\bar x_0, P_0)
        \end{align}

    with hidden state size :math:`m = k_{\text{states}}`, observed size :math:`p = k_{\text{endog}}`,
    and shock rank :math:`r = k_{\text{posdef}}`. The natural per-matrix shapes are

    +---------------------+-------------------------+
    | Matrix              | Natural shape           |
    +=====================+=========================+
    | ``initial_state``   | ``(m,)``                |
    +---------------------+-------------------------+
    | ``initial_state_cov`` | ``(m, m)``            |
    +---------------------+-------------------------+
    | ``state_intercept`` | ``(m,)``                |
    +---------------------+-------------------------+
    | ``obs_intercept``   | ``(p,)``                |
    +---------------------+-------------------------+
    | ``transition``      | ``(m, m)``              |
    +---------------------+-------------------------+
    | ``design``          | ``(p, m)``              |
    +---------------------+-------------------------+
    | ``selection``       | ``(m, r)``              |
    +---------------------+-------------------------+
    | ``obs_cov``         | ``(p, p)``              |
    +---------------------+-------------------------+
    | ``state_cov``       | ``(r, r)``              |
    +---------------------+-------------------------+

    Any matrix except :math:`\bar x_0` and :math:`P_0` may carry a leading dim of length
    ``n`` (the number of observations) to declare it time-varying. Downstream consumers
    decide static vs. time-varying by reading ``stored.ndim``.

    Examples
    --------
    Store and retrieve a transition matrix:

    .. code:: python

        from pymc_extras.statespace.core.representation import PytensorRepresentation
        ssm = PytensorRepresentation(k_endog=1, k_states=3, k_posdef=1)
        ssm["transition"].type.shape
        # (3, 3)

    Set individual entries:

    .. code:: python

        ssm["design", 0, 0] = 1.0

    Replace an entire matrix, optionally with a time dim:

    .. code:: python

        import numpy as np
        ssm["obs_intercept"] = np.arange(10)[:, None]    # 10 timesteps, k_endog=1
        ssm["obs_intercept"].type.shape
        # (10, 1)

    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
       Time Series Analysis by State Space Methods: Second Edition.
       Oxford University Press.
    """

    __slots__ = (
        "design",
        "initial_state",
        "initial_state_cov",
        "k_endog",
        "k_posdef",
        "k_states",
        "obs_cov",
        "obs_intercept",
        "selection",
        "state_cov",
        "state_intercept",
        "transition",
    )

    def __init__(
        self,
        k_endog: int,
        k_states: int,
        k_posdef: int | None = None,
        design: np.ndarray | pt.TensorVariable | None = None,
        obs_intercept: np.ndarray | pt.TensorVariable | None = None,
        obs_cov: np.ndarray | pt.TensorVariable | None = None,
        transition: np.ndarray | pt.TensorVariable | None = None,
        state_intercept: np.ndarray | pt.TensorVariable | None = None,
        selection: np.ndarray | pt.TensorVariable | None = None,
        state_cov: np.ndarray | pt.TensorVariable | None = None,
        initial_state: np.ndarray | pt.TensorVariable | None = None,
        initial_state_cov: np.ndarray | pt.TensorVariable | None = None,
    ) -> None:
        self.k_endog = k_endog
        self.k_states = k_states
        self.k_posdef = k_posdef if k_posdef is not None else k_states

        provided = {
            "design": design,
            "obs_intercept": obs_intercept,
            "obs_cov": obs_cov,
            "transition": transition,
            "state_intercept": state_intercept,
            "selection": selection,
            "state_cov": state_cov,
            "initial_state": initial_state,
            "initial_state_cov": initial_state_cov,
        }

        for name in _MATRIX_SPECS:
            value = provided[name]
            if value is None:
                tensor = pt.as_tensor_variable(
                    np.zeros(self._core_shape(name), dtype=floatX), name=name
                )
            else:
                tensor = self._coerce(name, value)
            setattr(self, name, tensor)

    def _core_shape(self, name: str) -> tuple[int, ...]:
        k_endog, k_states, k_posdef = self.k_endog, self.k_states, self.k_posdef
        return {
            "design": (k_endog, k_states),
            "obs_intercept": (k_endog,),
            "obs_cov": (k_endog, k_endog),
            "transition": (k_states, k_states),
            "state_intercept": (k_states,),
            "selection": (k_states, k_posdef),
            "state_cov": (k_posdef, k_posdef),
            "initial_state": (k_states,),
            "initial_state_cov": (k_states, k_states),
        }[name]

    def _validate_name(self, name: str) -> None:
        if name not in _MATRIX_SPECS:
            raise IndexError(f"{name!r} is an invalid state space matrix name")

    def _coerce(
        self, name: str, value: np.ndarray | pt.TensorVariable | float | int
    ) -> pt.TensorVariable:
        """Wrap ``value`` as a named TensorVariable after validating the trailing core dims.

        Anything to the left of the core dims is accepted unchanged: ``core``,
        ``(time, *core)``, ``(*batch, *core)``, or ``(*batch, time, *core)`` are all valid.
        ``time_varying=False`` matrices reject any leading dim.
        """
        core_ndim, time_varying = _MATRIX_SPECS[name]
        core_shape = self._core_shape(name)

        if isinstance(value, np.ndarray):
            tensor = pt.as_tensor_variable(value.astype(floatX), name=name)
        elif isinstance(value, int | float):
            tensor = pt.as_tensor_variable(np.asarray(value, dtype=floatX), name=name)
        else:
            tensor = pt.as_tensor_variable(value)
            tensor.name = name

        if tensor.ndim < core_ndim:
            raise ValueError(
                f"{name} must have at least {core_ndim} dimension(s); got ndim={tensor.ndim}"
            )

        if not time_varying and tensor.ndim != core_ndim:
            raise ValueError(
                f"{name} is never time-varying; expected ndim={core_ndim}, got ndim={tensor.ndim}"
            )

        trailing = tensor.type.shape[-core_ndim:]
        for got, want in zip(trailing, core_shape, strict=True):
            # Static-None means the dim is unknown until runtime; allow it.
            if got is not None and got != want:
                raise ValueError(f"Trailing dims of {name} are {trailing}, expected {core_shape}")

        return tensor

    def is_time_varying(self, name: str) -> bool:
        """Return True iff the stored tensor has a leading dim beyond its core shape."""
        self._validate_name(name)
        return getattr(self, name).ndim > _MATRIX_SPECS[name][0]

    def core_ndim(self, name: str) -> int:
        """Return the number of trailing dims that define ``name``'s core shape."""
        self._validate_name(name)
        return _MATRIX_SPECS[name][0]

    def __getitem__(self, key: KeyLike) -> pt.TensorVariable:
        if isinstance(key, str):
            self._validate_name(key)
            return getattr(self, key)

        if isinstance(key, tuple) and isinstance(key[0], str):
            name, *idx = key
            self._validate_name(name)
            tensor = getattr(self, name)
            if not idx:
                return tensor
            return tensor[tuple(idx)]

        raise IndexError("First index must the name of a valid state space matrix.")

    def __setitem__(
        self, key: KeyLike, value: float | int | np.ndarray | pt.TensorVariable
    ) -> None:
        if isinstance(key, str):
            self._validate_name(key)
            setattr(self, key, self._coerce(key, value))
            return

        if isinstance(key, tuple) and isinstance(key[0], str):
            name, *idx = key
            self._validate_name(name)
            existing = getattr(self, name)
            updated = pt.set_subtensor(existing[tuple(idx)], value)
            updated.name = name
            setattr(self, name, updated)
            return

        raise IndexError("First index must the name of a valid state space matrix.")

    def copy(self) -> "PytensorRepresentation":
        return copy.copy(self)
