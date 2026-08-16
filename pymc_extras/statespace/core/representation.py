import copy

import numpy as np
import pytensor
import pytensor.tensor as pt

from pymc_extras.statespace.core.assumptions import declare_time_varying

floatX = pytensor.config.floatX
KeyLike = tuple[str | int, ...] | str

# Number of trailing dims that define each matrix's "core" shape (1 for vectors, 2 for
# matrices). Anything to the left is a time and/or batch dim; downstream consumers
# distinguish them by consulting ``Representation.time_varying_names``.
_CORE_NDIM: dict[str, int] = {
    "design": 2,
    "obs_intercept": 1,
    "obs_cov": 2,
    "transition": 2,
    "state_intercept": 1,
    "selection": 2,
    "state_cov": 2,
    "initial_state": 1,
    "initial_state_cov": 2,
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

    Any matrix may carry a leading dim of length ``n`` (the number of observations) to
    make it time-varying. The model is responsible for declaring which matrices vary over
    time via :meth:`declare_time_varying`; downstream consumers (filter, smoother, etc.)
    read :attr:`time_varying_names` to dispatch.

    .. warning::

        The time dim must always be the **leftmost** axis of a time-varying matrix. The
        Kalman filter scan iterates over axis 0, ``pm.Deterministic`` registration prepends
        a ``TIME_DIM`` coord to the dim list, and forecast slicing reads ``matrix[n_train:]``
        — all of these assume the time axis sits in front of the core dims. Optional batch
        dims may sit even further left (``(*batch, time, *core)``).

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
        "_time_varying_names",
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

        self._time_varying_names: set[str] = set()

        for name in _CORE_NDIM:
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
        if name not in _CORE_NDIM:
            raise IndexError(f"{name!r} is an invalid state space matrix name")

    def _coerce(
        self, name: str, value: np.ndarray | pt.TensorVariable | float | int
    ) -> pt.TensorVariable:
        """Wrap ``value`` as a named TensorVariable after validating the trailing core dims.

        Anything to the left of the core dims is accepted unchanged -- ``core``,
        ``(time, *core)``, ``(*batch, *core)``, and ``(*batch, time, *core)`` are all valid.
        Whether a leading dim is "time" or "batch" is the model's decision, recorded via
        ``declare_time_varying``; the rep itself does not guess.
        """
        core_ndim = _CORE_NDIM[name]
        core_shape = self._core_shape(name)

        if isinstance(value, np.ndarray):
            tensor = pt.as_tensor_variable(value.astype(floatX), name=name)
        elif isinstance(value, int | float):
            tensor = pt.as_tensor_variable(np.asarray(value, dtype=floatX), name=name)
        else:
            # Name a derived variable, never the caller's own: model parameters are looked up by
            # name, so renaming one here would quietly break that lookup. ``copy`` drops tags, so
            # carry them across by hand.
            given = pt.as_tensor_variable(value)
            tensor = given.copy(name=name)
            tensor.tag = copy.copy(given.tag)

        if tensor.ndim < core_ndim:
            raise ValueError(
                f"{name} must have at least {core_ndim} dimension(s); got ndim={tensor.ndim}"
            )

        trailing = tensor.type.shape[-core_ndim:]
        for got, want in zip(trailing, core_shape, strict=True):
            # A None entry in ``tensor.type.shape`` is a runtime-only dim (pytensor doesn't
            # know its size at graph time). Skip; let it fail at runtime if wrong.
            if got is not None and got != want:
                raise ValueError(f"Trailing dims of {name} are {trailing}, expected {core_shape}")

        return tensor

    @property
    def time_varying_names(self) -> frozenset[str]:
        """Names of matrices the model declared as time-varying.

        The Kalman filter iterates over the leading dim of these tensors at scan time;
        all other matrices are passed in as scan non-sequences (i.e. used unchanged at
        every step).
        """
        return frozenset(self._time_varying_names)

    def declare_time_varying(self, *names: str) -> None:
        """Mark matrices as time-varying. The filter will iterate over the leading dim.

        The declaration rides on the tensor itself, so it survives substitution and slicing and
        travels with the matrix into any graph built from it.
        """
        for name in names:
            self._validate_name(name)
            self._time_varying_names.add(name)
            setattr(self, name, declare_time_varying(getattr(self, name)))

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
            updated.name = existing.name
            updated.tag = copy.copy(existing.tag)
            setattr(self, name, updated)
            return

        raise IndexError("First index must the name of a valid state space matrix.")
