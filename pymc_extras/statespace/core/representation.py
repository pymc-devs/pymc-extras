import numpy as np
import pytensor
import pytensor.tensor as pt

from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.xtensor import as_xtensor
from pytensor.xtensor.type import XTensorVariable

from pymc_extras.statespace.utils.constants import (
    LONG_MATRIX_NAMES,
    LONG_NAME_TO_SHORT,
    MATRIX_DIMS,
    RESERVED_DIMS,
    TIME_DIM,
)

floatX = pytensor.config.floatX
KeyLike = tuple[str | int, ...] | str

CORE_DIMS: dict[str, tuple[str, ...]] = {
    name: MATRIX_DIMS[LONG_NAME_TO_SHORT[name]] for name in LONG_MATRIX_NAMES
}

# Several array indices select pointwise in numpy but orthogonally in xtensor, unless they share a
# dim. This is the dim they share.
POINTWISE_INDEX_DIM = "pointwise_index"


def is_array_index(index) -> bool:
    """Return whether ``index`` selects with an array, as opposed to an integer or a slice."""
    if isinstance(index, list | np.ndarray):
        return True
    return isinstance(index, pt.TensorVariable) and index.ndim > 0


def pointwise_index(index: tuple) -> tuple:
    """Give the array indices in ``index`` a shared dim, so they select pointwise as in numpy."""
    if sum(is_array_index(entry) for entry in index) < 2:
        return index
    return tuple(
        as_xtensor(entry, dims=(POINTWISE_INDEX_DIM,)) if is_array_index(entry) else entry
        for entry in index
    )


class PytensorRepresentation:
    r"""
    Container for the nine state-space matrices of a linear Gaussian model.

    Each matrix is stored as an xtensor with the core dims given in ``MATRIX_DIMS``. A matrix with
    a ``time`` dim is time-varying, and the Kalman filter scans over it. Every other dim is a
    batch dim.

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

    A matrix assigned as a tensor or array gets the core dims, plus a leading ``time`` dim when
    it has one more axis than the core shape. To add batch dims, assign an xtensor that names
    them.

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
        ssm["obs_intercept"].dims
        # ('time', 'observed_state')

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

        for name in CORE_DIMS:
            value = provided[name]
            if value is None:
                value = np.zeros(self._core_shape(name), dtype=floatX)
            setattr(self, name, self._coerce(name, value))

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
        if name not in CORE_DIMS:
            raise IndexError(f"{name!r} is an invalid state space matrix name")

    def _coerce(
        self, name: str, value: np.ndarray | pt.TensorVariable | XTensorVariable | float | int
    ) -> XTensorVariable:
        """Wrap ``value`` as a named XTensorVariable after validating its core dims."""
        core_dims = CORE_DIMS[name]
        core_shape = self._core_shape(name)

        if isinstance(value, XTensorVariable):
            missing_dims = [dim for dim in core_dims if dim not in value.dims]
            if missing_dims:
                raise ValueError(f"{name} must have dims {core_dims}, got {value.dims}")
            foreign_dims = [
                dim
                for dim in value.dims
                if dim in RESERVED_DIMS and dim not in (TIME_DIM, *core_dims)
            ]
            if foreign_dims:
                raise ValueError(
                    f"{name} has the reserved dims {foreign_dims}, which are core dims of other "
                    f"matrices and so cannot be batch dims"
                )
            # Name a copy, never the caller's own variable: model parameters are looked up by name.
            matrix = value.copy(name=name)
        else:
            if isinstance(value, np.ndarray | int | float):
                tensor = pt.as_tensor_variable(np.asarray(value, dtype=floatX))
            else:
                tensor = pt.as_tensor_variable(value)

            if tensor.ndim == len(core_dims):
                dims = core_dims
            elif tensor.ndim == len(core_dims) + 1:
                dims = (TIME_DIM, *core_dims)
            else:
                raise ValueError(
                    f"{name} must have the {len(core_dims)} core dims {core_dims}, optionally "
                    f"after a leading time dim, but got ndim={tensor.ndim}. Assign an xtensor to "
                    f"add batch dims."
                )
            matrix = as_xtensor(tensor, dims=dims, name=name)

        sizes = dict(zip(matrix.dims, matrix.type.shape, strict=True))
        for dim, want in zip(core_dims, core_shape, strict=True):
            # A None size is only known at runtime. Let it fail there if wrong.
            if sizes[dim] is not None and sizes[dim] != want:
                raise ValueError(
                    f"Core dims of {name} have sizes {[sizes[dim] for dim in core_dims]}, "
                    f"expected {core_shape}"
                )

        return matrix

    @property
    def time_varying_names(self) -> frozenset[str]:
        """Names of the matrices that have a ``time`` dim."""
        return frozenset(name for name in CORE_DIMS if TIME_DIM in getattr(self, name).dims)

    def batch_dims(self, name: str) -> tuple[str, ...]:
        """Return the dims of matrix ``name`` that are not reserved."""
        return tuple(dim for dim in self[name].dims if dim not in RESERVED_DIMS)

    def to_tensor(self, name: str) -> pt.TensorVariable:
        """Return matrix ``name`` as a differentiable tensor laid out as ``(time?, *core)``."""
        batch_dims = self.batch_dims(name)
        if batch_dims:
            raise NotImplementedError(
                f"{name} has batch dims {batch_dims}, but batch dims on statespace matrices "
                f"are not supported yet"
            )

        matrix = self[name]
        time_dims = (TIME_DIM,) if TIME_DIM in matrix.dims else ()
        tensor = matrix.transpose(*time_dims, *CORE_DIMS[name]).values
        # Lowering can reduce the graph to a model parameter itself, so name a copy.
        return rewrite_graph(tensor, include=("lower_xtensor",)).copy(name=name)

    def __getitem__(self, key: KeyLike) -> XTensorVariable:
        if isinstance(key, str):
            self._validate_name(key)
            return getattr(self, key)

        if isinstance(key, tuple) and isinstance(key[0], str):
            name, *idx = key
            self._validate_name(name)
            matrix = getattr(self, name)
            if not idx:
                return matrix
            return matrix[pointwise_index(tuple(idx))]

        raise IndexError("First index must the name of a valid state space matrix.")

    def __setitem__(
        self, key: KeyLike, value: float | int | np.ndarray | pt.TensorVariable | XTensorVariable
    ) -> None:
        if isinstance(key, str):
            self._validate_name(key)
            setattr(self, key, self._coerce(key, value))
            return

        if isinstance(key, tuple) and isinstance(key[0], str):
            name, *idx = key
            self._validate_name(name)
            existing = getattr(self, name)
            target = existing[pointwise_index(tuple(idx))]
            if not isinstance(value, XTensorVariable):
                # Numpy broadcasting aligns a value with the trailing dims of its target.
                value = pt.as_tensor_variable(value)
                value = as_xtensor(value, dims=target.dims[target.ndim - value.ndim :])
            updated = target.set(value)
            updated.name = existing.name
            setattr(self, name, updated)
            return

        raise IndexError("First index must the name of a valid state space matrix.")
