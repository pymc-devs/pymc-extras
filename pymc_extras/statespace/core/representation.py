import copy

import numpy as np
import pytensor
import pytensor.tensor as pt

from pymc_extras.statespace.utils.constants import (
    NEVER_TIME_VARYING,
    VECTOR_VALUED,
)

floatX = pytensor.config.floatX
KeyLike = tuple[str | int, ...] | str


class PytensorRepresentation:
    r"""
    Core class to hold all objects required by linear gaussian statespace models

    Notation for the linear statespace model is taken from [1], while the specific implementation is adapted from
    the statsmodels implementation: https://github.com/statsmodels/statsmodels/blob/main/statsmodels/tsa/statespace/representation.py
    described in [2].

    Parameters
    ----------
    k_endog: int
        Number of observed states (called "endogeous states" in statsmodels)
    k_states: int
        Number of hidden states
    k_posdef: int
        Number of states that have exogenous shocks; also the rank of the selection matrix R.
    design: ArrayLike, optional
        Design matrix, denoted 'Z' in [1].
    obs_intercept: ArrayLike, optional
        Constant vector in the observation equation, denoted 'd' in [1]. Currently
        not used.
    obs_cov: ArrayLike, optional
        Covariance matrix for multivariate-normal errors in the observation equation. Denoted 'H' in
        [1].
    transition: ArrayLike, optional
        Transition equation that updates the hidden state between time-steps. Denoted 'T' in [1].
    state_intercept: ArrayLike, optional
        Constant vector for the observation equation, denoted 'c' in [1]. Currently not used.
    selection: ArrayLike, optional
        Selection matrix that matches shocks to hidden states, denoted 'R' in [1]. This is the identity
        matrix when k_posdef = k_states.
    state_cov: ArrayLike, optional
        Covariance matrix for state equations, denoted 'Q' in [1]. Null matrix when there is no observation
        noise.
    initial_state: ArrayLike, optional
        Experimental setting to allow for Bayesian estimation of the initial state, denoted `alpha_0` in [1]. Default
        It should potentially be removed in favor of the closed-form diffuse initialization.
    initial_state_cov: ArrayLike, optional
        Experimental setting to allow for Bayesian estimation of the initial state, denoted `P_0` in [1]. Default
        It should potentially be removed in favor of the closed-form diffuse initialization.

    Notes
    -----
    A linear statespace system is defined by two equations:

    .. math::
        \begin{align}
            x_t &= A_t x_{t-1} + c_t + R_t \varepsilon_t \tag{1} \\
            y_t &= Z_t x_t + d_t + \eta_t \tag{2} \\
        \end{align}

    Where :math:`\{x_t\}_{t=0}^T` is a trajectory of hidden states, and :math:`\{y_t\}_{t=0}^T` is a trajectory of
    observable states. Equation 1 is known as the "state transition equation", while describes how the system evolves
    over time. Equation 2 is the "observation equation", and maps the latent state processes to observed data.
    The system is Gaussian when the innovations, :math:`\varepsilon_t`, and the measurement errors, :math:`\eta_t`,
    are normally distributed. The definition is completed by specification of these distributions, as
    well as an initial state distribution:

    .. math::
        \begin{align}
            \varepsilon_t &\sim N(0, Q_t) \tag{3} \\
            \eta_t &\sim N(0, H_t) \tag{4} \\
            x_0 &\sim N(\bar{x}_0, P_0) \tag{5}
        \end{align}

    The 9 matrices that form equations 1 to 5 are summarized in the table below. We call :math:`N` the number of
    observations, :math:`m` the number of hidden states, :math:`p` the number of observed states, and :math:`r` the
    number of innovations.

    +-----------------------------------+-------------------+-----------------------+
    | Name                              | Symbol            | Shape                 |
    +===================================+===================+=======================+
    | Initial hidden state mean         | :math:`x_0`       | :math:`m \times 1`    |
    +-----------------------------------+-------------------+-----------------------+
    | Initial hidden state covariance   | :math:`P_0`       | :math:`m \times m`    |
    +-----------------------------------+-------------------+-----------------------+
    | Hidden state vector intercept     | :math:`c_t`       | :math:`m \times 1`    |
    +-----------------------------------+-------------------+-----------------------+
    | Observed state vector intercept   | :math:`d_t`       | :math:`p \times 1`    |
    +-----------------------------------+-------------------+-----------------------+
    | Transition matrix                 | :math:`T_t`       | :math:`m \times m`    |
    +-----------------------------------+-------------------+-----------------------+
    | Design matrix                     | :math:`Z_t`       | :math:`p \times m`    |
    +-----------------------------------+-------------------+-----------------------+
    | Selection matrix                  | :math:`R_t`       | :math:`m \times r`    |
    +-----------------------------------+-------------------+-----------------------+
    | Observation noise covariance      | :math:`H_t`       | :math:`p \times p`    |
    +-----------------------------------+-------------------+-----------------------+
    | Hidden state innovation covariance| :math:`Q_t`       | :math:`r \times r`    |
    +-----------------------------------+-------------------+-----------------------+

    The shapes listed above are the core shapes, but in the general case all of these matrices (except for :math:`x_0`
    and :math:`P_0`) can be time varying. In this case, a time dimension of shape :math:`n`, equal to the number of
    observations, can be added.

    .. warning:: The time dimension is used as a batch dimension during kalman filtering, and must thus **always**
                 be the **leftmost** dimension.

    The purpose of this class is to store these matrices, as well as to allow users to easily index into them. Matrices
    are stored as pytensor ``TensorVariables`` of known shape. Shapes are always accessible via the ``.type.shape``
    method, which should never return ``None``. Matrices can be accessed via normal numpy array slicing after first
    indexing by the name of the desired array. The time dimension is stored on the far left, and is automatically
    sliced away unless specifically requested by the user. See the examples for details.

    Examples
    --------
    .. code:: python

        from pymc_extras.statespace.core.representation import PytensorRepresentation
        ssm = PytensorRepresentation(k_endog=1, k_states=3, k_posdef=1)

        # Access matrices by their names
        print(ssm['transition'].type.shape)
        >>> (3, 3)

        # Slice a matrices
        print(ssm['observation_cov', 0, 0].eval())
        >>> 0.0

        # Set elements in a slice of a matrix
        ssm['design', 0, 0] = 1
        print(ssm['design'].eval())
        >>> np.array([[1, 0, 0]])

        # Setting an entire matrix is also permitted. If you set a time dimension, it must be the first dimension, and
        # the "core" dimensions must agree with those set when the ssm object was instantiated.
        ssm['obs_intercept'] = np.arange(10).reshape(10, 1) # 10 timesteps
        print(ssm['obs_intercept'].eval())
        >>> np.array([[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.]])

    References
    ----------
    .. [1] Durbin, James, and Siem Jan Koopman. 2012.
        Time Series Analysis by State Space Methods: Second Edition.
        Oxford University Press.
    .. [2] Fulton, Chad. "Estimating time series models by state space methods in Python: Statsmodels." (2015).
           http://www.chadfulton.com/files/fulton_statsmodels_2017_v1.pdf
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
        "shapes",
        "state_cov",
        "state_intercept",
        "transition",
    )

    def __init__(
        self,
        k_endog: int,
        k_states: int,
        k_posdef: int,
        design: np.ndarray | None = None,
        obs_intercept: np.ndarray | None = None,
        obs_cov=None,
        transition=None,
        state_intercept=None,
        selection=None,
        state_cov=None,
        initial_state=None,
        initial_state_cov=None,
    ) -> None:
        self.k_states = k_states
        self.k_endog = k_endog
        self.k_posdef = k_posdef if k_posdef is not None else k_states

        # The first dimension is for time varying matrices; it could be n_obs. Not thinking about that now.
        self.shapes = {
            "design": (1, self.k_endog, self.k_states),
            "obs_intercept": (1, self.k_endog),
            "obs_cov": (1, self.k_endog, self.k_endog),
            "transition": (1, self.k_states, self.k_states),
            "state_intercept": (1, self.k_states),
            "selection": (1, self.k_states, self.k_posdef),
            "state_cov": (1, self.k_posdef, self.k_posdef),
            # These are never time varying, so they don't have a dummy first dimension
            "initial_state": (self.k_states,),
            "initial_state_cov": (self.k_states, self.k_states),
        }

        # Initialize the representation matrices
        scope = locals()
        for name, shape in self.shapes.items():
            if scope[name] is not None:
                matrix = scope[name]
                if isinstance(matrix, np.ndarray):
                    matrix = self._numpy_to_pytensor(name, matrix)
                else:
                    matrix = self._check_provided_tensor(name, matrix)
                setattr(self, name, matrix)

            else:
                matrix = pt.as_tensor_variable(
                    np.zeros(shape, dtype=floatX), name=name, ndim=len(shape)
                )
                setattr(self, name, matrix)

    def _validate_key(self, key: KeyLike) -> None:
        if key not in self.shapes:
            raise IndexError(f"{key} is an invalid state space matrix name")

    def _update_shape(self, key: KeyLike, value: np.ndarray | pt.Variable) -> None:
        if isinstance(value, pt.TensorConstant | pt.TensorVariable):
            shape = value.type.shape
        else:
            shape = value.shape

        old_shape = self.shapes[key]
        ndim_core = 1 if key in VECTOR_VALUED else 2
        if not all([a == b for a, b in zip(shape[-ndim_core:], old_shape[-ndim_core:])]):
            raise ValueError(
                f"The last two dimensions of {key} must be {old_shape[-ndim_core:]}, found {shape[-ndim_core:]}"
            )

        # Add time dimension dummy if none present
        if key not in NEVER_TIME_VARYING:
            if len(shape) == 2 and key not in VECTOR_VALUED:
                shape = (1, *shape)
            elif len(shape) == 1:
                shape = (1, *shape)

        self.shapes[key] = shape

    def _add_time_dim_to_slice(
        self, name: str, slice_: list[int] | tuple[int], n_dim: int
    ) -> tuple[int | slice, ...]:
        # Case 1: There is never a time dim. No changes needed.
        if name in NEVER_TIME_VARYING:
            return slice_

        # Case 2: The matrix has a time dim, and it was requested. No changes needed.
        if len(slice_) == n_dim:
            return slice_

        # Case 3: There's no time dim on the matrix, and none requested. Slice away the dummy dim.
        if len(slice_) < n_dim:
            empty_slice = (slice(None, None, None),)
            n_omitted = n_dim - len(slice_) - 1
            return (0,) + tuple(slice_) + empty_slice * n_omitted

    @staticmethod
    def _validate_key_and_get_type(key: KeyLike) -> type[str]:
        if isinstance(key, tuple) and not isinstance(key[0], str):
            raise IndexError("First index must the name of a valid state space matrix.")

        return type(key)

    def _validate_matrix_shape(self, name: str, X: np.ndarray | pt.TensorVariable) -> None:
        time_dim, *expected_shape = self.shapes[name]
        expected_shape = tuple(expected_shape)
        shape = X.shape if isinstance(X, np.ndarray) else X.type.shape

        is_vector = name in VECTOR_VALUED
        not_time_varying = name in NEVER_TIME_VARYING

        if not_time_varying:
            if is_vector:
                if X.ndim != 1:
                    raise ValueError(
                        f"Array provided for {name} has {X.ndim} dimensions, but it must have exactly 1."
                    )

            else:
                if X.ndim != 2:
                    raise ValueError(
                        f"Array provided for {name} has {X.ndim} dimensions, but it must have exactly 2."
                    )

        else:
            if is_vector:
                if X.ndim not in [1, 2]:
                    raise ValueError(
                        f"Array provided for {name} has {X.ndim} dimensions, "
                        f"expecting 1 (static) or 2 (time-varying)"
                    )

                # Time varying vector case, check only the static shapes
                if X.ndim == 2 and X.shape[1:] != expected_shape:
                    raise ValueError(
                        f"Last dimension of array provided for {name} has shape {X.shape[1]}, "
                        f"expected {expected_shape}"
                    )

            else:
                if X.ndim not in [2, 3]:
                    raise ValueError(
                        f"Array provided for {name} has {X.ndim} dimensions, "
                        f"expecting 2 (static) or 3 (time-varying)"
                    )

                # Time varying matrix case, check only the static shapes
                if X.ndim == 3 and shape[1:] != expected_shape:
                    raise ValueError(
                        f"Last two dimensions of array provided for {name} have shapes {X.shape[1:]}, "
                        f"expected {expected_shape}"
                    )

            # TODO: Think of another way to validate shapes of time-varying matrices if we don't know the data
            #   when the PytensorRepresentation is recreated
            # if X.shape[-1] != self.data.shape[0]:
            #     raise ValueError(
            #         f"Last dimension (time dimension) of array provided for {name} has shape "
            #         f"{X.shape[-1]}, expected {self.data.shape[0]} (equal to the first dimension of the "
            #         f"provided data)"
            #     )

    def _check_provided_tensor(self, name: str, X: pt.TensorVariable) -> pt.TensorVariable:
        self._validate_matrix_shape(name, X)
        if name not in NEVER_TIME_VARYING:
            if X.ndim == 1 and name in VECTOR_VALUED:
                X = pt.expand_dims(X, (0,))
                X = pt.specify_shape(X, self.shapes[name])

            elif X.ndim == 2:
                X = pt.expand_dims(X, (0,))
                X = pt.specify_shape(X, self.shapes[name])

        return X

    def _numpy_to_pytensor(self, name: str, X: np.ndarray) -> pt.TensorVariable:
        X = X.copy()
        self._validate_matrix_shape(name, X)

        # Add a time dimension if one isn't provided
        if name not in NEVER_TIME_VARYING:
            if X.ndim == 1 and name in VECTOR_VALUED:
                X = X[None, ...]
            elif X.ndim == 2 and name not in VECTOR_VALUED:
                X = X[None, ...]

        X_pt = pt.as_tensor(X, name=name, dtype=floatX)
        return X_pt

    def __getitem__(self, key: KeyLike) -> pt.TensorVariable:
        _type = self._validate_key_and_get_type(key)

        # Case 1: user asked for an entire matrix by name
        if _type is str:
            self._validate_key(key)
            matrix = getattr(self, key)

            # Slice away the time dimension if it's a dummy
            if (matrix.type.shape[0] == 1) and (key not in NEVER_TIME_VARYING):
                X = matrix[(0,) + (slice(None),) * (matrix.ndim - 1)]
                X = pt.specify_shape(X, self.shapes[key][1:])
                X.name = key

                return X

            # If it's never time varying, return everything
            elif key in NEVER_TIME_VARYING:
                return matrix

            # Last possibility is that it's time varying -- also return everything (for now, might need some processing)
            else:
                return matrix

        # Case 2: user asked for a particular matrix and some slices of it
        elif _type is tuple:
            name, *slice_ = key
            slice_ = tuple(slice_)
            self._validate_key(name)

            matrix = getattr(self, name)
            # Case 2a: The user asked for the whole matrix, with time dummies. Return the whole thing
            # without slicing anything away
            if slice_ == (slice(None, None, None),) * matrix.ndim:
                return matrix

            # Case 2b: The user asked for the whole matrix except time dummies. Ignore the slice and act like we're in
            # case 1.
            elif slice_ == (slice(None, None, None),) * (matrix.ndim - 1):
                X = matrix[(0,) + (slice(None),) * (matrix.ndim - 1)]
                X = pt.specify_shape(X, self.shapes[name][1:])
                X.name = name
                return X

            # Case 3b: User asked for an arbitrary sub-matrix. Give it back -- nothing else to be done
            slice_ = self._add_time_dim_to_slice(name, slice_, matrix.ndim)
            return matrix[slice_]

        # Case 3: There is only one slice index, but it's not a string
        else:
            raise IndexError("First index must the name of a valid state space matrix.")

    def __setitem__(self, key: KeyLike, value: float | int | np.ndarray | pt.Variable) -> None:
        _type = type(key)

        # Case 1: key is a string: we are setting an entire matrix.
        if _type is str:
            self._validate_key(key)
            if isinstance(value, np.ndarray):
                value = self._numpy_to_pytensor(key, value)
            else:
                value.name = key

            setattr(self, key, value)
            self._update_shape(key, value)

        # Case 2: key is a string plus a slice: we are setting a subset of a matrix
        elif _type is tuple:
            name, *slice_ = key
            self._validate_key(name)

            matrix = getattr(self, name)

            slice_ = self._add_time_dim_to_slice(name, slice_, matrix.ndim)
            matrix = pt.set_subtensor(matrix[slice_], value)
            matrix = pt.specify_shape(matrix, self.shapes[name])
            matrix.name = name

            setattr(self, name, matrix)

    def copy(self):
        return copy.copy(self)
