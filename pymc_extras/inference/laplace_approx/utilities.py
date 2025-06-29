import logging

from typing import Literal, get_args

import numpy as np

from better_optimize.constants import MINIMIZE_MODE_KWARGS

_log = logging.getLogger(__name__)
GradientBackend = Literal["pytensor", "jax"]
VALID_BACKENDS = get_args(GradientBackend)


def set_optimizer_function_defaults(method, use_grad, use_hess, use_hessp):
    method_info = MINIMIZE_MODE_KWARGS[method].copy()

    if use_hess and use_hessp:
        _log.warning(
            'Both "use_hess" and "use_hessp" are set to True, but scipy.optimize.minimize never uses both at the '
            'same time. When possible "use_hessp" is preferred because its is computationally more efficient. '
            'Setting "use_hess" to False.'
        )
        use_hess = False

    use_grad = use_grad if use_grad is not None else method_info["uses_grad"]

    if use_hessp is not None and use_hess is None:
        use_hess = not use_hessp

    elif use_hess is not None and use_hessp is None:
        use_hessp = not use_hess

    elif use_hessp is None and use_hess is None:
        use_hessp = method_info["uses_hessp"]
        use_hess = method_info["uses_hess"]
        if use_hessp and use_hess:
            # If a method could use either hess or hessp, we default to using hessp
            use_hess = False

    return use_grad, use_hess, use_hessp


def get_nearest_psd(A: np.ndarray) -> np.ndarray:
    """
    Compute the nearest positive semi-definite matrix to a given matrix.

    This function takes a square matrix and returns the nearest positive semi-definite matrix using
    eigenvalue decomposition. It ensures all eigenvalues are non-negative. The "nearest" matrix is defined in terms
    of the Frobenius norm.

    Parameters
    ----------
    A : np.ndarray
        Input square matrix.

    Returns
    -------
    np.ndarray
        The nearest positive semi-definite matrix to the input matrix.
    """
    C = (A + A.T) / 2
    eigval, eigvec = np.linalg.eigh(C)
    eigval[eigval < 0] = 0

    return eigvec @ np.diag(eigval) @ eigvec.T
