import pytensor.tensor as pt

from pytensor.tensor.nlinalg import matrix_dot

from pymc_extras.statespace.utils.constants import (
    JITTER_DEFAULT,
    NEVER_TIME_VARYING,
    VECTOR_VALUED,
)

CORE_NDIM = (2, 1, 2, 1, 1, 2, 2, 2, 2, 2)
SMOOTHER_CORE_NDIM = (2, 2, 2, 2, 3)


def decide_if_x_time_varies(x, name):
    if name in NEVER_TIME_VARYING:
        return False

    ndim = x.ndim

    if name in VECTOR_VALUED:
        if ndim not in [1, 2]:
            raise ValueError(
                f"Vector {name} has {ndim} dimensions; it should have either 1 (static),"
                f" or 2 (time varying )"
            )

        return ndim == 2

    if ndim not in [2, 3]:
        raise ValueError(
            f"Matrix {name} has {ndim} dimensions; it should have either"
            f" 2 (static), or 3 (time varying)."
        )

    return ndim == 3


def split_vars_into_seq_and_nonseq(params, param_names):
    """
    Split inputs into those that are time varying and those that are not. This division is required by scan.
    """
    sequences, non_sequences = [], []
    seq_names, non_seq_names = [], []

    for param, name in zip(params, param_names):
        if decide_if_x_time_varies(param, name):
            sequences.append(param)
            seq_names.append(name)
        else:
            non_sequences.append(param)
            non_seq_names.append(name)

    return sequences, non_sequences, seq_names, non_seq_names


def stabilize(cov, jitter=JITTER_DEFAULT):
    # Ensure diagonal is non-zero
    cov = cov + pt.identity_like(cov) * jitter

    return cov


def quad_form_sym(A, B):
    out = matrix_dot(A, B, A.T)
    return 0.5 * (out + out.T)


def has_batched_input_smoother(T, R, Q, filtered_states, filtered_covariances):
    """
    Check if any of the inputs are batched.
    """
    return any(
        x.ndim > SMOOTHER_CORE_NDIM[i]
        for i, x in enumerate([T, R, Q, filtered_states, filtered_covariances])
    )


def get_dummy_core_inputs_smoother(T, R, Q, filtered_states, filtered_covariances):
    """
    Get dummy inputs for the core parameters.
    """
    out = []
    for x, core_ndim in zip([T, R, Q, filtered_states, filtered_covariances], SMOOTHER_CORE_NDIM):
        out.append(pt.tensor(f"{x.name}_core_case", dtype=x.dtype, shape=x.type.shape[-core_ndim:]))
    return out


def has_batched_input_filter(data, a0, P0, c, d, T, Z, R, H, Q):
    """
    Check if any of the inputs are batched.
    """
    return any(x.ndim > CORE_NDIM[i] for i, x in enumerate([data, a0, P0, c, d, T, Z, R, H, Q]))


def get_dummy_core_inputs_filter(data, a0, P0, c, d, T, Z, R, H, Q):
    """
    Get dummy inputs for the core parameters.
    """
    out = []
    for x, core_ndim in zip([data, a0, P0, c, d, T, Z, R, H, Q], CORE_NDIM):
        out.append(pt.tensor(f"{x.name}_core_case", dtype=x.dtype, shape=x.type.shape[-core_ndim:]))
    return out
