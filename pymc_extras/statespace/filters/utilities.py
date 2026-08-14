from collections.abc import Iterable

import pytensor.tensor as pt

from pymc_extras.statespace.utils.constants import (
    JITTER_DEFAULT,
    LONG_NAME_TO_SHORT,
    MATRIX_NAMES,
)

# The filter's scan takes x0 and P0 as outputs_info rather than inputs, so they are not
# among the matrices split into sequences and non-sequences.
PARAM_NAMES = MATRIX_NAMES[2:]


def dim_of(tensor, axis: int):
    """
    Return a tensor's length along ``axis``.

    Parameters
    ----------
    tensor : TensorVariable
        Tensor to measure.
    axis : int
        Axis to measure, which may be negative.

    Returns
    -------
    length : int or TensorVariable
        A Python int when the length is known statically, so downstream shapes stay static, and a
        symbolic scalar otherwise.
    """
    static_length = tensor.type.shape[axis]
    return static_length if static_length is not None else tensor.shape[axis]


def scan_sequence_names(time_varying_names: Iterable[str]) -> list[str]:
    """
    Return the names of the matrices the Kalman filter passes to ``scan`` as sequences.

    Parameters
    ----------
    time_varying_names : iterable of str
        Long names of the matrices the model declared time-varying, as given by
        :attr:`PytensorRepresentation.time_varying_names`.

    Returns
    -------
    seq_names : list of str
        Short matrix names, ordered as ``scan`` receives them.
    """
    time_varying_short = {LONG_NAME_TO_SHORT[name] for name in time_varying_names}
    return [name for name in PARAM_NAMES if name in time_varying_short]


def split_vars_into_seq_and_nonseq(params, param_names, time_varying_names: Iterable[str]):
    """Split filter inputs into scan sequences and non-sequences.

    Parameters
    ----------
    params : sequence of TensorVariable
        Filter input matrices in the order given by ``param_names``.
    param_names : sequence of str
        Long names of the matrices in ``params``.
    time_varying_names : iterable of str
        Names of matrices the model declared as time-varying. Anything in this set is
        treated as a scan sequence; everything else is a non-sequence.

    Returns
    -------
    sequences, non_sequences, seq_names, non_seq_names : four lists
        The split inputs and their names.
    """
    time_varying_names = frozenset(time_varying_names)
    sequences, non_sequences = [], []
    seq_names, non_seq_names = [], []

    for param, name in zip(params, param_names):
        if name in time_varying_names:
            sequences.append(param)
            seq_names.append(name)
        else:
            non_sequences.append(param)
            non_seq_names.append(name)

    return sequences, non_sequences, seq_names, non_seq_names


def stabilize(cov, jitter=JITTER_DEFAULT):
    cov = cov + pt.identity_like(cov) * jitter

    return cov


def quad_form_sym(A, B):
    out = A @ B @ A.mT
    return 0.5 * (out + out.mT)
