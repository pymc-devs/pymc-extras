from collections.abc import Iterable

import pytensor.tensor as pt

from pymc_extras.statespace.utils.constants import JITTER_DEFAULT


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
