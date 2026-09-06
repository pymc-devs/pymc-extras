import pytensor.tensor as pt

from pytensor.graph.basic import Variable

from pymc_extras.statespace.core.assumptions import is_time_varying
from pymc_extras.statespace.utils.constants import JITTER_DEFAULT, MATRIX_NAMES

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


def split_by_time_axis(
    matrices: dict[str, Variable],
) -> tuple[list[Variable], list[Variable], list[str], list[str]]:
    """
    Split matrices into ``scan`` sequences and non-sequences by asking each for its time axis.

    Parameters
    ----------
    matrices : dict mapping str to TensorVariable
        Matrices to split, keyed by short name.

    Returns
    -------
    sequences : list of TensorVariable
        Matrices carrying a time axis, in the order ``matrices`` gives them.
    non_sequences : list of TensorVariable
        The rest, in the same order.
    seq_names : list of str
        Names matching ``sequences``.
    non_seq_names : list of str
        Names matching ``non_sequences``.
    """
    varying = is_time_varying(*matrices.values())
    seq_names = [name for name, flag in zip(matrices, varying, strict=True) if flag]
    non_seq_names = [name for name, flag in zip(matrices, varying, strict=True) if not flag]

    return (
        [matrices[name] for name in seq_names],
        [matrices[name] for name in non_seq_names],
        seq_names,
        non_seq_names,
    )


def unpack_scan_step(
    args: tuple, seq_names: list[str], non_seq_names: list[str], order: tuple[str, ...]
) -> tuple[tuple, tuple]:
    """
    Split one ``scan`` step's arguments into its recurrent states and its matrices.

    ``scan`` passes sequences first, then ``outputs_info``, then non-sequences, so where a given
    matrix lands depends on how many of them carry a time axis.

    Parameters
    ----------
    args : tuple
        The step function's arguments, less any leading sequences the caller took for itself.
    seq_names : list of str
        Names of the matrices passed as sequences, in the order ``scan`` receives them.
    non_seq_names : list of str
        Names of the matrices passed as non-sequences, in the order ``scan`` receives them.
    order : tuple of str
        Names of the matrices to return, in the order the caller wants them.

    Returns
    -------
    recurrent : tuple
        The ``outputs_info`` values, which sit between the sequences and the non-sequences.
    matrices : tuple
        The matrices named by ``order``.
    """
    n_seq, n_non_seq = len(seq_names), len(non_seq_names)
    n_recurrent = len(args) - n_seq - n_non_seq

    seqs = args[:n_seq]
    recurrent = args[n_seq : n_seq + n_recurrent]
    non_seqs = args[n_seq + n_recurrent :]

    matrices = dict(zip(seq_names, seqs, strict=True)) | dict(
        zip(non_seq_names, non_seqs, strict=True)
    )
    return recurrent, tuple(matrices[name] for name in order)


def stabilize(cov, jitter=JITTER_DEFAULT):
    cov = cov + pt.identity_like(cov) * jitter

    return cov


def quad_form_sym(A, B):
    out = A @ B @ A.mT
    return 0.5 * (out + out.mT)
