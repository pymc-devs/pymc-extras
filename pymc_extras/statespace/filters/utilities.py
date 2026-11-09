import pytensor
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


def missing_mask(y, missing_fill_value):
    """Boolean mask that is True where ``y`` is ``NaN`` or equals ``missing_fill_value``."""
    return pt.or_(pt.isnan(y), pt.eq(y, missing_fill_value))


def mask_missing_values(y, Z, H, d, missing_fill_value):
    """
    Zero the observation rows that are missing, so they contribute nothing downstream.

    With ``y``, ``Z @ a`` and ``d`` all zero on a missing row, the innovation
    :math:`v = y - (Z a + d)` is exactly zero there.

    Parameters
    ----------
    y : TensorVariable
        Observation at a single timestep.
    Z : TensorVariable
        Design matrix.
    H : TensorVariable
        Observation noise covariance.
    d : TensorVariable
        Observation intercept.
    missing_fill_value : float
        Sentinel marking a missing observation, alongside ``nan``.

    Returns
    -------
    y_masked : TensorVariable
        Observation with missing entries replaced by zero.
    Z_masked : TensorVariable
        Design matrix with missing rows zeroed.
    H_masked : TensorVariable
        Observation noise covariance with missing rows and columns zeroed, so it stays symmetric.
    d_masked : TensorVariable
        Observation intercept with missing entries zeroed.
    nan_mask : TensorVariable
        Boolean vector, True where the observation is missing.
    """
    nan_mask = missing_mask(y, missing_fill_value)
    W = pt.diag(pt.bitwise_not(nan_mask).astype(pytensor.config.floatX))

    return (
        pt.set_subtensor(y[nan_mask], 0.0),
        W.dot(Z),
        W.dot(H).dot(W.mT),
        W.dot(d),
        nan_mask,
    )
