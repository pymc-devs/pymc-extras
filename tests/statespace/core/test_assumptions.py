import numpy as np
import pytensor.tensor as pt
import pytest

from pytensor.graph.replace import graph_replace

from pymc_extras.statespace.core.assumptions import declare_time_varying, is_time_varying

Z = declare_time_varying(pt.tensor("Z", shape=(100, 2, 3)))
# Square, so it can go through the Blockwise linalg ops the models and the filter use.
T = declare_time_varying(pt.tensor("T", shape=(100, 3, 3)))
# Declared on a core axis rather than a batch one, which a Blockwise may reshape away.
T_core = declare_time_varying(pt.tensor("T_core", shape=(3, 3)))


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (Z, True),
        (Z * 2, True),
        (Z[5:], True),
        (Z[5:20], True),
        (pt.specify_shape(Z, (100, 2, 3)), True),
        (pt.concatenate([Z[:5], Z[5:]], axis=0), True),
        (pt.concatenate([Z, Z], axis=1), True),
        (Z[:, :, np.array([2, 0, 1])], True),
        (pt.set_subtensor(Z[0, 0, 0], 1.0), True),
        (pt.set_subtensor(Z[:, 0, 0], pt.vector("v", shape=(100,))), True),
        (pt.set_subtensor(pt.tensor("S", shape=(2, 3))[0], Z[0, 0]), False),
        (Z[0], False),
        (Z[0][None], False),
        (Z[np.array([2, 0, 1])], False),
        (Z.sum(axis=0), False),
        (pt.zeros((100, 2, 3)) + Z.sum(), False),
        (pt.tensor("W", shape=(100, 2, 3)), False),
        (pt.linalg.block_diag(T, pt.tensor("S", shape=(2, 2))), True),
        (pt.linalg.block_diag(pt.tensor("S", shape=(2, 2)), T), True),
        (pt.linalg.cholesky(T), True),
        (pt.linalg.solve(T, T, b_ndim=2), True),
        (pt.linalg.block_diag(T_core, pt.tensor("S", shape=(2, 2))), False),
        (
            pt.linalg.block_diag(pt.tensor("S", shape=(2, 2)), pt.tensor("S2", shape=(3, 3))),
            False,
        ),
    ],
    ids=[
        "declared",
        "elemwise",
        "open_slice",
        "closed_slice",
        "specify_shape",
        "join_on_time",
        "join_off_time",
        "reorder_off_time",
        "write_one_element",
        "write_a_column",
        "write_a_time_varying_value_into_a_static_base",
        "index_time",
        "index_then_expand",
        "gather_time",
        "reduce_time",
        "static_from_a_reduction",
        "undeclared",
        "block_diag_time_varying_first",
        "block_diag_time_varying_second",
        "blockwise_cholesky",
        "blockwise_solve",
        "block_diag_declared_on_a_core_axis",
        "block_diag_of_static_matrices",
    ],
)
def test_declaration_survives_exactly_the_ops_that_keep_the_time_axis(expression, expected):
    assert is_time_varying(expression) == [expected]


def test_matrices_are_reported_in_the_order_they_are_given():
    static = pt.matrix("static")
    assert is_time_varying(static, Z, static, Z) == [False, True, False, True]


def test_declaration_survives_substitution():
    """Models declare on a placeholder and swap in the real parameter with ``graph_replace``."""
    placeholder = pt.tensor("Z", shape=(None, 2, 3))
    declared = declare_time_varying(placeholder)

    substituted = graph_replace(declared, {placeholder: pt.tensor("Z_real", shape=(100, 2, 3))})

    assert is_time_varying(substituted) == [True]
