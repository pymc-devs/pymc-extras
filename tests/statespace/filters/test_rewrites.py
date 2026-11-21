import operator

from functools import reduce

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose
from pytensor.compile.mode import get_default_mode
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.rewriting.utils import rewrite_graph
from pytensor.graph.traversal import ancestors
from pytensor.tensor.blas import BatchedDot
from pytensor.tensor.math import Dot, _matmul
from pytensor.tensor.random.basic import MvNormalRV
from pytensor.tensor.shape import Reshape

import pymc_extras.statespace.filters  # registers local_mvnormal_zero_cov

from pymc_extras.statespace.filters import StandardFilter
from pymc_extras.statespace.filters.batched import STEP_SIGNATURE
from pymc_extras.statespace.filters.rewrites import local_lift_batched_scale_out_of_matmul
from tests.statespace.shared_fixtures import (  # pylint: disable=unused-import
    rng,
)


def _has_mvnormal(draw) -> bool:
    canonical = rewrite_graph(draw, include=("canonicalize",))
    return any(
        var.owner is not None and isinstance(var.owner.op, MvNormalRV)
        for var in ancestors([canonical])
    )


@pytest.mark.parametrize("size", [None, (3,)], ids=["no_size", "size"])
def test_zero_cov_mvnormal_is_replaced_by_its_mean(size):
    mean = np.array([1.0, 2.0])
    draw = pm.MvNormal.dist(mu=mean, cov=np.zeros((2, 2)), method="cholesky", size=size)

    assert not _has_mvnormal(draw)
    assert_allclose(pytensor.function([], draw)(), np.broadcast_to(mean, (*(size or ()), 2)))


def test_nonzero_cov_mvnormal_is_kept():
    draw = pm.MvNormal.dist(mu=np.zeros(2), cov=np.diag([0.0, 1.0]), method="svd")

    assert _has_mvnormal(draw)


def _matmul_node(output):
    return next(
        var.owner
        for var in ancestors([output])
        if var.owner is not None and var.owner.op == _matmul
    )


def _lift_fires(output) -> bool:
    fgraph = FunctionGraph(outputs=[output], clone=False)
    return (
        local_lift_batched_scale_out_of_matmul.transform(fgraph, _matmul_node(output)) is not None
    )


def _symbolic_product(shapes, rng):
    variables = [pt.tensor(shape=shape) for shape in shapes]
    values = [
        rng.normal(size=tuple(5 if dim is None else dim for dim in shape)) for shape in shapes
    ]
    return reduce(operator.mul, variables), variables, reduce(np.multiply, values), values


LIFT_CASES = {
    "row_scale_on_left": ([(5, 3, 1), (1, 3, 6)], [(5, 6, 4)], True),
    "row_scale_with_unbatched_right": ([(5, 3, 1), (1, 3, 6)], [(1, 6, 4)], True),
    "column_scale_on_right": ([(5, 6, 6)], [(1, 6, 3), (5, 1, 3)], True),
    "scales_on_both_sides": ([(5, 3, 1), (1, 3, 6)], [(1, 6, 4), (5, 1, 4)], True),
    "per_batch_scalar": ([(5, 1, 1), (1, 3, 6)], [(5, 6, 4)], True),
    "two_batch_dims_one_shared": ([(4, 1, 3, 1), (1, 5, 3, 6)], [(4, 5, 6, 2)], True),
    "lifted_and_kept_scales": ([(5, 3, 1), (1, 3, 1), (1, 3, 6)], [(5, 6, 4)], True),
    "scale_shares_operand_batch": ([(5, 3, 1), (5, 3, 6)], [(1, 6, 4)], False),
    "unbatched_scale": ([(1, 3, 1), (1, 3, 6)], [(5, 6, 4)], False),
    "scale_on_contracted_axis": ([(5, 1, 6), (1, 3, 6)], [(5, 6, 4)], False),
    "unknown_batch_shape": ([(None, 3, 1), (None, 3, 6)], [(None, 6, 4)], False),
}


@pytest.mark.parametrize(
    "left_shapes, right_shapes, fires", LIFT_CASES.values(), ids=LIFT_CASES.keys()
)
def test_lift_fires_only_when_scale_adds_batch_dim(left_shapes, right_shapes, fires, rng):
    left, left_inputs, left_value, left_values = _symbolic_product(left_shapes, rng)
    right, right_inputs, right_value, right_values = _symbolic_product(right_shapes, rng)
    output = left @ right

    assert _lift_fires(output) == fires
    assert_allclose(
        pytensor.function([*left_inputs, *right_inputs], output)(*left_values, *right_values),
        left_value @ right_value,
    )


def test_lift_applies_to_matvec(rng):
    left, inputs, left_value, values = _symbolic_product([(5, 3, 1), (1, 3, 6)], rng)
    vector = pt.tensor(shape=(5, 6))
    vector_value = rng.normal(size=(5, 6))
    output = pt.matvec(left, vector)

    assert _lift_fires(output)
    assert_allclose(
        pytensor.function([*inputs, vector], output)(*values, vector_value),
        np.einsum("bij,bj->bi", left_value, vector_value),
    )


def test_lift_keeps_float_output_for_bool_mask(rng):
    mask = pt.tensor(shape=(5, 3, 1), dtype="bool")
    design = pt.tensor(shape=(1, 3, 6))
    right = pt.tensor(shape=(5, 6, 4))
    output = (mask * design) @ right
    mask_value = rng.random(size=(5, 3, 1)) > 0.5
    design_value, right_value = rng.normal(size=(1, 3, 6)), rng.normal(size=(5, 6, 4))

    assert _lift_fires(output)
    function = pytensor.function([mask, design, right], output)
    assert function.maker.fgraph.outputs[0].dtype == output.dtype
    assert_allclose(
        function(mask_value, design_value, right_value), (mask_value * design_value) @ right_value
    )


def test_lift_gradient_matches_unrewritten_graph(rng):
    left, left_inputs, _, left_values = _symbolic_product([(5, 3, 1), (1, 3, 6)], rng)
    right, right_inputs, _, right_values = _symbolic_product([(1, 6, 4), (5, 1, 4)], rng)
    inputs, values = [*left_inputs, *right_inputs], [*left_values, *right_values]
    loss = ((left @ right) * np.arange(60.0).reshape(5, 3, 4)).sum()
    gradients = pytensor.grad(loss, inputs)

    without_lift = get_default_mode().excluding("local_lift_batched_scale_out_of_matmul")
    rewritten = pytensor.function(inputs, gradients)(*values)
    reference = pytensor.function(inputs, gradients, mode=without_lift)(*values)
    for gradient, expected in zip(rewritten, reference, strict=True):
        assert_allclose(gradient, expected)


def test_lift_removes_batched_dots_from_batched_filter_step():
    # 50 series, 3 observed states, 6 states, 3 shocks, in STEP_SIGNATURE order
    shapes = [(50, 3), (50, 6), (50, 6, 6), (6,), (3,), (6, 6), (3, 6), (6, 3), (3, 3), (3, 3)]
    inputs = [pt.tensor(shape=shape) for shape in shapes]
    outputs = pt.vectorize(StandardFilter().filter_step, signature=STEP_SIGNATURE)(*inputs)

    def count_batched_dots(mode):
        function = pytensor.function(inputs, outputs, mode=mode)
        return sum(isinstance(node.op, BatchedDot) for node in function.maker.fgraph.apply_nodes)

    without_lift = get_default_mode().excluding("local_lift_batched_scale_out_of_matmul")
    assert count_batched_dots(get_default_mode()) < count_batched_dots(without_lift)


WITHOUT_LIFT = get_default_mode().excluding("local_lift_batched_scale_out_of_matmul")


def _compile_and_compare(inputs, outputs, values):
    with_lift = pytensor.function(inputs, outputs)
    without_lift = pytensor.function(inputs, outputs, mode=WITHOUT_LIFT)
    for lifted, reference in zip(with_lift(*values), without_lift(*values), strict=True):
        assert_allclose(lifted, reference, equal_nan=True)
    return with_lift


def _nodes_of_type(function, op_type):
    return [node for node in function.maker.fgraph.apply_nodes if isinstance(node.op, op_type)]


def test_lift_cascades_down_a_matmul_chain(rng):
    scale, A, B, C = (
        pt.tensor(shape=(5, 3, 1)),
        pt.tensor(shape=(3, 4)),
        pt.tensor(shape=(4, 6)),
        pt.tensor(shape=(6, 2)),
    )
    values = [rng.normal(size=variable.type.shape) for variable in (scale, A, B, C)]

    function = _compile_and_compare([scale, A, B, C], [(scale * A) @ B @ C], values)

    # Each lift leaves the next matmul's left operand scaled by the same row scale, so the scale
    # ends up outside the whole chain and every product is computed once, unbatched. Folding the
    # batch into a GEMM instead would need a Reshape.
    assert not _nodes_of_type(function, BatchedDot)
    assert not _nodes_of_type(function, Reshape)


def test_masked_quadratic_form_needs_no_batched_dot(rng):
    mask = pt.tensor(shape=(5, 3, 1))
    design = pt.tensor(shape=(3, 6))
    covariance = pt.tensor(shape=(5, 6, 6))
    masked = mask * design
    values = [
        (rng.random(size=(5, 3, 1)) > 0.3).astype(float),
        rng.normal(size=(3, 6)),
        rng.normal(size=(5, 6, 6)),
    ]

    function = _compile_and_compare(
        [mask, design, covariance], [masked @ covariance @ masked.mT], values
    )

    assert not _nodes_of_type(function, BatchedDot)


def test_scale_along_contracted_axis_after_transpose_is_not_lifted(rng):
    scale = pt.tensor(shape=(5, 3, 1))
    design = pt.tensor(shape=(1, 3, 6))
    right = pt.tensor(shape=(1, 3, 4))
    output = (scale * design).mT @ right
    values = [rng.normal(size=variable.type.shape) for variable in (scale, design, right)]

    assert not _lift_fires(output)
    _compile_and_compare([scale, design, right], [output], values)


def test_scale_transposed_into_right_operand_is_lifted(rng):
    scale = pt.tensor(shape=(5, 3, 1))
    design = pt.tensor(shape=(1, 3, 6))
    covariance = pt.tensor(shape=(5, 6, 6))
    values = [rng.normal(size=variable.type.shape) for variable in (scale, design, covariance)]

    function = _compile_and_compare(
        [scale, design, covariance], [covariance @ (scale * design).mT], values
    )

    assert not _nodes_of_type(function, BatchedDot)


def test_same_scaled_operand_on_both_sides(rng):
    scale = pt.tensor(shape=(5, 4, 1))
    square = pt.tensor(shape=(1, 4, 4))
    scaled = scale * square
    output = scaled @ scaled
    values = [rng.normal(size=variable.type.shape) for variable in (scale, square)]

    # Only the left use is a row scale. On the right the same factor scales the contracted axis.
    assert _lift_fires(output)
    _compile_and_compare([scale, square], [output], values)


def test_repeated_and_nested_scales_on_different_batch_axes(rng):
    first_scale = pt.tensor(shape=(4, 1, 3, 1))
    second_scale = pt.tensor(shape=(1, 5, 3, 1))
    left = pt.tensor(shape=(3, 6))
    right = pt.tensor(shape=(4, 5, 6, 2))
    output = (first_scale * (second_scale * (first_scale * left))) @ right
    inputs = [first_scale, second_scale, left, right]
    values = [rng.normal(size=variable.type.shape) for variable in inputs]

    function = _compile_and_compare(inputs, [output], values)

    assert not _nodes_of_type(function, BatchedDot)


def test_computed_scale_shared_with_the_output(rng):
    weights = pt.tensor(shape=(3, 6))
    states = pt.tensor(shape=(5, 6))
    right = pt.tensor(shape=(6, 4))
    scale = pt.sigmoid(pt.matvec(weights, states))[..., None]
    output = (scale * weights) @ right + scale
    values = [rng.normal(size=variable.type.shape) for variable in (weights, states, right)]

    assert _lift_fires(output.owner.inputs[0])
    _compile_and_compare([weights, states, right], [output], values)


def test_lift_bails_when_kept_factors_are_narrower_than_the_operand(rng):
    scale = pt.tensor(shape=(2, 2, 1), dtype="int16")
    design = pt.tensor(shape=(1, 2, 2), dtype="int8")
    right = pt.tensor(shape=(1, 2, 2), dtype="int8")
    output = (scale * design) @ right
    values = [
        np.ones((2, 2, 1), dtype="int16"),
        np.full((1, 2, 2), 100, dtype="int8"),
        np.full((1, 2, 2), 100, dtype="int8"),
    ]

    # A matmul in int8 would wrap 2 * 100 * 100 around to 32.
    assert not _lift_fires(output)
    assert_allclose(pytensor.function([scale, design, right], output)(*values), 20_000)


def test_lift_bails_on_float32_operand_with_float64_scale():
    scale = pt.tensor(shape=(5, 3, 1), dtype="float64")
    design = pt.tensor(shape=(1, 3, 6), dtype="float32")
    right = pt.tensor(shape=(5, 6, 4), dtype="float32")

    assert not _lift_fires((scale * design) @ right)
    assert _lift_fires((scale.astype("float32") * design) @ right)


def test_zero_scale_with_non_finite_operands_gives_the_same_nans(rng):
    mask = pt.tensor(shape=(4, 3, 1))
    design = pt.tensor(shape=(1, 3, 6))
    right = pt.tensor(shape=(1, 6, 2))
    mask_value = np.ones((4, 3, 1))
    mask_value[:, 1] = 0.0
    design_value = rng.normal(size=(1, 3, 6))
    design_value[0, 1, 2] = np.inf
    right_value = rng.normal(size=(1, 6, 2))
    right_value[0, 4, 0] = np.nan

    assert _lift_fires((mask * design) @ right)
    _compile_and_compare(
        [mask, design, right], [(mask * design) @ right], [mask_value, design_value, right_value]
    )


def test_gradient_with_respect_to_the_lifted_scale(rng):
    scale = pt.tensor(shape=(5, 3, 1))
    design = pt.tensor(shape=(1, 3, 6))
    covariance = pt.tensor(shape=(5, 6, 6))
    scaled = scale * design
    loss = pt.log1p((scaled @ covariance @ scaled.mT) ** 2).sum()
    inputs = [scale, design, covariance]
    values = [rng.normal(size=variable.type.shape) for variable in inputs]

    _compile_and_compare(inputs, pytensor.grad(loss, inputs), values)


def test_operands_of_different_rank(rng):
    scale = pt.tensor(shape=(7, 5, 3, 1))
    left = pt.tensor(shape=(3, 6))
    right = pt.tensor(shape=(6, 4))
    values = [rng.normal(size=variable.type.shape) for variable in (scale, left, right)]

    function = _compile_and_compare([scale, left, right], [(scale * left) @ right], values)

    assert not _nodes_of_type(function, BatchedDot)
    assert all(node.outputs[0].type.shape == (3, 4) for node in _nodes_of_type(function, Dot))


def test_unit_core_dims(rng):
    row_scale = pt.tensor(shape=(5, 1, 1))
    row = pt.tensor(shape=(1, 1, 6))
    outer_scale = pt.tensor(shape=(5, 3, 1))
    column = pt.tensor(shape=(1, 3, 1))
    right = pt.tensor(shape=(1, 6, 4))
    unit_right = pt.tensor(shape=(1, 1, 4))
    inputs = [row_scale, row, outer_scale, column, right, unit_right]
    values = [rng.normal(size=variable.type.shape) for variable in inputs]
    row_product = (row_scale * row) @ right
    outer_product = (outer_scale * column) @ unit_right

    # With k == 1 every factor of the left operand is constant along the contracted axis, so
    # nothing is left to keep inside the matmul.
    assert _lift_fires(row_product)
    assert not _lift_fires(outer_product)
    _compile_and_compare(inputs, [row_product, outer_product], values)
