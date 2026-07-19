import itertools

from collections.abc import Sequence
from itertools import zip_longest

from pymc import SymbolicRandomVariable
from pymc.model.fgraph import ModelVar
from pymc.variational.minibatch_rv import MinibatchRandomVariable
from pytensor.graph.basic import Variable
from pytensor.graph.traversal import ancestors, io_toposort
from pytensor.tensor import TensorType, TensorVariable
from pytensor.tensor.blockwise import Blockwise
from pytensor.tensor.elemwise import CAReduce, DimShuffle, Elemwise
from pytensor.tensor.math import Dot
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.shape import Shape
from pytensor.tensor.subtensor import AdvancedSubtensor, Subtensor, get_idx_list
from pytensor.tensor.type_other import NoneTypeT

from pymc_extras.model.marginal.distributions.core import MarginalRV


def static_shape_ancestors(vars):
    """Identify ancestors Shape Ops of static shapes (therefore constant in a valid graph)."""
    return [
        var
        for var in ancestors(vars)
        if (
            var.owner
            and isinstance(var.owner.op, Shape)
            # All static dims lengths of Shape input are known
            and None not in var.owner.inputs[0].type.shape
        )
    ]


def find_conditional_input_rvs(output_rvs, all_rvs):
    """Find conditionally independent input RVs."""
    other_rvs = [other_rv for other_rv in all_rvs if other_rv not in output_rvs]
    blockers = other_rvs + static_shape_ancestors(tuple(all_rvs) + tuple(output_rvs))
    return [var for var in ancestors(output_rvs, blockers=blockers) if var in other_rvs]


def is_conditional_dependent(
    dependent_rv: TensorVariable, dependable_rv: TensorVariable, all_rvs
) -> bool:
    """Check if dependent_rv is conditionally dependent on dependable_rv,
    given all conditionally independent all_rvs"""

    # Sibling outputs of the same node are conditionally dependent
    if (
        dependent_rv is not dependable_rv
        and dependent_rv.owner.inputs[0].owner is not None
        and dependent_rv.owner.inputs[0].owner is dependable_rv.owner.inputs[0].owner
    ):
        return True

    return dependable_rv in find_conditional_input_rvs((dependent_rv,), all_rvs)


def find_conditional_dependent_rvs(dependable_rv, all_rvs):
    """Find rvs than depend on dependable"""
    return [
        rv
        for rv in all_rvs
        if (rv is not dependable_rv and is_conditional_dependent(rv, dependable_rv, all_rvs))
    ]


def get_support_axes(op) -> tuple[tuple[int, ...], ...]:
    if hasattr(op, "support_axes"):
        return op.support_axes
    else:
        return (tuple(range(-op.ndim_supp, 0)),)


def _advanced_indexing_axis_and_ndim(idxs) -> tuple[int, int]:
    """Find the output axis and dimensionality of the advanced indexing group (i.e., array indexing).

    There is a special case: when there are non-consecutive advanced indexing groups, the advanced indexing
    group is always moved to the front.

    See: https://numpy.org/doc/stable/user/basics.indexing.html#combining-advanced-and-basic-indexing
    """

    def _is_tensor_idx(idx) -> bool:
        return isinstance(idx, Variable) and isinstance(idx.type, TensorType)

    adv_group_axis = None
    simple_group_after_adv = False
    for axis, idx in enumerate(idxs):
        if _is_tensor_idx(idx):
            if simple_group_after_adv:
                # Special non-consecutive case
                adv_group_axis = 0
                break
            elif adv_group_axis is None:
                adv_group_axis = axis
        elif adv_group_axis is not None:
            # Special non-consecutive case
            simple_group_after_adv = True

    adv_group_ndim = max(idx.type.ndim for idx in idxs if _is_tensor_idx(idx))
    return adv_group_axis, adv_group_ndim


DIMS = tuple[int | None, ...]
VAR_DIMS = dict[Variable, DIMS]


def _broadcast_dims(
    inputs_dims: Sequence[DIMS],
) -> DIMS:
    output_ndim = max((len(input_dim) for input_dim in inputs_dims), default=0)

    # Add missing dims
    inputs_dims = [
        (None,) * (output_ndim - len(input_dim)) + input_dim for input_dim in inputs_dims
    ]

    # Find which known dims show in the output, while checking no mixing
    output_dims = []
    for inputs_dim in zip(*inputs_dims):
        output_dim = None
        for input_dim in inputs_dim:
            if input_dim is None:
                continue
            if output_dim is not None and output_dim != input_dim:
                raise ValueError("Different known dimensions mixed via broadcasting")
            output_dim = input_dim
        output_dims.append(output_dim)

    # Check for duplicates
    known_dims = [dim for dim in output_dims if dim is not None]
    if len(known_dims) > len(set(known_dims)):
        raise ValueError("Same known dimension used in different axis after broadcasting")

    return tuple(output_dims)


def _subgraph_batch_dim_clients(var_dims: VAR_DIMS, input_vars, output_vars) -> VAR_DIMS:
    for node in io_toposort(input_vars, output_vars):
        inputs_dims = [
            var_dims.get(inp, ((None,) * inp.type.ndim) if hasattr(inp.type, "ndim") else ())
            for inp in node.inputs
        ]

        if all(dim is None for input_dims in inputs_dims for dim in input_dims):
            # None of the inputs are related to the batch_axes of the input_vars
            continue

        elif isinstance(node.op, ModelVar):
            var_dims[node.outputs[0]] = inputs_dims[0]

        elif isinstance(node.op, DimShuffle):
            [input_dims] = inputs_dims
            output_dims = tuple(None if i == "x" else input_dims[i] for i in node.op.new_order)
            var_dims[node.outputs[0]] = output_dims

        elif isinstance(node.op, MarginalRV) or (
            isinstance(node.op, SymbolicRandomVariable) and node.op.extended_signature is None
        ):
            # MarginalRV and SymbolicRandomVariables without signature are a wild-card,
            # so we need to introspect the inner graph.
            op = node.op
            inner_inputs = op.inner_inputs
            inner_outputs = op.inner_outputs

            inner_var_dims = _subgraph_batch_dim_clients(
                dict(zip(inner_inputs, inputs_dims)), inner_inputs, inner_outputs
            )

            support_axes = iter(get_support_axes(op))
            if isinstance(op, MarginalRV):
                # The first output is the marginalized variable for which we don't compute support axes
                support_axes = itertools.chain(((),), support_axes)
            for i, (out, inner_out) in enumerate(zip(node.outputs, inner_outputs)):
                if not isinstance(out.type, TensorType):
                    continue
                support_axes_out = next(support_axes)

                if inner_out in inner_var_dims:
                    out_dims = inner_var_dims[inner_out]
                    if any(
                        dim is not None for dim in (out_dims[axis] for axis in support_axes_out)
                    ):
                        raise ValueError(f"Known dim corresponds to core dimension of {node.op}")
                    var_dims[out] = out_dims

        elif isinstance(node.op, Blockwise) and isinstance(node.op.core_op, Dot):
            # matmul: (..., m, k) @ (..., k, p) -> (..., m, p). Each output entry is
            # ``sum_k A[..., m, k] * B[..., k, p]``, so the ``m`` and ``p`` dims are
            # unmixed (a known dim there maps 1:1 to an output dim), while the
            # contracted ``k`` dim mixes information and is a chokepoint.
            a_dims, b_dims = inputs_dims
            a_batch, (a_m, a_k) = a_dims[:-2], a_dims[-2:]
            b_batch, (b_k, b_p) = b_dims[:-2], b_dims[-2:]
            if a_k is not None or b_k is not None:
                raise ValueError(
                    f"Use of known dimensions as contracted dimensions of op {node.op} "
                    "not supported."
                )
            batch_dims = _broadcast_dims((a_batch, b_batch))
            var_dims[node.outputs[0]] = (*batch_dims, a_m, b_p)

        elif isinstance(node.op, Elemwise | Blockwise | RandomVariable | SymbolicRandomVariable):
            # NOTE: User-provided CustomDist may not respect core dimensions on the left.

            if isinstance(node.op, Elemwise):
                op_batch_ndim = node.outputs[0].type.ndim
            else:
                op_batch_ndim = node.op.batch_ndim(node)

            if isinstance(node.op, SymbolicRandomVariable):
                # SymbolicRandomVariable don't have explicit expand_dims unlike the other Ops considered in this
                [_, _, param_idxs], _ = node.op.get_input_output_type_idxs(
                    node.op.extended_signature
                )
                for param_idx, param_core_ndim in zip(param_idxs, node.op.ndims_params):
                    param_dims = inputs_dims[param_idx]
                    missing_ndim = op_batch_ndim - (len(param_dims) - param_core_ndim)
                    inputs_dims[param_idx] = (None,) * missing_ndim + param_dims

            if any(
                dim is not None for input_dim in inputs_dims for dim in input_dim[op_batch_ndim:]
            ):
                raise ValueError(
                    f"Use of known dimensions as core dimensions of op {node.op} not supported."
                )

            batch_dims = _broadcast_dims(
                tuple(input_dims[:op_batch_ndim] for input_dims in inputs_dims)
            )
            for out in node.outputs:
                if isinstance(out.type, TensorType):
                    core_ndim = out.type.ndim - op_batch_ndim
                    output_dims = batch_dims + (None,) * core_ndim
                    var_dims[out] = output_dims

        elif isinstance(node.op, CAReduce):
            [input_dims] = inputs_dims

            axes = node.op.axis
            if isinstance(axes, int):
                axes = (axes,)
            elif axes is None:
                axes = tuple(range(node.inputs[0].type.ndim))

            if any(input_dims[axis] is not None for axis in axes):
                raise ValueError(
                    f"Use of known dimensions as reduced dimensions of op {node.op} not supported."
                )

            output_dims = [dims for i, dims in enumerate(input_dims) if i not in axes]
            var_dims[node.outputs[0]] = tuple(output_dims)

        elif isinstance(node.op, Subtensor):
            value_dims, *keys_dims = inputs_dims
            # Dims in basic indexing must belong to the value variable, since indexing keys are always scalar
            assert not any(dim is None for dim in keys_dims)
            keys = get_idx_list(node.inputs, node.op.idx_list)

            output_dims = []
            for value_dim, idx in zip_longest(value_dims, keys, fillvalue=slice(None)):
                if idx == slice(None):
                    # Dim is kept
                    output_dims.append(value_dim)
                elif value_dim is not None:
                    raise ValueError(
                        "Partial slicing or indexing of known dimensions not supported."
                    )
                elif isinstance(idx, slice):
                    # Unknown dimensions kept by partial slice.
                    output_dims.append(None)

            var_dims[node.outputs[0]] = tuple(output_dims)

        elif isinstance(node.op, AdvancedSubtensor):
            # AdvancedSubtensor dimensions can show up as both the indexed variable and indexing variables
            value, *keys = node.inputs
            value_dims, *keys_dims = inputs_dims

            # Just to stay sane, we forbid any boolean indexing...
            if any(isinstance(idx.type, TensorType) and idx.type.dtype == "bool" for idx in keys):
                raise NotImplementedError(
                    f"Array indexing with boolean variables in node {node} not supported."
                )

            if any(dim is not None for dim in value_dims) and any(
                dim is not None for key_dims in keys_dims for dim in key_dims
            ):
                # Both indexed variable and indexing variables have known dimensions
                # I am to lazy to think through these, so we raise for now.
                raise NotImplementedError(
                    f"Simultaneous use of known dimensions in indexed and indexing variables in node {node} not supported."
                )

            full_keys = get_idx_list(node.inputs, node.op.idx_list)
            adv_group_axis, adv_group_ndim = _advanced_indexing_axis_and_ndim(full_keys)

            if any(dim is not None for dim in value_dims):
                # Indexed variable has known dimensions

                if any(isinstance(idx.type, NoneTypeT) for idx in keys):
                    # Corresponds to an expand_dims, for now not supported
                    raise NotImplementedError(
                        f"Advanced indexing in node {node} which introduces new axis is not supported."
                    )

                non_adv_dims = []
                for value_dim, idx in zip_longest(value_dims, full_keys, fillvalue=slice(None)):
                    if idx == slice(None):
                        non_adv_dims.append(value_dim)
                    elif value_dim is not None:
                        # We are trying to partially slice or index a known dimension
                        raise ValueError(
                            "Partial slicing or advanced integer indexing of known dimensions not supported."
                        )
                    elif isinstance(idx, slice):
                        # Unknown dimensions kept by partial slice.
                        non_adv_dims.append(None)

                # Insert unknown dimensions corresponding to advanced indexing
                output_dims = tuple(
                    non_adv_dims[:adv_group_axis]
                    + [None] * adv_group_ndim
                    + non_adv_dims[adv_group_axis:]
                )

            else:
                # Indexing keys have known dimensions.
                # Only array indices can have dimensions, the rest are just slices or newaxis

                # Advanced indexing variables broadcast together, so we apply same rules as in Elemwise
                adv_dims = _broadcast_dims(keys_dims)

                start_non_adv_dims = (None,) * adv_group_axis
                end_non_adv_dims = (None,) * (
                    node.outputs[0].type.ndim - adv_group_axis - adv_group_ndim
                )
                output_dims = start_non_adv_dims + adv_dims + end_non_adv_dims

            var_dims[node.outputs[0]] = output_dims

        elif isinstance(node.op, MinibatchRandomVariable):
            var_dims[node.outputs[0]] = inputs_dims[0]

        else:
            raise NotImplementedError(f"Marginalization through operation {node} not supported.")

    return var_dims


AXES = tuple[bool, ...]


def _backward_batch_axes(out_batch: AXES, inp: Variable, inp_batch_ndim: int) -> list[bool]:
    """Right-aligned backprop of carried output batch axes onto one input.

    ``out_batch`` are the carried flags over the op's output batch axes. Returns a
    per-axis flag list for ``inp`` (length ``inp.type.ndim``), marking an input batch
    axis when it feeds a carried output batch axis and is not broadcast (static size 1)
    there. Core/support axes of ``inp`` (the trailing ``ndim - inp_batch_ndim``) never
    carry a batch axis."""
    in_c = [False] * inp.type.ndim
    offset = len(out_batch) - inp_batch_ndim
    shape = inp.type.shape
    for j in range(inp_batch_ndim):
        oa = j + offset
        if 0 <= oa < len(out_batch) and out_batch[oa] and shape[j] != 1:
            in_c[j] = True
    return in_c


def _backward_node_axes(node, outs: list[AXES | None]) -> list[AXES | None]:
    """Backward batch-axis mapping for one node: given carried flags on its outputs,
    return carried flags for each input (``None`` for non-tensor inputs)."""
    op = node.op
    inputs = node.inputs
    none: list[AXES | None] = [None] * len(inputs)

    def tensor_out() -> AXES | None:
        # The single carried tensor output (RVs also output an rng we ignore).
        for o, no in zip(outs, node.outputs):
            if o is not None and isinstance(no.type, TensorType):
                return o
        return None

    def conservative() -> list[AXES | None]:
        carried = any(o is not None and any(o) for o in outs)
        return [
            (True,) * inp.type.ndim if (carried and isinstance(inp.type, TensorType)) else None
            for inp in inputs
        ]

    if isinstance(op, ModelVar | MinibatchRandomVariable):
        none[0] = tensor_out()
        return none

    if isinstance(op, DimShuffle):
        out = outs[0]
        in_c = [False] * inputs[0].type.ndim
        for i, o in enumerate(op.new_order):
            if o != "x" and out[i]:
                in_c[o] = True
        none[0] = tuple(in_c)
        return none

    if isinstance(op, CAReduce):
        out = outs[0]
        axes = op.axis
        if isinstance(axes, int):
            axes = (axes,)
        elif axes is None:
            axes = tuple(range(inputs[0].type.ndim))
        kept = [a for a in range(inputs[0].type.ndim) if a not in axes]
        in_c = [False] * inputs[0].type.ndim
        for out_axis, in_axis in enumerate(kept):
            if out[out_axis]:
                in_c[in_axis] = True
        none[0] = tuple(in_c)  # reduced axes stay False -> ancestor below is not reached
        return none

    if isinstance(op, Blockwise) and isinstance(op.core_op, Dot):
        out = outs[0]
        a, b = inputs
        out_batch = out[:-2]
        a_c = _backward_batch_axes(out_batch, a, a.type.ndim - 2)
        b_c = _backward_batch_axes(out_batch, b, b.type.ndim - 2)
        if out[-2]:  # output m comes from a's m
            a_c[-2] = True
        if out[-1]:  # output p comes from b's p
            b_c[-1] = True
        # the contracted k (a[-1], b[-2]) is never carried
        none[0], none[1] = tuple(a_c), tuple(b_c)
        return none

    if isinstance(op, Elemwise):
        out = outs[0]
        return [tuple(_backward_batch_axes(out, inp, inp.type.ndim)) for inp in inputs]

    if isinstance(op, RandomVariable):
        out = tensor_out()
        if out is None:
            return none
        op_batch_ndim = op.batch_ndim(node)
        out_batch = out[:op_batch_ndim]
        n_params = len(op.ndims_params)
        param_start = len(inputs) - n_params
        for k, core_ndim in enumerate(op.ndims_params):
            idx = param_start + k
            param = inputs[idx]
            none[idx] = tuple(_backward_batch_axes(out_batch, param, param.type.ndim - core_ndim))
        return none

    if isinstance(op, SymbolicRandomVariable) and op.extended_signature is not None:
        op_batch_ndim = op.batch_ndim(node)
        [_, _, param_idxs], _ = op.get_input_output_type_idxs(op.extended_signature)
        out_batch: AXES | None = None
        for o in outs:
            if o is None:
                continue
            ob = o[:op_batch_ndim]
            out_batch = ob if out_batch is None else tuple(x or y for x, y in zip(out_batch, ob))
        if out_batch is None:
            return none
        for pidx, core_ndim in zip(param_idxs, op.ndims_params):
            param = inputs[pidx]
            none[pidx] = tuple(_backward_batch_axes(out_batch, param, param.type.ndim - core_ndim))
        return none

    if isinstance(op, Subtensor):
        out = outs[0]
        value = inputs[0]
        keys = list(get_idx_list(inputs, op.idx_list))
        in_c = [False] * value.type.ndim
        out_axis = 0
        for v_axis in range(value.type.ndim):
            idx = keys[v_axis] if v_axis < len(keys) else slice(None)
            if isinstance(idx, slice):
                # Dim is kept (a partial slice still maps onto the same axis).
                if out_axis < len(out) and out[out_axis]:
                    in_c[v_axis] = True
                out_axis += 1
            # else: scalar integer index drops this axis (not present in output)
        none[0] = tuple(in_c)
        return none

    if isinstance(op, AdvancedSubtensor):
        out = outs[0]
        value = inputs[0]
        keys = inputs[1:]
        full_keys = get_idx_list(inputs, op.idx_list)
        adv_axis, adv_ndim = _advanced_indexing_axis_and_ndim(full_keys)
        # Value axes kept by a basic slice, in order, map to the non-advanced output axes.
        kept_value_axes = [
            v_axis
            for v_axis, idx in zip_longest(range(value.type.ndim), full_keys, fillvalue=slice(None))
            if v_axis is not None and isinstance(idx, slice)
        ]
        value_c = [False] * value.type.ndim
        idx_carries = False
        kept_ptr = 0
        for out_axis in range(len(out)):
            if adv_axis <= out_axis < adv_axis + adv_ndim:
                idx_carries = idx_carries or out[out_axis]
            else:
                if kept_ptr < len(kept_value_axes):
                    if out[out_axis]:
                        value_c[kept_value_axes[kept_ptr]] = True
                    kept_ptr += 1
        none[0] = tuple(value_c)
        if idx_carries:
            for ki, key in enumerate(keys):
                if isinstance(key.type, TensorType) and key.type.ndim > 0:
                    none[1 + ki] = (True,) * key.type.ndim
        return none

    # MarginalRV, SymbolicRandomVariable without signature, generic Blockwise, and any
    # other op: be conservative rather than risk dropping a carried ancestor.
    return conservative()


def subgraph_batch_dim_ancestors(output_var: Variable, axis: int) -> dict[Variable, AXES]:
    """Trace a single batch ``axis`` of ``output_var`` backward to its ancestors.

    Input-direction dual of :func:`_subgraph_batch_dim_clients`: instead of
    propagating an input's batch dims forward to its clients, propagate one output axis
    backward to the ancestors that feed it. Returns a map from every ancestor that
    carries the axis to a boolean tuple marking which of *its* axes carry it.

    Only presence/absence of the one axis is tracked (a bool per axis), so unlike the
    forward pass there is no mixing of distinct labels. An axis reduced or contracted
    away upstream is not propagated (the ancestor below it is not reached); an op whose
    inverse cannot be traced propagates the axis conservatively to all its inputs, so
    nothing that might carry it is silently dropped.
    """
    carries: dict[Variable, AXES] = {
        output_var: tuple(i == axis for i in range(output_var.type.ndim))
    }
    for node in reversed(list(io_toposort([], [output_var]))):
        outs = [carries.get(out) for out in node.outputs]
        if not any(o is not None and any(o) for o in outs):
            continue
        for inp, c in zip(node.inputs, _backward_node_axes(node, outs)):
            if not c or not any(c):
                continue
            prev = carries.get(inp)
            carries[inp] = c if prev is None else tuple(p or q for p, q in zip(prev, c))
    return carries


def subgraph_batch_dim_connection(input_var, output_vars) -> list[DIMS]:
    """Identify how the batch dims of input map to the batch dimensions of the output_rvs.

    Example:
    -------
    In the example below `idx` has two batch dimensions (indexed 0, 1 from left to right).
    The two uncommented dependent variables each have 2 batch dimensions where each entry
    results from a mapping of a single entry from one of these batch dimensions.

    This mapping is transposed in the case of the first dependent variable, and shows up in
    the same order for the second dependent variable. Each of the variables as a further
    batch dimension encoded as `None`.

    The commented out third dependent variable combines information from the batch dimensions
    of `idx` via the `sum` operation. A `ValueError` would be raised if we requested the
    connection of batch dims.

    .. code-block:: python
        import pymc as pm

        idx = pm.Bernoulli.dist(shape=(3, 2))
        dep1 = pm.Normal.dist(mu=idx.T[..., None] * 2, shape=(3, 2, 5))
        dep2 = pm.Normal.dist(mu=idx * 2, shape=(7, 2, 3))
        # dep3 = pm.Normal.dist(mu=idx.sum())  # Would raise if requested

        print(subgraph_batch_dim_connection(idx, [], [dep1, dep2]))
        # [(1, 0, None), (None, 0, 1)]

    Returns:
    -------
    list of tuples
        Each tuple corresponds to the batch dimensions of the output_rv in the order they are found in the output.
        None is used to indicate a batch dimension that is not mapped from the input.

    Raises:
    ------
    ValueError
        If input batch dimensions are mixed in the graph leading to output_vars.

    NotImplementedError
        If variable related to marginalized batch_dims is used in an operation that is not yet supported
    """
    var_dims = {input_var: tuple(range(input_var.type.ndim))}
    var_dims = _subgraph_batch_dim_clients(var_dims, [input_var], output_vars)
    ret = []
    for output_var in output_vars:
        output_dims = var_dims.get(output_var, (None,) * output_var.type.ndim)
        assert len(output_dims) == output_var.type.ndim
        ret.append(output_dims)
    return ret
