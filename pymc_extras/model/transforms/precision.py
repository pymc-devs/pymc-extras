import copy

from collections.abc import Sequence

import numpy as np
import pytensor
import pytensor.tensor as pt

from pymc.logprob.transforms import Transform
from pymc.model.core import Model
from pymc.model.fgraph import ModelValuedVar, fgraph_from_model, model_from_fgraph
from pytensor.compile import SharedVariable
from pytensor.compile.builders import construct_nominal_fgraph
from pytensor.graph import Constant, FunctionGraph, Op, Variable
from pytensor.graph.op import HasInnerGraph
from pytensor.graph.traversal import ancestors, io_toposort
from pytensor.scalar import Cast, discrete_dtypes
from pytensor.scan.op import Scan
from pytensor.tensor.elemwise import Elemwise
from pytensor.tensor.type import TensorType


def _is_dtype(dtype, ref_dtype: str) -> bool:
    """Whether `dtype` (a dtype-like or alias such as "float") is `ref_dtype` or follows floatX."""
    return dtype is not None and (dtype == "floatX" or np.dtype(dtype).name == ref_dtype)


def _cast_root(var: Variable, from_dtype: str, to_dtype: str) -> Variable:
    """Return a `to_dtype` clone of a root variable (constant, shared or input)."""
    if getattr(var.type, "dtype", None) != from_dtype:
        return var
    new_type = var.type.clone(dtype=to_dtype)
    if isinstance(var, Constant):
        return new_type.make_constant(var.data.astype(to_dtype), name=var.name)
    if isinstance(var, SharedVariable):
        value = var.get_value(borrow=False).astype(to_dtype)
        return type(var)(type=new_type, value=value, strict=False, name=var.name)
    return new_type(name=var.name)


def _restore_static_shape(new: Variable, old: Variable) -> Variable:
    if isinstance(new.type, TensorType) and new.type.shape != old.type.shape:
        new = pt.specify_shape(new, old.type.shape)
        new.name = old.name
    return new


def _cast_op(op: Op, from_dtype: str, to_dtype: str) -> Op:
    """Return `op` with the `from_dtype` baked into its attributes or inner graph cast."""
    # Fixed output dtype (RandomVariables, reductions, ARange, ...)
    changes: dict = {"dtype": to_dtype} if _is_dtype(getattr(op, "dtype", None), from_dtype) else {}
    for attr, value in vars(op).items():
        new_value: Variable | Op
        if isinstance(value, Constant):
            new_value = _cast_root(value, from_dtype, to_dtype)
        elif attr == "core_op" and isinstance(value, Op):
            new_value = _cast_op(value, from_dtype, to_dtype)
        else:
            continue
        if new_value is not value:
            changes[attr] = new_value

    if isinstance(op, HasInnerGraph) and any(
        getattr(var.type, "dtype", None) == from_dtype for var in op.fgraph.variables
    ):
        n_outs = len(op.inner_outputs)
        inner = _cast_graph_floats([*op.inner_outputs, *op.inner_inputs], from_dtype, to_dtype)
        # Static shapes frozen in the inner graph cannot be re-inferred from the inner inputs
        inner_outs = [
            _restore_static_shape(new, old)
            for new, old in zip(inner[:n_outs], op.inner_outputs, strict=True)
        ]
        inner_ins = inner[n_outs:]
        if isinstance(op, Scan):
            kwargs = ("mode", "truncate_gradient", "name", "profile", "allow_gc", "strict")
            op = Scan(inner_ins, inner_outs, op.info, **{k: getattr(op, k) for k in kwargs})
        else:
            op = op.clone_with_inner_graph(construct_nominal_fgraph(inner_ins, inner_outs))  # type: ignore[attr-defined]
    elif changes:
        op = copy.copy(op)
    vars(op).update(changes)
    return op


class _CastedTransform(Transform):
    """Wrap a transform whose graphs embed constants of a different float dtype.

    Such constants are not reachable from the model graph, so the graphs the wrapped
    transform builds are converted with `_cast_graph_floats` on every call.
    """

    def __init__(self, transform: Transform, from_dtype: str, to_dtype: str):
        self.transform = transform
        self.from_dtype = from_dtype
        self.to_dtype = to_dtype
        # Value variable names derive from the name
        self.name, self.ndim_supp = transform.name, transform.ndim_supp  # type: ignore[attr-defined]

    def _converted(self, method: str, value, *inputs):
        out = getattr(self.transform, method)(value, *inputs)
        (out,) = _cast_graph_floats([out], self.from_dtype, self.to_dtype, (value, *inputs))
        return out

    def forward(self, value, *inputs):
        return self._converted("forward", value, *inputs)

    def backward(self, value, *inputs):
        return self._converted("backward", value, *inputs)

    def log_jac_det(self, value, *inputs):
        return self._converted("log_jac_det", value, *inputs)


def _transform_keeps_dtype(transform: Transform, rv: Variable, value: Variable, dtype: str) -> bool:
    """Whether the transform's graphs on `rv`/`value` stay in `dtype`."""
    inputs = rv.owner.inputs  # type: ignore[union-attr]
    outs = (
        transform.forward(rv, *inputs),  # type: ignore[arg-type]
        transform.backward(value, *inputs),  # type: ignore[arg-type]
        transform.log_jac_det(value, *inputs),  # type: ignore[arg-type]
    )
    return all(out.type.dtype == dtype for out in outs)  # type: ignore[union-attr]


def _cast_graph_floats(
    outputs: Sequence[Variable],
    from_dtype: str,
    to_dtype: str,
    frozen: Sequence[Variable] = (),
) -> list[Variable]:
    """Clone the graph of `outputs`, casting every `from_dtype` variable to `to_dtype`.

    The `frozen` variables are kept as they are, and the graph above them is not visited.
    """
    # pytensor.sparse is heavy to import, so it is deferred to first use
    from pytensor.sparse.basic import Cast as SparseCast

    memo: dict[Variable, Variable] = {var: var for var in frozen}
    ops: dict[Op, Op] = {}

    def mapped(var):
        if var not in memo:
            memo[var] = _cast_root(var, from_dtype, to_dtype)
        return memo[var]

    def stale(new_outputs):
        return any(getattr(out.type, "dtype", None) == from_dtype for out in new_outputs)

    for node in io_toposort(frozen, outputs):
        op, new_inputs = node.op, [mapped(var) for var in node.inputs]
        if (
            isinstance(op, Elemwise)
            and isinstance(op.scalar_op, Cast)
            and op.scalar_op.o_type.dtype == from_dtype
        ) or (isinstance(op, SparseCast) and op.out_type == from_dtype):
            # Redirect explicit casts (e.g. `x.astype("float64")`)
            new_outputs = [new_inputs[0].astype(to_dtype)]
        else:
            if op not in ops:
                ops[op] = _cast_op(op, from_dtype, to_dtype)
            op = ops[op]
            if isinstance(op, ModelValuedVar) and (transform := op.transform) is not None:
                # Transforms may embed constants of the old dtype in the value-space graphs
                # (logp, initial point). Wrap those that do, keeping the others as they are.
                rv, value = new_inputs
                if isinstance(transform, _CastedTransform):
                    transform = transform.transform
                if not _transform_keeps_dtype(transform, rv, value, to_dtype):
                    transform = _CastedTransform(transform, from_dtype, to_dtype)
                if transform is not op.transform:
                    op = copy.copy(op)
                    op.transform = transform
            new_outputs = op.make_node(*new_inputs).outputs
            if stale(new_outputs):
                # Discrete inputs upcast `to_dtype` back to `from_dtype` (e.g. int32 * float32)
                new_inputs = [
                    i.astype(to_dtype) if getattr(i.type, "dtype", None) in discrete_dtypes else i
                    for i in new_inputs
                ]
                new_outputs = op.make_node(*new_inputs).outputs
        if stale(new_outputs):
            out = next(out for out in outputs if node.outputs[0] in ancestors([out]))
            raise NotImplementedError(
                f"Cannot convert {node.outputs[0]} (in the graph of {out}) to {to_dtype}: "
                f"{op} outputs {from_dtype} regardless of its inputs. "
                "Replace it by an operation that follows the dtype of its inputs."
            )
        for old, new in zip(node.outputs, new_outputs, strict=True):
            new.name = old.name
            memo[old] = new

    return [mapped(out) for out in outputs]


def _cast_model_floats(model: Model, from_dtype: str, to_dtype: str) -> Model:
    initial_values = {
        rv.name: iv for rv, iv in model.rvs_to_initial_values.items() if iv is not None
    }
    for name, initval in initial_values.items():
        if isinstance(initval, Variable) and not isinstance(initval, Constant):
            raise NotImplementedError(
                f"{name} has a symbolic initial value, which cannot be transplanted onto the "
                "new model. Only strategy strings and constant initial values are supported."
            )
    # fgraph_from_model rejects initial values, so they are cleared for the round-trip
    saved_initial_values = dict(model.rvs_to_initial_values)
    try:
        for rv in saved_initial_values:
            model.rvs_to_initial_values[rv] = None
        fg, _ = fgraph_from_model(model)
    finally:
        model.rvs_to_initial_values.update(saved_initial_values)

    # Dtype aliases and transform graphs follow floatX
    with pytensor.config.change_flags(floatX=to_dtype):
        new_outputs = _cast_graph_floats(fg.outputs, from_dtype, to_dtype)
    new_fg = FunctionGraph(outputs=new_outputs, clone=False)
    new_fg._coords, new_fg._dim_lengths = fg._coords, fg._dim_lengths  # type: ignore[attr-defined]
    new_model = model_from_fgraph(new_fg, mutate_fgraph=True)
    for name, initval in initial_values.items():
        if isinstance(initval, np.ndarray) and initval.dtype.kind == "f":
            initval = initval.astype(to_dtype)
        new_model.set_initval(new_model[name], initval)
    return new_model


def model_to_float32(model: Model) -> Model:
    """Recreate a Model with all float64 variables and data cast to float32.

    Every float64 variable is converted: data (constants and `pm.Data`), free and
    observed RVs (including the inner graphs of symbolic and scan-based RVs like
    `ZeroSumNormal` or `AR`), value variables, Deterministics and Potentials. Integer,
    boolean and RNG variables are unaffected. Explicit `.astype("float64")` casts are
    redirected to float32.

    This can speed up sampling at the cost of precision — most on GPUs and for
    compute-bound models; on CPU backends gains depend on how memory- and
    BLAS-bound the model's logp is.

    The new model must be compiled and sampled under ``floatX="float32"``, otherwise
    initial points are float64 and `pm.sample` raises.

    .. code-block:: python

        import pymc as pm
        import pytensor

        from pymc_extras.model.transforms.precision import model_to_float32

        with pm.Model() as m:
            x = pm.Data("x", [0.0, 1.0, 2.0])
            beta = pm.Normal("beta")
            pm.Normal("y", mu=beta * x, sigma=1.0, observed=[1.0, 2.0, 3.0])

        with pytensor.config.change_flags(floatX="float32"):
            with model_to_float32(m):
                idata = pm.sample()

    Raises
    ------
    NotImplementedError
        If an operation outputs float64 regardless of the dtype of its inputs, or a
        variable has a symbolic initial value.

    Notes
    -----
    Only the model graph is converted. Graphs built from it afterwards (logp, initial
    point) are as float32 as those of a model created under ``floatX="float32"``, e.g.
    the logp of `LKJCholeskyCov` still has float64 terms.

    The logp of a float32 `ZeroSumNormal` is only correct in PyMC releases that include
    pymc-devs/pymc#8464.

    ``pm.set_data`` casts new values to ``floatX``, so it also needs ``floatX="float32"``.

    Constant and strategy-string initial values are preserved (arrays are cast);
    symbolic initial values are not supported.
    """
    return _cast_model_floats(model, "float64", "float32")


def model_to_float64(model: Model) -> Model:
    """Recreate a Model with all float32 variables and data cast to float64.

    The counterpart of :func:`model_to_float32`, see its docstring for details. It is not
    an exact inverse: data rounded to float32 stays rounded.
    """
    return _cast_model_floats(model, "float32", "float64")


__all__ = ("model_to_float32", "model_to_float64")
