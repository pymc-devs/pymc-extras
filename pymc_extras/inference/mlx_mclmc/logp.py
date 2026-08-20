import numpy as np
import pymc as pm
import pytensor.tensor as pt

from arviz_base import dict_to_dataset
from pymc.backends.arviz import coords_and_dims_for_inferencedata
from pymc.util import get_untransformed_name, is_transformed_name
from pytensor.compile import mode
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph import graph_replace, vectorize_graph
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.traversal import graph_inputs
from pytensor.link.mlx.dispatch import mlx_funcify
from xarray import Dataset

from pymc_extras.inference.laplace_approx.laplace import unpack_last_axis

try:
    import mlx.core as mx
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The MCLMC sampler requires the `mlx` package, which is only available on Apple "
        "Silicon. Install it with `pip install mlx`."
    ) from exc


def _mlxify(inputs, outputs):
    """Rewrite a PyTensor graph under the MLX mode and funcify it to a raw MLX callable."""
    # Parameter checks are dropped rather than turned into a -inf switch: MLX tracing cannot keep
    # the assertion anyway, and unadjusted MCLMC has no accept step to reject the -inf region, so
    # the resulting nan gradient would poison the chain instead of bouncing it back.
    mlx_mode = mode.MLX.including("local_remove_check_parameter")

    fgraph = FunctionGraph(inputs=list(inputs), outputs=list(outputs), clone=True)
    mlx_mode.optimizer.rewrite(fgraph)

    return mlx_funcify(fgraph)


class MLXLogp:
    """
    The log-density of a PyMC model and its gradient, as one MLX callable over a flat vector.

    The free value variables are packed into a single unconstrained vector, and the density
    carries the transform Jacobian. Both the packing and the gradient live in the PyTensor graph,
    which is then funcified straight to ``mlx.core``: the gradient is PyTensor's own, so ops whose
    reverse rule MLX lacks -- matrix inverse and Cholesky among them -- still differentiate, and
    ``mx.vmap`` and ``mx.compile`` see through the result.

    Dim lengths and ``pm.Data`` containers stay as shared variables of the compiled graph, so
    the model is used as given and a later ``pm.set_data`` is picked up on the next call. Freeze
    the model yourself with :func:`~pymc.model.transform.optimization.freeze_dims_and_data` to
    fold them in as constants, which lets more of the graph constant-fold.

    Parameters
    ----------
    model : pm.Model
        The model whose log-density is compiled. It is used as given, not copied or rewritten.
    negative : bool
        Whether to return the negative log-density and its gradient. Default is False.

    Attributes
    ----------
    model : pm.Model
        The model the compiled density belongs to.
    dim : int
        Total flattened size of the unconstrained free variables.
    names : list of str
        Value-variable names in the flat vector's block order, so transformed names such as
        ``sigma_log__``.
    shapes : list of tuple
        Shape of each block.
    """

    def __init__(self, model: pm.Model, *, negative: bool = False):
        initial_point = model.initial_point()

        self.model = model
        self.names = [value_var.name for value_var in model.value_vars]
        self.shapes = [tuple(np.shape(initial_point[name])) for name in self.names]
        self.sizes = [int(np.prod(shape)) if shape else 1 for shape in self.shapes]
        self.dim = int(sum(self.sizes))
        self._initial_point = initial_point

        flat = pt.vector("flat_value", shape=(self.dim,), dtype=model.value_vars[0].dtype)
        value_vars = pt.unpack(flat, packed_shapes=self.shapes)
        logp = graph_replace(model.logp(), dict(zip(model.value_vars, value_vars, strict=True)))
        if negative:
            logp = -logp
        grad, _ = pt.pack(*pt.grad(logp, value_vars))

        # Dim lengths and pm.Data are shared variables, which are graph inputs rather than
        # constants unless the caller froze the model. Carrying them as trailing inputs of the
        # funcified callable keeps that choice hers.
        self._shared = [
            variable
            for variable in graph_inputs([logp, grad])
            if isinstance(variable, SharedVariable)
        ]
        self._shared_cache: list[tuple[object | None, mx.array | None]] = [(None, None)] * len(
            self._shared
        )

        self._raw = _mlxify([flat, *self._shared], [logp, grad])

    def _shared_values(self) -> list[mx.array]:
        """The current shared-variable values as MLX arrays, converted only when they change."""
        values = []
        for index, variable in enumerate(self._shared):
            # pm.set_data swaps in a new array rather than mutating the old one, so identity of
            # the borrowed value is enough to tell a stale conversion from a live one.
            borrowed = variable.get_value(borrow=True)
            cached_source, cached_value = self._shared_cache[index]
            if cached_source is not borrowed:
                cached_value = mx.array(np.asarray(borrowed))
                self._shared_cache[index] = (borrowed, cached_value)
            values.append(cached_value)

        return values

    def __call__(self, x: mx.array) -> mx.array:
        return self._raw(x, *self._shared_values())[0]

    def value_and_grad(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """Return the log-density and its gradient with respect to the flat vector ``x``."""
        value, grad = self._raw(x, *self._shared_values())

        return value, grad

    def flat_initial_point(self) -> np.ndarray:
        """Return the model's initial point as one flat float32 vector."""
        return np.concatenate(
            [np.asarray(self._initial_point[name], dtype="float32").ravel() for name in self.names]
        )


def check_model_is_sampleable(model: pm.Model) -> None:
    """
    Raise ValueError unless every value variable of ``model`` is float32.

    MCLMC is gradient-based, so it cannot sample discrete variables at all, and it runs on the
    Metal GPU, which has no float64 support, so a float64 graph would silently fall back to the
    CPU stream.
    """
    discrete = [
        value_var.name for value_var in model.value_vars if value_var.dtype.startswith("int")
    ]
    if discrete:
        raise ValueError(
            f"MCLMC cannot sample the discrete variables {', '.join(discrete)}, because it needs "
            "the gradient of the log-density with respect to every variable."
        )

    wrong_precision = [
        f"{value_var.name} ({value_var.dtype})"
        for value_var in model.value_vars
        if value_var.dtype != "float32"
    ]
    if wrong_precision:
        raise ValueError(
            "MCLMC requires a float32 model graph, but these value variables are not float32: "
            f'{", ".join(wrong_precision)}. Set `pytensor.config.floatX = "float32"` before '
            "building the model."
        )


def draws_to_datasets(
    flat_draws: np.ndarray,
    model: pm.Model,
    *,
    include_transformed: bool = False,
    compile_kwargs: dict | None = None,
) -> tuple[Dataset, Dataset | None]:
    """
    Map flat unconstrained draws back onto the model's free variables.

    Parameters
    ----------
    flat_draws : np.ndarray
        Draws in the flat unconstrained space, of shape ``(chains, draws, dim)``.
    model : pm.Model
        The model the draws came from.
    include_transformed : bool
        Whether to also return the unconstrained draws, keyed by value-variable name. Default is
        False.
    compile_kwargs : dict, optional
        Extra keyword arguments for ``pymc.pytensorf.compile``.

    Returns
    -------
    posterior : Dataset
        The free variables, with ``chain`` and ``draw`` dimensions. Deterministics are left to
        :func:`~pymc_extras.inference.laplace_approx.idata.add_data_to_inference_data`.
    unconstrained_posterior : Dataset or None
        The draws as the sampler saw them, or None unless ``include_transformed``.
    """
    compile_kwargs = {} if compile_kwargs is None else compile_kwargs

    chains, draws, dim = flat_draws.shape
    initial_point = model.initial_point()
    shapes = [initial_point[value_var.name].shape for value_var in model.value_vars]
    var_names = [free_RV.name for free_RV in model.free_RVs]

    outputs = model.replace_rvs_by_values(model.free_RVs)
    if include_transformed:
        outputs.extend(model.value_vars)

    packed = pt.matrix("packed_draws", shape=(None, dim), dtype=flat_draws.dtype)
    outputs = vectorize_graph(
        outputs, replace=dict(zip(model.value_vars, unpack_last_axis(packed, shapes)))
    )
    fn = pm.pytensorf.compile([packed], outputs, trust_input=True, **compile_kwargs)

    buffers = [
        buffer.reshape(chains, draws, *buffer.shape[1:])
        for buffer in fn(flat_draws.reshape(chains * draws, dim))
    ]

    model_coords, model_dims = coords_and_dims_for_inferencedata(model)
    posterior = dict_to_dataset(
        dict(zip(var_names, buffers, strict=not include_transformed)),
        coords=model_coords,
        dims=model_dims,
        inference_library=pm,
    )

    if not include_transformed:
        return posterior, None

    unconstrained = {
        value_var.name: buffer
        for value_var, buffer in zip(model.value_vars, buffers[len(var_names) :], strict=True)
    }
    unconstrained_posterior = dict_to_dataset(
        unconstrained,
        coords=model_coords,
        dims=_with_unconstrained_dims(unconstrained, model_coords, model_dims),
        inference_library=pm,
    )

    return posterior, unconstrained_posterior


def _with_unconstrained_dims(unconstrained, model_coords, model_dims) -> dict:
    """Extend ``model_dims`` with a transformed variable's dims, where its lengths still match."""
    dims = dict(model_dims)

    for var_name, var_draws in unconstrained.items():
        if not is_transformed_name(var_name):
            continue

        constrained_dims = dims.get(get_untransformed_name(var_name))
        if constrained_dims is None or len(constrained_dims) != (var_draws.ndim - 2):
            continue

        # A length mismatch, as a simplex transform gives, means the mapping is not one-to-one.
        dims[var_name] = [
            dim_name if len(model_coords.get(dim_name, ())) == length else f"{var_name}_dim_{axis}"
            for axis, (dim_name, length) in enumerate(
                zip(constrained_dims, var_draws.shape[2:], strict=True)
            )
        ]

    return dims
