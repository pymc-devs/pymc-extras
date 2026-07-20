import warnings

from collections.abc import Sequence

import pytensor.tensor as pt

from pymc.distributions.transforms import Chain
from pymc.logprob.transforms import IntervalTransform
from pymc.model import Model
from pymc.model.fgraph import (
    ModelDeterministic,
    ModelPotential,
    ModelValuedVar,
    fgraph_from_model,
    model_from_fgraph,
)
from pytensor.compile import SharedVariable
from pytensor.graph import (
    FunctionGraph,
    Variable,
    graph_inputs,
)
from pytensor.graph.rewriting.basic import in2out
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.graph.traversal import ancestors, io_toposort
from pytensor.tensor import TensorVariable

# Importing the distributions package registers the strategy rewrites
# (enumerable, laplace, normal) into marginal_rewrites_db.
import pymc_extras.model.marginal.distributions  # noqa: F401

from pymc_extras.model.marginal.distributions.core import (
    MarginalRV,
    inline_ofg_outputs,
)
from pymc_extras.model.marginal.distributions.laplace import MarginalLaplaceRV
from pymc_extras.model.marginal.graph_analysis import (
    find_conditional_dependent_rvs,
    find_conditional_input_rvs,
    is_conditional_dependent,
)
from pymc_extras.model.marginal.rewrites import (
    DEFAULT_MINIMIZER_KWARGS,
    DeferredLaplaceMarginalSubgraph,
    DeferredMarginalSubgraph,
    LaplaceMarginalSubgraph,
    MarginalSubgraph,
    MarginalSubgraphBase,
    local_unmarginalize,
    marginalize_rewrites_db,
)

ModelRVs = TensorVariable | Sequence[TensorVariable] | str | Sequence[str]


def _walk_marginal_ops(fgraph):
    """Yield every MarginalRV op in a (model or inner) fgraph, outermost first."""
    for node in fgraph.toposort():
        if isinstance(node.op, MarginalRV):
            yield node.op
            yield from _walk_marginal_ops(node.op.fgraph)


def _resolve_marginalized_names(var_names, marginalized_rv_names, arg_name: str) -> list[str]:
    """Resolve a user selection of marginalized variables to an ordered list of names.

    Normalizes a single name or None (= all), validates the selection, and
    returns it in ``marginalized_rv_names`` (outermost-first) order.
    Marginalized variables no longer exist in the model, so the user has no
    handle to pass; they can only be referred to by name.
    """
    if var_names is None:
        return list(marginalized_rv_names)
    if isinstance(var_names, str):
        var_names = (var_names,)
    var_names = tuple(var_names)
    if not all(isinstance(name, str) for name in var_names):
        raise TypeError(
            f"{arg_name} must be specified by name (str). Marginalized variables no longer "
            f"exist in the model, so they cannot be selected by variable. Got: {var_names}"
        )
    missing_names = [name for name in var_names if name not in marginalized_rv_names]
    if missing_names:
        raise ValueError(f"Unrecognized {arg_name}: {missing_names}")
    return [name for name in marginalized_rv_names if name in var_names]


def _replace_marginal_subgraph(
    fgraph, rv_to_marginalize, dependent_rvs, input_rvs, laplace_options=None
) -> None:
    """Replace a marginalized subgraph with a flat MarginalSubgraph marker Op.

    The subgraph stays alive in the fgraph — the MS node references both
    the subgraph outputs and boundary vars as its inputs. No cloning here;
    rewrites clone at resolution time when building the OpFromGraph.

    If `laplace_options` is given (a dict with ``Q`` and ``minimizer_kwargs``),
    a LaplaceMarginalSubgraph marker is created instead, with the precision
    matrix Q appended as the last boundary input and the minimizer options
    stored on the marker.
    """
    raw_marg = rv_to_marginalize.owner.inputs[0]
    raw_deps = [
        dep.owner.inputs[0] if isinstance(dep.owner.op, ModelValuedVar) else dep
        for dep in dependent_rvs
    ]

    subgraph_outputs = [raw_marg, *raw_deps]
    boundary = list(input_rvs)
    boundary += [
        inp
        for inp in graph_inputs(subgraph_outputs, blockers=boundary)
        if (isinstance(inp, SharedVariable) and inp not in boundary)
    ]

    # Unwrap ModelValuedVar inside the subgraph so the interior only
    # references raw RVs. This prevents cycles when rv_to_marginalize
    # is replaced by the MS output below.
    subgraph_nodes = set(io_toposort(boundary, subgraph_outputs))
    # TODO: Use the existing rewrite once we have rewrite_subgraph code:
    for node in list(subgraph_nodes):
        if not isinstance(node.op, ModelValuedVar):
            continue
        model_var = node.outputs[0]
        raw_rv = node.inputs[0]
        for client_node, client_idx in list(fgraph.clients.get(model_var, [])):
            if client_node in subgraph_nodes:
                fgraph.change_node_input(client_node, client_idx, raw_rv, import_missing=True)

    marginalized_dims = rv_to_marginalize.owner.op.dims
    n_dep = len(dependent_rvs)

    has_nested = any(
        rd.owner is not None and isinstance(rd.owner.op, MarginalSubgraphBase) for rd in raw_deps
    )

    output_types = [out.type for out in subgraph_outputs]
    if laplace_options is not None:
        # Q goes last so the logp implementation can pop it back
        boundary.append(laplace_options["Q"])
        cls = DeferredLaplaceMarginalSubgraph if has_nested else LaplaceMarginalSubgraph
        op = cls(
            n_dependent_rvs=n_dep,
            marginalized_name=rv_to_marginalize.name,
            marginalized_dims=marginalized_dims,
            output_types=output_types,
            minimizer_kwargs=laplace_options["minimizer_kwargs"],
        )
    else:
        cls = DeferredMarginalSubgraph if has_nested else MarginalSubgraph
        op = cls(
            n_dependent_rvs=n_dep,
            marginalized_name=rv_to_marginalize.name,
            marginalized_dims=marginalized_dims,
            output_types=output_types,
        )

    new_outputs = op(*subgraph_outputs, *boundary, return_list=True)

    for old, new in zip(subgraph_outputs, new_outputs):
        new.name = old.name

    # TODO: Why not a regular fgraph.replace_all?
    fgraph.replace(rv_to_marginalize, new_outputs[0], import_missing=True)

    # The marginalized variable is no longer a model variable: drop its output
    # (if any — variables recovered by unmarginalize_fgraph have none) so the
    # marker's first output is client-less rather than dangling.
    if new_outputs[0] in fgraph.outputs:
        fgraph.remove_output(fgraph.outputs.index(new_outputs[0]))

    for i, dep in enumerate(dependent_rvs):
        ms_dep = new_outputs[1 + i]
        if isinstance(dep.owner.op, ModelValuedVar):
            fgraph.change_node_input(dep.owner, 0, ms_dep, import_missing=True)


def marginalize_fgraph(
    fg: FunctionGraph,
    rvs_to_marginalize: Sequence[TensorVariable],
    *,
    laplace_approx: dict[TensorVariable, TensorVariable] | None = None,
    minimizer_kwargs: dict | None = None,
    rewrite_query=RewriteDatabaseQuery(include=["basic"]),
) -> None:
    """Marginalize model variables of a model fgraph, in place.

    Each variable (processed in reverse topological order: clients before
    their ancestors) has the subgraph connecting it to its dependents wrapped
    behind a flat marker Op delimiting its Markov blanket — a deferred marker
    when the dependents are still markers themselves. The marginal rewrite
    database then resolves each marker into a typed MarginalRV (enumeration,
    conjugacy, Laplace, ...), inside-out. A marker no rewrite claims raises
    NotImplementedError.

    All variables (including the Qs in ``laplace_approx``) must already belong
    to ``fg``'s variable space.
    """
    laplace_approx = laplace_approx or {}
    if minimizer_kwargs is None:
        minimizer_kwargs = DEFAULT_MINIMIZER_KWARGS

    toposort = fg.toposort()

    for rv_to_marginalize in sorted(
        rvs_to_marginalize,
        key=lambda rv: toposort.index(rv.owner),
        reverse=True,
    ):
        all_rvs = [node.out for node in fg.toposort() if isinstance(node.op, ModelValuedVar)]

        dependent_rvs = find_conditional_dependent_rvs(rv_to_marginalize, all_rvs)
        if not dependent_rvs:
            continue

        # Issue warning for IntervalTransform on dependent RVs
        for dependent_rv in dependent_rvs:
            transform = dependent_rv.owner.op.transform

            if isinstance(transform, IntervalTransform) or (
                isinstance(transform, Chain)
                and any(isinstance(tr, IntervalTransform) for tr in transform.transform_list)
            ):
                warnings.warn(
                    f"The transform {transform} for the variable {dependent_rv}, which depends on the "
                    f"marginalized {rv_to_marginalize} may no longer work if bounds depended on other variables.",
                    UserWarning,
                )

        # Check that no deterministics or potentials depend on the rv to marginalize
        for node in fg.toposort():
            if isinstance(node.op, ModelDeterministic):
                if is_conditional_dependent(node.outputs[0], rv_to_marginalize, all_rvs):
                    raise NotImplementedError(
                        f"Cannot marginalize {rv_to_marginalize} due to dependent "
                        f"Deterministic {node.outputs[0]}"
                    )
            elif isinstance(node.op, ModelPotential):
                if is_conditional_dependent(node.outputs[0], rv_to_marginalize, all_rvs):
                    raise NotImplementedError(
                        f"Cannot marginalize {rv_to_marginalize} due to dependent "
                        f"Potential {node.outputs[0]}"
                    )

        marginalized_rv_input_rvs = find_conditional_input_rvs([rv_to_marginalize], all_rvs)
        other_direct_rv_ancestors = [
            rv
            for rv in find_conditional_input_rvs(dependent_rvs, all_rvs)
            if rv is not rv_to_marginalize
        ]
        input_rvs = list(dict.fromkeys((*marginalized_rv_input_rvs, *other_direct_rv_ancestors)))

        laplace_options = None
        if rv_to_marginalize in laplace_approx:
            laplace_options = {
                "Q": laplace_approx[rv_to_marginalize],
                "minimizer_kwargs": minimizer_kwargs,
            }

        _replace_marginal_subgraph(fg, rv_to_marginalize, dependent_rvs, input_rvs, laplace_options)

    rewriter = marginalize_rewrites_db.query(rewrite_query)
    rewriter.rewrite(fg)

    remaining = [node for node in fg.toposort() if isinstance(node.op, MarginalSubgraphBase)]
    for node in remaining:
        marginalized_rv = node.inputs[0]
        n_dep = node.op.n_dependent_rvs
        dependent_rvs = node.inputs[1 : 1 + n_dep]
        raise NotImplementedError(
            f"Cannot marginalize {node.outputs[0]} with distribution "
            f"{marginalized_rv.owner.op} and dependent variables "
            f"{[rv.owner.op for rv in dependent_rvs]}. "
        )


def marginalize(
    model: Model,
    rvs_to_marginalize: ModelRVs = (),
    *,
    laplace_approx: dict[TensorVariable | str, Variable] | None = None,
    minimizer_kwargs: dict | None = None,
    rewrite_query=RewriteDatabaseQuery(include=["basic"]),
) -> Model:
    """Marginalize a subset of variables in a PyMC model.

    This creates a new `Model`, with the specified variables marginalized.

    Notes
    -----
    Deterministics and Potentials cannot be conditionally dependent on the
    marginalized variables.

    Marginalization is resolved via logprob rewrites. The supported cases
    include finite discrete variables (Bernoulli, Categorical,
    DiscreteUniform, DiscreteMarkovChain) and closed-form conjugate pairs
    such as Normal-Normal.

    For finite discrete marginalization with batched dimensions, any
    conditionally dependent variables must use information from an individual
    batched dimension (i.e., the connecting graph must be strictly Elemwise).
    If you want to bypass this restriction you can separate each dimension
    of the marginalized variable into scalar components and stack them
    together. Note that such graphs will grow exponentially in the number of
    marginalized variables.

    Parameters
    ----------
    model : Model
        PyMC model to marginalize. Original variables will be cloned.
    rvs_to_marginalize : Sequence[TensorVariable]
        Variables to marginalize exactly in the returned model.
    laplace_approx : dict, optional
        Variables to marginalize via Laplace approximation, mapped to their
        precision matrix ``Q``. These need not be repeated in
        ``rvs_to_marginalize``.
    minimizer_kwargs : dict, optional
        Options forwarded to the minimizer of Laplace-marginalized variables.

    Returns
    -------
    marginal_model: Model
        Marginal model with the specified variables marginalized.

    Examples
    --------
    .. code-block:: python

        import pymc as pm
        from pymc_extras.marginal import marginalize

        with pm.Model() as m:
            p = pm.Beta("p", 1, 1)
            x = pm.Bernoulli("x", p=p, shape=(3,))
            y = pm.Normal("y", pm.math.switch(x, -10, 10), observed=[10, 10, -10])

        marginal_m = marginalize(m, [x])
        idata = pm.sample(model=marginal_m)
    """
    if isinstance(rvs_to_marginalize, str | Variable):
        rvs_to_marginalize = (rvs_to_marginalize,)

    rvs_to_marginalize = [model[rv] if isinstance(rv, str) else rv for rv in rvs_to_marginalize]

    laplace_approx = {
        (model[rv] if isinstance(rv, str) else rv): Q for rv, Q in (laplace_approx or {}).items()
    }
    rvs_to_marginalize += [rv for rv in laplace_approx if rv not in rvs_to_marginalize]

    if not rvs_to_marginalize:
        return model

    for rv_to_marginalize in rvs_to_marginalize:
        if rv_to_marginalize not in model.free_RVs:
            raise ValueError(f"Marginalized RV {rv_to_marginalize} is not a free RV in the model")

    fg, memo = fgraph_from_model(model)

    # Remap rvs and Qs (which may reference model variables) to the fgraph clones
    laplace_approx_fg = {}
    for rv, Q in laplace_approx.items():
        if not isinstance(Q, Variable):
            Q = pt.as_tensor_variable(Q)
        laplace_approx_fg[memo[rv]] = memo.get(Q, Q).copy()

    marginalize_fgraph(
        fg,
        [memo[rv] for rv in rvs_to_marginalize],
        laplace_approx=laplace_approx_fg,
        minimizer_kwargs=minimizer_kwargs,
        rewrite_query=rewrite_query,
    )

    return model_from_fgraph(fg, mutate_fgraph=True)


def unmarginalize_fgraph(fg: FunctionGraph) -> None:
    """Unmarginalize all MarginalRVs of a model fgraph, in place.

    Each MarginalRV's generative inner graph is inlined (recursively — inlining
    exposes nested MarginalRVs, which are then inlined too) and its
    marginalized variable is restored as a model free RV, with the dependents
    rewired to the raw draws.
    """
    in2out(local_unmarginalize, ignore_newtrees=False).apply(fg)


def unmarginalize(model: Model, rvs_to_unmarginalize: str | Sequence[str] | None = None) -> Model:
    """Unmarginalize a subset of variables in a PyMC model.


    Parameters
    ----------
    model : Model
        PyMC model to unmarginalize. Original variables will be cloned.
    rvs_to_unmarginalize : str or sequence of str, optional
        Variables to unmarginalize in the returned model. If None, all variables are
        unmarginalized.

    Returns
    -------
    unmarginal_model: Model
        Model with the specified variables unmarginalized.
    """

    fg, _memo = fgraph_from_model(model)

    if rvs_to_unmarginalize is None:
        unmarginalize_fgraph(fg)
        return model_from_fgraph(fg, mutate_fgraph=True)

    marginalized_rv_names = [op.marginalized_name for op in _walk_marginal_ops(fg)]
    rvs_to_unmarginalize = _resolve_marginalized_names(
        rvs_to_unmarginalize, marginalized_rv_names, "rvs_to_unmarginalize"
    )
    kept_names = [name for name in marginalized_rv_names if name not in rvs_to_unmarginalize]

    # Capture the settings of kept Laplace marginalizations before
    # unmarginalizing. Nested MarginalRVs are reached by inlining their
    # containers, which rebuilds them over fg's variables (Q included).
    kept_laplace = {}
    worklist = [node for node in fg.toposort() if isinstance(node.op, MarginalRV)]
    while worklist:
        node = worklist.pop()
        if isinstance(node.op, MarginalLaplaceRV) and node.op.marginalized_name in kept_names:
            kept_laplace[node.op.marginalized_name] = (node.inputs[-1], node.op.minimizer_kwargs)
        worklist.extend(
            {
                var.owner
                for var in ancestors(inline_ofg_outputs(node.op, node.inputs))
                if var.owner is not None and isinstance(var.owner.op, MarginalRV)
            }
        )

    # Unmarginalize everything and re-marginalize the kept variables in place
    unmarginalize_fgraph(fg)
    model_vars = {
        node.outputs[0].name: node.outputs[0]
        for node in fg.toposort()
        if isinstance(node.op, ModelValuedVar)
    }
    plain_kept = [name for name in kept_names if name not in kept_laplace]
    if plain_kept:
        marginalize_fgraph(fg, [model_vars[name] for name in plain_kept])
    for name, (Q, minimizer_kwargs) in kept_laplace.items():
        marginalize_fgraph(
            fg,
            [model_vars[name]],
            laplace_approx={model_vars[name]: Q},
            minimizer_kwargs=minimizer_kwargs,
        )

    return model_from_fgraph(fg, mutate_fgraph=True)
