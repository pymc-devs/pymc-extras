import warnings

from collections.abc import Sequence

import numpy as np
import pytensor.tensor as pt

from pymc.distributions import Bernoulli, Categorical, DiscreteUniform
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import conditional_logp
from pymc.pytensorf import constant_fold
from pytensor.compile.mode import Mode
from pytensor.graph import Op, node_rewriter, vectorize_graph
from pytensor.graph.replace import graph_replace
from pytensor.scan import map as scan_map
from pytensor.tensor import TensorVariable

from pymc_extras.distributions import DiscreteMarkovChain
from pymc_extras.model.marginal.dims import output_dims_of
from pymc_extras.model.marginal.distributions.core import (
    MarginalRV,
    inline_ofg_outputs,
    marginalized_conditional,
)
from pymc_extras.model.marginal.graph_analysis import (
    get_support_axes,
    subgraph_batch_dim_connection,
)
from pymc_extras.model.marginal.rewrites import (
    MarginalSubgraph,
    extract_marginal_subgraph,
    finalize_marginal_rv,
    marginal_rewrites_db,
)


class EnumerableMarginalRV(MarginalRV):
    """Base class for enumerable Marginalized RVs with closed-form logp."""

    def __init__(
        self,
        *args,
        dims_connections: tuple[tuple[int | None], ...],
        **kwargs,
    ) -> None:
        self.dims_connections = dims_connections
        super().__init__(*args, **kwargs)

    @property
    def support_axes(self) -> tuple[tuple[int]]:
        """Dimensions of dependent RVs that belong to the core (non-batched) marginalized variable."""
        marginalized_ndim_supp = self.inner_outputs[0].owner.op.ndim_supp
        support_axes_vars = []
        for dims_connection in self.dims_connections:
            ndim = len(dims_connection)
            marginalized_supp_axes = ndim - marginalized_ndim_supp
            support_axes_vars.append(
                tuple(
                    -i
                    for i, dim in enumerate(reversed(dims_connection), start=1)
                    if (dim is None or dim > marginalized_supp_axes)
                )
            )
        return tuple(support_axes_vars)


class NonSeparableLogpWarning(UserWarning):
    pass


def warn_non_separable_logp(values):
    if len(values) > 1:
        warnings.warn(
            "There are multiple dependent variables in a FiniteDiscreteMarginalRV. "
            f"Their joint logp terms will be assigned to the first value: {values[0]}.",
            NonSeparableLogpWarning,
            stacklevel=2,
        )


def dummy_logps(op, values) -> tuple[TensorVariable, ...]:
    """Zero placeholders for the values whose density was folded into the first logp term.

    The joint logp of a MarginalRV cannot be split across its dependents, so it is all assigned
    to the first value and the rest get a placeholder. Each carries the shape a real term would
    have -- the value's shape minus the axes its logp reduces -- rather than a bare scalar, so
    that callers can reason about the term's shape (and, for dims models, label its dims)
    without special-casing the placeholder.
    """
    if len(values) < 2:
        return ()

    placeholders = []
    for value, supp_axes in zip(values[1:], get_support_axes(op)[1:]):
        ndim = value.type.ndim
        kept = [i for i in range(ndim) if (i - ndim) not in supp_axes]
        placeholders.append(pt.zeros([value.shape[i] for i in kept], dtype=value.type.dtype))
    return tuple(placeholders)


def align_logp_dims(dims: tuple[int | None, ...], logp: TensorVariable) -> TensorVariable:
    """Align the logp with the order specified in dims."""
    dims_alignment = [dim for dim in dims if dim is not None]
    return logp.transpose(*dims_alignment)


class MarginalFiniteDiscreteRV(EnumerableMarginalRV):
    """Base class for Marginalized Finite Discrete RVs"""


def get_domain_of_finite_discrete_rv(rv: TensorVariable) -> tuple[int, ...]:
    op = rv.owner.op
    dist_params = rv.owner.op.dist_params(rv.owner)
    if isinstance(op, Bernoulli):
        return (0, 1)
    elif isinstance(op, Categorical):
        [p_param] = dist_params
        [p_param_length] = constant_fold([p_param.shape[-1]])
        return tuple(range(p_param_length))
    elif isinstance(op, DiscreteUniform):
        lower, upper = constant_fold(dist_params)
        return tuple(np.arange(lower, upper + 1))
    elif isinstance(op, DiscreteMarkovChain):
        P, *_ = dist_params
        return tuple(range(pt.get_vector_length(P[-1])))

    raise NotImplementedError(f"Cannot compute domain for op {op}")


def reduce_batch_dependent_logps(
    dependent_dims_connections: Sequence[tuple[int | None, ...]],
    dependent_ops: Sequence[Op],
    dependent_logps: Sequence[TensorVariable],
) -> TensorVariable:
    """Combine the logps of dependent RVs and align them with the marginalized logp.

    This requires reducing extra batch dims and transposing when they are not aligned.

       idx = pm.Bernoulli(idx, shape=(3, 2))  # 0, 1
       pm.Normal("dep1", mu=idx.T[..., None] * 2, shape=(3, 2, 5))
       pm.Normal("dep2", mu=idx * 2, shape=(7, 2, 3))

       marginalize(idx)

       The marginalized op will have dims_connections = [(1, 0, None), (None, 0, 1)]
       which tells us we need to reduce the last axis of dep1 logp and the first of dep2 logp,
       as well as transpose the remaining axis of dep1 logp before adding the two element-wise.

    """
    reduced_logps = []
    for dependent_op, dependent_logp, dependent_dims_connection in zip(
        dependent_ops, dependent_logps, dependent_dims_connections
    ):
        if dependent_logp.type.ndim > 0:
            # Find which support axis implied by the MarginalRV need to be reduced
            # Some may have already been reduced by the logp expression of the dependent RV (e.g., multivariate RVs)
            dep_supp_axes = get_support_axes(dependent_op)[0]

            # Dependent RV support axes are already collapsed in the logp, so we ignore them.
            # The axes that remain must also be renumbered: they are counted against the
            # dependent RV, but the logp no longer has the collapsed ones, so each support axis
            # to the right of an axis shifts it one step towards zero.
            supp_axes = [
                -(i - sum(1 for supp_axis in dep_supp_axes if supp_axis > -i))
                for i, dim in enumerate(reversed(dependent_dims_connection), start=1)
                if (dim is None and -i not in dep_supp_axes)
            ]
            dependent_logp = dependent_logp.sum(supp_axes)

            # Finally, we need to align the dependent logp batch dimensions with the marginalized logp
            dims_alignment = [dim for dim in dependent_dims_connection if dim is not None]
            dependent_logp = dependent_logp.transpose(*dims_alignment)

        reduced_logps.append(dependent_logp)

    reduced_logp = pt.add(*reduced_logps)
    return reduced_logp


@_logprob.register(MarginalFiniteDiscreteRV)
def finite_discrete_marginal_rv_logp(op: MarginalFiniteDiscreteRV, values, *inputs, **kwargs):
    # Clone the inner RV graph of the Marginalized RV
    all_outputs = inline_ofg_outputs(op, inputs)
    marginalized_rv = all_outputs[0]
    inner_rvs = list(all_outputs[1 : 1 + op.n_dependent_rvs])

    # Obtain the joint_logp graph of the inner RV graph
    # strict: a caller that provides fewer values than there are dependents would
    # otherwise silently get the joint density of a subset of them
    inner_rv_values = dict(zip(inner_rvs, values, strict=True))
    marginalized_vv = marginalized_rv.clone()
    rv_values = inner_rv_values | {marginalized_rv: marginalized_vv}
    logps_dict = conditional_logp(rv_values=rv_values, **kwargs)

    # Reduce logp dimensions corresponding to broadcasted variables
    marginalized_logp = logps_dict.pop(marginalized_vv)
    joint_logp = marginalized_logp + reduce_batch_dependent_logps(
        dependent_dims_connections=op.dims_connections,
        dependent_ops=[inner_rv.owner.op for inner_rv in inner_rvs],
        dependent_logps=[logps_dict[value] for value in values],
    )

    # Compute the joint_logp for all possible n values of the marginalized RV. We assume
    # each original dimension is independent so that it suffices to evaluate the graph
    # n times, once with each possible value of the marginalized RV replicated across
    # batched dimensions of the marginalized RV

    # PyMC does not allow RVs in the logp graph, even if we are just using the shape
    marginalized_rv_shape = constant_fold(tuple(marginalized_rv.shape), raise_not_constant=False)
    marginalized_rv_domain = get_domain_of_finite_discrete_rv(marginalized_rv)
    marginalized_rv_domain_tensor = pt.moveaxis(
        pt.full(
            (*marginalized_rv_shape, len(marginalized_rv_domain)),
            marginalized_rv_domain,
            dtype=marginalized_rv.dtype,
        ),
        -1,
        0,
    )

    try:
        joint_logps = vectorize_graph(
            joint_logp, replace={marginalized_vv: marginalized_rv_domain_tensor}
        )
    except Exception:
        # Fallback to Scan
        def logp_fn(marginalized_rv_const, *non_sequences):
            return graph_replace(joint_logp, replace={marginalized_vv: marginalized_rv_const})

        joint_logps = scan_map(
            fn=logp_fn,
            sequences=marginalized_rv_domain_tensor,
            non_sequences=[*values, *inputs],
            mode=Mode().including("local_remove_check_parameter"),
            return_updates=False,
        )

    joint_logp = pt.logsumexp(joint_logps, axis=0)

    # Align logp with non-collapsed batch dimensions of first RV
    joint_logp = align_logp_dims(dims=op.dims_connections[0], logp=joint_logp)

    warn_non_separable_logp(values)
    # We have to add dummy logps for the remaining value variables, otherwise PyMC will raise
    return joint_logp, *dummy_logps(op, values)


@marginalized_conditional.register(MarginalFiniteDiscreteRV)
def finite_discrete_marginalized_conditional(op, inputs, dep_rvs):
    # The logp must be derived over root placeholders, not the real
    # inputs/dep_rvs: conditional_logp clones the rv graphs (leaking clones
    # of named upstream variables into the result), and dep_rvs have other
    # random variables in their ancestry, which trips the "RVs in logp graph"
    # warning in the conditional_logp calls that nested MarginalRV logps
    # perform internally (warn_rvs cannot be forwarded there). Work on the
    # inner (nominal) graph with value dummies and substitute the real
    # variables once at the end.
    # inner_inputs/inner_outputs are frozen (immutable) views; conditional_logp
    # and graph_replace below need mutable nodes, so work on an unfrozen copy.
    inner_graph = op.fgraph.unfreeze()
    inner_inputs = inner_graph.inputs
    marginalized = inner_graph.outputs[0]
    dependents = list(inner_graph.outputs[1 : 1 + op.n_dependent_rvs])

    marginalized_value = marginalized.clone()
    dep_dummies = [dep.type() for dep in dependents]
    rvs_to_values = {marginalized: marginalized_value}
    rvs_to_values.update(zip(dependents, dep_dummies, strict=True))

    logps_dict = conditional_logp(rvs_to_values)
    marginalized_logp = logps_dict[marginalized_value]
    dependent_logps = [logps_dict[dummy] for dummy in dep_dummies]

    joint_logp = marginalized_logp + reduce_batch_dependent_logps(
        op.dims_connections,
        [dep.owner.op for dep in dependents],
        dependent_logps,
    )

    rv_shape = constant_fold(tuple(marginalized.shape), raise_not_constant=False)
    rv_domain = get_domain_of_finite_discrete_rv(marginalized)
    rv_domain_tensor = pt.moveaxis(
        pt.full(
            (*rv_shape, len(rv_domain)),
            rv_domain,
            dtype=marginalized.dtype,
        ),
        -1,
        0,
    )

    batched_joint_logp = vectorize_graph(
        joint_logp,
        replace={marginalized_value: rv_domain_tensor},
    )
    batched_joint_logp = pt.moveaxis(batched_joint_logp, 0, -1)

    sample_graph = Categorical.dist(logit_p=batched_joint_logp)
    if isinstance(marginalized.owner.op, DiscreteUniform):
        # rv_domain[0] is folded to a float; adding it directly would insert a
        # Cast{float64} that breaks logp derivation. Keep the offset integral and
        # matching the marginalized dtype so the conditional stays loggable.
        sample_graph += rv_domain[0].astype(marginalized.dtype)

    replacements = dict(zip(inner_inputs, inputs, strict=True))
    replacements.update(zip(dep_dummies, dep_rvs, strict=True))
    [sample_graph] = graph_replace([sample_graph], replace=replacements, strict=False)
    return sample_graph


def build_enumerable_marginal_rv(node, inputs, outer_inputs, outputs, constructor):
    """Build an :class:`EnumerableMarginalRV` of type ``constructor`` from a marginal subgraph.

    Shared by the per-distribution rewriters (e.g. finite discrete and DiscreteMarkovChain).
    Computes the dependent-RV dim connections, instantiates the typed op, and returns the
    replacement outptus aligned with ``node``.
    """
    op = node.op
    n_dep = op.n_dependent_rvs
    marginalized_rv = outputs[0]

    try:
        dependent_rvs_dim_connections = subgraph_batch_dim_connection(
            marginalized_rv, outputs[1 : 1 + n_dep]
        )
    except (ValueError, NotImplementedError) as e:
        raise type(e)(
            "The graph between the marginalized and dependent RVs cannot be marginalized efficiently. "
            "You can try splitting the marginalized RV into separate components and marginalizing "
            f"them separately. {e}"
        ) from e

    typed_op = constructor(
        inputs=inputs,
        outputs=outputs,
        dims_connections=dependent_rvs_dim_connections,
        marginalized_name=op.marginalized_name,
        marginalized_dims=op.marginalized_dims,
        n_dependent_rvs=n_dep,
        output_dims=output_dims_of(node),
    )
    return finalize_marginal_rv(node, typed_op, outer_inputs)


@node_rewriter(tracks=[MarginalSubgraph])
def finite_discrete_marginal(fgraph, node):
    inputs, outer_inputs, outputs = extract_marginal_subgraph(node)
    marginalized_rv_op = outputs[0].owner.op
    if not isinstance(marginalized_rv_op, Bernoulli | Categorical | DiscreteUniform):
        return None
    return build_enumerable_marginal_rv(
        node, inputs, outer_inputs, outputs, MarginalFiniteDiscreteRV
    )


marginal_rewrites_db.register("finite_discrete_marginal", finite_discrete_marginal, "basic")
