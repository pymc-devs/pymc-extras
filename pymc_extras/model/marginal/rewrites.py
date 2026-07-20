from pymc.model.fgraph import ModelValuedVar, model_free_rv
from pymc.pytensorf import collect_default_updates
from pytensor.compile import SharedVariable
from pytensor.compile.mode import optdb
from pytensor.graph import Apply, Op, node_rewriter
from pytensor.graph.replace import graph_replace
from pytensor.graph.rewriting.db import EquilibriumDB, SequenceDB
from pytensor.graph.traversal import ancestors, graph_inputs

from pymc_extras.model.marginal.distributions.core import MarginalRV, inline_ofg_outputs


class MarginalSubgraphBase(Op):
    """Base for flat IR markers representing marginalized subgraphs.

    Inputs: [*subgraph_outputs, *boundary_vars]
    Outputs: [marginalized_rv, *dependent_rvs]

    The marker delimits the Markov blanket of the marginalized RV: the
    dependent RVs are its children, and the boundary contains its parents
    and the children's other parents. Given the boundary, the marginalized
    RV is conditionally independent of the rest of the model, so rewrites
    can resolve the marker locally.

    The actual subgraph lives in the fgraph between the boundary vars
    and the subgraph outputs. At rewrite time, the subgraph is cloned
    out of the fgraph to build the OpFromGraph (MarginalRV subclass).
    RNG updates are discovered at clone time, not stored on the marker.
    """

    def __init__(self, n_dependent_rvs, marginalized_name, marginalized_dims, output_types):
        self.n_dependent_rvs = n_dependent_rvs
        self.marginalized_name = marginalized_name
        self.marginalized_dims = marginalized_dims
        self.output_types = output_types
        super().__init__()

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def make_node(self, *inputs):
        outputs = [t() for t in self.output_types]
        return Apply(self, list(inputs), outputs)

    @property
    def n_subgraph_outputs(self):
        return 1 + self.n_dependent_rvs

    def split_node_inputs(self, node):
        """Split node.inputs into (subgraph_outputs, boundary)."""
        n = self.n_subgraph_outputs
        return list(node.inputs[:n]), list(node.inputs[n:])

    def perform(self, node, inputs, outputs):
        raise NotImplementedError("MarginalSubgraph should be resolved by rewrites")


DEFAULT_MINIMIZER_KWARGS = {"method": "L-BFGS-B", "optimizer_kwargs": {"tol": 1e-8}}


class MarginalSubgraph(MarginalSubgraphBase):
    """Ready-to-resolve marginalized subgraph marker."""


class LaplaceMarginalSubgraph(MarginalSubgraphBase):
    """Marginalized subgraph to be resolved via Laplace approximation.

    Created when the user calls ``marginalize(..., laplace_approx={rv: Q})``.
    The precision matrix Q of the marginalized variable is appended as the
    last boundary input; the minimizer options are stored on the marker and
    forwarded to the MarginalLaplaceRV.
    """

    def __init__(
        self,
        *args,
        minimizer_kwargs: dict = DEFAULT_MINIMIZER_KWARGS,
        **kwargs,
    ):
        self.minimizer_kwargs = minimizer_kwargs
        super().__init__(*args, **kwargs)


class AbstractDeferredMarginalSubgraph(MarginalSubgraphBase):
    """Base for markers whose dependent subgraphs still contain other markers.

    Marginalization is resolved inside-out, so a marker whose dependents come
    from unresolved markers cannot be turned into an OpFromGraph yet.
    Type-specific rewrites track only the ready marker classes and never see
    deferred ones; once the inner markers are resolved by the EquilibriumDB,
    resolve_deferred_marginal_subgraph promotes the node to its ``concrete_cls``.
    """

    concrete_cls: type[MarginalSubgraphBase]

    def concrete_kwargs(self) -> dict:
        return {
            "n_dependent_rvs": self.n_dependent_rvs,
            "marginalized_name": self.marginalized_name,
            "marginalized_dims": self.marginalized_dims,
            "output_types": self.output_types,
        }


class DeferredMarginalSubgraph(AbstractDeferredMarginalSubgraph):
    """Deferred counterpart of MarginalSubgraph."""

    concrete_cls = MarginalSubgraph


class DeferredLaplaceMarginalSubgraph(AbstractDeferredMarginalSubgraph):
    """Deferred counterpart of LaplaceMarginalSubgraph."""

    concrete_cls = LaplaceMarginalSubgraph

    def __init__(
        self,
        *args,
        minimizer_kwargs: dict = DEFAULT_MINIMIZER_KWARGS,
        **kwargs,
    ):
        self.minimizer_kwargs = minimizer_kwargs
        super().__init__(*args, **kwargs)

    def concrete_kwargs(self) -> dict:
        return {
            **super().concrete_kwargs(),
            "minimizer_kwargs": self.minimizer_kwargs,
        }


def extract_marginal_subgraph(node):
    """Extract inputs/outputs from a MarginalSubgraph node for building an OpFromGraph.

    ModelValuedVar nodes inside the subgraph were already unwrapped by
    _unwrap_subgraph_model_vars during replace_marginal_subgraph. RNG
    updates are discovered here. The OpFromGraph constructor handles cloning.

    Returns (inputs, outputs) where outputs = [marginalized_rv, *deps, *rng_updates].
    """
    subgraph_outputs, boundary = node.op.split_node_inputs(node)

    n_rvs = 1 + node.op.n_dependent_rvs
    rng_updates = collect_default_updates(
        subgraph_outputs[:n_rvs], inputs=boundary, must_be_shared=False
    )

    outputs = subgraph_outputs + list(rng_updates.values())
    return boundary, outputs


@node_rewriter(tracks=[MarginalRV])
def local_unmarginalize(fgraph, node):
    all_outputs = inline_ofg_outputs(node.op, node.inputs)
    n_dep = node.op.n_dependent_rvs
    unmarginalized_rv = all_outputs[0]
    dependent_rvs = list(all_outputs[1 : 1 + n_dep])
    rngs = list(all_outputs[1 + n_dep :])

    # Variable names are not preserved reliably through cloning/rewrites;
    # restore the model variable name from the op metadata.
    unmarginalized_rv.name = node.op.marginalized_name
    value = unmarginalized_rv.clone()
    transform = None
    unmarginalized_free_rv = model_free_rv(
        unmarginalized_rv, value, transform, node.op.marginalized_name, *node.op.marginalized_dims
    )

    # Restore the model-variable output that was dropped when the variable was
    # marginalized, so the recovered RV survives even with no dependent clients.
    # import_missing imports the new value variable as an input.
    fgraph.add_output(unmarginalized_free_rv, reason="unmarginalize", import_missing=True)

    # Pin already-built model-var wrappers (opaque ModelValuedVar) as boundaries so
    # graph_replace does not clone their subgraphs — otherwise a shared upstream RV
    # they wrap (e.g. a previously unmarginalized parent) gets duplicated.
    pinned = {
        a: a
        for a in ancestors(dependent_rvs)
        if a.owner is not None and isinstance(a.owner.op, ModelValuedVar)
    }
    dependent_rvs = graph_replace(
        dependent_rvs, {**pinned, unmarginalized_rv: unmarginalized_free_rv}, strict=False
    )

    return [unmarginalized_free_rv, *dependent_rvs, *rngs]


marginal_rewrites_db = EquilibriumDB()
marginal_rewrites_db.name = "marginal_rewrites_db"
# The strategy-specific rewrites (finite discrete, Laplace, Normal-Normal)
# live next to their MarginalRV subclasses in ``distributions/`` and register
# themselves here on import.

# Canonicalize the marker subgraphs (flattening Add/Mul, folding constants, ...)
# before resolving them, mirroring pymc.logprob's pre-canonicalize -> IR sequence.
# The structure detectors (e.g. affine_coefficients) can then assume canonical
# graphs instead of re-implementing canonicalization.
marginalize_rewrites_db = SequenceDB()
marginalize_rewrites_db.name = "marginalize_rewrites_db"
marginalize_rewrites_db.register(
    "pre-canonicalize",
    optdb.query("+canonicalize", "-local_eager_useless_unbatched_blockwise"),
    "basic",
    position=1,
)
marginalize_rewrites_db.register(
    "marginal_ir_rewrites",
    marginal_rewrites_db,
    "basic",
    position=2,
)


@node_rewriter(tracks=[MarginalSubgraph, LaplaceMarginalSubgraph])
def remarginalize_absorbed_dependent(fgraph, node):
    """Re-nest a marginalization whose target was absorbed by an earlier marginalize() call.

    Marginalization is resolved inside-out: dependents are marginalized before
    the variables they depend on. Within a single marginalize() call this
    ordering holds by construction, but a later call can target a variable
    that an earlier call already absorbed as a dependent into a resolved
    MarginalRV — committing the two marginalizations in the wrong order.

    This rewrite restores the inside-out order. It inlines the earlier
    MarginalRV and rebuilds both marginalizations as nested markers: an inner
    marker for this node's target and an outer marker re-marginalizing the
    earlier variable. Each marker preserves the type and settings of the
    marginalization it stands for (e.g. Laplace Q and minimizer options), so
    successive marginalize() calls with different settings compose.
    """
    subgraph_outputs, boundary = node.op.split_node_inputs(node)
    marginalized_rv = subgraph_outputs[0]
    outer_dep_outputs = subgraph_outputs[1:]

    if not (marginalized_rv.owner and isinstance(marginalized_rv.owner.op, MarginalRV)):
        return None

    marg_rv_node = marginalized_rv.owner
    marg_rv_op = marg_rv_node.op

    # Inline the MarginalRV to get raw variables
    inlined = inline_ofg_outputs(marg_rv_op, marg_rv_node.inputs)
    inlined_marginalized = inlined[0]

    # Map the node's subgraph outputs to the inlined raw variables. The target
    # is an output of the earlier MarginalRV; each dependent either is one too,
    # or lives outside it and is rebuilt over the inlined variables.
    out_to_inlined = dict(zip(marg_rv_node.outputs, inlined))
    target_inlined = out_to_inlined[marginalized_rv]
    deps_inlined = [
        out_to_inlined[d]
        if d in out_to_inlined
        else graph_replace([d], replace=out_to_inlined, strict=False)[0]
        for d in outer_dep_outputs
    ]

    def _shared_boundary(outputs, base_boundary):
        return base_boundary + [
            inp
            for inp in graph_inputs(outputs, blockers=base_boundary)
            if isinstance(inp, SharedVariable) and inp not in base_boundary
        ]

    # The current node's settings carry over to the inner marker. For Laplace,
    # Q is the last boundary input and belongs to this node's target.
    inner_op_kwargs = {}
    inner_extra_boundary = []
    if isinstance(node.op, LaplaceMarginalSubgraph):
        inner_extra_boundary.append(boundary.pop())
        inner_op_kwargs = {
            "minimizer_kwargs": node.op.minimizer_kwargs,
        }

    # Inner marker: marginalize the target variable (e.g. sub_idx), deps are outer deps
    # Compute boundary from scratch — only shared vars actually used by this subgraph.
    # Block inlined_marginalized so idx's RNG doesn't leak into the inner boundary.
    inner_subgraph = [target_inlined, *deps_inlined]
    inner_boundary = _shared_boundary(inner_subgraph, [inlined_marginalized])
    inner_boundary += inner_extra_boundary
    inner_ms = type(node.op)(
        n_dependent_rvs=len(deps_inlined),
        marginalized_name=node.op.marginalized_name,
        marginalized_dims=node.op.marginalized_dims,
        output_types=[o.type for o in inner_subgraph],
        **inner_op_kwargs,
    )
    inner_outs = inner_ms(*(inner_subgraph + inner_boundary))
    if not isinstance(inner_outs, list):
        inner_outs = list(inner_outs)

    # The earlier marginalization's settings carry over to the outer marker,
    # which is deferred until the inner marker resolves. For Laplace, Q was
    # kept as the (dummy) last input of the MarginalRV node.
    # Local import: laplace.py imports this module to register its rewrite.
    from pymc_extras.model.marginal.distributions.laplace import MarginalLaplaceRV

    if isinstance(marg_rv_op, MarginalLaplaceRV):
        outer_cls = DeferredLaplaceMarginalSubgraph
        outer_op_kwargs = {
            "minimizer_kwargs": marg_rv_op.minimizer_kwargs,
        }
        outer_q = marg_rv_node.inputs[-1]
    else:
        outer_cls = DeferredMarginalSubgraph
        outer_op_kwargs = {}
        outer_q = None

    # Outer marker: marginalize the previously-marginalized variable (e.g. idx).
    # Use original boundary (not inner_boundary) so inlined_marginalized stays internal
    outer_subgraph = [inlined_marginalized, *inner_outs[1:]]
    outer_boundary = _shared_boundary(outer_subgraph, list(boundary))
    if outer_q is not None:
        if outer_q in outer_boundary:
            outer_boundary.remove(outer_q)
        outer_boundary.append(outer_q)
    outer_ms = outer_cls(
        n_dependent_rvs=len(deps_inlined),
        marginalized_name=marg_rv_op.marginalized_name,
        marginalized_dims=marg_rv_op.marginalized_dims,
        output_types=[o.type for o in outer_subgraph],
        **outer_op_kwargs,
    )
    outer_outs = outer_ms(*(outer_subgraph + outer_boundary))
    if not isinstance(outer_outs, list):
        outer_outs = list(outer_outs)

    # The node's dependent draws map to the outer marker's dependent outputs.
    # The target draw output is client-less (the marginalized draw is dropped
    # from the fgraph outputs at marker-creation time); inner_outs[0] stands in
    # as a type-correct replacement without being added to the graph.
    return [inner_outs[0], *outer_outs[1 : len(node.outputs)]]


marginal_rewrites_db.register(
    "remarginalize_absorbed_dependent", remarginalize_absorbed_dependent, "basic"
)


@node_rewriter(tracks=[AbstractDeferredMarginalSubgraph])
def resolve_deferred_marginal_subgraph(fgraph, node):
    """Promote a deferred marker to its concrete class once inner markers resolve.

    The EquilibriumDB resolves inner markers first (they live in the same
    fgraph). Once none of this node's inputs come from a marker, the node is
    rebuilt as its ``concrete_cls`` so the type-specific rewrites can fire.
    """
    for inp in node.inputs:
        if inp.owner is not None and isinstance(inp.owner.op, MarginalSubgraphBase):
            return None

    op = node.op
    resolved_op = op.concrete_cls(**op.concrete_kwargs())
    new_outputs = resolved_op(*node.inputs)
    if not isinstance(new_outputs, list):
        new_outputs = list(new_outputs)
    return new_outputs


marginal_rewrites_db.register(
    "resolve_deferred_marginal_subgraph",
    resolve_deferred_marginal_subgraph,
    "basic",
)
