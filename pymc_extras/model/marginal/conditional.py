import warnings

from collections.abc import Sequence

from pymc.model import Model, modelcontext
from pymc.model.fgraph import (
    ModelObservedRV,
    ModelValuedVar,
    fgraph_from_model,
    model_free_rv,
    model_from_fgraph,
)
from pymc.sampling.forward import sample_posterior_predictive
from pymc.util import RandomState
from pytensor.graph import FunctionGraph, Variable
from pytensor.graph.replace import graph_replace
from xarray import DataTree

from pymc_extras.model.marginal.distributions.core import (
    MarginalRV,
    marginalized_conditional,
)
from pymc_extras.model.marginal.marginalize import (
    _resolve_marginalized_names,
    _walk_marginal_ops,
    marginalize_fgraph,
    unmarginalize_fgraph,
)


def _find_marg_rv(fg, var_name):
    """Find the MarginalRV in ``fg`` whose marginalized variable is ``var_name``."""
    for node in fg.toposort():
        if isinstance(node.op, MarginalRV) and node.op.marginalized_name == var_name:
            return node
    return None


def _model_var_of(fg, rv_output):
    """The fg model variable wrapping ``rv_output`` (or its data, if observed)."""
    mv_client = next(
        (c for c, _ in fg.clients[rv_output] if isinstance(c.op, ModelValuedVar)), None
    )
    if mv_client is None:
        raise ValueError(f"No model variable found wrapping dependent output {rv_output}")
    if isinstance(mv_client.op, ModelObservedRV):
        return mv_client.inputs[1]
    return mv_client.outputs[0]


def conditional_fgraph(
    fg: FunctionGraph,
    var_names_to_recover: Sequence[str],
) -> None:
    """Add conditional distributions for marginalized variables to a model fgraph, in place.

    The base stays the marginal model — dependents keep their marginal
    distribution via the MarginalRV, so the conditional can reference them
    without cycles (chain-rule factorization of the joint posterior).

    For each variable to recover (outermost first), its MarginalRV is located
    in ``fg`` — or, when an outer marginalization absorbed it, rebuilt on a
    scratch clone via unmarginalize_fgraph + marginalize_fgraph — and
    ``marginalized_conditional`` is dispatched on it to build
    ``p(var | dependents, recovered vars)``, added to ``fg`` as a new free RV.
    Variables recovered only to complete the chain rule are re-marginalized
    at the end.
    """
    marginalized_rv_names = [op.marginalized_name for op in _walk_marginal_ops(fg)]

    # Check if all requested vars can be found directly in fg.
    # If any are nested, recover ALL vars via the chain-rule, then
    # re-marginalize the unwanted ones (the chain-rule model IS the joint
    # posterior, so Σ_unwanted p(all|y) = p(kept|y)).
    all_direct = all(_find_marg_rv(fg, name) is not None for name in var_names_to_recover)
    vars_to_add = var_names_to_recover if all_direct else list(marginalized_rv_names)

    recovered: dict[str, Variable] = {}
    for var_name in vars_to_add:
        marg_node = _find_marg_rv(fg, var_name)
        source_fg = fg
        remap = {}
        if marg_node is None:
            # var_name was absorbed inside another MarginalRV (joint
            # marginalization), so there is no node to dispatch on. Rebuild one
            # on a scratch clone of the model IR: unmarginalize everything and
            # re-marginalize var_name together with the not-yet-recovered
            # variables. Those are all downstream (recovery is outermost-first),
            # so they nest inside var_name's MarginalRV and are integrated out,
            # while the recovered ones are left as conditioning free RVs — the
            # chain-rule factor p(var_name | recovered, dependents).
            source_fg, equiv = fg.clone_get_equiv(attach_feature=False)
            unmarginalize_fgraph(source_fg)
            clones = set(equiv.values())
            unmarginalized_vars = {
                node.outputs[0].name: node.outputs[0]
                for node in source_fg.toposort()
                if isinstance(node.op, ModelValuedVar) and node.outputs[0] not in clones
            }
            marginalize_fgraph(
                source_fg,
                [var for name, var in unmarginalized_vars.items() if name not in recovered],
            )
            marg_node = _find_marg_rv(source_fg, var_name)

            # Map the scratch variables home: clones back to the fg originals,
            # and free RVs created by unmarginalize to their recovered conditionals.
            remap = {clone: orig for orig, clone in equiv.items() if isinstance(orig, Variable)}
            for node in source_fg.toposort():
                if isinstance(node.op, ModelValuedVar) and node.outputs[0] not in remap:
                    name = node.outputs[0].name
                    if name not in recovered:
                        raise NotImplementedError(
                            f"Cannot build conditional for '{var_name}': it requires "
                            f"'{name}', which was neither recovered nor re-marginalized."
                        )
                    remap[node.outputs[0]] = recovered[name]

        op = marg_node.op
        inputs = list(marg_node.inputs)
        dep_rvs = [
            _model_var_of(source_fg, dep_output)
            for dep_output in marg_node.outputs[1 : 1 + op.n_dependent_rvs]
        ]
        if remap:
            replaced = graph_replace([*inputs, *dep_rvs], replace=remap, strict=False)
            inputs, dep_rvs = replaced[: len(inputs)], replaced[len(inputs) :]

        sample_graph = marginalized_conditional(op, inputs, dep_rvs)

        # Add the conditional as a new free RV. import_missing imports its value
        # variable and any new shared RNGs (e.g. from Categorical.dist) as inputs.
        sample_graph.name = var_name
        value = sample_graph.clone()
        conditional_free_rv = model_free_rv(
            sample_graph, value, None, var_name, *op.marginalized_dims
        )
        fg.add_output(conditional_free_rv, reason="conditionalize", import_missing=True)
        recovered[var_name] = conditional_free_rv

    # Re-marginalize vars that were recovered only for the chain-rule
    # but weren't requested by the user.
    vars_to_remarginalize = [n for n in vars_to_add if n not in var_names_to_recover]
    if vars_to_remarginalize:
        marginalize_fgraph(fg, [recovered[name] for name in vars_to_remarginalize])


def conditional(
    model: Model,
    rvs_to_recover: str | Sequence[str] | None = None,
) -> Model:
    """Replace marginalized variables with their conditional distributions.

    Returns a new model where the specified marginalized variables become
    free RVs whose distributions are their conditionals given the dependents.
    Unspecified marginalized variables stay marginalized (integrated out).

    The returned model can be used with ``pm.sample_posterior_predictive``
    to draw conditional posterior samples, or with ``model.compile_logp``
    to evaluate conditional log-probabilities.

    The input is a marginalized model. Starting from an original model
    factored as ``p(mu) * p(x|mu) * p(y|x)``, marginalizing ``x`` yields
    ``p(mu) * p(y|mu)``. ``conditional`` adds ``x`` back as its conditional
    distribution, giving ``p(mu) * p(y|mu) * p(x|y, mu)`` -- a re-factorization
    of the same full joint ``p(mu, x, y)``: the recovered variable follows the
    conditional ``p(x|y, mu)``, while each dependent stays marginalized over it.

    Selecting variables matters when evaluating logp:
    ``model.compile_logp(vars=[model["x"]])`` gives the conditional
    ``p(x|y, mu)``, while the unqualified ``model.compile_logp()`` is the full
    joint ``p(mu, x, y)``.

    Parameters
    ----------
    model : Model
        PyMC model with marginalized variables.
    rvs_to_recover : str, sequence of str, or None
        Marginalized variables to recover. Defaults to all.

    Returns
    -------
    Model
        Model with the specified variables as free RVs with conditional
        distributions.

    Examples
    --------
    **Basic usage** — recover a marginalized variable:

    .. code-block:: python

        import pymc as pm
        from pymc_extras.marginal import marginalize, conditional

        with pm.Model() as m:
            p = pm.Beta("p", 1, 1)
            idx = pm.Bernoulli("idx", p=p, shape=(3,))
            y = pm.Normal("y", pm.math.switch(idx, -10, 10), observed=[10, 10, -10])

        marginal_m = marginalize(m, [idx])
        idata = pm.sample(model=marginal_m)

        # Get model with idx's conditional posterior as its distribution
        cond_m = conditional(marginal_m)
        logp_fn = cond_m.compile_logp(vars=[cond_m["idx"]])
        pm.sample_posterior_predictive(idata, model=cond_m, sample_vars=["idx"])

    **Nested marginalization** — recover a subset (marginal posterior):

    When multiple variables are marginalized, specifying a subset recovers
    those variables with the others integrated out (marginal posterior).

    .. code-block:: python

        with pm.Model() as m:
            idx = pm.Bernoulli("idx", p=0.5)
            sub_idx = pm.Bernoulli("sub_idx", p=f(idx))
            y = pm.Normal("y", mu=idx + sub_idx, sigma=1)

        marginal_m = marginalize(m, ["idx", "sub_idx"])

        # Marginal posterior of idx (sub_idx integrated out):
        # P(idx | y, σ) = Σ_sub_idx P(idx, sub_idx | y, σ)
        cond_idx = conditional(marginal_m, "idx")

        # Marginal posterior of sub_idx (idx integrated out):
        # P(sub_idx | y, σ) = Σ_idx P(idx, sub_idx | y, σ)
        cond_sub = conditional(marginal_m, "sub_idx")

    **Recovering all nested variables** — joint posterior factorization:

    When recovering all marginalized variables at once, the joint
    posterior is factored via the chain rule in recovery order.  Each
    variable integrates out the not-yet-recovered ones and conditions
    on the already-recovered ones:

    .. code-block:: python

        # P(idx, sub_idx | y) = P(idx | y) · P(sub_idx | idx, y)
        cond_all = conditional(marginal_m)

        # idx's logp does NOT depend on sub_idx (sub_idx is integrated out):
        logp_idx = cond_all.compile_logp(vars=[cond_all["idx"]])

        # sub_idx's logp depends on idx:
        logp_sub = cond_all.compile_logp(vars=[cond_all["sub_idx"]])

    The result is a valid generative DAG — draw exact joint posterior
    samples by forward-sampling through it.

    **Full conditional via unmarginalize:**

    To get the full conditional ``P(idx | sub_idx, y)`` (conditioning
    on ``sub_idx`` rather than integrating it out), first unmarginalize
    ``sub_idx`` so it becomes a free RV with its original prior, then
    conditionalize ``idx``:

    .. code-block:: python

        from pymc_extras.marginal import unmarginalize

        partial_m = unmarginalize(marginal_m, "sub_idx")
        cond_full = conditional(partial_m, "idx")
        # User must provide sub_idx values when evaluating
    """
    fg, _memo = fgraph_from_model(model)
    marginalized_rv_names = [op.marginalized_name for op in _walk_marginal_ops(fg)]
    var_names_to_recover = _resolve_marginalized_names(
        rvs_to_recover, marginalized_rv_names, "rvs_to_recover"
    )

    if not var_names_to_recover:
        return model

    conditional_fgraph(fg, var_names_to_recover)
    return model_from_fgraph(fg, mutate_fgraph=True)


def recover(
    idata: DataTree,
    *,
    model: Model | None = None,
    var_names: Sequence[str] | None = None,
    extend_inferencedata: bool = True,
    random_seed: RandomState = None,
):
    """Sample marginalized variables from their conditional posterior.

    Builds the chain-rule factorization of the joint posterior via
    :func:`conditional` and forward-samples all recovered variables
    together.  For more control, use :func:`conditional` directly.

    Parameters
    ----------
    idata : DataTree
        DataTree with posterior group.
    model : Model, optional
        PyMC model with marginalized variables.
    var_names : sequence of str, optional
        Variables to recover. Defaults to all marginalized variables.
    extend_inferencedata : bool, default True
        Whether to extend the original DataTree or return a new Dataset.
    random_seed : int, array-like of int or SeedSequence, optional
        Seed for generating samples.

    Returns
    -------
    idata : DataTree or Dataset
        DataTree with recovered samples added to posterior, or a new Dataset.

    Examples
    --------
    .. code-block:: python

        import pymc as pm
        from pymc_extras.marginal import marginalize, recover

        with pm.Model() as m:
            p = pm.Beta("p", 1, 1)
            x = pm.Bernoulli("x", p=p, shape=(3,))
            y = pm.Normal("y", pm.math.switch(x, -10, 10), observed=[10, 10, -10])

        marginal_m = marginalize(m, [x])
        idata = pm.sample(model=marginal_m)
        recover(idata, model=marginal_m)
    """
    if isinstance(idata, Model):
        raise TypeError(
            "The order of arguments of `recover` changed. The first input must be an idata"
        )

    model = modelcontext(model)

    # Build a single conditional model recovering all requested variables
    # via the chain-rule factorization.  This handles nested variables
    # correctly (each conditions on the already-recovered ones and
    # integrates out the not-yet-recovered ones).  Sample all recovered
    # variables together so the chain-rule dependencies are satisfied
    # (e.g. sub_idx's conditional uses idx's sampled value).
    cond_model = conditional(model, var_names)

    # The recovered variables are the free RVs that weren't free before
    base_names = {rv.name for rv in model.free_RVs}
    var_names_to_recover = [rv.name for rv in cond_model.free_RVs if rv.name not in base_names]
    if not var_names_to_recover:
        return idata

    freeze = [rv.name for rv in cond_model.free_RVs if rv.name not in var_names_to_recover]

    sample_result = sample_posterior_predictive(
        idata,
        model=cond_model,
        sample_vars=var_names_to_recover,
        freeze_vars=freeze,
        random_seed=random_seed,
        progressbar=False,
    )
    pp = sample_result.posterior_predictive
    pp_ds = pp.dataset if isinstance(pp, DataTree) else pp
    rv_dataset = pp_ds[var_names_to_recover]

    if extend_inferencedata:
        idata["posterior"] = idata["posterior"].assign(rv_dataset)
        return idata
    else:
        return rv_dataset


def recover_marginals(*args, return_samples: bool = True, **kwargs):
    """Deprecated alias for :func:`recover`.

    .. deprecated::
        ``recover_marginals`` has been renamed to :func:`recover` (available as
        ``pymc_extras.marginal.recover``).  Unlike the old implementation, it no
        longer returns the posterior log-probabilities of the marginalized
        variables (the ``lp_*`` arrays / ``return_samples=False`` mode); use
        :func:`conditional` together with ``Model.compile_logp`` to evaluate
        those instead.
    """
    warnings.warn(
        "`recover_marginals` has been renamed to `recover` and moved to the "
        "`pymc_extras.marginal` namespace (`pymc_extras.marginal.recover`).",
        FutureWarning,
        stacklevel=2,
    )
    if not return_samples:
        raise NotImplementedError(
            "`recover` no longer returns posterior log-probabilities of the "
            "marginalized variables. Use `conditional(...)` with "
            "`Model.compile_logp` to evaluate them instead."
        )
    return recover(*args, **kwargs)
