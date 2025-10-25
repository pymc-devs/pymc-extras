import warnings

from collections.abc import Sequence

import numpy as np
import pymc
import pytensor.tensor as pt

from arviz import InferenceData, dict_to_dataset
from pymc.backends.arviz import coords_and_dims_for_inferencedata, dataset_to_point_list
from pymc.distributions.discrete import Bernoulli, Categorical, DiscreteUniform
from pymc.distributions.transforms import Chain
from pymc.logprob.transforms import IntervalTransform
from pymc.model import Model
from pymc.model.fgraph import (
    ModelFreeRV,
    ModelValuedVar,
    fgraph_from_model,
    model_free_rv,
    model_from_fgraph,
)
from pymc.pytensorf import collect_default_updates, constant_fold, toposort_replace
from pymc.pytensorf import compile as compile_pymc
from pymc.util import RandomState, _get_seeds_per_chain
from pytensor import In, Out
from pytensor.compile import SharedVariable
from pytensor.graph import (
    FunctionGraph,
    Variable,
    clone_replace,
    graph_inputs,
    graph_replace,
    node_rewriter,
    vectorize_graph,
)
from pytensor.graph.rewriting.basic import in2out
from pytensor.tensor import TensorVariable

__all__ = ["MarginalModel", "marginalize"]

from pytensor.tensor.random.type import RandomType
from pytensor.tensor.special import log_softmax

from pymc_extras.distributions import DiscreteMarkovChain
from pymc_extras.model.marginal.distributions import (
    MarginalDiscreteMarkovChainRV,
    MarginalFiniteDiscreteRV,
    MarginalRV,
    NonSeparableLogpWarning,
    get_domain_of_finite_discrete_rv,
    inline_ofg_outputs,
    reduce_batch_dependent_logps,
)
from pymc_extras.model.marginal.graph_analysis import (
    find_conditional_dependent_rvs,
    find_conditional_input_rvs,
    is_conditional_dependent,
    subgraph_batch_dim_connection,
)

ModelRVs = TensorVariable | Sequence[TensorVariable] | str | Sequence[str]


class MarginalModel(Model):
    """Subclass of PyMC Model that implements functionality for automatic
    marginalization of variables in the logp transformation

    After defining the full Model, the `marginalize` method can be used to indicate a
    subset of variables that should be marginalized

    Notes
    -----
    Marginalization functionality is still very restricted. Only finite discrete
    variables can be marginalized. Deterministics and Potentials cannot be conditionally
    dependent on the marginalized variables.

    Furthermore, not all instances of such variables can be marginalized. If a variable
    has batched dimensions, it is required that any conditionally dependent variables
    use information from an individual batched dimension. In other words, the graph
    connecting the marginalized variable(s) to the dependent variable(s) must be
    composed strictly of Elemwise Operations. This is necessary to ensure an efficient
    logprob graph can be generated. If you want to bypass this restriction you can
    separate each dimension of the marginalized variable into the scalar components
    and then stack them together. Note that such graphs will grow exponentially in the
    number of  marginalized variables.

    For the same reason, it's not possible to marginalize RVs with multivariate
    dependent RVs.

    Examples
    --------
    Marginalize over a single variable

    .. code-block:: python

        import pymc as pm
        from pymc_extras import MarginalModel

        with MarginalModel() as m:
            p = pm.Beta("p", 1, 1)
            x = pm.Bernoulli("x", p=p, shape=(3,))
            y = pm.Normal("y", pm.math.switch(x, -10, 10), observed=[10, 10, -10])

            m.marginalize([x])

            idata = pm.sample()

    """

    def __init__(self, *args, **kwargs):
        raise TypeError(
            "MarginalModel was deprecated in favor of `marginalize` which now returns a PyMC model"
        )


def _warn_interval_transform(rv_to_marginalize, replaced_vars: Sequence[ModelValuedVar]) -> None:
    for replaced_var in replaced_vars:
        if not isinstance(replaced_var.owner.op, ModelValuedVar):
            raise TypeError(f"{replaced_var} is not a ModelValuedVar")

        if not isinstance(replaced_var.owner.op, ModelFreeRV):
            continue

        if replaced_var is rv_to_marginalize:
            continue

        transform = replaced_var.owner.op.transform

        if isinstance(transform, IntervalTransform) or (
            isinstance(transform, Chain)
            and any(isinstance(tr, IntervalTransform) for tr in transform.transform_list)
        ):
            warnings.warn(
                f"The transform {transform} for the variable {replaced_var}, which depends on the "
                f"marginalized {rv_to_marginalize} may no longer work if bounds depended on other variables.",
                UserWarning,
            )


def _unique(seq: Sequence) -> list:
    """Copied from https://stackoverflow.com/a/480227"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def marginalize(model: Model, rvs_to_marginalize: ModelRVs) -> MarginalModel:
    """Marginalize a subset of variables in a PyMC model.

    This creates a class of `MarginalModel` from an existing `Model`, with the specified
    variables marginalized.

    See documentation for `MarginalModel` for more information.

    Parameters
    ----------
    model : Model
        PyMC model to marginalize. Original variables well be cloned.
    rvs_to_marginalize : Sequence[TensorVariable]
        Variables to marginalize in the returned model.

    Returns
    -------
    marginal_model: MarginalModel
        Marginal model with the specified variables marginalized.
    """
    if isinstance(rvs_to_marginalize, str | Variable):
        rvs_to_marginalize = (rvs_to_marginalize,)

    rvs_to_marginalize = [model[rv] if isinstance(rv, str) else rv for rv in rvs_to_marginalize]

    if not rvs_to_marginalize:
        return model

    for rv_to_marginalize in rvs_to_marginalize:
        if rv_to_marginalize not in model.free_RVs:
            raise ValueError(f"Marginalized RV {rv_to_marginalize} is not a free RV in the model")

        rv_op = rv_to_marginalize.owner.op
        if isinstance(rv_op, DiscreteMarkovChain):
            if rv_op.n_lags > 1:
                raise NotImplementedError(
                    "Marginalization for DiscreteMarkovChain with n_lags > 1 is not supported"
                )
            if rv_to_marginalize.owner.inputs[0].type.ndim > 2:
                raise NotImplementedError(
                    "Marginalization for DiscreteMarkovChain with non-matrix transition probability is not supported"
                )
        elif not isinstance(rv_op, Bernoulli | Categorical | DiscreteUniform):
            raise NotImplementedError(
                f"Marginalization of RV with distribution {rv_to_marginalize.owner.op} is not supported"
            )

    fg, memo = fgraph_from_model(model)
    rvs_to_marginalize = [memo[rv] for rv in rvs_to_marginalize]
    toposort = fg.toposort()

    for rv_to_marginalize in sorted(
        rvs_to_marginalize,
        key=lambda rv: toposort.index(rv.owner),
        reverse=True,
    ):
        all_rvs = [node.out for node in fg.toposort() if isinstance(node.op, ModelValuedVar)]

        dependent_rvs = find_conditional_dependent_rvs(rv_to_marginalize, all_rvs)
        if not dependent_rvs:
            # TODO: This should at most be a warning, not an error
            raise ValueError(f"No RVs depend on marginalized RV {rv_to_marginalize}")

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
        for det in model.deterministics:
            if is_conditional_dependent(memo[det], rv_to_marginalize, all_rvs):
                raise NotImplementedError(
                    f"Cannot marginalize {rv_to_marginalize} due to dependent Deterministic {det}"
                )
        for pot in model.potentials:
            if is_conditional_dependent(memo[pot], rv_to_marginalize, all_rvs):
                raise NotImplementedError(
                    f"Cannot marginalize {rv_to_marginalize} due to dependent Potential {pot}"
                )

        marginalized_rv_input_rvs = find_conditional_input_rvs([rv_to_marginalize], all_rvs)
        other_direct_rv_ancestors = [
            rv
            for rv in find_conditional_input_rvs(dependent_rvs, all_rvs)
            if rv is not rv_to_marginalize
        ]
        input_rvs = _unique((*marginalized_rv_input_rvs, *other_direct_rv_ancestors))

        replace_finite_discrete_marginal_subgraph(fg, rv_to_marginalize, dependent_rvs, input_rvs)

    return model_from_fgraph(fg, mutate_fgraph=True)


@node_rewriter(tracks=[MarginalRV])
def local_unmarginalize(fgraph, node):
    unmarginalized_rv, *dependent_rvs_and_rngs = inline_ofg_outputs(node.op, node.inputs)
    rngs = [rng for rng in dependent_rvs_and_rngs if isinstance(rng.type, RandomType)]
    dependent_rvs = [rv for rv in dependent_rvs_and_rngs if rv not in rngs]

    # Wrap the marginalized RV in a FreeRV
    # TODO: Preserve dims and transform in MarginalRV
    value = unmarginalized_rv.clone()
    fgraph.add_input(value)
    transform = None
    unmarginalized_free_rv = model_free_rv(unmarginalized_rv, value, transform, *node.op.dims)

    # Replace references to the marginalized RV with the FreeRV in the dependent RVs
    dependent_rvs = graph_replace(dependent_rvs, {unmarginalized_rv: unmarginalized_free_rv})

    return [unmarginalized_free_rv, *dependent_rvs, *rngs]


unmarginalize_rewrite = in2out(local_unmarginalize, ignore_newtrees=False)


def unmarginalize(model: Model, rvs_to_unmarginalize: str | Sequence[str] | None = None) -> Model:
    """Unmarginalize a subset of variables in a PyMC model.


    Parameters
    ----------
    model : Model
        PyMC model to unmarginalize. Original variables well be cloned.
    rvs_to_unmarginalize : str or sequence of str, optional
        Variables to unmarginalize in the returned model. If None, all variables are
        unmarginalized.

    Returns
    -------
    unmarginal_model: Model
        Model with the specified variables unmarginalized.
    """

    # Unmarginalize all the MarginalRVs
    fg, memo = fgraph_from_model(model)
    unmarginalize_rewrite(fg)
    unmarginalized_model = model_from_fgraph(fg, mutate_fgraph=True)
    if rvs_to_unmarginalize is None:
        return unmarginalized_model

    # Re-marginalize the variables we want to keep marginalized
    if not isinstance(rvs_to_unmarginalize, list | tuple):
        rvs_to_unmarginalize = (rvs_to_unmarginalize,)
    rvs_to_unmarginalize = set(rvs_to_unmarginalize)

    old_free_rv_names = set(rv.name for rv in model.free_RVs)
    new_free_rv_names = set(
        rv.name for rv in unmarginalized_model.free_RVs if rv.name not in old_free_rv_names
    )
    if rvs_to_unmarginalize - new_free_rv_names:
        raise ValueError(
            f"Unrecognized rvs_to_unmarginalize: {rvs_to_unmarginalize - new_free_rv_names}"
        )
    rvs_to_keep_marginalized = tuple(new_free_rv_names - rvs_to_unmarginalize)
    return marginalize(unmarginalized_model, rvs_to_keep_marginalized)


def transform_posterior_pts(model, posterior_pts):
    """Create a function from the untransformed space to the transformed space"""
    # TODO: This should be a utility in PyMC
    transformed_rvs = []
    transformed_names = []

    for rv in model.free_RVs:
        transform = model.rvs_to_transforms.get(rv)
        if transform is None:
            transformed_rvs.append(rv)
            transformed_names.append(rv.name)
        else:
            transformed_rv = transform.forward(rv, *rv.owner.inputs)
            transformed_rvs.append(transformed_rv)
            transformed_names.append(model.rvs_to_values[rv].name)

    fn = compile_pymc(
        inputs=[In(inp, borrow=True) for inp in model.free_RVs],
        outputs=[Out(out, borrow=True) for out in transformed_rvs],
    )
    fn.trust_input = True

    # TODO: This should work with vectorized inputs
    return [dict(zip(transformed_names, fn(**point))) for point in posterior_pts]


def recover_marginals(
    model: Model,
    idata: InferenceData,
    var_names: Sequence[str] | None = None,
    return_samples: bool = True,
    extend_inferencedata: bool = True,
    random_seed: RandomState = None,
):
    """Computes posterior log-probabilities and samples of marginalized variables
    conditioned on parameters of the model given InferenceData with posterior group

    When there are multiple marginalized variables, each marginalized variable is
    conditioned on both the parameters and the other variables still marginalized

    All log-probabilities are within the transformed space

    Parameters
    ----------
    model: Model
        PyMC model with marginalized variables to recover
    idata : InferenceData
        InferenceData with posterior group
    var_names : sequence of str, optional
        List of variable names for which to compute posterior log-probabilities and samples. Defaults to all marginalized variables
    return_samples : bool, default True
        If True, also return samples of the marginalized variables
    extend_inferencedata : bool, default True
        Whether to extend the original InferenceData or return a new one
    random_seed: int, array-like of int or SeedSequence, optional
        Seed used to generating samples

    Returns
    -------
    idata : InferenceData
        InferenceData with where a lp_{varname} and {varname} for each marginalized variable in var_names added to the posterior group

    .. code-block:: python

        import pymc as pm
        from pymc_extras import MarginalModel

        with MarginalModel() as m:
            p = pm.Beta("p", 1, 1)
            x = pm.Bernoulli("x", p=p, shape=(3,))
            y = pm.Normal("y", pm.math.switch(x, -10, 10), observed=[10, 10, -10])

            m.marginalize([x])

            idata = pm.sample()
            m.recover_marginals(idata, var_names=["x"])


    """
    unmarginal_model = unmarginalize(model)

    # Find the names of the marginalized variables
    model_var_names = set(rv.name for rv in model.free_RVs)
    marginalized_rv_names = [
        rv.name for rv in unmarginal_model.free_RVs if rv.name not in model_var_names
    ]

    if var_names is None:
        var_names = marginalized_rv_names

    var_names = [var if isinstance(var, str) else var.name for var in var_names]
    var_names_to_recover = [name for name in marginalized_rv_names if name in var_names]
    missing_names = [name for name in var_names_to_recover if name not in marginalized_rv_names]
    if missing_names:
        raise ValueError(f"Unrecognized var_names: {missing_names}")

    if return_samples and random_seed is not None:
        seeds = _get_seeds_per_chain(random_seed, len(var_names_to_recover))
    else:
        seeds = [None] * len(var_names_to_recover)

    posterior_pts, stacked_dims = dataset_to_point_list(
        # Remove Deterministics
        idata.posterior[[rv.name for rv in model.free_RVs]],
        sample_dims=("chain", "draw"),
    )
    transformed_posterior_pts = transform_posterior_pts(model, posterior_pts)

    rv_dict = {}
    rv_dims = {}
    for seed, var_name_to_recover in zip(seeds, var_names_to_recover):
        var_to_recover = unmarginal_model[var_name_to_recover]
        supported_dists = (Bernoulli, Categorical, DiscreteUniform)
        if not isinstance(var_to_recover.owner.op, supported_dists):
            raise NotImplementedError(
                f"RV with distribution {var_to_recover.owner.op} cannot be recovered. "
                f"Supported distribution include {supported_dists}"
            )

        other_marginalized_rvs_names = marginalized_rv_names.copy()
        other_marginalized_rvs_names.remove(var_name_to_recover)
        dependent_rvs = [
            rv
            for rv in find_conditional_dependent_rvs(var_to_recover, unmarginal_model.basic_RVs)
            if rv.name not in other_marginalized_rvs_names
        ]
        # Handle batch dims for marginalized value and its dependent RVs
        dependent_rvs_dim_connections = subgraph_batch_dim_connection(var_to_recover, dependent_rvs)

        marginalized_model = marginalize(unmarginal_model, other_marginalized_rvs_names)

        marginalized_var_to_recover = marginalized_model[var_name_to_recover]
        dependent_rvs = [marginalized_model[rv.name] for rv in dependent_rvs]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=NonSeparableLogpWarning)
            logps = marginalized_model.logp(
                vars=[marginalized_var_to_recover, *dependent_rvs], sum=False
            )

        marginalized_logp, *dependent_logps = logps
        joint_logp = marginalized_logp + reduce_batch_dependent_logps(
            dependent_rvs_dim_connections,
            [dependent_var.owner.op for dependent_var in dependent_rvs],
            dependent_logps,
        )

        marginalized_value = marginalized_model.rvs_to_values[marginalized_var_to_recover]
        other_values = [v for v in marginalized_model.value_vars if v is not marginalized_value]

        rv_shape = constant_fold(tuple(var_to_recover.shape), raise_not_constant=False)
        rv_domain = get_domain_of_finite_discrete_rv(var_to_recover)
        rv_domain_tensor = pt.moveaxis(
            pt.full(
                (*rv_shape, len(rv_domain)),
                rv_domain,
                dtype=var_to_recover.dtype,
            ),
            -1,
            0,
        )

        batched_joint_logp = vectorize_graph(
            joint_logp,
            replace={marginalized_value: rv_domain_tensor},
        )
        batched_joint_logp = pt.moveaxis(batched_joint_logp, 0, -1)

        joint_logp_norm = log_softmax(batched_joint_logp, axis=-1)
        if return_samples:
            rv_draws = Categorical.dist(logit_p=batched_joint_logp)
            if isinstance(var_to_recover.owner.op, DiscreteUniform):
                rv_draws += rv_domain[0]
            outputs = [joint_logp_norm, rv_draws]
        else:
            outputs = joint_logp_norm

        rv_loglike_fn = compile_pymc(
            inputs=other_values,
            outputs=outputs,
            on_unused_input="ignore",
            random_seed=seed,
        )

        logvs = [rv_loglike_fn(**vs) for vs in transformed_posterior_pts]

        if return_samples:
            logps, samples = zip(*logvs)
            logps = np.asarray(logps)
            samples = np.asarray(samples)
            rv_dict[var_name_to_recover] = samples.reshape(
                tuple(len(coord) for coord in stacked_dims.values()) + samples.shape[1:],
            )
        else:
            logps = np.asarray(logvs)

        rv_dict["lp_" + var_name_to_recover] = logps.reshape(
            tuple(len(coord) for coord in stacked_dims.values()) + logps.shape[1:],
        )
        if var_name_to_recover in unmarginal_model.named_vars_to_dims:
            rv_dims[var_name_to_recover] = list(
                unmarginal_model.named_vars_to_dims[var_name_to_recover]
            )
            rv_dims["lp_" + var_name_to_recover] = rv_dims[var_name_to_recover] + [
                "lp_" + var_name_to_recover + "_dim"
            ]

    coords, dims = coords_and_dims_for_inferencedata(unmarginal_model)
    dims.update(rv_dims)
    rv_dataset = dict_to_dataset(
        rv_dict,
        library=pymc,
        dims=dims,
        coords=coords,
        skip_event_dims=True,
    )

    if extend_inferencedata:
        idata.posterior = idata.posterior.assign(rv_dataset)
        return idata
    else:
        return rv_dataset


def collect_shared_vars(outputs, blockers):
    return [
        inp
        for inp in graph_inputs(outputs, blockers=blockers)
        if (isinstance(inp, SharedVariable) and inp not in blockers)
    ]


def remove_model_vars(vars):
    """Remove ModelVars from the graph of vars."""
    model_vars = [var for var in vars if isinstance(var.owner.op, ModelValuedVar)]
    replacements = [(model_var, model_var.owner.inputs[0]) for model_var in model_vars]
    fgraph = FunctionGraph(outputs=vars, clone=False)
    toposort_replace(fgraph, replacements)
    return fgraph.outputs


def replace_finite_discrete_marginal_subgraph(
    fgraph, rv_to_marginalize, dependent_rvs, input_rvs
) -> None:
    # If the marginalized RV has multiple dimensions, check that graph between
    # marginalized RV and dependent RVs does not mix information from batch dimensions
    # (otherwise logp would require enumerating over all combinations of batch dimension values)
    try:
        dependent_rvs_dim_connections = subgraph_batch_dim_connection(
            rv_to_marginalize, dependent_rvs
        )
    except (ValueError, NotImplementedError) as e:
        # For the perspective of the user this is a NotImplementedError
        raise NotImplementedError(
            "The graph between the marginalized and dependent RVs cannot be marginalized efficiently. "
            "You can try splitting the marginalized RV into separate components and marginalizing them separately."
        ) from e

    output_rvs = [rv_to_marginalize, *dependent_rvs]
    rng_updates = collect_default_updates(output_rvs, inputs=input_rvs, must_be_shared=False)
    outputs = output_rvs + list(rng_updates.values())
    inputs = input_rvs + list(rng_updates.keys())
    # Add any other shared variable inputs
    inputs += collect_shared_vars(output_rvs, blockers=inputs)

    inner_inputs = [inp.clone() for inp in inputs]
    inner_outputs = clone_replace(outputs, replace=dict(zip(inputs, inner_inputs)))
    inner_outputs = remove_model_vars(inner_outputs)

    if isinstance(inner_outputs[0].owner.op, DiscreteMarkovChain):
        marginalize_constructor = MarginalDiscreteMarkovChainRV
    else:
        marginalize_constructor = MarginalFiniteDiscreteRV

    _, _, *dims = rv_to_marginalize.owner.inputs
    marginalization_op = marginalize_constructor(
        inputs=inner_inputs,
        outputs=inner_outputs,
        dims_connections=dependent_rvs_dim_connections,
        dims=dims,
    )

    new_outputs = marginalization_op(*inputs)
    for old_output, new_output in zip(outputs, new_outputs):
        new_output.name = old_output.name

    model_replacements = []
    for old_output, new_output in zip(outputs, new_outputs):
        if old_output is rv_to_marginalize or not isinstance(old_output.owner.op, ModelValuedVar):
            # Replace the marginalized ModelFreeRV (or non model-variables) themselves
            var_to_replace = old_output
        else:
            # Replace the underlying RV, keeping the same value, transform and dims
            var_to_replace = old_output.owner.inputs[0]
        model_replacements.append((var_to_replace, new_output))

    fgraph.replace_all(model_replacements)
