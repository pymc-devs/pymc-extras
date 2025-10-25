import sys

from pymc.model.core import Model
from pymc.sampling.mcmc import sample
from pytensor.graph.rewriting.basic import GraphRewriter

from pymc_extras.sampling.optimizations.optimize import (
    TAGS_TYPE,
    optimize_model_for_mcmc_sampling,
)


def opt_sample(
    *args,
    model: Model | None = None,
    include: TAGS_TYPE = ("default",),
    exclude: TAGS_TYPE = None,
    rewriter: GraphRewriter | None = None,
    verbose: bool = False,
    **kwargs,
):
    """Sample from a model after applying optimizations.

    Parameters
    ----------
    model : Model, optinoal
        The model to sample from. If None, use the model associated with the context.
    include : TAGS_TYPE
        The tags to include in the optimizations. Ignored if `rewriter` is not None.
    exclude : TAGS_TYPE
        The tags to exclude from the optimizations. Ignored if `rewriter` is not None.
    rewriter : RewriteDatabaseQuery (optional)
        The rewriter to use. If None, use the default rewriter with the given `include` and `exclude` tags.
    verbose : bool, default=False
        Print information about the optimizations applied.
    *args, **kwargs:
        Passed to `pm.sample`

    Returns
    -------
    sample_output:
        The output of `pm.sample`

    Examples
    --------
    .. code:: python
        import pymc as pm
        import pymc_experimental as pmx

        with pm.Model() as m:
            p = pm.Beta("p", 1, 1, shape=(1000,))
            y = pm.Binomial("y", n=100, p=p, observed=[1, 50, 99, 50]*250)

            idata = pmx.opt_sample(verbose=True)

        # Applied optimization: beta_binomial_conjugacy 1x
        # ConjugateRVSampler: [p]


    You can control which optimizations are applied using the `include` and `exclude` arguments:

    .. code:: python
        import pymc as pm
        import pymc_experimental as pmx

        with pm.Model() as m:
            p = pm.Beta("p", 1, 1, shape=(1000,))
            y = pm.Binomial("y", n=100, p=p, observed=[1, 50, 99, 50]*250)

            idata = pmx.opt_sample(exclude="conjugacy", verbose=True)

        # No optimizations applied
        # NUTS: [p]

    .. code:: python
        import pymc as pm
        import pymc_experimental as pmx

        with pm.Model() as m:
            a = pm.InverseGamma("a", 1, 1)
            b = pm.InverseGamma("b", 1, 1)
            p = pm.Beta("p", a, b, shape=(1000,))
            y = pm.Binomial("y", n=100, p=p, observed=[1, 50, 99, 50]*250)

            # By default, the conjugacy of p will not be applied because it depends on other free variables
            idata = pmx.opt_sample(include="conjugacy-eager", verbose=True)

        # Applied optimization: beta_binomial_conjugacy_eager 1x
        # CompoundStep
        # >NUTS: [a, b]
        # >ConjugateRVSampler: [p]

    """
    if kwargs.get("step", None) is not None:
        raise ValueError(
            "The `step` argument is not supported in `opt_sample`, as custom steps would refer to the original model.\n"
            "You can manually transform the model with `pymc_experimental.sampling.optimizations.optimize_model_for_mcmc_sampling` "
            "and then define the custom steps and forward them to `pymc.sample`."
        )

    opt_model, rewrite_counters = optimize_model_for_mcmc_sampling(
        model, include=include, exclude=exclude, rewriter=rewriter
    )

    if verbose:
        applied_opt = False
        for rewrite_counter in rewrite_counters:
            for rewrite, counts in rewrite_counter.items():
                applied_opt = True
                print(f"Applied optimization: {rewrite} {counts}x", file=sys.stdout)
        if not applied_opt:
            print("No optimizations applied", file=sys.stdout)

    return sample(*args, model=opt_model, **kwargs)
