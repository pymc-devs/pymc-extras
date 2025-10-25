from collections import Counter
from collections.abc import Sequence
from typing import TypeAlias

from pymc.model.core import Model, modelcontext
from pymc.model.fgraph import fgraph_from_model, model_from_fgraph
from pytensor.graph.rewriting.db import EquilibriumDB, RewriteDatabaseQuery

posterior_optimization_db = EquilibriumDB()
posterior_optimization_db.failure_callback = None  # Raise an error if an optimization fails
posterior_optimization_db.name = "posterior_optimization_db"

TAGS_TYPE: TypeAlias = str | Sequence[str] | None


def optimize_model_for_mcmc_sampling(
    model: Model,
    include: TAGS_TYPE = ("default",),
    exclude: TAGS_TYPE = None,
    rewriter=None,
) -> tuple[Model, Sequence[Counter]]:
    if isinstance(include, str):
        include = (include,)
    if isinstance(exclude, str):
        exclude = (exclude,)

    model = modelcontext(model)
    fgraph, _ = fgraph_from_model(model)

    if rewriter is None:
        rewriter = posterior_optimization_db.query(
            RewriteDatabaseQuery(include=include, exclude=exclude)
        )
    _, _, rewrite_counters, *_ = rewriter.rewrite(fgraph)

    opt_model = model_from_fgraph(fgraph, mutate_fgraph=True)
    return opt_model, rewrite_counters
