#   Copyright 2024 - present The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import logging

from collections.abc import Sequence

import pytensor

from pymc.data import Minibatch
from pymc.distributions.shape_utils import change_dist_size
from pymc.model.core import Model
from pymc.model.fgraph import (
    ModelDeterministic,
    ModelFreeRV,
    ModelNamed,
    ModelObservedRV,
    ModelPotential,
    extract_dims,
    fgraph_from_model,
    model_deterministic,
    model_from_fgraph,
    model_observed_rv,
)
from pymc.model.transform.basic import parse_vars
from pymc.variational.minibatch_rv import create_minibatch_rv
from pytensor.graph.basic import Variable
from pytensor.graph.traversal import ancestors

from pymc_extras.model.marginal.graph_analysis import (
    _subgraph_batch_dim_clients,
    subgraph_batch_dim_ancestors,
)

_log = logging.getLogger("pmx")

ModelVariable = Variable | str


def _to_seq(x):
    return x if isinstance(x, tuple | list) else (x,)


def minibatch(
    model: Model,
    observed: ModelVariable | Sequence[ModelVariable] | None = None,
    data: ModelVariable | Sequence[ModelVariable] | None = None,
    *,
    batch_size: int,
    validate: bool = True,
) -> Model:
    """Replace observed data in a model by random minibatches.

    The returned model draws a fresh random slice from the leading dimension of
    the relevant ``Data`` variables on every evaluation, and rescales the logp of
    the affected observed variables by ``total_size`` (the full number of rows of
    the original data) so that the expected gradient matches that of the full
    model. This is the transform-based equivalent of building a model with
    :func:`pymc.Minibatch` and passing ``total_size`` to the observed variable.

    All minibatched data is sliced with the *same* random indices, so every
    selected ``Data`` variable must share the same leading dimension length. The
    minibatched (leading) dimension is relabeled to ``"<dim>_minibatch"`` on the
    rewritten variables, and any ``Deterministic`` downstream of the minibatch is
    relabeled and resized to match. The original full-size ``Data`` variables and
    their dims are left untouched.

    .. warning::

        Only the *leading* dimension is minibatched, and the logp is rescaled
        assuming each observed variable has an independent batch dimension aligned
        with it. If the data enters the likelihood batched across non-leading
        dimensions, or the dependent observed variable does not carry a batch
        dimension aligned with the sliced axis (e.g. the data is reduced before
        reaching it), the returned model is silently inconsistent. Use
        ``validate=True`` (the default) to catch these cases where possible.

    Parameters
    ----------
    model : Model
        The model whose observed data should be minibatched. The original model
        is left untouched; a new model is returned.
    observed : model variable or sequence of model variables, optional
        Observed variables whose logp should be rescaled. Defaults to every
        observed variable with at least one dimension.
    data : model variable or sequence of model variables, optional
        ``Data`` variables to minibatch. Defaults to every named variable that
        feeds a selected observed along the minibatched axis, plus the observed
        values themselves.
    batch_size : int
        Number of rows to draw from the leading dimension on each evaluation.
    validate : bool, default True
        Run soundness checks and raise if the minibatch would be incorrect:
        each observed variable must have a batch dimension aligned with the
        minibatched leading axis (and, where the graph can be traced, the data
        must connect to it); no free RV or Potential may depend on the minibatched
        data / resized observeds or be depended on by a selected observed along the
        minibatched axis (their logp is not rescaled); and every Data variable
        feeding a selected observed along that axis (including the observed value
        itself) must be among the minibatched ``data``. A free RV that merely
        shares the minibatched dim name without being connected is fine. Set to
        False to skip the checks and minibatch anyway. See the warning above.

    Returns
    -------
    new_model : Model
        A distinct model with the selected data minibatched and the observed
        variables rescaled. The full-size ``Data`` variables remain available
        under their original names. If there is nothing to minibatch (no
        batchable observed variable, or no data feeding it) an unchanged copy of
        the model is returned.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import pymc as pm

        from pymc_extras.model.transforms.minibatch import minibatch

        with pm.Model(coords={"obs": range(5000), "feature": range(3)}) as model:
            x = pm.Data("x", np.ones((5000, 3)), dims=("obs", "feature"))
            y = pm.Data("y", np.ones(5000), dims=("obs",))
            beta = pm.Normal("beta", dims="feature")
            noise = pm.HalfNormal("noise")
            pm.Normal("llike", beta @ x.T, noise, observed=y, dims="obs")

        mb_model = minibatch(model, batch_size=100)

        with mb_model:
            approx = pm.fit()

    """
    if not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer")

    if observed is None:
        observed = [rv for rv in model.observed_RVs if rv.ndim > 0]
    else:
        observed = parse_vars(model, observed)
        if any(rv not in model.observed_RVs for rv in observed):
            raise ValueError("All `observed` variables must be observed variables in the model")

    fgraph, memo = fgraph_from_model(model, inlined_views=True)
    observed = [memo[rv] for rv in observed]

    # The minibatched dimension is the leading axis (axis 0) of each observed, matching
    # pm.Minibatch. We connect it through the graph in both directions, for different
    # questions:
    #   backward (subgraph_batch_dim_ancestors), observed axis -> ancestors: which Data to
    #     slice, and which free RVs / Potentials share the axis (e.g. a per-observation
    #     effect) and would need rescaling.
    #   forward (_subgraph_batch_dim_clients), data and resized observeds -> clients: which
    #     Deterministics to relabel, and which free RVs / Potentials are evaluated on the
    #     subsample without rescaling.

    # Backward: ancestors carrying the subsampled axis. An observed whose leading axis is a
    # support or scalar dim has no batch dimension to subsample.
    mixed_observed: list[Variable] = []
    carried: set[Variable] = set()
    for obs_rv in observed:
        rv = obs_rv.owner.inputs[0]
        ndim_supp = getattr(getattr(rv.owner, "op", None), "ndim_supp", 0)
        if rv.type.ndim - ndim_supp < 1:
            mixed_observed.append(obs_rv)
            continue
        carried.update(v for v, axes in subgraph_batch_dim_ancestors(rv, 0).items() if any(axes))

    # Data to slice: named ancestors carrying the axis, plus the observed values (sliced to
    # match the resized RVs). Everything else is left full size.
    named_ancestors = [
        out
        for out in fgraph.outputs
        if isinstance(out.owner.op, ModelNamed) and out.owner.inputs[0] in set(ancestors(observed))
    ]
    observed_values = {obs_rv.owner.inputs[1] for obs_rv in observed}
    if data is None:
        data = [d for d in named_ancestors if d in carried or d in observed_values]
    else:
        data = [memo[var] for var in parse_vars(model, data)]
        for var in data:
            if not (var.owner is not None and isinstance(var.owner.op, ModelNamed)):
                raise ValueError(f"{var} is not a named (Data) variable in the model")

    if validate and mixed_observed:
        raise ValueError(
            f"The leading dimension of observed {mixed_observed[0].name!r} is not an "
            "independent batch dimension (it is a support or scalar dimension); its logp "
            "cannot be rescaled by total_size. Pass validate=False to minibatch anyway."
        )

    # Nothing to minibatch: no data feeds the observeds along the batch axis.
    if not observed or not data:
        _log.info("Nothing to minibatch; returning an unchanged copy of the model")
        return model_from_fgraph(fgraph, mutate_fgraph=True)

    _log.info(
        "Minibatching data %s feeding observed %s with batch_size=%d",
        [var.name for var in data],
        [rv.name for rv in observed],
        batch_size,
    )

    # The minibatch dim's original name (the data's leading dim) and its relabel.
    data_axes = {d: (0,) + (None,) * (d.type.ndim - 1) for d in data}
    d0_dims = extract_dims(data[0])
    orig_dim = getattr(d0_dims[0], "data", None) if d0_dims else None
    mb_name = f"{orig_dim}_minibatch" if orig_dim is not None else None

    # Forward sources: the data and the (resized) observed RVs, each on its leading axis.
    # Seeding the observeds reaches clients of a resized observed whose distribution does
    # not depend on data (and so is not reachable from the data alone).
    mb_sources = dict(data_axes)
    for obs_rv in observed:
        rv = obs_rv.owner.inputs[0]
        mb_sources[rv] = (0,) + (None,) * (rv.type.ndim - 1)

    def forward_axis(var: Variable, sources: dict) -> int | None:
        """Axis of ``var`` carrying the subsampled dim traced forward from ``sources``
        (None if unconnected). Raises if the dim is used twice or flows through an op the
        analysis cannot trace."""
        conn = _subgraph_batch_dim_clients(dict(sources), list(sources), [var])
        axes = [i for i, m in enumerate(conn.get(var, (None,) * var.type.ndim)) if m == 0]
        if len(axes) > 1:
            raise ValueError(f"{var.name!r} uses the minibatched dimension more than once")
        return axes[0] if axes else None

    def is_client(var: Variable, sources: dict) -> bool:
        try:
            return forward_axis(var, sources) is not None
        except (ValueError, NotImplementedError):
            return True  # untraceable: conservatively connected

    if validate:
        # A free RV or Potential is invalid if it is connected to the minibatch either
        # way: a client of the data or a resized observed (forward), or an ancestor of an
        # observed on the subsampled axis (backward, e.g. a per-observation effect). Its
        # logp is not rescaled, so a subsample would be wrong. Sharing the dim name without
        # being connected is fine.
        for node in fgraph.apply_nodes:
            if not isinstance(node.op, ModelFreeRV | ModelPotential):
                continue
            inner = node.inputs[0]
            if is_client(inner, mb_sources) or inner in carried or node.outputs[0] in carried:
                raise ValueError(
                    f"{node.outputs[0].name!r} is a free RV or Potential affected by the "
                    "minibatch; its logp cannot be rescaled. Pass validate=False to "
                    "minibatch anyway."
                )
        # Data feeding a non-selected observed would be resized without its total_size
        # rescaling.
        selected = set(observed)
        for node in fgraph.apply_nodes:
            if isinstance(node.op, ModelObservedRV) and node.outputs[0] not in selected:
                if is_client(node.inputs[0], data_axes):
                    raise ValueError(
                        f"Minibatched data also feeds the non-selected observed "
                        f"{node.outputs[0].name!r}; minibatch it as well or pass "
                        "validate=False."
                    )
        # Every Data on the subsampled axis must be sliced, including each observed value,
        # else it stays full size while the observed is sliced and rescaled.
        selected_data = set(data)
        for obs_rv in observed:
            value = obs_rv.owner.inputs[1]
            if value not in selected_data:
                raise ValueError(
                    f"The value of observed {obs_rv.name!r} is not among the minibatched "
                    "`data`; it must be minibatched too. Add it to `data` (as a pm.Data) "
                    "or pass validate=False."
                )
        for cand in named_ancestors:
            if cand in carried and cand not in selected_data:
                raise ValueError(
                    f"Data {cand.name!r} feeds a selected observed along the minibatched "
                    "axis but is not in `data`; minibatch it as well or pass validate=False."
                )

    # A single Minibatch slices every selected data variable with the same random indices.
    # The original full-size data variables stay in the model; only the selected observeds
    # and downstream Deterministics use the slices.
    mb_data = _to_seq(Minibatch(*data, batch_size=batch_size))
    data_to_mb = dict(zip(data, mb_data))

    # The minibatch is built from the same data it replaces, so swapping directly would
    # create a cycle. We go through a temporary copy: data -> copy -> minibatch.
    replacements = []
    data_copies = [var.copy() for var in data]
    replacements.extend(zip(data, data_copies))
    replacements.extend(zip(data_copies, mb_data))

    # A fresh shared length for the minibatch dim, used to resize the observed RVs and, if
    # the dim is named, as the registered length of the new dim.
    mb_len = pytensor.shared(batch_size, name=mb_name)
    if mb_name is not None:
        fgraph._coords[mb_name] = None
        fgraph._dim_lengths[mb_name] = mb_len

    def relabel_axis(dims, axis: int | None):
        new_dims = list(dims)
        if mb_name is not None and axis is not None and axis < len(new_dims):
            new_dims[axis] = mb_name
        return new_dims

    # Full size along the subsampled axis. Stay symbolic so it tracks a later pm.set_data.
    full_size = data[0].shape[0]
    for obs_rv in observed:
        rv, value, *dims = obs_rv.owner.inputs

        # Resize the RV so its baked-in size follows the minibatch on the sliced axis. An
        # RV created with dims bakes in the full dim lengths, which the data substitution
        # does not touch, so otherwise e.g. pm.draw would fail.
        batch_ndim = rv.type.ndim - getattr(rv.owner.op, "ndim_supp", 0)
        new_size = [mb_len if i == 0 else rv.shape[i] for i in range(batch_ndim)]
        mb_dist = change_dist_size(rv, new_size)

        # Rescale the logp by the full size along the subsampled axis.
        total_size = [None] * mb_dist.type.ndim
        total_size[0] = full_size
        mb_rv = create_minibatch_rv(mb_dist, total_size)
        mb_rv.name = rv.name  # the rebuilt ModelObservedRV takes its name from this

        # Rebuilding the observed node makes a fresh value edge the data replacements above
        # would not reach, so wire the minibatch in directly.
        value = data_to_mb.get(value, value)
        replacements.append((obs_rv, model_observed_rv(mb_rv, value, *relabel_axis(dims, 0))))

    # Deterministics that are clients of the minibatch are resized too; relabel their
    # subsampled axis so the labels match their batch-sized values.
    for node in fgraph.apply_nodes:
        if not isinstance(node.op, ModelDeterministic):
            continue
        det = node.outputs[0]
        var, *dims = node.inputs
        try:
            axis = forward_axis(var, mb_sources)
        except (ValueError, NotImplementedError):
            continue  # subsampled dim is reduced/mixed here; nothing to relabel
        if axis is None:
            continue  # not a client of the minibatch
        new_var = data_to_mb.get(var, var)
        new_det = model_deterministic(new_var, *relabel_axis(dims, axis))
        new_det.name = det.name
        replacements.append((det, new_det))

    fgraph.replace_all(replacements, import_missing=True, reason="minibatch")
    return model_from_fgraph(fgraph, mutate_fgraph=True)


__all__ = ("minibatch",)
