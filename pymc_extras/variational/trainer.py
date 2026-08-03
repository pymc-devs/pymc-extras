#   Copyright 2026 - present The PyMC Developers
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
"""Drive variational inference over a :class:`DataLoader` with no user callbacks."""

from __future__ import annotations

import warnings

from collections.abc import Iterator

import numpy as np

from pymc.model import modelcontext
from pymc.variational.inference import Inference
from pymc.variational.inference import fit as _fit
from pymc.variational.minibatch_rv import MinibatchRandomVariable
from pytensor.graph.basic import Constant

from pymc_extras.variational.dataloader import DataLoader

__all__ = ["Trainer"]


def _cycle(loader: DataLoader) -> Iterator[np.ndarray]:
    """Repeat the loader's epochs forever. A pass that yields nothing never ends."""
    while True:
        empty = True
        for batch in loader:
            empty = False
            yield batch
        if empty:
            raise RuntimeError("dataloader yielded no batches")


def _warn_if_scaling_mismatches(model, total_size: int) -> None:
    """Warn if the model cannot rescale the minibatch likelihood to ``total_size``.

    ``total_size`` reaches the graph as a constant on a ``MinibatchRandomVariable``.
    Both an absent and a disagreeing value are silent otherwise: the posterior comes
    out wrong rather than the fit failing.
    """
    declared = {
        int(t.data)
        for rv in model.observed_RVs
        if isinstance(rv.owner.op, MinibatchRandomVariable)
        for t in rv.owner.inputs[1:]
        if isinstance(t, Constant) and t.data is not None
    }
    if not declared:
        warnings.warn(
            "no observed variable declares total_size, so the minibatch "
            "log-likelihood is never rescaled and is underweighted against the prior "
            "by about N / batch_size. Pass total_size=len(dataloader) to the observed "
            "distribution.",
            UserWarning,
            stacklevel=3,
        )
    elif declared != {total_size}:
        warnings.warn(
            f"the model declares total_size={sorted(declared)} but the loader streams "
            f"N={total_size}; the minibatch rescaling does not match the data.",
            UserWarning,
            stacklevel=3,
        )


class Trainer:
    """Drive variational inference over a :class:`DataLoader` without user callbacks.

    Follows the design in PyMC's variational-inference rework and PyTorch
    Lightning: the ``Trainer`` owns the training loop, the
    :class:`DataLoader` owns batching (and ``len(dataloader)`` is the dataset size
    ``N``), and the model owns the math. The model exposes a ``pm.Data`` placeholder;
    the ``Trainer`` streams minibatches into it with ``model.set_data`` once per
    step; no user callbacks are needed.

    Parameters
    ----------
    method : str or Inference, default "advi"
        Variational method, forwarded to :func:`pymc.fit`: a name (``"advi"``,
        ``"fullrank_advi"``, ...) or an :class:`~pymc.variational.inference.Inference`
        instance. ``pm.fit`` applies ``model`` and ``random_seed`` only to a name;
        an instance is already bound to a model, so configure it at construction
        (e.g. ``ADVI(random_seed=...)``).
    dataloader : DataLoader
        The minibatch source. ``len(dataloader)`` is ``N``; the model should pass
        it to the observed distribution's ``total_size``.
    model : pymc.Model, optional
        Defaults to the model on the context stack.
    data_name : str, default "batch"
        Name of the ``pm.Data`` placeholder minibatches are streamed into. Must
        match the name used for ``pm.Data(name, ...)`` in the model.
    **fit_kwargs
        Default keyword arguments forwarded to :func:`pymc.fit` (e.g.
        ``obj_optimizer``); per-call kwargs to :meth:`fit` override them.

    Examples
    --------
    .. code-block:: python

        loader = DataLoader(
            parquet_source("shuffled/"), batch_size=4096, sample_shape=(4,), total_size="auto"
        )
        with pm.Model() as model:
            b = pm.Normal("b", 0.0, 3.0, shape=4)
            batch = pm.Data("batch", np.zeros((4096, 4)))  # placeholder
            logit = b[0] + b[1] * batch[:, 0] + b[2] * batch[:, 1] + b[3] * batch[:, 2]
            pm.Bernoulli("y", logit_p=logit, observed=batch[:, 3], total_size=len(loader))
            approx = Trainer(method="advi", dataloader=loader, data_name="batch").fit(20_000)
    """

    def __init__(
        self,
        *,
        method: str | Inference = "advi",
        dataloader: DataLoader,
        model=None,
        data_name: str = "batch",
        **fit_kwargs,
    ):
        self.method = method
        self.dataloader = dataloader
        self.model = model
        self.data_name = data_name
        self._fit_kwargs = fit_kwargs

    def fit(self, n: int = 10_000, **kwargs):
        """Fit for ``n`` steps, streaming minibatches into the model's placeholder.

        Step ``i`` trains on batch ``i``: the first batch seeds the placeholder
        before step 0 and every step loads the next one, so ``n`` steps train ``n``
        batches and leave batch ``n`` loaded for whatever runs next --
        :meth:`~pymc.variational.inference.Inference.refine` then continues the
        stream instead of repeating a batch. User ``callbacks`` run while the batch
        that produced the latest loss is still in place, and a ``StopIteration``
        from one ends the fit before another batch is loaded. Keyword arguments are
        forwarded to :func:`pymc.fit` on top of the constructor's ``fit_kwargs``
        (per-call wins); ``progressbar`` defaults to ``False`` unless either sets it.

        Returns
        -------
        :class:`Approximation`
            The fitted approximation, as returned by :func:`pymc.fit`.
        """
        if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
            raise ValueError(f"n must be a positive integer (the number of fit steps), got {n!r}")
        loader = self.dataloader
        if not isinstance(loader, DataLoader):
            raise TypeError(
                f"Trainer needs a DataLoader for `dataloader`, got {type(loader).__name__}."
            )
        model = modelcontext(self.model)
        if self.data_name not in model:
            raise KeyError(
                f"data_name {self.data_name!r} is not a variable in the model; it "
                f"must name the pm.Data placeholder the minibatches are streamed into."
            )
        if isinstance(self.method, Inference) and self.method.approx.model is not model:
            raise ValueError(
                "`method` is an Inference instance bound to a different model than the "
                "one being trained, so the minibatches would stream into a model the "
                "fit never reads. Build it under this model, or pass a method name."
            )
        if loader.total_size is not None:
            _warn_if_scaling_mismatches(model, loader.total_size)

        batches = _cycle(loader)
        model.set_data(self.data_name, next(batches))

        def _advance(*_):
            model.set_data(self.data_name, next(batches))

        merged = {**self._fit_kwargs, **kwargs}
        merged.setdefault("progressbar", False)
        user_callbacks = merged.pop("callbacks", None) or []
        return _fit(
            n,
            method=self.method,
            model=model,
            callbacks=[*user_callbacks, _advance],
            **merged,
        )
