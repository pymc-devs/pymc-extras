import itertools

from collections.abc import Mapping, Sequence

import numpy as np
import pytensor.tensor as pt

from pymc.model import Model
from pymc.model.fgraph import ModelFreeRV, fgraph_from_model
from pymc.pytensorf import toposort_replace
from pytensor.graph.fg import FunctionGraph
from pytensor.tensor import TensorLike
from pytensor.tensor.variable import TensorVariable

from pymc_extras.statespace.core.statespace import PyMCStateSpace


def observed_state_names(*, model: Model, rv_names: Sequence[str]) -> dict[str, list[str]]:
    """
    Return the observed-state names that stand for each element of the named random variables.

    Build the statespace components from these names. An element is named after its coordinates
    when every dim of its variable has coordinate values, and after its position otherwise; a
    scalar variable is named after itself. Elements are listed in row-major order.

    Parameters
    ----------
    model : Model
        The cross-sectional model.
    rv_names : sequence of str
        Names of free random variables of ``model``.

    Returns
    -------
    names : dict of str to list of str
        For each variable, the names of its elements.
    """
    return {
        name: _element_labels(model=model, rv=model[name]).ravel().tolist() for name in rv_names
    }


class FusedStateSpace:
    r"""
    Combine a cross-sectional PyMC model with a statespace model over its parameters.

    The elements of the free random variables named in ``rv_names`` are the observed states of
    ``statespace``, under the names :func:`observed_state_names` gives them. The observed states,
    in the statespace's order, form the parameter vector :math:`\theta`; :meth:`unpack` returns
    a vector in that layout as the variables' own shapes. Every other free random variable is
    held at a value given in ``frozen``.

    Parameters
    ----------
    model : Model
        The cross-sectional model.
    statespace : PyMCStateSpace
        The statespace model whose observed states are the variables' elements.
    rv_names : sequence of str
        Names of the free random variables that are observed states.
    frozen : mapping of str to TensorLike
        Values for the free random variables that are not observed states, each shaped like its
        variable.
    """

    def __init__(
        self,
        *,
        model: Model,
        statespace: PyMCStateSpace,
        rv_names: Sequence[str],
        frozen: Mapping[str, TensorLike],
    ):
        self.model = model
        self.statespace = statespace
        self.rv_names = list(rv_names)
        self.frozen = dict(frozen)

        free_names = [rv.name for rv in model.free_RVs]
        not_free = [name for name in [*self.rv_names, *frozen] if name not in free_names]
        if not_free:
            raise ValueError(f"{not_free} are not free random variables of the model.")
        mapped_and_frozen = [name for name in self.rv_names if name in frozen]
        if mapped_and_frozen:
            raise ValueError(f"{mapped_and_frozen} are both in rv_names and frozen.")
        unlisted = [name for name in free_names if name not in self.rv_names and name not in frozen]
        if unlisted:
            raise ValueError(
                f"Free random variables {unlisted} are neither in rv_names nor frozen."
            )

        wrong_shapes = {}
        for name, value in self.frozen.items():
            expected = tuple(int(length) for length in model[name].shape.eval())
            given = pt.as_tensor(value).type.shape
            if len(given) != len(expected) or any(
                length is not None and length != target for length, target in zip(given, expected)
            ):
                wrong_shapes[name] = (given, expected)
        if wrong_shapes:
            raise ValueError(
                f"Frozen values have the wrong shape (given, expected): {wrong_shapes}."
            )

        labels = {name: _element_labels(model=model, rv=model[name]) for name in self.rv_names}
        observed_states = list(statespace.observed_states)
        expected_states = {
            str(state) for name_labels in labels.values() for state in name_labels.flat
        }
        missing = [state for state in expected_states if state not in observed_states]
        extra = [state for state in observed_states if state not in expected_states]
        if missing or extra:
            raise ValueError(
                "The statespace's observed states have to be the names from "
                f"observed_state_names; missing: {sorted(missing)}, unexpected: {extra}."
            )

        state_positions = {state: position for position, state in enumerate(observed_states)}
        # Packing the variables in rv_names order gives a vector whose entry i sits at
        # _packed_order[i] in theta, which follows the statespace's order.
        self._packed_order = np.array(
            [state_positions[str(state)] for name in self.rv_names for state in labels[name].flat],
            dtype=int,
        )
        self._packed_shapes = [labels[name].shape for name in self.rv_names]

    def unpack(self, vector: TensorVariable) -> dict[str, TensorVariable]:
        """
        Return the entries of a vector laid out like :math:`\theta`, in each variable's shape.

        Parameters
        ----------
        vector : TensorVariable
            Vector of shape ``(k,)``.

        Returns
        -------
        values : dict of str to TensorVariable
            For each variable in ``rv_names``, its entries of ``vector``.
        """
        values = pt.unpack(vector[self._packed_order], packed_shapes=self._packed_shapes)
        return dict(zip(self.rv_names, values, strict=True))

    def _rewrite(self) -> FunctionGraph:
        """Return a new graph of the model with the frozen variables replaced by constants."""
        fgraph, _ = fgraph_from_model(self.model, inlined_views=True)
        free_rvs = {
            node.op.name: node.outputs[0]
            for node in fgraph.apply_nodes
            if isinstance(node.op, ModelFreeRV)
        }
        toposort_replace(
            fgraph,
            [
                (free_rvs[name], free_rvs[name].type.filter_variable(value))
                for name, value in self.frozen.items()
            ],
        )
        return fgraph


def _element_labels(*, model: Model, rv: TensorVariable) -> np.ndarray:
    shape = tuple(int(length) for length in rv.shape.eval())
    if not shape:
        return np.array(rv.name)

    dims = model.named_vars_to_dims.get(rv.name)
    if dims is not None and all(model.coords.get(dim) is not None for dim in dims):
        axes = [model.coords[dim] for dim in dims]
    else:
        axes = [range(length) for length in shape]

    labels = [
        f"{rv.name}[{', '.join(str(entry) for entry in entries)}]"
        for entries in itertools.product(*axes)
    ]
    return np.array(labels).reshape(shape)
