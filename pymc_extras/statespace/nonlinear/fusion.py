import itertools

from collections.abc import Mapping, Sequence

import numpy as np
import pytensor.tensor as pt

from pymc.distributions.shape_utils import change_dist_size
from pymc.logprob.basic import conditional_logp
from pymc.model import Model
from pymc.model.fgraph import (
    ModelDeterministic,
    ModelFreeRV,
    ModelNamed,
    ModelObservedRV,
    fgraph_from_model,
)
from pymc.pytensorf import toposort_replace
from pytensor.graph.basic import Apply
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import clone_replace, graph_replace
from pytensor.graph.traversal import ancestors
from pytensor.tensor import TensorLike
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.variable import TensorConstant, TensorVariable

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

        reshaping = [
            name
            for name in self.rv_names
            if not np.array_equal(
                _unconstrained_shape(model=model, rv=model[name]), model[name].shape.eval()
            )
        ]
        if reshaping:
            raise ValueError(
                f"{reshaping} have transforms that change their shape, so their elements are not "
                "observed states."
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

    def observation_logp(
        self, *, parameters: TensorVariable, data: Mapping[str, TensorVariable]
    ) -> dict[str, tuple[TensorVariable, list[TensorVariable] | None]]:
        r"""
        Return the log-likelihood of each observed variable.

        Parameters
        ----------
        parameters : TensorVariable
            Vector laid out like :math:`\theta`.
        data : mapping of str to TensorVariable
            A value for each data variable and for each observed variable whose observations are
            constants, by name. Every observed variable takes its batch shape from its value.

        Returns
        -------
        logp : dict of str to tuple
            For each observed variable, the log-likelihood of each observation, shaped like its
            batch dims, and its distribution parameters that depend on ``parameters``, broadcast
            to the shape of its value and contained in that log-likelihood. The distribution
            parameters exist only for a vector of scalar draws from a distribution with scalar
            parameters, and are None otherwise, where the curvature has to be exact.
        """
        fgraph = self._rewrite(parameters=parameters, data=data)

        observations = {}
        distribution_parameters = {}
        stand_ins = {}
        for node in fgraph.toposort():
            if not isinstance(node.op, ModelObservedRV):
                continue
            rv, value = node.inputs
            op = rv.owner.op
            if (
                isinstance(op, RandomVariable)
                and op.ndim_supp == 0
                and not any(op.ndims_params)
                and value.ndim == 1
            ):
                broadcast = [
                    pt.broadcast_to(parameter, value.shape)
                    for parameter in op.dist_params(rv.owner)
                ]

                # conditional_logp rewrites the graph it is given, so its log-likelihood would not
                # contain these parameters. Building the variable on stand-ins and substituting
                # the parameters afterwards keeps them in it, where the Gauss-Newton curvature
                # cuts the graph.
                parameter_stand_ins = [parameter.type() for parameter in broadcast]
                stand_ins.update(zip(parameter_stand_ins, broadcast, strict=True))
                rng, size, *_ = rv.owner.inputs
                rv = op.make_node(rng, size, *parameter_stand_ins).default_output()
                distribution_parameters[node.op.name] = [
                    parameter for parameter in broadcast if parameters in ancestors([parameter])
                ]
            else:
                distribution_parameters[node.op.name] = None
            observations[node.op.name] = (rv, value)

        value_logps = conditional_logp(dict(observations.values()))
        pointwise_logps = graph_replace(
            [value_logps[value] for _, value in observations.values()],
            replace=stand_ins,
            strict=False,
        )
        return {
            name: (pointwise_logp, distribution_parameters[name])
            for name, pointwise_logp in zip(observations, pointwise_logps, strict=True)
        }

    def statespace_matrices(
        self,
        *,
        statespace_parameters: Mapping[str, TensorLike],
        statespace_data: Mapping[str, TensorLike],
    ) -> list[TensorVariable]:
        """
        Return the statespace matrices at given values of the statespace's parameters and data.

        Parameters
        ----------
        statespace_parameters : mapping of str to TensorLike
            A value, constant or symbolic, for each name in the statespace's ``param_names``.
        statespace_data : mapping of str to TensorLike
            A value, constant or symbolic, for each name in the statespace's ``data_names``.

        Returns
        -------
        matrices : list of TensorVariable
            The matrices ``x0, P0, c, d, T, Z, R, H, Q``.
        """
        replacements = {}
        for argument, values, placeholders in [
            ("statespace_parameters", statespace_parameters, self.statespace._name_to_variable),
            ("statespace_data", statespace_data, self.statespace._name_to_data),
        ]:
            if set(values) != set(placeholders):
                raise ValueError(
                    f"{argument} has to give a value for exactly {sorted(placeholders)}; missing: "
                    f"{sorted(set(placeholders) - set(values))}, unexpected: "
                    f"{sorted(set(values) - set(placeholders))}."
                )
            replacements.update(
                {
                    placeholder: placeholder.type.filter_variable(values[name])
                    for name, placeholder in placeholders.items()
                }
            )

        return graph_replace(
            list(self.statespace._unpack_statespace_with_placeholders()),
            replace=replacements,
            strict=True,
        )

    def _rewrite(
        self, *, parameters: TensorVariable, data: Mapping[str, TensorVariable]
    ) -> FunctionGraph:
        """Return a new graph of the model with its free variables and data replaced."""
        fgraph, _ = fgraph_from_model(self.model, inlined_views=True)
        free_rvs = {
            node.op.name: node.outputs[0]
            for node in fgraph.apply_nodes
            if isinstance(node.op, ModelFreeRV)
        }
        constrained = {
            name: _backward_transform(free_rv=free_rvs[name], value=value)
            for name, value in self.unpack(parameters).items()
        }
        replacements = [
            (free_rvs[name], free_rvs[name].type.filter_variable(value))
            for name, value in {**constrained, **self.frozen}.items()
        ] + [
            (node.outputs[0], node.inputs[0])
            for node in fgraph.apply_nodes
            if isinstance(node.op, ModelDeterministic)
        ]
        # A backward transform reads its inputs, which may be other free variables, from the graph
        # as it stands. Replacing dependents first lets the later replacements reach into them.
        toposort_replace(fgraph, replacements, reverse=True)

        return _replace_data(fgraph=fgraph, data=data)


def _replace_data(*, fgraph: FunctionGraph, data: Mapping[str, TensorVariable]) -> FunctionGraph:
    named_data = {
        node.op.name: node.outputs[0]
        for node in fgraph.apply_nodes
        if isinstance(node.op, ModelNamed)
    }
    constant_observations = {
        node.op.name: node.inputs[1]
        for node in fgraph.apply_nodes
        if isinstance(node.op, ModelObservedRV) and isinstance(node.inputs[1], TensorConstant)
    }
    replaced = {**named_data, **constant_observations}
    if set(data) != set(replaced):
        raise ValueError(
            f"data has to give a value for exactly {sorted(replaced)}; missing: "
            f"{sorted(set(replaced) - set(data))}, unexpected: {sorted(set(data) - set(replaced))}."
        )

    # Rebuilding the graph, unlike replacing in place, lets each type lose the static shape of
    # the value it was built from.
    outputs = clone_replace(
        fgraph.outputs,
        replace={variable: data[name] for name, variable in replaced.items()},
        rebuild_strict=False,
    )
    observed = [
        variable
        for variable in outputs
        if variable.owner is not None and isinstance(variable.owner.op, ModelObservedRV)
    ]
    outputs = clone_replace(
        outputs,
        replace={variable: _resized_to_value(node=variable.owner) for variable in observed},
        rebuild_strict=False,
    )
    return FunctionGraph(outputs=outputs, clone=False)


def _backward_transform(*, free_rv: TensorVariable, value: TensorVariable) -> TensorVariable:
    transform = free_rv.owner.op.transform
    if transform is None:
        return value
    rv = free_rv.owner.inputs[0]
    return transform.backward(value, *rv.owner.inputs)


def _resized_to_value(*, node: Apply) -> TensorVariable:
    rv, value = node.inputs
    batch_shape = value.shape[: value.ndim - rv.owner.op.ndim_supp]
    return node.op(change_dist_size(rv, new_size=batch_shape), value)


def _unconstrained_shape(*, model: Model, rv: TensorVariable) -> np.ndarray:
    transform = model.rvs_to_transforms[rv]
    if transform is None:
        return rv.shape.eval()
    return transform.forward(rv, *rv.owner.inputs).shape.eval()


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
