import itertools

from collections.abc import Mapping, Sequence
from typing import Literal, NamedTuple

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from numpy.typing import ArrayLike
from pymc.distributions.shape_utils import change_dist_size
from pymc.logprob.basic import conditional_logp
from pymc.logprob.transforms import Transform
from pymc.model import Model, modelcontext
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
from pytensor.tensor.random.basic import multivariate_normal
from pytensor.tensor.random.op import RandomVariable
from pytensor.tensor.variable import TensorConstant, TensorVariable
from pytensor_ml.optim.line_search import LineSearch
from xarray import DataTree

from pymc_extras.statespace.core import dummy_graph
from pymc_extras.statespace.core.properties import Coord, Data, Parameter, Shock, State
from pymc_extras.statespace.core.statespace import PyMCStateSpace
from pymc_extras.statespace.nonlinear.filter import scan_filter, validated_data
from pymc_extras.statespace.utils.constants import (
    FILTER_OUTPUT_TYPES,
    OBSERVED_LIKELIHOOD_NAME,
    TIME_DIM,
)

MASK_SUFFIX = "_observed_mask"


class FusedFitData(NamedTuple):
    """The padded per-date data and observations of a fit, and each observed variable's mask."""

    data: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]


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


class FusedStateSpace(PyMCStateSpace):
    r"""
    Combine a cross-sectional PyMC model with a statespace model over its parameters.

    The elements of the free random variables named in ``rv_names``, on the unconstrained scale
    of each variable's transform, are the observed states of ``statespace``, under the names
    :func:`observed_state_names` gives them. The observed states, in the statespace's order, form
    the parameter vector :math:`\theta`; :meth:`unpack` returns a vector in that layout as the
    variables' own shapes. Every other free random variable is held at a value given in
    ``frozen``, on the variable's own constrained scale.

    The model's likelihood has to consist of observed variables that do not depend on each other,
    without potentials or imputed observations.

    It is used like the statespace it wraps, whose parameters, states, shocks, coords and matrices
    it takes on: :meth:`build_statespace_graph` registers the observations, whose
    log-probability is the filter's from :func:`scan_filter`, in a model holding priors for the
    statespace's parameters, and the conditional samplers run that filter and its smoother for
    each parameter draw. Smoothed states
    are drawn from each date's marginal.

    Parameters
    ----------
    model : Model
        The cross-sectional model.
    statespace : PyMCStateSpace
        The statespace model whose observed states are the variables' elements.
    rv_names : sequence of str
        Names of the free random variables that are observed states.
    frozen : mapping of str to TensorLike
        Values for the free random variables that are not observed states, each on its
        variable's constrained scale and shaped like it.
    curvature : {"gauss_newton", "exact"}
        The curvature of the log-likelihood in the Newton iterations: the generalized Gauss-Newton
        matrix of each observed variable, or the exact negative Hessian.
    line_search : LineSearch
        Search along each Newton direction.
    tolerance : float
        Largest absolute entry of the whitened Newton direction at which the iterations stop.
    max_iterations : int
        Most Newton iterations on each date.
    """

    def __init__(
        self,
        *,
        model: Model,
        statespace: PyMCStateSpace,
        rv_names: Sequence[str],
        frozen: Mapping[str, TensorLike],
        curvature: Literal["gauss_newton", "exact"],
        line_search: LineSearch,
        tolerance: float,
        max_iterations: int,
    ):
        self.model = model
        self.statespace = statespace
        self.rv_names = list(rv_names)
        self.frozen = dict(frozen)
        self.curvature = curvature
        self.line_search = line_search
        self.tolerance = tolerance
        self.max_iterations = max_iterations

        if model.potentials:
            raise ValueError(
                f"Potentials {[potential.name for potential in model.potentials]} are not part of "
                "the observation log-likelihood."
            )

        dependent_observations = {}
        for rv in model.observed_RVs:
            rv_ancestors = set(ancestors(rv.owner.inputs))
            others = [other.name for other in model.observed_RVs if other in rv_ancestors]
            if others:
                dependent_observations[rv.name] = others
        if dependent_observations:
            raise ValueError(
                "Observed variables depend on other observed variables (variable: the ones it "
                f"depends on): {dependent_observations}."
            )

        free_names = [rv.name for rv in model.free_RVs]
        observed_names = [rv.name for rv in model.observed_RVs]
        imputed = [
            name
            for name in free_names
            if name.endswith("_unobserved")
            and f"{name.removesuffix('_unobserved')}_observed" in observed_names
        ]
        if imputed:
            raise ValueError(
                f"{imputed} impute missing observations, whose number is fixed by the model's data."
            )

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
        # The position in theta of each element of each variable, in the variable's shape. Indexing
        # theta with it gives the variable without a size-1 intermediate for a scalar, whose static
        # shape would differ from the same value's inside a scan.
        self._theta_positions = {
            name: np.vectorize(lambda state: state_positions[str(state)], otypes=[int])(
                labels[name]
            )
            for name in self.rv_names
        }

        super().__init__(
            k_endog=statespace.k_endog,
            k_states=statespace.k_states,
            k_posdef=statespace.k_posdef,
            joint_smoothed_draws=False,
            verbose=False,
            measurement_error=statespace.measurement_error,
            mode=statespace.mode,
            cov_jitter=statespace.cov_jitter,
        )

    def set_parameters(self) -> tuple[Parameter, ...]:
        return tuple(self.statespace._param_info)

    def set_states(self) -> tuple[State, ...]:
        return tuple(self.statespace._state_info)

    def set_shocks(self) -> tuple[Shock, ...]:
        return tuple(self.statespace._shock_info)

    def set_coords(self) -> tuple[Coord, ...]:
        return tuple(self.statespace._coords_info)

    def set_data_info(self) -> tuple[Data, ...] | None:
        return tuple(self.statespace._data_info) or None

    def make_symbolic_graph(self) -> None:
        self.ssm = self.statespace.ssm
        self._tensor_variable_info = self.statespace._tensor_variable_info
        self._tensor_data_info = self.statespace._tensor_data_info
        self._n_timesteps_placeholder = self.statespace._n_timesteps_placeholder

    def build_statespace_graph(
        self, data: Mapping[str, Sequence[ArrayLike]], dates: ArrayLike
    ) -> None:
        """
        Register the observations in the active model, with the filter's log-likelihood.

        The active model holds priors for the statespace's parameters and data under the names in
        ``param_names`` and ``data_names``. Each date's values are padded to a common size. The
        model's data variables are registered as ``pm.Data`` along the ``time`` dim, with a
        boolean mask for each observed variable under its name followed by ``_observed_mask``.
        The observations of every observed variable, flattened on each date and concatenated in
        the order of the model's observed variables, are the value of one observed variable named
        ``obs`` along ``time``. Its log-probability is each date's log-likelihood from
        :func:`scan_filter`, and its draws simulate the statespace and then the model's
        observations given each date's parameters.

        Parameters
        ----------
        data : mapping of str to sequence of ArrayLike
            For each data variable of the model and each observed variable whose observations are
            constants, its finite value on each date. Missing observations are left out of a
            date's arrays.
        dates : ArrayLike
            Label of each date, registered as the coords of the ``time`` dim.
        """
        model = modelcontext(None)
        padded, masks = self._pad(validated_data(fused=self, data=data, n_dates=len(dates)))
        model.add_coord(TIME_DIM, dates)
        sources = self._observation_sources()
        data_variables = [
            pm.Data(name, padded[name], dims=_padded_dims(name=name, ndim=padded[name].ndim))
            for name in self._filter_data_names()
            if name not in sources.values()
        ]
        mask_variables = [
            pm.Data(
                f"{name}{MASK_SUFFIX}",
                masks[name],
                dims=_padded_dims(name=f"{name}{MASK_SUFFIX}", ndim=masks[name].ndim),
            )
            for name in sources
        ]
        observations = np.concatenate(
            [padded[source].reshape(len(dates), -1) for source in sources.values()], axis=1
        ).astype("float64")
        inputs = [
            *data_variables,
            *mask_variables,
            *(model[name] for name in self.param_names),
            *(model[name] for name in self.data_names),
        ]
        date_shapes = {source: padded[source].shape[1:] for source in sources.values()}

        def logp(value, *values):
            data_values, mask_values, parameters, statespace_data = self._split_inputs(values)
            scanned = scan_filter(
                fused=self,
                statespace_parameters=parameters,
                statespace_data=statespace_data,
                data={**data_values, **self._unpack_observations(value, date_shapes=date_shapes)},
                observed_masks=mask_values,
            )
            return scanned["loglike_obs"]

        def simulate(*values_and_size):
            # The observations are registered without a size, so the last argument is unused.
            data_values, mask_values, parameters, statespace_data = self._split_inputs(
                values_and_size[:-1]
            )
            return self._simulate(
                n_dates=next(iter(mask_values.values())).shape[0],
                data=data_values,
                date_shapes=date_shapes,
                statespace_parameters=parameters,
                statespace_data=statespace_data,
            )

        model.add_coord(f"{OBSERVED_LIKELIHOOD_NAME}_dim_1", length=observations.shape[1])
        pm.CustomDist(
            OBSERVED_LIKELIHOOD_NAME,
            *inputs,
            logp=logp,
            dist=simulate,
            signature=_signature(inputs),
            dtype="float64",
            observed=observations,
            dims=(TIME_DIM, f"{OBSERVED_LIKELIHOOD_NAME}_dim_1"),
        )

    def _fit_data(self, idata: DataTree) -> FusedFitData:
        constant_data = idata["constant_data"]
        sources = self._observation_sources()
        masks = {name: constant_data[f"{name}{MASK_SUFFIX}"].values for name in sources}
        data = {
            name: constant_data[name].values
            for name in self._filter_data_names()
            if name not in sources.values()
        }
        date_shapes = {
            source: masks[observed].shape[1:] + self._support_shape(observed)
            for observed, source in sources.items()
        }
        observations = idata["observed_data"][OBSERVED_LIKELIHOOD_NAME].values
        data.update(
            {
                source: values.eval()
                for source, values in self._unpack_observations(
                    pt.as_tensor(observations), date_shapes=date_shapes
                ).items()
            }
        )
        return FusedFitData(data=data, masks=masks)

    def _kalman_filter_outputs_from_dummy_graph(
        self,
        data: FusedFitData,
        *,
        coords: dict[str, Sequence],
        dims: dict[str, list[str]],
        exog: dict[str, dict],
        data_dims: str | tuple[str] | list[str] | None = None,
        scenario: dict | None = None,
    ) -> tuple[list[TensorVariable], list[tuple[TensorVariable, TensorVariable]]]:
        if scenario is not None:
            raise NotImplementedError("A FusedStateSpace does not take scenario data.")
        model = modelcontext(None)
        dummy_graph.build_dummy_graph(self, coords=coords, dims=dims)
        matrices = self._insert_random_variables()
        for name in self.data_names:
            if name not in model:
                pm.Data(**exog[name])
        matrices = self._insert_data_variables(matrices)

        scanned = scan_filter(
            fused=self,
            statespace_parameters={name: model[name] for name in self.param_names},
            statespace_data={name: model[name] for name in self.data_names},
            data=data.data,
            observed_masks=data.masks,
        )
        grouped_outputs = [
            (scanned[f"{kind}_states"], scanned[f"{kind}_covariances"])
            for kind in FILTER_OUTPUT_TYPES
        ]
        return matrices, grouped_outputs

    def _split_inputs(
        self, values: Sequence[TensorVariable]
    ) -> tuple[
        dict[str, TensorVariable],
        dict[str, TensorVariable],
        dict[str, TensorVariable],
        dict[str, TensorVariable],
    ]:
        """Return the observation distribution's inputs by name, in the order they were given."""
        sources = self._observation_sources()
        data_names = [name for name in self._filter_data_names() if name not in sources.values()]
        values = list(values)
        data_values = dict(zip(data_names, values[: len(data_names)], strict=True))
        values = values[len(data_names) :]
        mask_values = dict(zip(sources, values[: len(sources)], strict=True))
        values = values[len(sources) :]
        parameters = dict(zip(self.param_names, values[: len(self.param_names)], strict=True))
        statespace_data = dict(zip(self.data_names, values[len(self.param_names) :], strict=True))
        return data_values, mask_values, parameters, statespace_data

    def _unpack_observations(
        self, value: TensorVariable, *, date_shapes: Mapping[str, tuple[int, ...]]
    ) -> dict[str, TensorVariable]:
        """Return each observed variable's padded observations from the concatenated value."""
        observations = {}
        start = 0
        for observed, source in self._observation_sources().items():
            size = int(np.prod(date_shapes[source]))
            observations[source] = pt.cast(
                value[:, start : start + size].reshape((value.shape[0], *date_shapes[source])),
                self.model[observed].dtype,
            )
            start += size
        return observations

    def _simulate(
        self,
        *,
        n_dates: TensorVariable,
        data: Mapping[str, TensorVariable],
        date_shapes: Mapping[str, tuple[int, ...]],
        statespace_parameters: Mapping[str, TensorVariable],
        statespace_data: Mapping[str, TensorVariable],
    ) -> TensorVariable:
        """Return draws of the concatenated observations, simulating the statespace forward."""
        x0, P0, c, d, T, Z, R, H, Q = self.statespace_matrices(
            statespace_parameters=statespace_parameters, statespace_data=statespace_data
        )
        sources = self._observation_sources()
        data_names = list(data)
        rng, initial_state = multivariate_normal(
            x0, P0, rng=pytensor.shared(np.random.default_rng()), method="svd", return_next_rng=True
        )

        def step(*arguments):
            date_data = dict(zip(data_names, arguments[: len(data_names)], strict=True))
            state, rng = arguments[len(data_names) :]
            rng, parameters = multivariate_normal(
                Z @ state + d, H, rng=rng, method="svd", return_next_rng=True
            )
            placeholders = {
                source: pt.zeros(date_shapes[source], dtype=self.model[observed].dtype)
                for observed, source in sources.items()
            }
            draws, rng = self._observation_draws(
                parameters=parameters, data={**date_data, **placeholders}, rng=rng
            )
            observations = pt.concatenate(
                [pt.cast(draws[observed], "float64").ravel() for observed in sources]
            )
            rng, shock = multivariate_normal(
                pt.zeros(Q.shape[0]), Q, rng=rng, method="svd", return_next_rng=True
            )
            return c + T @ state + R @ shock, rng, observations

        _, _, observations = pytensor.scan(
            step,
            sequences=[pt.as_tensor_variable(values) for values in data.values()],
            outputs_info=[initial_state, rng, None],
            n_steps=n_dates,
            name="fused_simulation",
            return_updates=False,
        )
        return observations

    def _observation_draws(
        self,
        *,
        parameters: TensorVariable,
        data: Mapping[str, TensorVariable],
        rng: TensorVariable,
    ) -> tuple[dict[str, TensorVariable], TensorVariable]:
        """Return a draw of each observed variable at the parameters, and the generator after."""
        fgraph = self._rewrite(parameters=parameters, data=data)
        draws = {}
        for node in fgraph.toposort():
            if not isinstance(node.op, ModelObservedRV):
                continue
            rv = node.inputs[0]
            if not isinstance(rv.owner.op, RandomVariable):
                raise NotImplementedError(
                    f"Observed variable {node.op.name} is not a RandomVariable, so it cannot be "
                    "drawn on a given generator."
                )
            redrawn = rv.owner.op.make_node(rng, *rv.owner.inputs[1:])
            rng, draws[node.op.name] = redrawn.outputs
        return draws, rng

    def _support_shape(self, observed: str) -> tuple[int, ...]:
        value = self.model.rvs_to_values[self.model[observed]]
        ndim_supp = self.model[observed].owner.op.ndim_supp
        return tuple(int(length) for length in value.shape.eval()[value.ndim - ndim_supp :])

    def _filter_data_names(self) -> list[str]:
        """Return the names the filter takes per-date values for."""
        constant_observations = [
            rv.name
            for rv in self.model.observed_RVs
            if isinstance(self.model.rvs_to_values[rv], TensorConstant)
        ]
        return [variable.name for variable in self.model.data_vars] + constant_observations

    def _observation_sources(self) -> dict[str, str]:
        """Return, for each observed variable, the name its per-date values are given under."""
        return {
            rv.name: rv.name
            if isinstance(self.model.rvs_to_values[rv], TensorConstant)
            else self.model.rvs_to_values[rv].name
            for rv in self.model.observed_RVs
        }

    def _pad(
        self, data: Mapping[str, Sequence[np.ndarray]]
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Return the per-date values padded to a common size, and each observed variable's mask."""
        # Padding every name with the first entry of one date where all of them have entries pads
        # each observation with a real observation and its own data, so its log-likelihood is finite.
        n_dates = len(next(iter(data.values())))
        reference_date = next(
            (date for date in range(n_dates) if all(len(values[date]) for values in data.values())),
            None,
        )
        if reference_date is None:
            raise ValueError("No date has entries for every name in data.")

        padded = {}
        for name, values in data.items():
            reference = values[reference_date][0]
            width = max(len(value) for value in values)
            stacked = np.empty((len(values), width, *reference.shape), dtype=reference.dtype)
            stacked[...] = reference
            for date, value in enumerate(values):
                stacked[date, : len(value)] = value
            padded[name] = stacked

        masks = {}
        for observed, source in self._observation_sources().items():
            ndim_supp = self.model[observed].owner.op.ndim_supp
            stacked = padded[source]
            batch_shape = stacked.shape[1 : stacked.ndim - ndim_supp]
            counts = np.array([len(value) for value in data[source]])
            real = np.arange(batch_shape[0]) < counts[:, None]
            masks[observed] = np.broadcast_to(
                real.reshape(*real.shape, *(1,) * (len(batch_shape) - 1)),
                (len(counts), *batch_shape),
            ).copy()
        return padded, masks

    def unpack(self, vector: TensorVariable) -> dict[str, TensorVariable]:
        r"""
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
        return {name: vector[positions] for name, positions in self._theta_positions.items()}

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
            name: _backward_transform(
                transform=self.model.rvs_to_transforms[self.model[name]],
                free_rv=free_rvs[name],
                value=value,
            )
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


def _signature(inputs: Sequence[TensorVariable]) -> str:
    core_dims = [
        "(" + ",".join(f"i{position}_{axis}" for axis in range(variable.ndim)) + ")"
        for position, variable in enumerate(inputs)
    ]
    return f"{','.join(core_dims)}->(time,observation)"


def _padded_dims(*, name: str, ndim: int) -> tuple[str, ...]:
    return (TIME_DIM, *(f"{name}_dim_{axis}" for axis in range(1, ndim)))


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


def _backward_transform(
    *, transform: Transform | None, free_rv: TensorVariable, value: TensorVariable
) -> TensorVariable:
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
