import numpy as np

from pymc import STEP_METHODS
from pymc.distributions.distribution import _support_point
from pymc.initial_point import PointType
from pymc.logprob.abstract import MeasurableOp, _logprob
from pymc.model.core import modelcontext
from pymc.pytensorf import compile
from pymc.step_methods.compound import BlockedStep, Competence, StepMethodState
from pymc.util import get_value_vars_from_user_vars
from pytensor import shared
from pytensor.compile.builders import OpFromGraph
from pytensor.link.jax.linker import JAXLinker
from pytensor.tensor.random.type import RandomGeneratorType

from pymc_extras.utils.ofg import inline_ofg_outputs


class ConjugateRV(OpFromGraph, MeasurableOp):
    """Wrapper for ConjugateRVs, that outputs the original RV and the conjugate posterior expression.

    For partial step samplers to work, the logp and initial point correspond to the original RV
    while the variable itself is sampled by default by the `ConjugateRVSampler` by evaluating directly the
    conjugate posterior expression (i.e., taking forward random draws).
    """


@_logprob.register(ConjugateRV)
def conjugate_rv_logp(op, values, rv, *params, **kwargs):
    # Logp is the same as the original RV
    return _logprob(rv.owner.op, values, *rv.owner.inputs)


@_support_point.register(ConjugateRV)
def conjugate_rv_support_point(op, conjugate_rv, rv, *params):
    # Support point is the same as the original RV
    return _support_point(rv.owner.op, rv, *rv.owner.inputs)


class ConjugateRVSampler(BlockedStep):
    name = "conjugate_rv_sampler"
    _state_class = StepMethodState

    def __init__(self, vars, model=None, rng=None, compile_kwargs: dict | None = None, **kwargs):
        if len(vars) != 1:
            raise ValueError("ConjugateRVSampler can only be assigned to one variable at a time")

        model = modelcontext(model)
        [value] = get_value_vars_from_user_vars(vars, model=model)
        rv = model.values_to_rvs[value]
        self.vars = (value,)
        self.rv_name = value.name

        if model.rvs_to_transforms[rv] is not None:
            raise ValueError("Variable assigned to ConjugateRVSampler cannot be transformed")

        rv_and_posterior_rv_node = rv.owner
        op = rv_and_posterior_rv_node.op
        if not isinstance(op, ConjugateRV):
            raise ValueError("Variable must be a ConjugateRV")

        # Replace RVs in inputs of rv_posterior_rv_node by the corresponding value variables
        value_inputs = model.replace_rvs_by_values(
            [rv_and_posterior_rv_node.outputs[1]],
        )[0].owner.inputs
        # Inline the ConjugateRV graph to only compile `posterior_rv`
        _, posterior_rv, *_ = inline_ofg_outputs(op, value_inputs)

        if compile_kwargs is None:
            compile_kwargs = {}
        self.posterior_fn = compile(
            model.value_vars,
            posterior_rv,
            random_seed=rng,
            on_unused_input="ignore",
            **compile_kwargs,
        )
        self.posterior_fn.trust_input = True
        if isinstance(self.posterior_fn.maker.linker, JAXLinker):
            # Reseeding RVs in JAX backend requires a different logic, becuase the SharedVariables
            # used internally are not the ones that `function.get_shared()` returns.
            raise ValueError("ConjugateRVSampler is not compatible with JAX backend")

    def set_rng(self, rng: np.random.Generator):
        # Copy the function and replace any shared RNGs
        # This is needed so that it can work correctly with multiple traces
        # This will be costly if set_rng is called too often!
        shared_rngs = [
            var
            for var in self.posterior_fn.get_shared()
            if isinstance(var.type, RandomGeneratorType)
        ]
        n_shared_rngs = len(shared_rngs)
        swap = {
            old_shared_rng: shared(rng, borrow=True)
            for old_shared_rng, rng in zip(shared_rngs, rng.spawn(n_shared_rngs), strict=True)
        }
        self.posterior_fn = self.posterior_fn.copy(swap=swap)

    def step(self, point: PointType) -> tuple[PointType, list]:
        new_point = point.copy()
        new_point[self.rv_name] = self.posterior_fn(**point)
        return new_point, []

    @staticmethod
    def competence(var, has_grad):
        """BinaryMetropolis is only suitable for Bernoulli and Categorical variables with k=2."""
        if isinstance(var.owner.op, ConjugateRV):
            return Competence.IDEAL

        return Competence.INCOMPATIBLE


# Register the ConjugateRVSampler
STEP_METHODS.append(ConjugateRVSampler)
