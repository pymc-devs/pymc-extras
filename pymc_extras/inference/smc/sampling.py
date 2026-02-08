#   Copyright 2023 The PyMC Developers
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
import time
import warnings
from collections.abc import Callable
from typing import NamedTuple, cast

import arviz as az
import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.smc import extend_params
from blackjax.smc.resampling import systematic
from pymc import draw, modelcontext, to_inference_data
from pymc.backends import NDArray
from pymc.backends.base import MultiTrace
from pymc.initial_point import make_initial_point_expression
from pymc.sampling.jax import get_jaxified_graph
from pymc.util import RandomState, _get_seeds_per_chain

log = logging.getLogger(__name__)


def sample_smc_blackjax(
    n_particles: int = 2000,
    random_seed: RandomState = None,
    kernel: str = "HMC",
    target_essn: float = 0.5,
    num_mcmc_steps: int = 10,
    inner_kernel_params: dict | None = None,
    model=None,
    iterations_to_diagnose: int = 100,
):
    """Samples using BlackJax's implementation of Sequential Monte Carlo.

    Parameters
    ----------
    n_particles: int
     number of particles used to sample from the posterior. This is also the number of draws. Defaults to 2000.
    random_seed: RandomState
     seed used for random number generator, set for reproducibility. Otherwise a random one will be used (default).
    kernel: str
     Either 'HMC' (default) or 'NUTS'. The kernel to be used to mutate the particles in each SMC iteration.
    target_essn: float
     Proportion (0 < target_essn < 1) of the total number of particles, to be used for incrementing the exponent
     of the tempered posterior between iterations. The higher the number, each increment is going to be smaller,
     leading to more steps and computational cost. Defaults to 0.5. See https://arxiv.org/abs/1602.03572
    num_mcmc_steps: int
      fixed number of steps of each inner kernel markov chain for each SMC mutation step.
    inner_kernel_params: Optional[dict]
     a dictionary with parameters for the inner kernel.
        For HMC it must have 'step_size' and 'integration_steps'
        For NUTS it must have 'step_size'
     these parameters are fixed for all iterations.
    model:
     PyMC model to sample from
    iterations_to_diagnose: int
     Number of iterations to generate diagnosis for. By default, will diagnose the first 100 iterations. Increase
     this number for further diagnosis (it can be bigger than the actual number of iterations executed by the algorithm,
     at the expense of allocating memory to store the diagnosis).

    Returns
    -------
    An Arviz Inference data.

    Note
    ----
    A summary of the algorithm is:

     1. Initialize :math:`\beta` at zero and stage at zero.
     2. Generate N samples :math:`S_{\beta}` from the prior (because when :math `\beta = 0` the
        tempered posterior is the prior).
     3. Increase :math:`\beta` in order to make the effective sample size equal some predefined
        value (target_essn)
     4. Compute a set of N importance weights W. The weights are computed as the ratio of the
        likelihoods of a sample at stage i+1 and stage i.
     5. Obtain :math:`S_{w}` by re-sampling according to W.
     6. Run N independent MCMC chains, starting each one from a different sample
        in :math:`S_{w}`. For that, set the kernel and inner_kernel_params.
     7. The N chains are run for num_mcmc_steps each.
     8. Repeat from step 3 until :math:`\beta \\ge 1`.
     9. The final result is a collection of N samples from the posterior

    """

    model = modelcontext(model)
    random_seed = np.random.default_rng(seed=random_seed)

    if inner_kernel_params is None:
        inner_kernel_params = {}

    log.info(
        f"Will only diagnose the first {iterations_to_diagnose} SMC iterations,"
        f"this number can be increased by setting iterations_to_diagnose parameter"
        f" in sample_with_blackjax_smc"
    )

    key = jax.random.PRNGKey(_get_seeds_per_chain(random_seed, 1)[0])

    key, initial_particles_key, iterations_key = jax.random.split(key, 3)

    initial_particles = blackjax_particles_from_pymc_population(
        model, initialize_population(model, n_particles, random_seed)
    )

    var_map = var_map_from_model(
        model, model.initial_point(random_seed=random_seed.integers(2**30))
    )

    posterior_dimensions = sum(var_map[k][1] for k in var_map)

    if kernel == "HMC":
        mcmc_kernel = blackjax.mcmc.hmc
        mcmc_parameters = extend_params(
            dict(
                step_size=inner_kernel_params["step_size"],
                inverse_mass_matrix=jnp.eye(posterior_dimensions),
                num_integration_steps=inner_kernel_params["integration_steps"],
            )
        )
    elif kernel == "NUTS":
        mcmc_kernel = blackjax.mcmc.nuts
        mcmc_parameters = extend_params(
            dict(
                step_size=inner_kernel_params["step_size"],
                inverse_mass_matrix=jnp.eye(posterior_dimensions),
            )
        )
    else:
        raise ValueError(f"Invalid kernel {kernel}, valid options are 'HMC' and 'NUTS'")

    sampler = build_smc_with_kernel(
        prior_log_prob=get_jaxified_logprior(model),
        loglikelihood=get_jaxified_loglikelihood(model),
        target_ess=target_essn,
        num_mcmc_steps=num_mcmc_steps,
        kernel_parameters=mcmc_parameters,
        mcmc_kernel=mcmc_kernel,
    )

    start = time.time()
    total_iterations, particles, diagnosis = inference_loop(
        iterations_key,
        sampler.init(initial_particles),
        sampler,
        iterations_to_diagnose,
        n_particles,
    )
    end = time.time()
    running_time = end - start

    inference_data = arviz_from_particles(model, particles)

    add_to_inference_data(
        inference_data,
        n_particles,
        target_essn,
        num_mcmc_steps,
        kernel,
        diagnosis,
        total_iterations,
        iterations_to_diagnose,
        inner_kernel_params,
        running_time,
    )

    if total_iterations < iterations_to_diagnose:
        log.warning(
            f"Only the first {iterations_to_diagnose} were included in diagnosed quantities out of {total_iterations}."
        )

    return inference_data


def arviz_from_particles(model, particles):
    """
    Given Particles in Blackjax format,
    builds an Arviz Inference Data object.
    In order to do so in a consistent way,
    particles are assumed to be encoded in
    model.value_vars order.

    Parameters
    ----------
    model: Pymc Model
    particles: output of Blackjax SMC.


    Returns an Arviz Inference Data Object
    -------
    """
    n_particles = jax.tree_util.tree_flatten(particles)[0][0].shape[0]
    by_varname = {
        k.name: v.squeeze()[np.newaxis, :].astype(k.dtype)
        for k, v in zip(model.value_vars, particles)
    }
    varnames = [v.name for v in model.value_vars]
    with model:
        strace = NDArray(name=model.name)
        strace.setup(n_particles, 0)
    for particle_index in range(0, n_particles):
        strace.record(
            point={k: np.asarray(by_varname[k][0][particle_index]) for k in varnames}
        )
        multitrace = MultiTrace((strace,))
    return to_inference_data(multitrace, log_likelihood=False)


class SMCDiagnostics(NamedTuple):
    """
    A Jax-compilable object to track
    quantities of interest of an SMC run.
    Note that initial_diagnosis and update_diagnosis
    must return copies and not modify in place for the class
    to be Jax Compilable, reason why they are static methods.
    """

    lmbda_evolution: jax.Array
    log_likelihood_increment_evolution: jax.Array
    ancestors_evolution: jax.Array
    weights_evolution: jax.Array

    @staticmethod
    def update_diagnosis(i, history, info, state):
        le, lli, ancestors, weights_evolution = history
        return SMCDiagnostics(
            le.at[i].set(state.tempering_param),
            lli.at[i].set(info.log_likelihood_increment),
            ancestors.at[i].set(info.ancestors),
            weights_evolution.at[i].set(state.weights),
        )

    @staticmethod
    def initial_diagnosis(iterations_to_diagnose, n_particles):
        return SMCDiagnostics(
            jnp.zeros(iterations_to_diagnose),
            jnp.zeros(iterations_to_diagnose),
            jnp.zeros((iterations_to_diagnose, n_particles)),
            jnp.zeros((iterations_to_diagnose, n_particles)),
        )


def flatten_single_particle(particle):
    return jnp.hstack([v.squeeze() for v in particle])


def inference_loop(rng_key, initial_state, kernel, iterations_to_diagnose, n_particles):
    """
    SMC inference loop that keeps tracks of diagnosis quantities.
    """

    def cond(carry):
        i, state, _, _ = carry
        return state.tempering_param < 1

    def one_step(carry):
        i, state, k, previous_info = carry
        k, subk = jax.random.split(k, 2)
        state, info = kernel.step(subk, state)
        full_info = SMCDiagnostics.update_diagnosis(i, previous_info, info, state)

        return i + 1, state, k, full_info

    n_iter, final_state, _, diagnosis = jax.lax.while_loop(
        cond,
        one_step,
        (
            0,
            initial_state,
            rng_key,
            SMCDiagnostics.initial_diagnosis(iterations_to_diagnose, n_particles),
        ),
    )

    return n_iter, final_state.particles, diagnosis


def blackjax_particles_from_pymc_population(model, pymc_population):
    """
    Transforms a pymc population of particles into the format
    accepted by BlackJax. Particles must be a PyTree, each leave represents
    a variable from the posterior, being an array of size n_particles
    * the variable's dimensionality.
    Note that the order in which variables are stored in the Pytree
    must be the same order used to calculate the logprior and loglikelihood.

    Parameters
    ----------
    pymc_population : A dictionary with variables as keys, and arrays
    with samples as values.
    """

    order_of_vars = model.value_vars

    def _format(var):
        variable = pymc_population[var.name]
        if len(variable.shape) == 1:
            return variable[:, np.newaxis]
        else:
            return variable

    return [_format(var) for var in order_of_vars]


def add_to_inference_data(
    inference_data: az.InferenceData,
    n_particles: int,
    target_ess: float,
    num_mcmc_steps: int,
    kernel: str,
    diagnosis: SMCDiagnostics,
    total_iterations: int,
    iterations_to_diagnose: int,
    kernel_parameters: dict,
    running_time_seconds: float,
):
    """
    Adds several SMC parameters into the az.InferenceData result

    Parameters
    ----------
    inference_data: arviz object to add attributes to.
    n_particles: number of particles present in the result
    target_ess: target effective sampling size between SMC iterations, used
    to calculate the tempering exponent
    num_mcmc_steps: number of steps of the inner kernel when mutating particles
    kernel: string representing the kernel used to mutate particles
    diagnosis: SMCDiagnostics, containing quantities of interest for the full
    SMC run
    total_iterations: the total number of iterations executed by the sampler
    iterations_to_diagnose: the number of iterations represented in the diagnosed
    quantities
    kernel_parameters: dict parameters from the inner kernel used to mutate particles
    running_time_seconds: float sampling time
    """
    experiment_parameters = {
        "particles": n_particles,
        "target_ess": target_ess,
        "num_mcmc_steps": num_mcmc_steps,
        "iterations": total_iterations,
        "iterations_to_diagnose": iterations_to_diagnose,
        "sampler": f"Blackjax SMC with {kernel} kernel",
    }

    inference_data.posterior.attrs["lambda_evolution"] = np.array(
        diagnosis.lmbda_evolution
    )[:iterations_to_diagnose]
    inference_data.posterior.attrs["log_likelihood_increments"] = np.array(
        diagnosis.log_likelihood_increment_evolution
    )[:iterations_to_diagnose]
    inference_data.posterior.attrs["ancestors_evolution"] = np.array(
        diagnosis.ancestors_evolution
    )[:iterations_to_diagnose]
    inference_data.posterior.attrs["weights_evolution"] = np.array(
        diagnosis.weights_evolution
    )[:iterations_to_diagnose]

    for k in experiment_parameters:
        inference_data.posterior.attrs[k] = experiment_parameters[k]

    for k in kernel_parameters:
        inference_data.posterior.attrs[k] = kernel_parameters[k]

    inference_data.posterior.attrs["running_time_seconds"] = running_time_seconds

    return inference_data


def get_jaxified_logprior(model) -> Callable:
    return get_jaxified_particles_fn(model, model.varlogp)


def get_jaxified_loglikelihood(model) -> Callable:
    return get_jaxified_particles_fn(model, model.datalogp)


def get_jaxified_particles_fn(model, graph_outputs):
    """
    Builds a Jaxified version of a value_vars function,
    that is applyable to Blackjax particles format.
    """
    logp_fn = get_jaxified_graph(inputs=model.value_vars, outputs=[graph_outputs])

    def logp_fn_wrap(particles):
        return logp_fn(*[p.squeeze() for p in particles])[0]

    return logp_fn_wrap


def initialize_population(model, draws, random_seed) -> dict[str, np.ndarray]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="The effect of Potentials"
        )

        prior_expression = make_initial_point_expression(
            free_rvs=model.free_RVs,
            rvs_to_transforms=model.rvs_to_transforms,
            initval_strategies={},
            default_strategy="prior",
            return_transformed=True,
        )
        prior_values = draw(prior_expression, draws=draws, random_seed=random_seed)

        names = [model.rvs_to_values[rv].name for rv in model.free_RVs]
        dict_prior = {k: np.stack(v) for k, v in zip(names, prior_values)}

    return cast(dict[str, np.ndarray], dict_prior)


def var_map_from_model(model, initial_point) -> dict:
    """
    Computes a dictionary that maps
    variable names to tuples (shape, size)
    """
    var_info = {}
    for v in model.value_vars:
        var_info[v.name] = (initial_point[v.name].shape, initial_point[v.name].size)
    return var_info


def build_smc_with_kernel(
    prior_log_prob,
    loglikelihood,
    target_ess,
    num_mcmc_steps,
    kernel_parameters,
    mcmc_kernel,
):
    return blackjax.adaptive_tempered_smc(
        prior_log_prob,
        loglikelihood,
        mcmc_kernel.build_kernel(),
        mcmc_kernel.init,
        mcmc_parameters=kernel_parameters,
        resampling_fn=systematic,
        target_ess=target_ess,
        num_mcmc_steps=num_mcmc_steps,
    )
