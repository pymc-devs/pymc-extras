from step_methods.rjmcmc import *

import functools
import itertools

from scipy.linalg import expm
import scipy
import numpy as np

import pymc as pm
from pymc.aesaraf import compile_pymc

import dill as pickle
import arviz as az

import aesara.tensor as tt


def get_analytical_solution_for_matrix(deltas, ks, initial_values):
    """Solution of the form s[t] = exp(Mt)s[0]"""
    # build matrix M to exponentiate
    M = np.zeros_like(deltas)
    for i, (drow, krow) in enumerate(zip(deltas, ks)):
        for j, (d, k) in enumerate(zip(drow, krow)):
            M[j, i] += d * k
            if i == j:
                # along the diagonal we subtract the column trace
                M[i, j] -= np.sum(deltas[i, ...] * ks[i, ...])

    def f(t):
        return expm(M * t) @ initial_values

    return f


def get_vectorized_solution_for_matrix(deltas, ks, initial_values):
    f = get_analytical_solution_for_matrix(deltas, ks, initial_values)

    def function_for_substance(i):
        return lambda t: f(t)[i]

    # Just splitting this into a dictionary for each species cause I'm confused
    return {
        i: np.vectorize(function_for_substance(i)) for i in range(len(initial_values))
    }


class LogLike(tt.Op):
    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [tt.dvector]  # expects a vector of parameter values when called
    otypes = [tt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        """

        # add inputs as class attributes
        self.loglikelihood = loglike
        self.logpgrad = LogLikeGrad(self.loglikelihood)

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.loglikelihood(theta)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

    def grad(self, inputs, g):
        # TODO test this somehow
        (theta,) = inputs
        return [g[0] * self.logpgrad(theta)]


class LogLikeGrad(tt.Op):
    """
    TODO test this somehow
    This Op will be called with a vector of values and also return a vector of
    values - the gradients in each dimension.
    """

    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def __init__(self, loglike):
        """
        Initialise with various things that the function requires. Below
        are the things that are needed in this particular example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that out function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike

    def perform(self, node, inputs, outputs):
        (theta,) = inputs

        outputs[0][0] = scipy.optimize.approx_fprime(
            theta, self.likelihood, epsilon=np.sqrt(np.finfo(float).eps)
        )


class UniformJump(Jump):
    """Placeholder class for jumps"""

    # TODO figure out how to generate and gather stats properly
    # generates_stats = True
    # stats_dtypes = [
    #     {
    #         "accept": np.float64
    #         ,"accepted": bool
    #         ,"diverged": bool
    #         , "tune": bool
    #     }
    # ]

    @classmethod
    def get_deltas_to_flip(self, src, dest):
        return {d for i, d in enumerate(delta_parameters) if src[i] != dest[i]}

    def __new__(self, k_parameters, *args, **kwargs):
        return super().__new__(self, *args, **kwargs)

    def __init__(self, k_parameters, *args, **kwargs):
        """
        Creates a very simple Jump that randomely selects new k's from a uniform distribution
        """

        self.uniform_scaling = 10
        self.k_parameters = k_parameters

        super().__init__(*args, **kwargs)

    def get_subspace_updater(self):
        # Precompute setting ks
        ks_to_randomize = [
            (str(k.tag.value_var), self.get_forward_transform(k))
            for i, k in enumerate(self.k_parameters)
            if self.dest[i] == 1 and self.src[i] == 0
        ]

        def set_new_ks(point):
            for k, transform in ks_to_randomize:
                point[k] = transform(np.random.random() * self.uniform_scaling)

        return set_new_ks

    def get_deltas_updater(self):
        # precompute delta update
        targets = [
            str(d)
            for i, d in enumerate(self.delta_parameters)
            if self.src[i] != self.dest[i]
        ]

        def f(point):
            for delta in targets:
                point[delta] = 1 - point[delta]

        return f

    def get_log_acceptance_fraction_calculator(self):
        dimension_mismatch = np.sum(self.dest) - np.sum(self.src)
        self.q_term = +dimension_mismatch * np.log(self.uniform_scaling)

        def f(new_point, point):
            return (
                self.logp(new_point) - self.logp(point) + self.diff_logj + self.q_term
            )

        return f


if __name__ == "__main__":
    # Generate Data from the true model
    timepoints = np.linspace(0, 5, 1000)
    true_deltas = np.array([[0, 1], [0, 0]])
    true_ks = np.array([[0, 2], [0, 0]])
    initial_values = np.array([5, 0])
    true_model = get_vectorized_solution_for_matrix(
        true_deltas, true_ks, initial_values
    )
    true_data = [true_model[i](timepoints) for i in range(2)]

    # Define loglikelihood
    def my_loglike(vec):
        # we'll just hardcode this cause I'm tired

        if (vec[0] != 0 and vec[0] != 1) and (vec[1] != 0 and vec[1] != 1):
            raise (ValueError("Delta parameters can only be one or zero"))

        deltas = np.array([[0, vec[0]], [vec[1], 0]])
        ks = np.array([[0, vec[2]], [vec[3], 0]])
        new_model = get_vectorized_solution_for_matrix(deltas, ks, initial_values)
        simulated_data = [new_model[i](timepoints) for i in range(2)]
        sum = 0
        for i in range(2):
            sum += np.sum(
                np.log(scipy.stats.norm.pdf(simulated_data[i] - true_data[i]))
            )

        if not np.isfinite(sum):
            raise (ValueError("loglikelihood diverged"))

        return sum

    # Generate the likelihood
    ndraws = 10
    nburn = 15
    pymc_model = pm.Model()
    theano_loglike = LogLike(lambda x: my_loglike(x))
    with pymc_model:
        # Generate priors based on the Supermodel priors
        var_assoc = []
        delta_parameters = []
        k_parameters = []
        for i in range(2):
            for j in range(2):
                if i != j:
                    delta = pm.Bernoulli("delta_{}_{}".format(i, j), 0.5)
                    k = pm.Uniform("k_{}_{}".format(i, j), lower=0, upper=10)
                    var_assoc.append((delta, k))
                    delta_parameters.append(delta)
                    k_parameters.append(k)

        pymc_parameters = [d for d, _ in var_assoc] + [k for _, k in var_assoc]

        theta = tt.as_tensor_variable(pymc_parameters)

        pm.Potential("loglikelihood", theano_loglike(theta))

        # Delta configurations and subspace sets
        configurations = [(0, 1), (1, 0), (1, 1)]
        configurations_to_subspaces = {
            (0, 1): [pymc_model.named_vars["k_0_1"]],
            (1, 0): [pymc_model.named_vars["k_1_0"]],
            (1, 1): [pymc_model.named_vars["k_0_1"], pymc_model.named_vars["k_1_0"]],
        }

        # Inter subspace jump steppers
        jump_probas = collections.defaultdict(dict)
        # we filter out self maps and maps to/from the zero dynamics model
        for c1, c2 in filter(
            lambda x: x[0] != x[1], itertools.product(configurations, repeat=2)
        ):
            # we'll have everything be equiprobable but no jumping towards zero dynamics
            jump_probas[c1][c2] = 1 if c2 != tuple([0] * len(delta_parameters)) else 0

        # renormalize the jump probabilities
        for c1 in jump_probas:
            sum = np.sum(list(jump_probas[c1].values()))
            jump_probas[c1] = {c2: val / sum for c2, val in jump_probas[c1].items()}

        # Remap jump probabilities into steppers
        jumps = {
            c1: {
                c2: UniformJump(
                    k_parameters,
                    delta_parameters,
                    configurations_to_subspaces,
                    c1,
                    c2,
                    jump_probas[c1][c2],
                    jump_probas[c2][c1],
                )
                for c2 in jump_probas[c1]
            }
            for c1 in jump_probas
        }

        step = RJMCMC(delta_parameters, configurations_to_subspaces, jumps, jump_probas)

        trace = pm.sample(
            ndraws, tune=nburn, step=step, discard_tuned_samples=True, cores=1
        )

    # # Just dump all the data inzto a pickle
    # with open('pymc_trace.pkl', 'wb') as buff:
    #     pickle.dump({'model':pymc_model, 'trace':trace}, buff)

    # with open('pymc_trace.pkl', 'rb') as buff:
    #     data = pickle.load(buff)
    #     trace = data['trace']

    # figure_directory = 'figures/simple_reaction_reversibility/'
    # with pymc_model:
    #     az.plot_trace(trace.posterior, var_names = ['delta_0_1', 'delta_1_0'])
    #     full_screen_save(figure_directory, 'delta_trace.png')
    #     az.plot_trace(trace.posterior, var_names = ['k_0_1', 'k_1_0'])
    #     full_screen_save(figure_directory, 'k_trace.png')
