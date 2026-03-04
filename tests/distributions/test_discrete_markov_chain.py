import numpy as np
import pymc as pm

# general imports
import pytensor.tensor as pt
import pytest

from pymc.distributions import Categorical
from pymc.distributions.shape_utils import change_dist_size
from pymc.logprob.utils import ParameterValueError
from pymc.sampling.mcmc import assign_step_methods

from pymc_extras.distributions.timeseries import (
    DiscreteMarkovChain,
    DiscreteMarkovChainGibbsMetropolis,
)


def transition_probability_tests(steps, n_states, n_lags, n_draws, atol):
    P = np.full((n_states,) * (n_lags + 1), 1 / n_states)
    x0 = pm.Categorical.dist(p=np.ones(n_states) / n_states)

    chain = DiscreteMarkovChain.dist(
        P=pt.as_tensor_variable(P), init_dist=x0, steps=steps, n_lags=n_lags
    )

    draws = pm.draw(chain, n_draws, random_seed=172)

    # Test x0 is uniform over n_states
    for i in range(n_lags):
        assert np.allclose(
            np.histogram(draws[:, ..., i], bins=n_states)[0] / n_draws,
            1 / n_states,
            atol=atol,
        )

    n_grams = [[tuple(row[i : i + n_lags + 1]) for i in range(len(row) - n_lags)] for row in draws]
    freq_table = np.zeros((n_states,) * (n_lags + 1))

    for row in n_grams:
        for ngram in row:
            freq_table[ngram] += 1
    freq_table /= freq_table.sum(axis=-1)[:, None]

    # Test continuation probabilities match P
    assert np.allclose(P, freq_table, atol=atol)


class TestDiscreteMarkovRV:
    def test_fail_if_P_not_square(self):
        P = pt.eye(3, 2)
        x0 = pm.Categorical.dist(p=np.ones(3) / 3)

        chain = DiscreteMarkovChain.dist(P=P, init_dist=x0, steps=3)
        with pytest.raises(ParameterValueError):
            pm.logp(chain, np.zeros((3,))).eval()

    def test_fail_if_P_not_valid(self):
        P = pt.zeros((3, 3))
        x0 = pm.Categorical.dist(p=np.ones(3) / 3)
        chain = DiscreteMarkovChain.dist(P=P, init_dist=x0, steps=3)
        with pytest.raises(ParameterValueError):
            pm.logp(chain, np.zeros((3,))).eval()

    def test_high_dimensional_P(self):
        P = pm.Dirichlet.dist(a=pt.ones(3), size=(3, 3, 3))
        n_lags = 3
        x0 = pm.Categorical.dist(p=np.ones(3) / 3)
        chain = DiscreteMarkovChain.dist(P=P, steps=10, init_dist=x0, n_lags=n_lags)
        draws = pm.draw(chain, 10)
        logp = pm.logp(chain, draws)

    def test_default_init_dist_warns_user(self):
        P = pt.as_tensor_variable(np.array([[0.1, 0.5, 0.4], [0.3, 0.4, 0.3], [0.9, 0.05, 0.05]]))

        with pytest.warns(UserWarning):
            DiscreteMarkovChain.dist(P=P, steps=3)

    def test_logp_shape(self):
        P = pt.as_tensor_variable(np.array([[0.1, 0.5, 0.4], [0.3, 0.4, 0.3], [0.9, 0.05, 0.05]]))
        x0 = pm.Categorical.dist(p=np.ones(3) / 3)

        # Test with steps
        chain = DiscreteMarkovChain.dist(P=P, init_dist=x0, steps=3)
        draws = pm.draw(chain, 5)
        logp = pm.logp(chain, draws).eval()

        assert logp.shape == (5,)

        # Test with shape
        chain = DiscreteMarkovChain.dist(P=P, init_dist=x0, shape=(3,))
        draws = pm.draw(chain, 5)
        logp = pm.logp(chain, draws).eval()

        assert logp.shape == (5,)

    def test_logp_with_default_init_dist(self):
        P = pt.as_tensor_variable(np.array([[0.1, 0.5, 0.4], [0.3, 0.4, 0.3], [0.9, 0.05, 0.05]]))
        x0 = pm.Categorical.dist(p=np.ones(3) / 3)

        value = np.array([0, 1, 2])
        logp_expected = np.log((1 / 3) * 0.5 * 0.3)

        # Test dist directly
        chain = DiscreteMarkovChain.dist(P=P, init_dist=x0, steps=3)
        logp_eval = pm.logp(chain, value).eval()
        np.testing.assert_allclose(logp_eval, logp_expected, rtol=1e-6)

        # Test via Model
        with pm.Model() as m:
            DiscreteMarkovChain("chain", P=P, init_dist=x0, steps=3)
        model_logp_eval = m.compile_logp()({"chain": value})
        np.testing.assert_allclose(model_logp_eval, logp_expected, rtol=1e-6)

    def test_logp_with_user_defined_init_dist(self):
        P = pt.as_tensor_variable(np.array([[0.1, 0.5, 0.4], [0.3, 0.4, 0.3], [0.9, 0.05, 0.05]]))
        x0 = pm.Categorical.dist(p=[0.2, 0.6, 0.2])
        chain = DiscreteMarkovChain.dist(P=P, init_dist=x0, steps=3)

        logp = pm.logp(chain, [0, 1, 2]).eval()
        assert logp == np.log(0.2 * 0.5 * 0.3)

    def test_moment_function(self):
        P_np = np.array([[0.1, 0.5, 0.4], [0.3, 0.4, 0.3], [0.9, 0.05, 0.05]])

        x0_np = np.array([0, 1, 0])

        P = pt.as_tensor_variable(P_np)
        x0 = pm.Categorical.dist(p=x0_np.tolist())
        n_steps = 3

        chain = DiscreteMarkovChain.dist(P=P, init_dist=x0, steps=n_steps)

        chain_np = np.empty(shape=n_steps + 1, dtype="int8")
        chain_np[0] = np.argmax(x0_np)
        for i in range(n_steps):
            state = chain_np[i]
            chain_np[i + 1] = np.argmax(P_np[state])

        dmc_chain = pm.distributions.distribution.support_point(chain).eval()

        assert np.allclose(dmc_chain, chain_np)

    def test_define_steps_via_shape_arg(self):
        P = pt.full((3, 3), 1 / 3)
        x0 = pm.Categorical.dist(p=np.ones(3) / 3)

        chain = DiscreteMarkovChain.dist(P=P, init_dist=x0, shape=(3,))
        assert chain.eval().shape == (3,)

        chain = DiscreteMarkovChain.dist(P=P, init_dist=x0, shape=(3, 2))
        assert chain.eval().shape == (3, 2)

    def test_define_steps_via_dim_arg(self):
        coords = {"steps": [1, 2, 3]}

        with pm.Model(coords=coords):
            P = pt.full((3, 3), 1 / 3)
            x0 = pm.Categorical.dist(p=np.ones(3) / 3)

            chain = DiscreteMarkovChain("chain", P=P, init_dist=x0, dims=["steps"])

        assert chain.eval().shape == (3,)

    def test_dims_when_steps_are_defined(self):
        coords = {"steps": [1, 2, 3, 4]}

        with pm.Model(coords=coords):
            P = pt.full((3, 3), 1 / 3)
            x0 = pm.Categorical.dist(p=np.ones(3) / 3)

            chain = DiscreteMarkovChain("chain", P=P, steps=3, init_dist=x0, dims=["steps"])

        assert chain.eval().shape == (4,)

    def test_multiple_dims_with_steps(self):
        coords = {"steps": [1, 2, 3], "mc_chains": [1, 2, 3]}

        with pm.Model(coords=coords):
            P = pt.full((3, 3), 1 / 3)
            x0 = pm.Categorical.dist(p=np.ones(3) / 3)

            chain = DiscreteMarkovChain(
                "chain", P=P, steps=2, init_dist=x0, dims=["steps", "mc_chains"]
            )

        assert chain.eval().shape == (3, 3)

    def test_mutiple_dims_with_steps_and_init_dist(self):
        coords = {"steps": [1, 2, 3], "mc_chains": [1, 2, 3]}

        with pm.Model(coords=coords):
            P = pt.full((3, 3), 1 / 3)
            x0 = pm.Categorical.dist(p=[0.1, 0.1, 0.8], size=(3,))
            chain = DiscreteMarkovChain(
                "chain", P=P, init_dist=x0, steps=2, dims=["steps", "mc_chains"]
            )

        assert chain.eval().shape == (3, 3)

    def test_multiple_lags_with_data(self):
        with pm.Model():
            P = pt.full((3, 3, 3), 1 / 3)
            x0 = pm.Categorical.dist(p=[0.1, 0.1, 0.8], size=2)
            data = pm.draw(x0, 100)

            chain = DiscreteMarkovChain("chain", P=P, init_dist=x0, n_lags=2, observed=data)

        assert chain.eval().shape == (100, 2)

    def test_random_draws(self):
        transition_probability_tests(steps=3, n_states=2, n_lags=1, n_draws=2500, atol=0.05)
        transition_probability_tests(steps=3, n_states=2, n_lags=3, n_draws=7500, atol=0.05)

    def test_change_size_univariate(self):
        P = pt.as_tensor_variable(np.array([[0.1, 0.5, 0.4], [0.3, 0.4, 0.3], [0.9, 0.05, 0.05]]))
        x0 = pm.Categorical.dist(p=np.ones(3) / 3)

        chain = DiscreteMarkovChain.dist(P=P, init_dist=x0, shape=(100, 5))

        new_rw = change_dist_size(chain, new_size=(7,))
        assert tuple(new_rw.shape.eval()) == (7, 5)

        new_rw = change_dist_size(chain, new_size=(4, 3), expand=True)
        assert tuple(new_rw.shape.eval()) == (4, 3, 100, 5)

    def test_mcmc_sampling(self):
        with pm.Model(coords={"step": range(100)}) as model:
            init_dist = Categorical.dist(p=[0.5, 0.5])
            markov_chain = DiscreteMarkovChain(
                "markov_chain",
                P=[[0.1, 0.9], [0.1, 0.9]],
                init_dist=init_dist,
                shape=(100,),
                dims="step",
            )

            _, assigned_step_methods = assign_step_methods(model)
            assert assigned_step_methods[DiscreteMarkovChainGibbsMetropolis] == [
                model.rvs_to_values[markov_chain]
            ]

            # Sampler needs no tuning
            idata = pm.sample(
                tune=0,
                chains=4,
                draws=250,
                progressbar=False,
                compute_convergence_checks=False,
            )

        np.testing.assert_allclose(
            idata.posterior["markov_chain"].isel(step=0).mean(("chain", "draw")),
            0.5,
            atol=0.05,
        )

        np.testing.assert_allclose(
            idata.posterior["markov_chain"].isel(step=slice(1, None)).mean(("chain", "draw")),
            0.9,
            atol=0.05,
        )

        assert pm.stats.ess(idata, method="tail").min() > 950
