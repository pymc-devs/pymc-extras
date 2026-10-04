import numpy as np
import pymc as pm
import pytest

from arviz_base import from_dict
from pymc.model.fgraph import fgraph_from_model

from pymc_extras.marginal import (
    approximate_conditional,
    approximate_marginalize,
    approximate_recover,
    conditional,
    marginalize,
    recover,
    unmarginalize,
)
from pymc_extras.model.marginal.distributions.laplace import MarginalLaplaceRV

# SciPy's optimizer runs through PyTensor's supported Numba object-mode fallback.
pytestmark = pytest.mark.filterwarnings(
    r"ignore:Numba will use object mode to run MinimizeOp\(:UserWarning"
)


def test_mixed_laplace_marginalization():
    """Laplace settings survive re-marginalization, and the joint mixed call works."""

    def build_model():
        with pm.Model() as m:
            x = pm.MvNormal("x", mu=np.zeros(2), tau=np.eye(2))
            z = pm.Bernoulli("z", p=pm.math.sigmoid(x.sum()))
            pm.Normal("y", mu=z * 2.0, sigma=1.0, observed=1.5)
        return m

    minimizer_kwargs = {"method": "L-BFGS-B", "optimizer_kwargs": {"tol": 1e-6}}

    def assert_laplace_preserved(marginal_model):
        fg, _ = fgraph_from_model(marginal_model)
        [laplace_op] = [n.op for n in fg.apply_nodes if isinstance(n.op, MarginalLaplaceRV)]
        assert laplace_op.marginalized_name == "x"
        assert laplace_op.minimizer_kwargs == minimizer_kwargs

    # Sequential: laplace first, then a variable it absorbed as dependent
    laplace_m = approximate_marginalize(
        build_model(), laplace_approx={"x": np.eye(2)}, minimizer_kwargs=minimizer_kwargs
    )
    assert_laplace_preserved(approximate_marginalize(laplace_m, "z"))

    # Joint single call with mixed settings
    joint_m = approximate_marginalize(
        build_model(), "z", laplace_approx={"x": np.eye(2)}, minimizer_kwargs=minimizer_kwargs
    )
    assert_laplace_preserved(joint_m)

    # Partial unmarginalize recovers z but keeps x marginalized with its settings
    assert_laplace_preserved(unmarginalize(joint_m, "z"))


def test_laplace_gaussian_conditional_and_recover():
    with pm.Model() as model:
        offset = pm.Normal("offset")
        x = pm.MvNormal("x", mu=np.zeros(2), tau=np.eye(2))
        pm.Normal("y", mu=x + offset, sigma=1, observed=[1.0, 2.0])

    marginal_model = approximate_marginalize(model, laplace_approx={"x": np.eye(2)})
    for exact_api in (marginalize, conditional):
        with pytest.raises(ValueError):
            exact_api(marginal_model)
    with pytest.raises(TypeError):
        marginalize(model, laplace_approx={"x": np.eye(2)})
    offsets = np.repeat([-0.5, 1.5], 1000)
    idata = from_dict({"posterior": {"offset": offsets[None, :]}})
    with pytest.raises(ValueError):
        recover(idata, model=marginal_model)
    samples = approximate_recover(idata, model=marginal_model, random_seed=42)["posterior"][
        "x"
    ].values[0]
    for offset in (-0.5, 1.5):
        conditional_samples = samples[offsets == offset]
        np.testing.assert_allclose(
            conditional_samples.mean(axis=0), (np.array([1, 2]) - offset) / 2, atol=0.08
        )
        np.testing.assert_allclose(
            np.cov(conditional_samples, rowvar=False), np.eye(2) / 2, atol=0.08
        )


def test_laplace_multiple_dependents_logp():
    with pm.Model() as model:
        x = pm.MvNormal("x", mu=np.zeros(1), tau=np.eye(1))
        pm.Normal("y", mu=x, sigma=1, observed=[1.0])
        pm.Normal("z", mu=x, sigma=1, observed=[2.0])

    marginal_model = approximate_marginalize(model, laplace_approx={"x": np.eye(1)})
    conditional_model = approximate_conditional(marginal_model)
    logp = conditional_model.compile_logp(vars=[conditional_model["x"]])
    np.testing.assert_allclose(logp({"x": np.ones(1)}), 0.5 * np.log(3 / (2 * np.pi)), atol=1e-6)
    np.testing.assert_allclose(
        marginal_model.compile_logp()({}), -np.log(2 * np.pi) - 0.5 * np.log(3) - 1, atol=1e-6
    )


def test_laplace_nonlinear_conditional_curvature_at_mode():
    with pm.Model() as model:
        offset = pm.Normal("offset")
        x = pm.Normal("x", sigma=1 / np.sqrt(3), shape=1)
        pm.Poisson("y", mu=pm.math.exp(x + offset), observed=[2])

    marginal_model = approximate_marginalize(model, laplace_approx={"x": 3 * np.eye(1)})
    conditional_model = approximate_conditional(marginal_model)
    rv = conditional_model["x"]
    mean, covariance = rv.owner.op.dist_params(rv.owner)
    parameters = conditional_model.compile_fn(
        conditional_model.replace_rvs_by_values([mean, covariance])
    )
    # At x=0.5 and rate=0.5, the posterior gradient is zero and precision is 3.5.
    mean, covariance = parameters({"offset": np.log(0.5) - 0.5})
    np.testing.assert_allclose(mean, [0.5], atol=1e-5)
    np.testing.assert_allclose(covariance, [[2 / 7]], atol=1e-6)
