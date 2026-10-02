import numpy as np
import pymc as pm
import pytest

from arviz_base import from_dict
from pymc.model.fgraph import fgraph_from_model

from pymc_extras.marginal import conditional, marginalize, recover, unmarginalize
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
    laplace_m = marginalize(
        build_model(), laplace_approx={"x": np.eye(2)}, minimizer_kwargs=minimizer_kwargs
    )
    assert_laplace_preserved(marginalize(laplace_m, "z"))

    # Joint single call with mixed settings
    joint_m = marginalize(
        build_model(), "z", laplace_approx={"x": np.eye(2)}, minimizer_kwargs=minimizer_kwargs
    )
    assert_laplace_preserved(joint_m)

    # Partial unmarginalize recovers z but keeps x marginalized with its settings
    assert_laplace_preserved(unmarginalize(joint_m, "z"))


def test_laplace_gaussian_conditional_and_recover():
    """An affine Gaussian likelihood gives an exact Gaussian conditional."""
    with pm.Model(coords={"feature": ["a", "b"]}) as model:
        offset = pm.Normal("offset")
        x = pm.MvNormal("x", mu=np.zeros(2), tau=np.eye(2), dims="feature")
        pm.Normal("y", mu=x + offset, sigma=1, observed=[1.0, 2.0])

    marginal_model = marginalize(model, laplace_approx={"x": np.eye(2)})
    conditional_model = conditional(marginal_model)
    rv = conditional_model["x"]
    mean, covariance = rv.owner.op.dist_params(rv.owner)
    parameters = conditional_model.compile_fn(
        conditional_model.replace_rvs_by_values([mean, covariance])
    )
    for offset in (-0.5, 1.5):
        actual_mean, actual_covariance = parameters({"offset": offset})
        np.testing.assert_allclose(actual_mean, (np.array([1, 2]) - offset) / 2, atol=1e-6)
        np.testing.assert_allclose(actual_covariance, np.eye(2) / 2, atol=1e-6)

    offsets = np.repeat([-0.5, 1.5], 1000)
    idata = from_dict({"posterior": {"offset": offsets[None, :]}})
    posterior = recover(idata, model=marginal_model, random_seed=42)["posterior"]
    assert posterior["x"].dims == ("chain", "draw", "feature")
    np.testing.assert_array_equal(posterior.coords["feature"], ["a", "b"])
    for offset in (-0.5, 1.5):
        samples = posterior["x"].values[0, offsets == offset]
        np.testing.assert_allclose(samples.mean(axis=0), (np.array([1, 2]) - offset) / 2, atol=0.08)
        np.testing.assert_allclose(np.cov(samples, rowvar=False), np.eye(2) / 2, atol=0.08)


def test_laplace_conditional_nonzero_mean_precision_and_multiple_dependents():
    """All dependent values enter the mode and curvature, including free dependents."""
    prior_precision = np.array([[2.0, 0.4], [0.4, 1.5]])
    prior_mean = np.array([1.0, -0.5])
    design = np.array([[1.0, 2.0], [-0.5, 1.0]])
    observed = np.array([0.5, 2.0])
    with pm.Model() as model:
        parent = pm.Normal("parent")
        x = pm.MvNormal("x", mu=prior_mean + parent, tau=prior_precision)
        pm.Normal("y", mu=design @ x + parent, sigma=2, observed=observed)
        pm.Normal("z", mu=x - parent, sigma=0.5, shape=2)

    marginal_model = marginalize(model, laplace_approx={"x": prior_precision})
    conditional_model = conditional(marginal_model, "x")
    rv = conditional_model["x"]
    mean, covariance = rv.owner.op.dist_params(rv.owner)
    parameters = conditional_model.compile_fn(
        conditional_model.replace_rvs_by_values([mean, covariance])
    )
    # Compiling the conditional logp also rejects leaked marginalized RVs.
    logp = conditional_model.compile_logp(vars=[rv])
    marginal_logp = marginal_model.compile_logp()
    joint_design = np.vstack([design, np.eye(2)])
    joint_covariance = joint_design @ np.linalg.solve(prior_precision, joint_design.T) + np.diag(
        [4.0, 4.0, 0.25, 0.25]
    )
    expected_precision = prior_precision + design.T @ design / 4 + np.eye(2) * 4
    for parent, z in [(0.7, np.array([2.0, -1.0])), (-1.2, np.array([-0.5, 3.0]))]:
        actual_mean, actual_covariance = parameters({"parent": parent, "z": z})
        expected_mean = np.linalg.solve(
            expected_precision,
            prior_precision @ (prior_mean + parent)
            + design.T @ (observed - parent) / 4
            + 4 * (z + parent),
        )
        np.testing.assert_allclose(actual_mean, expected_mean, atol=1e-6)
        np.testing.assert_allclose(actual_covariance, np.linalg.inv(expected_precision), atol=1e-6)
        point = {"x": expected_mean, "parent": parent, "z": z}
        expected_logp = -np.log(2 * np.pi) + np.linalg.slogdet(expected_precision)[1] / 2
        np.testing.assert_allclose(logp(point), expected_logp, atol=1e-6)
        joint_mean = np.concatenate([design @ (prior_mean + parent) + parent, prior_mean])
        residual = np.concatenate([observed, z]) - joint_mean
        expected_marginal_logp = -0.5 * (
            4 * np.log(2 * np.pi)
            + np.linalg.slogdet(joint_covariance)[1]
            + residual @ np.linalg.solve(joint_covariance, residual)
        ) - 0.5 * (np.log(2 * np.pi) + parent**2)
        np.testing.assert_allclose(
            marginal_logp({"parent": parent, "z": z}), expected_marginal_logp, atol=1e-6
        )


def test_laplace_nonlinear_conditional_curvature_at_mode():
    """Poisson curvature changes with x and must be evaluated at the optimized mode."""
    prior_mean = 1.2
    prior_precision = 3.0
    observed = 4
    with pm.Model() as model:
        offset = pm.Normal("offset")
        x = pm.Normal("x", mu=prior_mean, sigma=1 / np.sqrt(prior_precision), shape=1)
        pm.Poisson("y", mu=pm.math.exp(x + offset), observed=[observed])

    marginal_model = marginalize(
        model,
        laplace_approx={"x": np.array([[prior_precision]])},
        minimizer_kwargs={"method": "BFGS", "optimizer_kwargs": {"tol": 1e-10}},
    )
    conditional_model = conditional(marginal_model)
    rv = conditional_model["x"]
    mean, covariance = rv.owner.op.dist_params(rv.owner)
    parameters = conditional_model.compile_fn(
        conditional_model.replace_rvs_by_values([mean, covariance])
    )
    for mode in (0.2, 1.4):
        # Choose offset so Q * (mode - mu) + exp(mode + offset) - y == 0.
        rate = observed - prior_precision * (mode - prior_mean)
        offset = np.log(rate) - mode
        actual_mean, actual_covariance = parameters({"offset": offset})
        np.testing.assert_allclose(actual_mean, [mode], atol=1e-6)
        np.testing.assert_allclose(actual_covariance, [[1 / (prior_precision + rate)]], atol=1e-6)


def test_laplace_conditional_with_nested_normal_marginal():
    """Nested recovery keeps the recovered Laplace variable as the child's parent."""
    with pm.Model() as model:
        x = pm.MvNormal("x", mu=np.zeros(2), tau=np.eye(2))
        z = pm.Normal("z", mu=x[0], sigma=1)
        pm.Normal("y", mu=z, sigma=1, observed=3.0)

    marginal_model = marginalize(model, "z", laplace_approx={"x": np.eye(2)})
    conditional_model = conditional(marginal_model)
    rv = conditional_model["x"]
    mean, covariance = rv.owner.op.dist_params(rv.owner)
    actual_mean, actual_covariance = conditional_model.compile_fn([mean, covariance])({})
    np.testing.assert_allclose(actual_mean, [1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(actual_covariance, np.diag([2 / 3, 1]), atol=1e-6)

    child = conditional_model["z"]
    child_parameters = conditional_model.compile_fn(
        conditional_model.replace_rvs_by_values(child.owner.op.dist_params(child.owner))
    )
    for x_value in (np.array([0.0, -1.0]), np.array([2.0, 1.0])):
        child_mean, child_sigma = child_parameters({"x": x_value})
        np.testing.assert_allclose(child_mean, (x_value[0] + 3) / 2)
        np.testing.assert_allclose(child_sigma, np.sqrt(0.5))
