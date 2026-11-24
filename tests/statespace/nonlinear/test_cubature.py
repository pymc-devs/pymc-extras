import numpy as np
import pytest

from numpy.testing import assert_allclose

from pymc_extras.statespace.nonlinear import cubature_points


@pytest.mark.parametrize("columns", [3, 2], ids=["square", "fewer_columns_than_rows"])
def test_cubature_exact_for_quadratics(columns):
    rng = np.random.default_rng(0)
    mean = rng.normal(size=3)
    factor = rng.normal(size=(3, columns))
    covariance = factor @ factor.T

    points = cubature_points(mean=mean, factor=factor).eval()

    assert points.shape == (2 * columns, 3)
    assert_allclose(points.mean(axis=0), mean, atol=1e-12)
    assert_allclose(
        np.einsum("pi,pj->ij", points, points) / len(points),
        covariance + np.outer(mean, mean),
        atol=1e-12,
    )
