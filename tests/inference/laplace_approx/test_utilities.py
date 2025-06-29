import numpy as np
import pytest

from pymc_extras.inference.laplace_approx import utilities


@pytest.fixture
def rng():
    return np.random.default_rng()


def test_get_nearest_psd_returns_psd(rng):
    # Matrix with negative eigenvalues
    A = np.array([[2, -3], [-3, 2]])
    psd = utilities.get_nearest_psd(A)

    # Should be symmetric
    np.testing.assert_allclose(psd, psd.T)

    # All eigenvalues should be >= 0
    eigvals = np.linalg.eigvalsh(psd)
    assert np.all(eigvals >= -1e-12), "All eigenvalues should be non-negative"


def test_get_nearest_psd_given_psd_input(rng):
    L = rng.normal(size=(2, 2))
    A = L @ L.T
    psd = utilities.get_nearest_psd(A)

    # Given PSD input, should return the same matrix
    assert np.allclose(psd, A)


def test_set_optimizer_function_defaults_warns_and_prefers_hessp(caplog):
    # "trust-ncg" uses_grad=True, uses_hess=True, uses_hessp=True
    method = "trust-ncg"
    with caplog.at_level("WARNING"):
        use_grad, use_hess, use_hessp = utilities.set_optimizer_function_defaults(
            method, True, True, True
        )

    message = caplog.messages[0]
    assert message.startswith('Both "use_hess" and "use_hessp" are set to True')

    assert use_grad
    assert not use_hess
    assert use_hessp


def test_set_optimizer_function_defaults_infers_hess_and_hessp():
    # "trust-ncg" uses_grad=True, uses_hess=True, uses_hessp=True
    method = "trust-ncg"

    # If only use_hessp is set, use_hess should be False but use_grad should be inferred as True
    use_grad, use_hess, use_hessp = utilities.set_optimizer_function_defaults(
        method, None, None, True
    )
    assert use_grad
    assert not use_hess
    assert use_hessp

    # Only use_hess is set
    use_grad, use_hess, use_hessp = utilities.set_optimizer_function_defaults(
        method, None, True, None
    )
    assert use_hess
    assert not use_hessp


def test_set_optimizer_function_defaults_defaults():
    # "trust-ncg" uses_grad=True, uses_hess=True, uses_hessp=True
    method = "trust-ncg"
    use_grad, use_hess, use_hessp = utilities.set_optimizer_function_defaults(
        method, None, None, None
    )
    assert use_grad
    assert not use_hess
    assert use_hessp
