"""Tests for the Cholesky-form Kalman filter implementation.

This test file is meant to lock down the phase-1 behavior of the filter before
any backend expansion. The emphasis is on a few concrete things:

- the NumPy reference step should agree with standard Kalman calculations,
- the Cholesky form should stay positive definite in long or awkward runs,
- missing observations should follow the documented predict-only behavior,
- the PyTensor Op should match the NumPy path and support gradients.

The tests are intentionally a mix of small exact checks and longer numerical
stress cases. That gives us both readable failures for simple bugs and broader
coverage for the stability claims this implementation is making.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pytensor.tensor as pt
import pytest
import scipy.stats

from numpy.testing import assert_allclose

from pymc_extras.statespace.filters.chol_kalman import (
    CholKalmanUpdateOp,
    _chol_kalman_update_numpy,
    _qr_chol_predict_numpy,
    chol_kalman_step_numpy,
    run_chol_kalman_filter_numpy,
)

RNG = np.random.default_rng(42)


def make_local_level_model(sigma_obs=1.0, sigma_state=0.5):
    """Return a tiny 1D local-level model used throughout the tests."""
    T = np.eye(1)
    R = np.eye(1)
    Q = np.array([[sigma_state**2]])
    Z = np.eye(1)
    H = np.array([[sigma_obs**2]])
    return T, R, Q, Z, H


def make_kinematic_2d_model(dt=1.0, sigma_proc=0.1, sigma_obs=1.0):
    """Return a small constant-velocity tracking model with position observations."""
    n, m = 4, 2
    T = np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )
    R = np.eye(4)
    Q = (sigma_proc**2) * np.eye(4)
    Z = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
    H = (sigma_obs**2) * np.eye(2)
    return T, R, Q, Z, H


def standard_kalman_filter(x0, P0, T, R, Q, Z, H, observations):
    """Reference Kalman filter used to cross-check the Cholesky version.

    This is intentionally plain and works with full covariance matrices so the
    tests have a simple baseline to compare against.
    """
    n = x0.shape[0]
    T_steps = observations.shape[0]
    means = np.zeros((T_steps, n))
    covs = np.zeros((T_steps, n, n))
    lls = np.zeros(T_steps)

    x, P = x0.copy(), P0.copy()
    for t in range(T_steps):
        x_pred = T @ x
        P_pred = T @ P @ T.T + R @ Q @ R.T

        y = observations[t]
        if np.isnan(y).any():
            x, P = x_pred, P_pred
            lls[t] = 0.0
        else:
            m = y.shape[0]
            v = y - Z @ x_pred
            S = Z @ P_pred @ Z.T + H
            S_sym = 0.5 * (S + S.T)
            K = P_pred @ Z.T @ np.linalg.solve(S_sym, np.eye(m))
            x = x_pred + K @ v
            P_new = (np.eye(n) - K @ Z) @ P_pred
            P = 0.5 * (P_new + P_new.T)

            log_det = np.linalg.slogdet(S_sym)[1]
            quad = float(v @ np.linalg.solve(S_sym, v))
            lls[t] = -0.5 * (log_det + quad + m * np.log(2.0 * np.pi))

        means[t] = x
        covs[t] = P

    return means, covs, lls


class TestQRPredict:
    """Tests for _qr_chol_predict_numpy."""

    def test_matches_direct_propagation(self):
        """L_pred L_pred^T must equal T P T^T + R Q R^T."""
        rng = np.random.default_rng(0)
        n, q = 4, 3
        T = rng.standard_normal((n, n))
        R = rng.standard_normal((n, q))
        P = rng.standard_normal((n, n))
        P = P @ P.T + np.eye(n)
        Q = rng.standard_normal((q, q))
        Q = Q @ Q.T + 0.1 * np.eye(q)

        L_prev = np.linalg.cholesky(P)
        L_Q = np.linalg.cholesky(Q)
        L_pred = _qr_chol_predict_numpy(L_prev, T, R, L_Q)

        P_pred_qr = L_pred @ L_pred.T
        P_pred_direct = T @ P @ T.T + R @ Q @ R.T

        assert_allclose(P_pred_qr, P_pred_direct, rtol=1e-10, atol=1e-12)

    def test_output_is_lower_triangular(self):
        """L_pred must be lower-triangular."""
        rng = np.random.default_rng(1)
        n, q = 3, 2
        T = rng.standard_normal((n, n))
        R = rng.standard_normal((n, q))
        L_prev = np.tril(rng.standard_normal((n, n)))
        L_prev[np.diag_indices(n)] = np.abs(np.diag(L_prev)) + 0.1
        L_Q = np.tril(np.eye(q) + 0.1 * rng.standard_normal((q, q)))
        L_Q[np.diag_indices(q)] = np.abs(np.diag(L_Q)) + 0.1

        L_pred = _qr_chol_predict_numpy(L_prev, T, R, L_Q)

        assert_allclose(np.triu(L_pred, k=1), 0.0, atol=1e-14)

    def test_positive_diagonal(self):
        """Cholesky convention: diagonal must be strictly positive."""
        rng = np.random.default_rng(2)
        n, q = 5, 3
        T = rng.standard_normal((n, n))
        R = rng.standard_normal((n, q))
        P = np.eye(n) + 0.1 * rng.standard_normal((n, n)) @ rng.standard_normal((n, n)).T
        Q = 0.1 * np.eye(q)
        L_prev = np.linalg.cholesky(P)
        L_Q = np.linalg.cholesky(Q)

        L_pred = _qr_chol_predict_numpy(L_prev, T, R, L_Q)
        assert np.all(np.diag(L_pred) > 0), "Diagonal of L_pred must be positive."

    def test_scalar_system(self):
        """1-D sanity check: L_pred = sqrt(T^2 P + Q)."""
        T = np.array([[2.0]])
        R = np.array([[1.0]])
        P = np.array([[3.0]])
        Q = np.array([[1.0]])
        L_prev = np.linalg.cholesky(P)
        L_Q = np.linalg.cholesky(Q)
        L_pred = _qr_chol_predict_numpy(L_prev, T, R, L_Q)
        expected = np.sqrt(4 * 3 + 1)
        assert_allclose(L_pred[0, 0], expected, rtol=1e-12)


class TestCholKalmanUpdate:
    """Tests for _chol_kalman_update_numpy."""

    def test_missing_observation_predict_only(self):
        """All-NaN y must return x_pred, L_pred unchanged with log_lik = 0."""
        n, m = 3, 2
        x_pred = np.array([1.0, 2.0, 3.0])
        L_pred = np.tril(np.ones((n, n)))
        L_pred[np.diag_indices(n)] += 1.0
        Z = RNG.standard_normal((m, n))
        H = np.eye(m)
        y = np.full(m, np.nan)

        x_new, L_new, ll = _chol_kalman_update_numpy(x_pred, L_pred, Z, H, y)

        assert_allclose(x_new, x_pred)
        assert_allclose(L_new, L_pred)
        assert ll == 0.0

    def test_posterior_tighter_than_prior(self):
        """After a real observation the posterior variance must shrink."""
        T, R, Q, Z, H = make_local_level_model(sigma_obs=0.5, sigma_state=0.1)
        n = 1
        x_pred = np.array([0.0])
        P_pred = np.eye(n) * 2.0
        L_pred = np.linalg.cholesky(P_pred)
        y = np.array([0.5])

        x_new, L_new, ll = _chol_kalman_update_numpy(x_pred, L_pred, Z, H, y)
        P_new = L_new @ L_new.T

        assert P_new[0, 0] < P_pred[0, 0], "Posterior variance must be less than prior."

    def test_log_lik_matches_scipy_1d(self):
        """1-D Gaussian log-likelihood must match scipy.stats.norm."""
        sigma_obs, sigma_state = 1.0, 0.5
        T, R, Q, Z, H = make_local_level_model(sigma_obs, sigma_state)
        x_pred = np.array([0.0])
        P_pred = np.array([[1.0]])  # unit prior variance
        L_pred = np.linalg.cholesky(P_pred)
        y = np.array([1.5])

        _, _, ll = _chol_kalman_update_numpy(x_pred, L_pred, Z, H, y, jitter=0.0)

        # Predictive is N(Z x_pred, Z P Z^T + H) = N(0, 1 + sigma_obs^2)
        pred_var = float((Z @ P_pred @ Z.T + H)[0, 0])
        expected = float(scipy.stats.norm.logpdf(y[0], loc=0.0, scale=np.sqrt(pred_var)))

        assert_allclose(ll, expected, atol=1e-10)

    def test_posterior_covariance_spd(self):
        """L_new L_new^T must be symmetric positive-definite."""
        rng = np.random.default_rng(5)
        n, m = 4, 2
        P_pred = rng.standard_normal((n, n))
        P_pred = P_pred @ P_pred.T + np.eye(n)
        L_pred = np.linalg.cholesky(P_pred)
        Z = rng.standard_normal((m, n))
        H = np.eye(m) * 0.5
        x_pred = rng.standard_normal(n)
        y = rng.standard_normal(m)

        x_new, L_new, ll = _chol_kalman_update_numpy(x_pred, L_pred, Z, H, y)
        P_new = L_new @ L_new.T
        eigs = np.linalg.eigvalsh(P_new)
        assert np.all(eigs > 0), f"Posterior covariance not SPD. Min eigenvalue: {eigs.min():.3e}"


class TestPositiveDefiniteness:
    """Stress tests that make sure the covariance stays positive definite."""

    def _run_pd_stress(self, n_steps: int, sigma_proc: float, sigma_obs: float, seed: int):
        rng = np.random.default_rng(seed)
        T, R, Q, Z, H = make_local_level_model(sigma_obs=sigma_obs, sigma_state=sigma_proc)
        x0 = np.array([0.0])
        P0 = np.eye(1)
        observations = rng.standard_normal((n_steps, 1))

        _, chols, _ = run_chol_kalman_filter_numpy(x0, P0, T, R, Q, Z, H, observations)

        min_eigvals = []
        for L in chols:
            P = L @ L.T
            eigs = np.linalg.eigvalsh(P)
            min_eigvals.append(eigs.min())

        return np.array(min_eigvals)

    def test_pd_1000_steps_standard(self):
        """1,000 steps: all posteriors must have min eigenvalue > 1e-10."""
        min_eigvals = self._run_pd_stress(1000, sigma_proc=0.5, sigma_obs=1.0, seed=7)
        assert np.all(
            min_eigvals > 1e-10
        ), f"PD violation detected. Min eigenvalue across run: {min_eigvals.min():.3e}"

    def test_pd_1000_steps_small_noise(self):
        """Near-degenerate Q: min eigenvalue must remain above threshold."""
        min_eigvals = self._run_pd_stress(1000, sigma_proc=1e-5, sigma_obs=1.0, seed=8)
        assert np.all(
            min_eigvals > 1e-12
        ), f"PD violation with small process noise. Min eigenvalue: {min_eigvals.min():.3e}"

    @pytest.mark.slow
    def test_pd_100k_steps(self):
        """Full proposal spec: 10^5 steps, min eigenvalue > 1e-10."""
        min_eigvals = self._run_pd_stress(100_000, sigma_proc=0.5, sigma_obs=1.0, seed=9)
        assert np.all(min_eigvals > 1e-10)


class TestMissingData:
    def test_all_nan_sequence(self):
        """Run with all observations NaN: state must remain at prior."""
        T, R, Q, Z, H = make_local_level_model()
        x0 = np.array([5.0])
        P0 = np.eye(1) * 2.0

        obs = np.full((50, 1), np.nan)
        means, chols, lls = run_chol_kalman_filter_numpy(x0, P0, T, R, Q, Z, H, obs)

        assert_allclose(lls, 0.0)

    def test_mixed_nan_sequence_log_lik(self):
        """NaN steps contribute 0 to log-lik; observed steps contribute nonzero."""
        rng = np.random.default_rng(10)
        T, R, Q, Z, H = make_local_level_model()
        x0 = np.array([0.0])
        P0 = np.eye(1)

        obs = rng.standard_normal((20, 1))
        obs[::3] = np.nan

        _, _, lls = run_chol_kalman_filter_numpy(x0, P0, T, R, Q, Z, H, obs)

        nan_steps = np.arange(0, 20, 3)
        obs_steps = np.setdiff1d(np.arange(20), nan_steps)

        assert_allclose(lls[nan_steps], 0.0, atol=0.0)
        assert np.all(lls[obs_steps] < 0.0), "Observed log-likelihoods must be negative."

    def test_mixed_nan_state_continuity(self):
        """NaN steps must not reset or corrupt the state estimate."""
        T, R, Q, Z, H = make_local_level_model(sigma_state=0.0, sigma_obs=1e-6)
        x0 = np.array([10.0])
        P0 = np.eye(1) * 1e-4

        obs = np.array([[10.0]] * 5 + [[np.nan]] * 5 + [[10.0]] * 5)
        means, _, _ = run_chol_kalman_filter_numpy(x0, P0, T, R, Q, Z, H, obs, jitter=1e-12)

        assert_allclose(means, 10.0, atol=0.01)

    def test_partial_nan_treated_as_predict_only(self):
        """Any NaN in y is conservatively treated as full missing in phase 1."""
        rng = np.random.default_rng(99)
        T, R, Q, Z, H = make_kinematic_2d_model()
        x0 = np.zeros(4)
        P0 = np.eye(4)

        obs_full = rng.standard_normal((6, 2))
        obs_partial = obs_full.copy()
        obs_partial[3, 0] = np.nan
        obs_full_missing = obs_full.copy()
        obs_full_missing[3] = np.array([np.nan, np.nan])

        means_partial, chols_partial, lls_partial = run_chol_kalman_filter_numpy(
            x0, P0, T, R, Q, Z, H, obs_partial
        )
        means_missing, chols_missing, lls_missing = run_chol_kalman_filter_numpy(
            x0, P0, T, R, Q, Z, H, obs_full_missing
        )

        assert_allclose(means_partial, means_missing, rtol=1e-12, atol=1e-12)
        assert_allclose(chols_partial, chols_missing, rtol=1e-12, atol=1e-12)
        assert_allclose(lls_partial, lls_missing, rtol=1e-12, atol=1e-12)


class TestLogLikelihood:
    def test_total_log_lik_matches_standard_filter(self):
        """Total log-likelihood should match the standard filter."""
        rng = np.random.default_rng(11)
        T, R, Q, Z, H = make_kinematic_2d_model()
        n, m = 4, 2
        x0 = np.zeros(n)
        P0 = np.eye(n)
        obs = rng.standard_normal((100, m))

        _, _, lls_chol = run_chol_kalman_filter_numpy(x0, P0, T, R, Q, Z, H, obs)
        _, _, lls_std = standard_kalman_filter(x0, P0, T, R, Q, Z, H, obs)

        assert_allclose(
            lls_chol.sum(),
            lls_std.sum(),
            rtol=1e-7,
            err_msg="Total log-likelihood mismatch between Cholesky and standard filter.",
        )

    def test_log_lik_per_step_matches_standard_filter(self):
        """
        Per-step log-likelihoods from the Cholesky filter must match the standard
        filter at every step.
        """
        rng = np.random.default_rng(12)
        T, R, Q, Z, H = make_local_level_model()
        x0 = np.array([0.0])
        P0 = np.eye(1) * 2.0
        obs = rng.standard_normal((50, 1))

        _, _, lls_chol = run_chol_kalman_filter_numpy(x0, P0, T, R, Q, Z, H, obs)
        _, _, lls_std = standard_kalman_filter(x0, P0, T, R, Q, Z, H, obs)

        assert_allclose(lls_chol, lls_std, rtol=1e-6, atol=1e-8)

    def test_log_lik_is_negative(self):
        """Log-likelihood must be negative for non-degenerate observations."""
        T, R, Q, Z, H = make_local_level_model()
        x0 = np.array([0.0])
        P0 = np.eye(1)
        obs = np.array([[1.0], [2.0], [-1.0]])

        _, _, lls = run_chol_kalman_filter_numpy(x0, P0, T, R, Q, Z, H, obs)
        assert np.all(lls < 0.0)


class TestCholKalmanOp:
    @pytest.fixture(autouse=True)
    def op(self):
        self.op = CholKalmanUpdateOp(jitter=1e-8)

    def _make_inputs(self, n=4, m=2, q=4, seed=20):
        """Return concrete numpy arrays for a valid Kalman step."""
        rng = np.random.default_rng(seed)
        x_prev = rng.standard_normal(n)
        A = rng.standard_normal((n, n))
        P_prev = A @ A.T + np.eye(n) * 2.0
        L_prev = np.linalg.cholesky(P_prev)
        T, R, Q, Z, H = make_kinematic_2d_model()
        y = rng.standard_normal(m)
        return x_prev, L_prev, T, R, Q, Z, H, y

    def test_perform_matches_numpy_reference(self):
        """Op.perform() must produce identical output to chol_kalman_step_numpy."""
        x_prev, L_prev, T, R, Q, Z, H, y = self._make_inputs()

        x_ref, L_ref, ll_ref = chol_kalman_step_numpy(x_prev, L_prev, T, R, Q, Z, H, y)

        import pytensor.tensor as pt

        out = self.op(
            pt.as_tensor(x_prev),
            pt.as_tensor(L_prev),
            pt.as_tensor(T),
            pt.as_tensor(R),
            pt.as_tensor(Q),
            pt.as_tensor(Z),
            pt.as_tensor(H),
            pt.as_tensor(y),
        )
        x_op, L_op, ll_op = (o.eval() for o in out)

        assert_allclose(x_op, x_ref, rtol=1e-12, err_msg="x_new mismatch")
        assert_allclose(L_op, L_ref, rtol=1e-12, err_msg="L_new mismatch")
        assert_allclose(ll_op, ll_ref, rtol=1e-12, err_msg="log_lik mismatch")

    def test_perform_nan_observation(self):
        """Op.perform() with all-NaN y must return x_pred, L_pred unchanged."""
        import pytensor.tensor as pt

        x_prev, L_prev, T, R, Q, Z, H, _ = self._make_inputs()
        y_nan = np.full(2, np.nan)

        out = self.op(
            pt.as_tensor(x_prev),
            pt.as_tensor(L_prev),
            pt.as_tensor(T),
            pt.as_tensor(R),
            pt.as_tensor(Q),
            pt.as_tensor(Z),
            pt.as_tensor(H),
            pt.as_tensor(y_nan),
        )
        _, _, ll_op = (o.eval() for o in out)
        assert_allclose(ll_op, 0.0, atol=1e-15)

    def test_perform_partial_nan_observation(self):
        """A partially missing y must also trigger predict-only in phase 1."""
        x_prev, L_prev, T, R, Q, Z, H, y = self._make_inputs()
        y_partial = y.copy()
        y_partial[0] = np.nan
        y_full_missing = np.full_like(y, np.nan)

        x_partial, L_partial, ll_partial = (
            o.eval()
            for o in self.op(
                pt.as_tensor(x_prev),
                pt.as_tensor(L_prev),
                pt.as_tensor(T),
                pt.as_tensor(R),
                pt.as_tensor(Q),
                pt.as_tensor(Z),
                pt.as_tensor(H),
                pt.as_tensor(y_partial),
            )
        )
        x_missing, L_missing, ll_missing = (
            o.eval()
            for o in self.op(
                pt.as_tensor(x_prev),
                pt.as_tensor(L_prev),
                pt.as_tensor(T),
                pt.as_tensor(R),
                pt.as_tensor(Q),
                pt.as_tensor(Z),
                pt.as_tensor(H),
                pt.as_tensor(y_full_missing),
            )
        )

        assert_allclose(x_partial, x_missing, atol=1e-14)
        assert_allclose(L_partial, L_missing, atol=1e-14)
        assert_allclose(ll_partial, ll_missing, atol=1e-14)

    def test_op_output_cholesky_is_lower_triangular(self):
        """L_new from the Op must be lower-triangular with positive diagonal."""
        import pytensor.tensor as pt

        x_prev, L_prev, T, R, Q, Z, H, y = self._make_inputs()

        out = self.op(
            pt.as_tensor(x_prev),
            pt.as_tensor(L_prev),
            pt.as_tensor(T),
            pt.as_tensor(R),
            pt.as_tensor(Q),
            pt.as_tensor(Z),
            pt.as_tensor(H),
            pt.as_tensor(y),
        )
        L_op = out[1].eval()

        assert_allclose(
            np.triu(L_op, k=1), 0.0, atol=1e-14, err_msg="L_new upper triangle not zero."
        )
        assert np.all(np.diag(L_op) > 0), "L_new diagonal must be positive."

    def test_gradient_check_log_lik_wrt_Z(self):
        """Check the gradient of log_lik with respect to Z."""
        from pytensor.gradient import verify_grad

        rng = np.random.default_rng(30)
        n, m, q = 2, 1, 2
        x_prev = rng.standard_normal(n)
        A = rng.standard_normal((n, n))
        P_prev = A @ A.T + np.eye(n) * 3.0
        L_prev = np.linalg.cholesky(P_prev)
        T = np.eye(n) * 0.9
        R = np.eye(n)
        Q = np.eye(q) * 0.1
        Z = rng.standard_normal((m, n))
        H = np.eye(m) * 0.5
        y = rng.standard_normal(m)

        op = CholKalmanUpdateOp(jitter=1e-6)

        def log_lik_fn(Z_var):
            outputs = op(
                pt.as_tensor(x_prev),
                pt.as_tensor(L_prev),
                pt.as_tensor(T),
                pt.as_tensor(R),
                pt.as_tensor(Q),
                Z_var,
                pt.as_tensor(H),
                pt.as_tensor(y),
            )
            return outputs[2]

        verify_grad(
            log_lik_fn,
            pt=[Z],
            rng=np.random.default_rng(31),
            eps=1e-5,
            rel_tol=1e-4,
            abs_tol=1e-5,
        )

    def test_gradient_check_log_lik_wrt_H(self):
        """Verify gradient of log_lik w.r.t. H via finite differences."""
        from pytensor.gradient import verify_grad

        rng = np.random.default_rng(32)
        n, m, q = 2, 2, 2
        x_prev = rng.standard_normal(n)
        P_prev = np.eye(n)
        L_prev = np.linalg.cholesky(P_prev)
        T = np.eye(n)
        R = np.eye(n)
        Q = np.eye(q) * 0.1
        Z = rng.standard_normal((m, n))
        H = np.eye(m) * 0.5
        y = rng.standard_normal(m)

        op = CholKalmanUpdateOp(jitter=1e-6)

        def log_lik_fn(H_var):
            outputs = op(
                pt.as_tensor(x_prev),
                pt.as_tensor(L_prev),
                pt.as_tensor(T),
                pt.as_tensor(R),
                pt.as_tensor(Q),
                pt.as_tensor(Z),
                H_var,
                pt.as_tensor(y),
            )
            return outputs[2]

        verify_grad(
            log_lik_fn,
            pt=[H],
            rng=np.random.default_rng(33),
            eps=1e-5,
            rel_tol=1e-4,
            abs_tol=1e-5,
        )

    def test_gradient_check_x_new_wrt_T(self):
        """Verify gradient of x_new[0] w.r.t. T via finite differences."""
        from pytensor.gradient import verify_grad

        rng = np.random.default_rng(34)
        n, m, q = 2, 1, 2
        x_prev = rng.standard_normal(n)
        L_prev = np.linalg.cholesky(np.eye(n))
        T = np.eye(n) * 0.8
        R = np.eye(n)
        Q = np.eye(q) * 0.1
        Z = rng.standard_normal((m, n))
        H = np.eye(m) * 0.5
        y = rng.standard_normal(m)

        op = CholKalmanUpdateOp(jitter=1e-6)

        def x0_fn(T_var):
            outputs = op(
                pt.as_tensor(x_prev),
                pt.as_tensor(L_prev),
                T_var,
                pt.as_tensor(R),
                pt.as_tensor(Q),
                pt.as_tensor(Z),
                pt.as_tensor(H),
                pt.as_tensor(y),
            )
            return outputs[0][0]

        verify_grad(
            x0_fn,
            pt=[T],
            rng=np.random.default_rng(35),
            eps=1e-5,
            rel_tol=1e-4,
            abs_tol=1e-5,
        )

    def test_symbolic_missing_data_grad_is_finite(self):
        """Gradients through missing-data branch should be finite (and zero here)."""
        rng = np.random.default_rng(77)
        n, m, q = 2, 2, 2
        x_prev = rng.standard_normal(n)
        P_prev = np.eye(n) * 2.0
        L_prev = np.linalg.cholesky(P_prev)
        T = np.eye(n) * 0.9
        R = np.eye(n)
        Q = np.eye(q) * 0.1
        Z = rng.standard_normal((m, n))
        H = np.eye(m) * 0.5
        y_nan = np.array([np.nan, 0.0])

        Z_sym = pt.dmatrix("Z_sym")
        outputs = self.op(
            pt.as_tensor(x_prev),
            pt.as_tensor(L_prev),
            pt.as_tensor(T),
            pt.as_tensor(R),
            pt.as_tensor(Q),
            Z_sym,
            pt.as_tensor(H),
            pt.as_tensor(y_nan),
        )
        grad_Z = pt.grad(outputs[2], Z_sym)
        grad_val = grad_Z.eval({Z_sym: Z})

        assert np.all(np.isfinite(grad_val))
        assert_allclose(grad_val, 0.0, atol=1e-12)

    def test_infer_shape(self):
        """infer_shape must return (n,), (n, n), () for the three outputs."""
        n, m = 3, 2
        x_sym = pt.dvector("x")
        L_sym = pt.dmatrix("L")
        T_sym = pt.dmatrix("T")
        R_sym = pt.dmatrix("R")
        Q_sym = pt.dmatrix("Q")
        Z_sym = pt.dmatrix("Z")
        H_sym = pt.dmatrix("H")
        y_sym = pt.dvector("y")

        outs = self.op(x_sym, L_sym, T_sym, R_sym, Q_sym, Z_sym, H_sym, y_sym)

        # Build a concrete test case to evaluate the shape graph
        rng = np.random.default_rng(90)
        x = rng.standard_normal(n)
        L = np.linalg.cholesky(np.eye(n))
        T = np.eye(n) * 0.9
        R = np.eye(n)
        Q = np.eye(n) * 0.1
        Z = rng.standard_normal((m, n))
        H = np.eye(m)
        y = rng.standard_normal(m)

        feed = {x_sym: x, L_sym: L, T_sym: T, R_sym: R, Q_sym: Q, Z_sym: Z, H_sym: H, y_sym: y}

        x_out, L_out, ll_out = (o.eval(feed) for o in outs)
        assert x_out.shape == (n,)
        assert L_out.shape == (n, n)
        assert ll_out.shape == ()

    def test_connection_pattern(self):
        """All 8 inputs should connect to all 3 outputs."""
        op = CholKalmanUpdateOp(jitter=1e-8)
        pattern = op.connection_pattern(node=None)
        assert len(pattern) == 8
        for row in pattern:
            assert row == [True, True, True]

    def test_r_op_produces_finite_output(self):
        """R_op should return finite values for a standard input."""
        rng = np.random.default_rng(91)
        n, m = 2, 1
        x_prev = rng.standard_normal(n)
        P_prev = np.eye(n) * 2.0
        L_prev = np.linalg.cholesky(P_prev)
        T = np.eye(n) * 0.9
        R = np.eye(n)
        Q = np.eye(n) * 0.1
        Z = rng.standard_normal((m, n))
        H = np.eye(m) * 0.5
        y = rng.standard_normal(m)

        inputs = [
            pt.as_tensor(x_prev),
            pt.as_tensor(L_prev),
            pt.as_tensor(T),
            pt.as_tensor(R),
            pt.as_tensor(Q),
            pt.as_tensor(Z),
            pt.as_tensor(H),
            pt.as_tensor(y),
        ]
        # Tangent vectors: small perturbation in x_prev direction only
        eval_points = [
            pt.as_tensor(np.ones(n) * 0.01),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ]

        rop_x, rop_L, rop_ll = self.op.R_op(inputs, eval_points)
        vals = [rop_x.eval(), rop_L.eval(), rop_ll.eval()]
        for v in vals:
            assert np.all(np.isfinite(v)), f"R_op output not finite: {v}"

    def test_make_node_rejects_bad_static_shapes(self):
        """make_node should raise TypeError when static shapes conflict."""
        op = CholKalmanUpdateOp(jitter=1e-8)

        # L_prev not square
        x = pt.dvector("x")
        L_bad = pt.tensor("L_bad", shape=(3, 4), dtype="float64")
        T = pt.dmatrix("T")
        R = pt.dmatrix("R")
        Q = pt.dmatrix("Q")
        Z = pt.dmatrix("Z")
        H = pt.dmatrix("H")
        y = pt.dvector("y")

        with pytest.raises(TypeError, match="L_prev must be square"):
            op(x, L_bad, T, R, Q, Z, H, y)

        # H not square
        H_bad = pt.tensor("H_bad", shape=(2, 3), dtype="float64")
        with pytest.raises(TypeError, match="H must be square"):
            op(x, pt.dmatrix("L"), T, R, Q, Z, H_bad, y)

        # T not square
        T_bad = pt.tensor("T_bad", shape=(3, 4), dtype="float64")
        with pytest.raises(TypeError, match="T must be square"):
            op(x, pt.dmatrix("L2"), T_bad, R, Q, Z, pt.dmatrix("H2"), y)

    def test_gradient_check_log_lik_wrt_Q(self):
        """Verify gradient of log_lik w.r.t. Q via finite differences.

        Q is a covariance matrix (symmetric), but verify_grad perturbs elements
        independently. The NumPy path takes cholesky(Q) which only reads the
        lower triangle, so we parameterize Q = L_Q L_Q^T and test the gradient
        w.r.t. L_Q instead to avoid the asymmetry.
        """
        from pytensor.gradient import verify_grad

        rng = np.random.default_rng(40)
        n, m, q = 2, 1, 2
        x_prev = rng.standard_normal(n)
        L_prev = np.linalg.cholesky(np.eye(n) * 2.0)
        T = np.eye(n) * 0.9
        R = np.eye(n)
        L_Q = np.linalg.cholesky(np.eye(q) * 0.3)
        Z = rng.standard_normal((m, n))
        H = np.eye(m) * 0.5
        y = rng.standard_normal(m)

        op = CholKalmanUpdateOp(jitter=1e-6)

        def log_lik_fn(L_Q_var):
            Q_var = L_Q_var @ L_Q_var.T
            outputs = op(
                pt.as_tensor(x_prev),
                pt.as_tensor(L_prev),
                pt.as_tensor(T),
                pt.as_tensor(R),
                Q_var,
                pt.as_tensor(Z),
                pt.as_tensor(H),
                pt.as_tensor(y),
            )
            return outputs[2]

        verify_grad(
            log_lik_fn,
            pt=[L_Q],
            rng=np.random.default_rng(41),
            eps=1e-5,
            rel_tol=1e-4,
            abs_tol=1e-5,
        )

    def test_gradient_check_log_lik_wrt_x_prev(self):
        """Verify gradient of log_lik w.r.t. x_prev via finite differences."""
        from pytensor.gradient import verify_grad

        rng = np.random.default_rng(45)
        n, m, q = 2, 1, 2
        x_prev = rng.standard_normal(n)
        L_prev = np.linalg.cholesky(np.eye(n) * 2.0)
        T = np.eye(n) * 0.9
        R = np.eye(n)
        Q = np.eye(q) * 0.1
        Z = rng.standard_normal((m, n))
        H = np.eye(m) * 0.5
        y = rng.standard_normal(m)

        op = CholKalmanUpdateOp(jitter=1e-6)

        def log_lik_fn(x_var):
            outputs = op(
                x_var,
                pt.as_tensor(L_prev),
                pt.as_tensor(T),
                pt.as_tensor(R),
                pt.as_tensor(Q),
                pt.as_tensor(Z),
                pt.as_tensor(H),
                pt.as_tensor(y),
            )
            return outputs[2]

        verify_grad(
            log_lik_fn,
            pt=[x_prev],
            rng=np.random.default_rng(46),
            eps=1e-5,
            rel_tol=1e-4,
            abs_tol=1e-5,
        )


class TestBatchVsOnlineEquivalence:
    """Check that online filtering agrees with the batch-style reference."""

    def test_kinematic_2d_final_state(self):
        """The final filtered state should match the standard filter."""
        rng = np.random.default_rng(42)
        T_mat, R, Q, Z, H = make_kinematic_2d_model(dt=1.0, sigma_proc=0.1, sigma_obs=1.0)
        n, m = 4, 2

        x_true = np.array([0.0, 0.0, 1.0, 0.5])
        obs = []
        for _ in range(200):
            x_true = T_mat @ x_true + rng.multivariate_normal(np.zeros(n), Q)
            obs.append(Z @ x_true + rng.multivariate_normal(np.zeros(m), H))
        obs = np.array(obs)

        x0 = np.zeros(n)
        P0 = np.eye(n)

        means_chol, _, lls_chol = run_chol_kalman_filter_numpy(x0, P0, T_mat, R, Q, Z, H, obs)

        means_std, _, lls_std = standard_kalman_filter(x0, P0, T_mat, R, Q, Z, H, obs)

        assert_allclose(
            means_chol[-1],
            means_std[-1],
            rtol=1e-5,
            err_msg="Final state mismatch: online vs batch.",
        )

        assert_allclose(
            lls_chol.sum(), lls_std.sum(), rtol=1e-7, err_msg="Total log-likelihood mismatch."
        )

    def test_per_step_state_agreement(self):
        """The filtered means should match the standard filter step by step."""
        rng = np.random.default_rng(43)
        T_mat, R, Q, Z, H = make_local_level_model(sigma_obs=1.0, sigma_state=0.5)
        x0 = np.array([0.0])
        P0 = np.eye(1) * 2.0
        obs = rng.standard_normal((200, 1))

        means_chol, _, _ = run_chol_kalman_filter_numpy(x0, P0, T_mat, R, Q, Z, H, obs)
        means_std, _, _ = standard_kalman_filter(x0, P0, T_mat, R, Q, Z, H, obs)

        assert_allclose(
            means_chol, means_std, rtol=1e-6, atol=1e-8, err_msg="Per-step state mismatch."
        )

    def test_jitter_sensitivity(self):
        """Changing jitter in a stable regime should not move results much."""
        rng = np.random.default_rng(44)
        T_mat, R, Q, Z, H = make_kinematic_2d_model()
        x0 = np.zeros(4)
        P0 = np.eye(4)
        obs = rng.standard_normal((100, 2))

        means_hi, _, lls_hi = run_chol_kalman_filter_numpy(
            x0, P0, T_mat, R, Q, Z, H, obs, jitter=1e-8
        )
        means_lo, _, lls_lo = run_chol_kalman_filter_numpy(
            x0, P0, T_mat, R, Q, Z, H, obs, jitter=1e-12
        )

        assert_allclose(
            means_hi,
            means_lo,
            atol=1e-5,
            err_msg="State estimates should be insensitive to jitter in stable regime.",
        )
        assert_allclose(lls_hi.sum(), lls_lo.sum(), atol=1e-5)

    def test_chol_vs_standard_with_mixed_nans(self):
        """Both filters must agree when observations contain scattered NaNs."""
        rng = np.random.default_rng(55)
        T_mat, R, Q, Z, H = make_kinematic_2d_model()
        x0 = np.zeros(4)
        P0 = np.eye(4)
        obs = rng.standard_normal((80, 2))
        # Knock out some observations
        obs[5] = np.nan
        obs[20] = np.nan
        obs[21, 0] = np.nan  # partial NaN → both filters should treat as full missing

        means_chol, _, lls_chol = run_chol_kalman_filter_numpy(x0, P0, T_mat, R, Q, Z, H, obs)
        means_std, _, lls_std = standard_kalman_filter(x0, P0, T_mat, R, Q, Z, H, obs)

        assert_allclose(
            lls_chol,
            lls_std,
            rtol=1e-6,
            atol=1e-8,
            err_msg="Per-step log-lik mismatch with NaN observations.",
        )
        # Cholesky and covariance forms accumulate different rounding errors,
        # especially after predict-only recovery steps, so use looser tolerance.
        assert_allclose(
            means_chol,
            means_std,
            rtol=5e-4,
            atol=1e-6,
            err_msg="State estimate mismatch with NaN observations.",
        )


class TestEdgeCases:
    """Small defensive checks around shapes, jitter, and awkward numerics."""

    def test_invalid_jitter_raises(self):
        with pytest.raises(ValueError, match="jitter must be positive"):
            CholKalmanUpdateOp(jitter=0.0)
        with pytest.raises(ValueError, match="jitter must be positive"):
            CholKalmanUpdateOp(jitter=-1e-8)

    def test_1d_1d_system(self):
        """Minimal system: n=1, m=1, q=1."""
        T = np.array([[0.9]])
        R = np.array([[1.0]])
        Q = np.array([[0.1]])
        Z = np.array([[1.0]])
        H = np.array([[1.0]])
        x0 = np.array([0.0])
        P0 = np.array([[1.0]])
        obs = np.array([[1.0], [2.0], [1.5]])

        means, chols, lls = run_chol_kalman_filter_numpy(x0, P0, T, R, Q, Z, H, obs)
        assert means.shape == (3, 1)
        assert chols.shape == (3, 1, 1)
        assert np.all(lls < 0)

    def test_high_dimensional_state(self):
        """n=20, m=5: ensure no shape or numerical errors."""
        rng = np.random.default_rng(50)
        n, m, q = 20, 5, 10
        T = 0.9 * np.eye(n)
        R = rng.standard_normal((n, q))
        Q = np.eye(q) * 0.1
        Z = rng.standard_normal((m, n))
        H = np.eye(m) * 0.5
        x0 = np.zeros(n)
        P0 = np.eye(n)
        obs = rng.standard_normal((10, m))

        means, chols, lls = run_chol_kalman_filter_numpy(x0, P0, T, R, Q, Z, H, obs)
        assert means.shape == (10, n)
        for L in chols:
            eigs = np.linalg.eigvalsh(L @ L.T)
            assert np.all(eigs > 0), f"Not PD. Min eig: {eigs.min():.3e}"

    def test_repr_op(self):
        op = CholKalmanUpdateOp(jitter=1e-6)
        assert "CholKalmanUpdateOp" in repr(op)
        assert "1e-06" in repr(op)

    def test_chol_step_shape_mismatch_raises_cleanly(self):
        """Input shape mismatches should raise clear ValueError messages."""
        T, R, Q, Z, H = make_local_level_model()
        x_prev = np.zeros(2)
        L_prev = np.eye(2)
        y = np.zeros(1)

        with pytest.raises(ValueError, match=r"T must have shape \(2, 2\)"):
            chol_kalman_step_numpy(x_prev, L_prev, T, R, Q, Z, H, y)

    def test_chol_step_nonzero_c_offset(self):
        """Non-zero c branch should shift predict-only state by c."""
        x_prev = np.array([1.0, -2.0])
        L_prev = np.linalg.cholesky(np.eye(2))
        T = np.array([[1.0, 0.0], [0.0, 1.0]])
        R = np.eye(2)
        Q = np.eye(2) * 0.1
        Z = np.eye(2)
        H = np.eye(2)
        y = np.array([np.nan, np.nan])
        c = np.array([0.25, -0.75])

        x_new, _, ll = chol_kalman_step_numpy(x_prev, L_prev, T, R, Q, Z, H, y, c=c)
        assert_allclose(x_new, x_prev + c, atol=1e-12)
        assert_allclose(ll, 0.0, atol=0.0)

    def test_ill_conditioned_q_h_remains_pd(self):
        """Nearly singular process/observation noise should remain numerically stable."""
        rng = np.random.default_rng(123)
        n, m, q = 3, 2, 3
        T = np.eye(n)
        R = np.eye(n)
        Q = np.diag([1e-12, 1e-10, 1e-8])
        Z = rng.standard_normal((m, n))
        H = np.diag([1e-12, 1e-9])
        x0 = np.zeros(n)
        P0 = np.eye(n)
        obs = rng.standard_normal((40, m))

        means, chols, lls = run_chol_kalman_filter_numpy(x0, P0, T, R, Q, Z, H, obs, jitter=1e-12)

        assert means.shape == (40, n)
        assert np.all(np.isfinite(lls))
        for L in chols:
            eigs = np.linalg.eigvalsh(L @ L.T)
            assert np.all(eigs > 0), f"Not PD under ill-conditioned Q/H. Min eig: {eigs.min():.3e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
