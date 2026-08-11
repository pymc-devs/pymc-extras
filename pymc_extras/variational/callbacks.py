#   Copyright 2024 - present The PyMC Developers
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
"""Loss-based early stopping for noisy (minibatch) ELBO traces."""

import numpy as np

from pymc.variational.callbacks import Callback

__all__ = ["CheckLossConvergence"]


class CheckLossConvergence(Callback):
    """Stop ``pm.fit`` when the loss improvement rate decays to noise level.

    The monitored quantity must be *minimized*: an ELBO is maximized, so pass
    ``-elbo``.

    Let ``delta_t = loss[t-1] - loss[t]`` (positive while optimizing). Each
    increment is standardized by a robust exponentially-weighted scale estimate
    built from successive differences (von Neumann, 1941) and winsorized at
    ``+/- 4``, giving ``z_t``; the monitor accumulates the one-sided CUSUM
    (Page, 1954)::

        S_t = max(0, S_{t-1} + (kappa - max(z_t, 0)))

    and declares convergence once ``S > h``; the CUSUM accumulates only after
    ``min_steps``.

    Reaching the threshold while the loss trends upwards is reported as divergence,
    not convergence.

    Parameters
    ----------
    kappa : float
        Allowance per step, in units of the scale that standardizes ``delta``. Must
        exceed what a stalled trace spends on noise alone (0.3 to 0.4); see Notes.
    h : float
        CUSUM decision threshold, trading detection delay against false alarms.
    halflife : float
        Half-life, in steps, of the exponentially-weighted scale estimate.
    min_steps : int
        Steps before the CUSUM is armed; needs a few half-lives for the scale to
        settle. Also the number of consecutive non-finite losses tolerated:
        ``pm.fit`` aborts on NaN but runs to completion on ``+inf``.

    Notes
    -----
    The defaults fix a *rate*, the per-step improvement over the per-step noise sd, so
    a fit improving more slowly is stopped however long it would have kept improving.
    Calibrated on 1000 traces per cell of ``loss[t] = f(t) + sigma[t] * eps[t]``, 6000
    steps, four families (linear, power law, alternating ``sigma``, Student-t ``df=3``):
    at rate 1.0 none of the 4000 still-improving traces stopped, the stall boundary sits
    between 0.6 and 0.7, and ``kappa`` moves it. Frozen to a plateau after improving at
    rate 1.0, every trace stopped, none early, median delay 25 to 54 steps, p95 at most 86.

    Examples
    --------
    .. code-block:: python

        monitor = CheckLossConvergence()
        approx = pm.fit(100_000, callbacks=[monitor])  # stops early if converged
    """

    # Scales mean |successive difference| into the standardizer for delta. Its z is
    # unit-variance only for independent increments, so kappa and h are calibrated
    # against it as it stands rather than derived from it.
    _SCALE_TO_SIGMA = float(np.sqrt(np.pi) / 2.0)
    # Winsorization bound on z, applied to the scale update too so one spike cannot
    # inflate the scale for hundreds of steps.
    _Z_CLIP = 4.0
    # Divide guard, not a knob: an exactly-constant stretch of loss drives the
    # successive-difference scale to zero, and dividing by it raises.
    _SIGMA_FLOOR = 1e-12

    def __init__(self, kappa=0.5, h=10.0, halflife=200.0, min_steps=1000):
        for name, value in (("kappa", kappa), ("h", h), ("halflife", halflife)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if self._Z_CLIP <= kappa:
            raise ValueError(f"kappa ({kappa}) must be below _Z_CLIP ({self._Z_CLIP})")
        if not np.isfinite(min_steps) or min_steps != int(min_steps) or min_steps < 0:
            raise ValueError(f"min_steps must be a non-negative integer, got {min_steps!r}")
        self.kappa = float(kappa)
        self.h = float(h)
        self.halflife = float(halflife)
        self.min_steps = int(min_steps)

        self._lam = float(np.exp(np.log(0.5) / self.halflife))
        # The smoothed z settles at the mean z, so the stop is reported as a divergence
        # once the loss has been climbing by _rise_tol times the scale, per step.
        self._rise_tol = 4.0 * float(np.sqrt((1.0 - self._lam) / (1.0 + self._lam)))
        self.n_nonfinite = 0
        self._prev_loss = None
        self._prev_delta = None
        self._scale = None  # EW mean of |delta_t - delta_{t-1}|
        self._z_bar = 0.0
        self._S = 0.0

    def __call__(self, approx, loss, i):
        if loss is None:
            raise TypeError(
                f"{type(self).__name__} needs per-step losses; run pm.fit with score=True "
                "(the default for ADVI) or remove this callback."
            )
        current = float(loss[-1])
        if not np.isfinite(current):
            self.n_nonfinite += 1
            if self.n_nonfinite > self.min_steps:
                raise StopIteration(
                    f"{type(self).__name__}: the loss has been non-finite for "
                    f"{self.n_nonfinite} steps; stopping at step {i}"
                )
            return
        self.n_nonfinite = 0
        if self._prev_loss is None:
            self._prev_loss = current
            return

        delta = self._prev_loss - current
        self._prev_loss = current

        if self._prev_delta is None:
            self._prev_delta = delta
            return
        abs_diff = abs(delta - self._prev_delta)
        self._prev_delta = delta

        # Standardize with the *previous* scale so a step never judges itself.
        if self._scale is None:
            if np.isfinite(abs_diff):
                self._scale = abs_diff
            return
        sigma = self._scale * self._SCALE_TO_SIGMA + self._SIGMA_FLOOR
        if np.isfinite(abs_diff):
            update = min(abs_diff, self._Z_CLIP * sigma)
            self._scale = self._lam * self._scale + (1.0 - self._lam) * update
        z = float(np.clip(delta / sigma, -self._Z_CLIP, self._Z_CLIP))
        self._z_bar = self._lam * self._z_bar + (1.0 - self._lam) * z

        if i >= self.min_steps:
            self._S = max(0.0, self._S + (self.kappa - max(z, 0.0)))

        if self._S > self.h:
            if self._z_bar < -self._rise_tol:
                raise StopIteration(
                    f"{type(self).__name__}: the loss is trending up, not converging "
                    f"(step {i}, mean z={self._z_bar:.2f}); if it is maximized, negate it"
                )
            raise StopIteration(
                f"{type(self).__name__}: converged at step {i} (S={self._S:.2f} > h={self.h:g})"
            )
