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
"""Tests for CheckLossConvergence, the loss-based early-stopping callback."""

import numpy as np
import pymc as pm
import pytest

from pymc_extras.variational import CheckLossConvergence


def run_monitor(monitor, losses):
    """Feed a loss trace step by step; return the stop step or None."""
    losses = np.asarray(losses, dtype=float)
    for i in range(len(losses)):
        try:
            monitor(None, losses[: i + 1], i)
        except StopIteration:
            return i
    return None


def improving_then_plateau(n_improve, n_plateau, step=1.0, noise=0.5, seed=0):
    """Loss trace that decreases by ~step per iteration, then sits at a noisy level.

    The plateau is a fixed level plus independent noise, matching a converged fit's
    Monte-Carlo score noise -- not a random walk, which would mean a fit still
    wandering and is deliberately not treated as converged.
    """
    rng = np.random.default_rng(seed)
    descent = 1000.0 - np.cumsum(rng.normal(step, noise, size=n_improve))
    plateau = descent[-1] + rng.normal(0.0, noise, size=n_plateau)
    return np.concatenate([descent, plateau])


# ------------------------------------------------------------------ construction


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window": 1},
        {"window": 100.0},
        {"rel_tol": -1e-4},
        {"rel_tol": np.nan},
        {"window": 100, "min_steps": 399},
        {"window": 100, "min_steps": 400.0},
    ],
)
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        CheckLossConvergence(**kwargs)


def test_min_steps_defaults_to_four_windows():
    assert CheckLossConvergence(window=250).min_steps == 1000


def test_min_steps_may_only_be_raised():
    assert CheckLossConvergence(window=100, min_steps=1000).min_steps == 1000


def test_none_losses_raises_typeerror():
    with pytest.raises(TypeError, match="score=True"):
        CheckLossConvergence()(None, None, 0)


# ------------------------------------------------------------------ core behavior


def test_no_stop_before_min_steps():
    """Even a perfect plateau from step 0 waits out the warm-up."""
    rng = np.random.default_rng(0)
    stop = run_monitor(
        CheckLossConvergence(window=50), 100.0 + rng.normal(0, 1.0, 2000)
    )
    assert stop is not None and stop >= 200


def test_stops_on_a_plateau_after_convergence():
    tr = improving_then_plateau(600, 4000, seed=1)
    stop = run_monitor(CheckLossConvergence(window=50), tr)
    assert stop is not None and stop >= 600


def test_never_stops_while_steadily_improving():
    rng = np.random.default_rng(2)
    tr = 10_000.0 - 1.0 * np.arange(5000) + rng.normal(0, 0.5, 5000)
    assert run_monitor(CheckLossConvergence(window=50), tr) is None


def test_stop_message_names_the_step_and_the_negate_hint():
    tr = improving_then_plateau(600, 4000, seed=3)
    monitor = CheckLossConvergence(window=50)
    with pytest.raises(StopIteration, match=r"step \d+.*negate"):
        for i in range(len(tr)):
            monitor(None, tr[: i + 1], i)


def test_a_fired_monitor_keeps_raising():
    tr = improving_then_plateau(600, 4000, seed=4)
    monitor = CheckLossConvergence(window=50)
    stop = run_monitor(monitor, tr)
    assert stop is not None
    with pytest.raises(StopIteration):
        monitor(None, tr[: stop + 2], stop + 1)


# ------------------------------------------------------------------ the two yardsticks


def test_statistical_yardstick_fires_alone():
    """With rel_tol=0 only the noise criterion is active; a plateau still stops."""
    tr = improving_then_plateau(600, 4000, noise=2.0, seed=5)
    assert run_monitor(CheckLossConvergence(window=50, rel_tol=0.0), tr) is not None


def test_practical_yardstick_fires_alone():
    """A statistically clear but negligible residual drift stops only via rel_tol.

    After a 1000-unit drop, the loss keeps improving by 1e-5 per step with 1e-4
    noise: z stays far above 1 at every horizon, so with rel_tol=0 the monitor
    (correctly) never calls it a plateau; the practical yardstick does.
    """
    rng = np.random.default_rng(6)
    n = 6000
    f = np.where(np.arange(n) < 500, 1000.0 - 2.0 * np.arange(n), 0.0)
    tail = -1e-5 * np.arange(n) + rng.normal(0, 1e-4, n)
    tr = f + tail
    assert run_monitor(CheckLossConvergence(window=50, rel_tol=0.0), tr) is None
    assert run_monitor(CheckLossConvergence(window=50, rel_tol=3e-4), tr) is not None


def test_stop_step_is_invariant_to_affine_relabelling():
    """Rescaling or shifting the loss must not move the stop step."""
    tr = improving_then_plateau(600, 4000, seed=7)
    reference = run_monitor(CheckLossConvergence(window=50), tr)
    for scale, offset in [(7.5, 0.0), (1.0, -3e4), (0.02, 1e6)]:
        assert run_monitor(CheckLossConvergence(window=50), scale * tr + offset) == reference


# ------------------------------------------------------------------ robustness


def test_a_shelf_before_further_descent_does_not_stop():
    """A flat stretch with real improvement still ahead is vetoed by the long horizon."""
    rng = np.random.default_rng(8)
    i = np.arange(4000, dtype=float)
    f = np.where(i < 1000, -1.0 * i, np.where(i < 1300, -1000.0, -1000.0 - 1.0 * (i - 1300)))
    tr = f + rng.normal(0, 0.5, 4000)
    assert run_monitor(CheckLossConvergence(window=50), tr) is None


def test_persistence_resets_when_improvement_resumes():
    """Plateau checks interrupted by a descent burst must not accumulate."""
    rng = np.random.default_rng(9)
    pieces = []
    level = 1000.0
    for _ in range(12):
        pieces.append(level + rng.normal(0, 0.5, 60))  # brief flat
        drop = np.cumsum(rng.normal(1.0, 0.5, 300))  # then real descent
        pieces.append(level - drop)
        level -= drop[-1]
    tr = np.concatenate(pieces)
    assert run_monitor(CheckLossConvergence(window=50), tr) is None


def test_scattered_nonfinite_losses_never_stop_a_healthy_fit():
    rng = np.random.default_rng(10)
    tr = 10_000.0 - 1.0 * np.arange(3000) + rng.normal(0, 0.5, 3000)
    tr[::250] = np.inf
    assert run_monitor(CheckLossConvergence(window=50), tr) is None


def test_persistently_nonfinite_loss_stops():
    tr = np.full(500, np.inf)
    monitor = CheckLossConvergence(window=50)
    with pytest.raises(StopIteration, match="non-finite"):
        for i in range(len(tr)):
            monitor(None, tr[: i + 1], i)


def test_rising_loss_stops_rather_than_running_forever():
    """A maximized objective passed unnegated is stopped, with the hint."""
    rng = np.random.default_rng(11)
    tr = 1000.0 + 1.0 * np.arange(3000) + rng.normal(0, 0.5, 3000)
    monitor = CheckLossConvergence(window=50)
    with pytest.raises(StopIteration, match="negate"):
        for i in range(len(tr)):
            monitor(None, tr[: i + 1], i)


# ------------------------------------------------------------------ integration


def test_pm_fit_integration_smoke():
    rng = np.random.default_rng(12)
    y = rng.normal(1.0, 2.0, 200)
    with pm.Model():
        mu = pm.Normal("mu", 0, 10)
        pm.Normal("y", mu, 2.0, observed=y)
        approx = pm.fit(
            8000,
            method="advi",
            progressbar=False,
            random_seed=13,
            callbacks=[CheckLossConvergence(window=50)],
        )
    assert len(approx.hist) < 8000  # stopped early on an easy model
