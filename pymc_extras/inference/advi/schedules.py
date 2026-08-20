"""Learning rate schedules.

A schedule maps the step number within a :meth:`Trainer.fit` call to a learning rate.
It is evaluated in Python before the step loop runs and fed to the compiled step
function as an ordinary input, so it is never part of the training state.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Schedule = Callable[[int], float]
ScalarOrSchedule = float | Schedule


def linear_onecycle_schedule(
    transition_steps: int,
    peak_value: float,
    pct_start: float = 0.3,
    pct_final: float = 0.85,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
) -> Schedule:
    """Linear one-cycle learning rate schedule (Smith & Topin, 2018), as in optax.

    The learning rate ramps from ``peak_value / div_factor`` to ``peak_value`` over the
    first ``pct_start`` fraction of ``transition_steps``, anneals back down by
    ``pct_final``, and decays to ``peak_value / div_factor / final_div_factor`` at the end.
    """
    init_value = peak_value / div_factor
    end_value = init_value / final_div_factor
    boundaries = np.array([0.0, pct_start, pct_final, 1.0]) * transition_steps
    values = np.array([init_value, peak_value, init_value, end_value])

    def schedule(count: int) -> float:
        return float(np.interp(count, boundaries, values))

    return schedule
