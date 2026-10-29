import numpy as np
import pytest

from rich.console import Console

mx = pytest.importorskip("mlx.core", reason="MCLMC requires mlx, which needs Apple Silicon")

from pymc_extras.inference.mlx_mclmc.kernel import warmup_and_sample, warmup_schedule
from pymc_extras.inference.mlx_mclmc.progress import (
    MAX_SPLIT_BARS,
    MCLMCProgressBarManager,
    SectionedProgressBar,
    section_widths,
)
from pymc_extras.inference.mlx_mclmc.settings import AdaptationSettings


def test_reports_cover_every_step_in_phase_order():
    """The bar's total comes from warmup_schedule, so the reports must add up to it exactly."""
    settings = AdaptationSettings()
    reports = []

    warmup_and_sample(
        lambda x: -0.5 * mx.sum(x**2),
        np.zeros(3),
        num_tune=700,
        draws=150,
        discard=20,
        chains=2,
        settings=settings,
        progress=reports.append,
    )

    phases = list(dict.fromkeys(report.phase for report in reports))
    assert phases == ["step size", "metric", "L", "sampling"]
    assert sum(report.steps for report in reports) == (
        warmup_schedule(700, settings).total + 20 + 150
    )
    assert sum(report.nan_steps for report in reports) == 0


def test_nan_steps_are_counted_in_warmup_and_sampling():
    """A nan band straddling the mode forces reverted steps in every phase."""

    def banded(x):
        return mx.where(mx.abs(x[0]) < 0.2, mx.array(float("nan")), -0.5 * mx.sum(x**2))

    reports = []
    warmup_and_sample(
        banded, np.array([1.0, 0.0]), num_tune=600, draws=200, chains=4, progress=reports.append
    )

    nan_steps = dict.fromkeys(["step size", "metric", "L", "sampling"], 0)
    for report in reports:
        nan_steps[report.phase] += report.nan_steps
    assert all(count > 0 for count in nan_steps.values()), nan_steps


@pytest.mark.parametrize(
    "chains, progressbar, combined",
    [(4, True, True), (4, "split", False), (MAX_SPLIT_BARS + 1, "split+stats", True)],
)
def test_split_bars_are_capped(chains, progressbar, combined):
    manager = MCLMCProgressBarManager(chains=chains, sections=[50, 50], progressbar=progressbar)

    assert manager.combined_progress is combined


def test_draw_count_restarts_at_each_section():
    manager = MCLMCProgressBarManager(chains=2, sections=[100, 0, 40, 64], progressbar=False)
    counts = []
    for completed in [64, 100, 128, 140, 204]:
        manager.completed_steps = completed
        counts.append(manager._draw_count())

    # The empty section is dropped, and a section's last step reads as its full length.
    assert counts == [64, 100, 28, 40, 64]


def test_every_section_keeps_a_character():
    """At 1.9 characters' worth, the burn-in would round to nothing and merge its two gaps."""
    widths = section_widths([300, 400, 300, 100, 1000], available=36)

    assert widths.sum() == 36
    assert widths.min() >= 1
    assert widths[3] == 2


def test_sectioned_bar_draws_one_gap_between_each_pair_of_sections():
    bar = SectionedProgressBar(
        total=2100, completed=2100, width=40, sections=(300, 400, 300, 100, 1000)
    )
    console = Console(width=80, color_system=None)

    with console.capture() as captured:
        console.print(bar)
    rendered = captured.get().rstrip("\n")

    assert len(rendered) == 40
    assert [len(run) for run in rendered.split(" ")] == list(section_widths(bar.sections, 36))
