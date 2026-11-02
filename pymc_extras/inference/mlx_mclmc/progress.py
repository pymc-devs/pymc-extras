import logging

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from pymc.progress_bar import ProgressBarOptions
from pymc.progress_bar.marimo_progress import in_marimo_notebook
from pymc.progress_bar.progress import ProgressBackend, ProgressBarManager
from pymc.progress_bar.rich_progress import CustomBarColumn, CustomProgress, RichProgressBackend
from rich.console import Console, ConsoleOptions, RenderResult
from rich.progress import ProgressBar, Task, TextColumn
from rich.segment import Segment
from rich.table import Column
from rich.theme import Theme

if TYPE_CHECKING:
    from pymc_extras.inference.mlx_mclmc.kernel import ProgressReport

_log = logging.getLogger(__name__)

# Every chain advances in lockstep as one array, so per-chain bars only differ in their stats.
# Past this many they stop being readable and the manager draws one combined bar instead.
MAX_SPLIT_BARS = 16

# rich's default, which the ADVI bar keeps.
_BAR_WIDTH = 40


def section_widths(lengths: Sequence[int], available: int) -> np.ndarray:
    """
    Split ``available`` characters across sections in proportion to their lengths.

    Every section gets at least one character, so a short one such as the burn-in stays visible.
    The characters left over from rounding down go to the sections that lost the most to it.
    """
    weights = np.asarray(lengths, dtype=float)
    exact = weights * available / weights.sum()
    widths = np.maximum(1, np.floor(exact)).astype(int)

    while widths.sum() < available:
        widths[np.argmax(exact - widths)] += 1
    while widths.sum() > available:
        widths[np.argmax(widths)] -= 1

    return widths


class SectionedProgressBar(ProgressBar):
    """A progress bar drawn as one sub-bar per section, separated by single-character gaps."""

    def __init__(self, *args, sections: Sequence[int] = (), **kwargs):
        super().__init__(*args, **kwargs)
        self.sections = sections

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        width = min(self.width or options.max_width, options.max_width)
        available = width - (len(self.sections) - 1)
        if len(self.sections) < 2 or not self.total or self.pulse or available < len(self.sections):
            yield from super().__rich_console__(console, options)
            return

        remaining = self.completed
        for index, (length, section_width) in enumerate(
            zip(self.sections, section_widths(self.sections, available), strict=True)
        ):
            if index:
                yield Segment(" ")
            section = ProgressBar(
                total=length,
                completed=min(max(remaining, 0), length),
                width=int(section_width),
                style=self.style,
                complete_style=self.complete_style,
                finished_style=self.finished_style,
                pulse_style=self.pulse_style,
            )
            yield from section.__rich_console__(console, options)
            remaining -= length


class SectionedBarColumn(CustomBarColumn):
    """pymc's bar column, red on failure, drawn in the sections listed in the task's fields."""

    def render(self, task: Task) -> ProgressBar:
        return SectionedProgressBar(
            total=max(0, task.total) if task.total is not None else None,
            completed=max(0, task.completed),
            width=None if self.bar_width is None else max(1, self.bar_width),
            pulse=not task.started,
            animation_time=task.get_time(),
            style=self.style,
            complete_style=self.complete_style,
            finished_style=self.finished_style,
            pulse_style=self.pulse_style,
            sections=task.fields.get("sections", ()),
        )


class SectionedRichBackend(RichProgressBackend):
    """pymc's rich backend with a sectioned bar, sized to its contents as the ADVI bar is."""

    def _create_progress_bar(self, progress_columns: list, theme: Theme) -> CustomProgress:
        progress = super()._create_progress_bar(progress_columns=progress_columns, theme=theme)
        stretched = progress.columns[0]
        progress.columns = (
            SectionedBarColumn(
                bar_width=_BAR_WIDTH,
                table_column=stretched.get_table_column(),
                complete_style=stretched.default_complete_style,
                finished_style=stretched.default_finished_style,
            ),
            *progress.columns[1:],
        )
        progress.expand = False
        return progress


class MCLMCProgressBarManager(ProgressBarManager):
    """
    Progress bars for MCLMC, fed by the kernel's :class:`ProgressReport` callbacks.

    The bar is split into one section per nonzero entry of ``sections`` (the warmup phases, the
    burn-in, and the kept draws), and the draw count restarts at each one. Chains advance in
    lockstep, so every bar counts the steps of one chain. ``progressbar=True`` draws a single bar
    showing the median step size across chains; ``"split"`` and ``"split+stats"`` draw one bar per
    chain, up to :data:`MAX_SPLIT_BARS` chains. NaN steps counts the chain-steps that came out NaN
    or infinite and were reverted, summed across chains over the whole run, and turns the bar red
    once there are any.
    """

    step_name: str = "Draw"

    def __init__(
        self,
        chains: int,
        sections: Sequence[int],
        progressbar: bool | ProgressBarOptions = True,
        progressbar_theme: Theme | str | None = None,
    ):
        super().__init__(
            n_bars=chains,
            progressbar="combined+stats" if progressbar is True else progressbar,
            progressbar_theme=progressbar_theme,
        )
        if not self.combined_progress and chains > MAX_SPLIT_BARS:
            _log.info(
                "Drawing one combined progress bar: split bars are capped at %d chains, got %d.",
                MAX_SPLIT_BARS,
                chains,
            )
            self.combined_progress = True

        self.chains = chains
        self.completed_steps = 0
        self.nan_steps = 0

        self._sections = tuple(int(length) for length in sections if length)
        self._section_ends = np.cumsum(self._sections)
        self._section_starts = np.concatenate([[0], self._section_ends[:-1]])
        self.total_steps = int(self._section_ends[-1])

        bars = 1 if self.combined_progress else chains
        progress_columns = [
            TextColumn("{task.fields[step_size]}", table_column=Column("Step size")),
            TextColumn("{task.fields[nan_steps]}", table_column=Column("NaN steps")),
        ]
        progress_stats = {
            "step_size": [""] * bars,
            "nan_steps": [0] * bars,
            "draw": [0] * bars,
            "sections": [self._sections] * bars,
        }
        self._backend = self._create_backend(
            total=self.total_steps,
            progress_columns=progress_columns,
            progress_stats=progress_stats,
        )

    def _create_backend(
        self, total: int | float | None, progress_columns: list, progress_stats: dict[str, list]
    ) -> ProgressBackend:
        if not self._show_progress or in_marimo_notebook():
            return super()._create_backend(
                total=total, progress_columns=progress_columns, progress_stats=progress_stats
            )

        theme = self._progressbar_theme
        return SectionedRichBackend(
            step_name=self.step_name,
            n_bars=1 if self.combined_progress else self.n_bars,
            total=total,
            combined=self.combined_progress,
            full_stats=self.full_stats,
            progress_columns=progress_columns,
            progress_stats=progress_stats,
            theme=theme if isinstance(theme, Theme) else None,
        )

    def _draw_count(self) -> int:
        """Steps finished in the current section, which reads full once its last step is."""
        section = min(
            int(np.searchsorted(self._section_ends, self.completed_steps)),
            len(self._section_ends) - 1,
        )
        return self.completed_steps - int(self._section_starts[section])

    def update(self, report: "ProgressReport") -> None:
        if not self._show_progress:
            return

        self.completed_steps += report.steps
        self.nan_steps += report.nan_steps
        is_last = self.completed_steps >= self.total_steps
        step_size = report.step_size

        if self.combined_progress:
            bars = [(0, np.median(step_size))]
        else:
            bars = enumerate(step_size.tolist())

        for task_id, bar_step_size in bars:
            self._backend.update(
                task_id=task_id,
                advance=report.steps,
                failing=self.nan_steps > 0,
                stats={
                    "step_size": f"{bar_step_size:.3g}",
                    "nan_steps": self.nan_steps,
                    "draw": self._draw_count(),
                    "sections": self._sections,
                },
                is_last=is_last,
                total=self.total_steps,
            )
