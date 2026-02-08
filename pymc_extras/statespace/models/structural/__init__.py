import warnings

from pymc_extras.statespace.models.structural.components.autoregressive import \
    Autoregressive
from pymc_extras.statespace.models.structural.components.cycle import Cycle
from pymc_extras.statespace.models.structural.components.level_trend import \
    LevelTrend
from pymc_extras.statespace.models.structural.components.measurement_error import \
    MeasurementError
from pymc_extras.statespace.models.structural.components.regression import \
    Regression
from pymc_extras.statespace.models.structural.components.seasonality import (
    FrequencySeasonality, TimeSeasonality)

_DEPRECATED_NAMES = {
    "LevelTrendComponent": LevelTrend,
    "CycleComponent": Cycle,
    "RegressionComponent": Regression,
    "AutoregressiveComponent": Autoregressive,
}


def __getattr__(name: str):
    if name in _DEPRECATED_NAMES:
        warnings.warn(
            f"{name} is deprecated and will be removed in a future release. "
            f"Use {_DEPRECATED_NAMES[name].__name__} instead.",
            FutureWarning,
            stacklevel=2,
        )
        return _DEPRECATED_NAMES[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Autoregressive",
    "Cycle",
    "FrequencySeasonality",
    "LevelTrend",
    "MeasurementError",
    "Regression",
    "TimeSeasonality",
]
