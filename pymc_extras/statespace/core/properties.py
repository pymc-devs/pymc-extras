from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Generic, Protocol, Self, TypeVar

from pytensor.tensor.variable import TensorVariable

from pymc_extras.statespace.utils.constants import (
    ALL_STATE_AUX_DIM,
    ALL_STATE_DIM,
    OBS_STATE_AUX_DIM,
    OBS_STATE_DIM,
    SHOCK_AUX_DIM,
    SHOCK_DIM,
)


class StateSpaceLike(Protocol):
    @property
    def state_names(self) -> tuple[str, ...]: ...

    @property
    def observed_states(self) -> tuple[str, ...]: ...

    @property
    def shock_names(self) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class Property:
    def __str__(self) -> str:
        return "\n".join(f"{f.name}: {getattr(self, f.name)}" for f in fields(self))


T = TypeVar("T", bound=Property)


@dataclass(frozen=True)
class Info(Generic[T]):
    items: tuple[T, ...] | None
    key_field: str | tuple[str, ...] = "name"
    _index: dict[str | tuple, T] | None = None

    def __post_init__(self):
        index = {}
        if self.items is None:
            object.__setattr__(self, "items", ())
        else:
            object.__setattr__(self, "items", tuple(self.items))

        for item in self.items:
            key = self._key(item)
            if key in index:
                raise ValueError(f"Duplicate {self.key_field} '{key}' detected.")
            index[key] = item
        object.__setattr__(self, "_index", index)

    def _key(self, item: T) -> str | tuple:
        if isinstance(self.key_field, tuple):
            return tuple(getattr(item, f) for f in self.key_field)
        return getattr(item, self.key_field)

    def get(self, key: str | tuple, default=None) -> T | None:
        return self._index.get(key, default)

    def __getitem__(self, key: str | tuple) -> T:
        try:
            return self._index[key]
        except KeyError as e:
            available = ", ".join(str(k) for k in self._index.keys())
            raise KeyError(f"No {self.key_field} '{key}'. Available: [{available}]") from e

    def __contains__(self, key: object) -> bool:
        return key in self._index

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __str__(self) -> str:
        return f"{self.key_field}s: {tuple(self._index.keys())}"

    def add(self, new_item: T) -> Self:
        return type(self)((*self.items, new_item))

    def merge(self, other: Self, overwrite_duplicates: bool = False) -> Self:
        if not isinstance(other, type(self)):
            raise TypeError(f"Cannot merge {type(other).__name__} with {type(self).__name__}")

        overlapping = set(self._index.keys()) & set(other._index.keys())
        if overlapping and overwrite_duplicates:
            return type(self)(
                (
                    *self.items,
                    *(item for item in other.items if self._key(item) not in overlapping),
                )
            )

        return type(self)(self.items + other.items)

    @property
    def names(self) -> tuple[str, ...]:
        if isinstance(self.key_field, tuple):
            return tuple(item.name for item in self.items)
        return tuple(self._index.keys())

    def copy(self) -> Info[T]:
        return deepcopy(self)


@dataclass(frozen=True)
class Parameter(Property):
    name: str
    shape: tuple[int, ...] | None = None
    dims: tuple[str, ...] | None = None
    constraints: str | None = None


@dataclass(frozen=True)
class ParameterInfo(Info[Parameter]):
    def __init__(self, parameters: tuple[Parameter, ...] | None):
        super().__init__(items=parameters, key_field="name")

    def to_dict(self):
        return {
            param.name: {
                "shape": param.shape,
                "constraints": param.constraints,
                "dims": param.dims,
            }
            for param in self.items
        }


@dataclass(frozen=True)
class Data(Property):
    name: str
    shape: tuple[int | None, ...] | None = None
    dims: tuple[str, ...] | None = None
    is_exogenous: bool = False


@dataclass(frozen=True)
class DataInfo(Info[Data]):
    def __init__(self, data: tuple[Data, ...] | None):
        super().__init__(items=data, key_field="name")

    @property
    def needs_exogenous_data(self) -> bool:
        return any(d.is_exogenous for d in self.items)

    @property
    def exogenous_names(self) -> tuple[str, ...]:
        return tuple(d.name for d in self.items if d.is_exogenous)

    def __str__(self) -> str:
        return f"data: {[d.name for d in self.items]}\nneeds exogenous data: {self.needs_exogenous_data}"

    def to_dict(self):
        return {
            data.name: {
                "shape": data.shape,
                "dims": data.dims,
                "exogenous": data.is_exogenous,
            }
            for data in self.items
        }


@dataclass(frozen=True)
class Coord(Property):
    dimension: str
    labels: tuple[str | int, ...]


@dataclass(frozen=True)
class CoordInfo(Info[Coord]):
    def __init__(self, coords: tuple[Coord, ...] | None = None):
        super().__init__(items=coords, key_field="dimension")

    def __str__(self) -> str:
        base = "coordinates:"
        for coord in self.items:
            coord_str = str(coord)
            indented = "\n".join("  " + line for line in coord_str.splitlines())
            base += "\n" + indented + "\n"
        return base

    @classmethod
    def default_coords_from_model(cls, model: StateSpaceLike) -> CoordInfo:
        states = tuple(model.state_names)
        obs_states = tuple(model.observed_states)
        shocks = tuple(model.shock_names)

        dim_to_labels = (
            (ALL_STATE_DIM, states),
            (ALL_STATE_AUX_DIM, states),
            (OBS_STATE_DIM, obs_states),
            (OBS_STATE_AUX_DIM, obs_states),
            (SHOCK_DIM, shocks),
            (SHOCK_AUX_DIM, shocks),
        )

        coords = tuple(Coord(dimension=dim, labels=labels) for dim, labels in dim_to_labels)
        return cls(coords=coords)

    def to_dict(self):
        return {coord.dimension: tuple(coord.labels) for coord in self.items}


@dataclass(frozen=True)
class State(Property):
    name: str
    observed: bool
    shared: bool = False


@dataclass(frozen=True)
class StateInfo(Info[State]):
    def __init__(self, states: tuple[State, ...] | None):
        super().__init__(items=states, key_field=("name", "observed"))

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            return any(s.name == key for s in self.items)
        return key in self._index

    def __str__(self) -> str:
        return (
            f"states: {[s.name for s in self.items]}\nobserved: {[s.observed for s in self.items]}"
        )

    @property
    def observed_state_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.items if s.observed)

    @property
    def unobserved_state_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.items if not s.observed)


@dataclass(frozen=True)
class Shock(Property):
    name: str


@dataclass(frozen=True)
class ShockInfo(Info[Shock]):
    def __init__(self, shocks: tuple[Shock, ...] | None):
        super().__init__(items=shocks, key_field="name")


@dataclass(frozen=True)
class SymbolicVariable(Property):
    name: str
    symbolic_variable: TensorVariable


@dataclass(frozen=True)
class SymbolicVariableInfo(Info[SymbolicVariable]):
    def __init__(self, symbolic_variables: tuple[SymbolicVariable, ...] | None = None):
        super().__init__(items=symbolic_variables, key_field="name")

    def to_dict(self):
        return {variable.name: variable.symbolic_variable for variable in self.items}


@dataclass(frozen=True)
class SymbolicData(Property):
    name: str
    symbolic_data: TensorVariable


@dataclass(frozen=True)
class SymbolicDataInfo(Info[SymbolicData]):
    def __init__(self, symbolic_data: tuple[SymbolicData, ...] | None = None):
        super().__init__(items=symbolic_data, key_field="name")

    def to_dict(self):
        return {data.name: data.symbolic_data for data in self.items}
