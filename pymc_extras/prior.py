"""Class that represents a prior distribution.

The `Prior` class is a wrapper around PyMC distributions that allows the user
to create outside of the PyMC model.

Examples
--------
Create a normal prior.

.. code-block:: python

    from pymc_extras.prior import Prior

    normal = Prior("Normal")

Create a hierarchical normal prior by using distributions for the parameters
and specifying the dims.

.. code-block:: python

    hierarchical_normal = Prior(
        "Normal",
        mu=Prior("Normal"),
        sigma=Prior("HalfNormal"),
        dims="channel",
    )

Create a non-centered hierarchical normal prior with the `centered` parameter.

.. code-block:: python

    non_centered_hierarchical_normal = Prior(
        "Normal",
        mu=Prior("Normal"),
        sigma=Prior("HalfNormal"),
        dims="channel",
        # Only change needed to make it non-centered
        centered=False,
    )

Create a hierarchical beta prior by using Beta distribution, distributions for
the parameters, and specifying the dims.

.. code-block:: python

    hierarchical_beta = Prior(
        "Beta",
        alpha=Prior("HalfNormal"),
        beta=Prior("HalfNormal"),
        dims="channel",
    )

Create a transformed hierarchical normal prior by using the `transform`
parameter. Here the "sigmoid" transformation comes from `pm.math`.

.. code-block:: python

    transformed_hierarchical_normal = Prior(
        "Normal",
        mu=Prior("Normal"),
        sigma=Prior("HalfNormal"),
        transform="sigmoid",
        dims="channel",
    )

Create a prior with a custom transform function by registering it with
`register_tensor_transform`.

.. code-block:: python

    from pymc_extras.prior import register_tensor_transform


    def custom_transform(x):
        return x**2


    register_tensor_transform("square", custom_transform)

    custom_distribution = Prior("Normal", transform="square")

"""

from __future__ import annotations

import copy
import typing

from collections.abc import Callable
from functools import partial
from inspect import signature
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xarray as xr

from pydantic import InstanceOf, validate_call
from pydantic.dataclasses import dataclass
from pymc.distributions.shape_utils import Dims
from pytensor.graph import Variable
from pytensor.tensor import TensorVariable

from pymc_extras.deserialize import deserialize, register_deserialization

if typing.TYPE_CHECKING:
    # Lazy import of experimental modules
    from pymc.dims import DimDistribution
    from pytensor.tensor import TensorLike
    from pytensor.xtensor.type import XTensorVariable
    from xarray import DataArray

    XTensorLike: TypeAlias = TensorLike | DataArray


class UnsupportedShapeError(Exception):
    """Error for when the shapes from variables are not compatible."""


class UnsupportedDistributionError(Exception):
    """Error for when an unsupported distribution is used."""


class UnsupportedParameterizationError(Exception):
    """The follow parameterization is not supported."""


class MuAlreadyExistsError(Exception):
    """Error for when 'mu' is present in Prior."""

    def __init__(self, distribution: Prior) -> None:
        self.distribution = distribution
        self.message = f"The mu parameter is already defined in {distribution}"
        super().__init__(self.message)


class UnknownTransformError(Exception):
    """Error for when an unknown transform is used."""


def _remove_leading_xs(args: list[str | int]) -> list[str | int]:
    """Remove leading 'x' from the args."""
    while args and args[0] == "x":
        args.pop(0)

    return args


def handle_dims(x: pt.TensorLike, dims: Dims, desired_dims: Dims) -> pt.TensorVariable:
    """Take a tensor of dims `dims` and align it to `desired_dims`.

    Doesn't check for validity of the dims

    Parameters
    ----------
    x : pt.TensorLike
        The tensor to align.
    dims : Dims
        The current dimensions of the tensor.
    desired_dims : Dims
        The desired dimensions of the tensor.

    Returns
    -------
    pt.TensorVariable
        The aligned tensor.

    Examples
    --------
    Handle transpose 1D to 2D with new dimension.

    .. code-block:: python

        x = np.array([1, 2, 3])
        dims = "channel"

        desired_dims = ("channel", "group")

        handle_dims(x, dims, desired_dims)

    """
    x = pt.as_tensor_variable(x)

    if np.ndim(x) == 0:
        return x

    if dims is None:
        raise ValueError("handle_dims requires explicit dims, got None")
    if desired_dims is None:
        raise ValueError("handle_dims requires explicit derised_dims, got None")
    dims = dims if isinstance(dims, tuple) else (dims,)
    desired_dims = desired_dims if isinstance(desired_dims, tuple) else (desired_dims,)

    if difference := set(dims).difference(desired_dims):
        raise UnsupportedShapeError(
            f"Dims {dims} of data are not a subset of the desired dims {desired_dims}. "
            f"{difference} is missing from the desired dims."
        )

    aligned_dims = np.array(dims)[:, None] == np.array(desired_dims)

    missing_dims = aligned_dims.sum(axis=0) == 0
    new_idx = aligned_dims.argmax(axis=0)

    args = ["x" if missing else idx for (idx, missing) in zip(new_idx, missing_dims, strict=False)]
    args = _remove_leading_xs(args)
    return x.dimshuffle(*args)


DimHandler = Callable[[pt.TensorLike, Dims], pt.TensorLike]
"""A function that takes a tensor and its current dims and makes it broadcastable to the desired dims."""


def create_dim_handler(desired_dims: Dims) -> DimHandler:
    """Wrap the :func:`handle_dims` function to always use the same desired_dims.

    Parameters
    ----------
    desired_dims : Dims
        The desired dimensions to align to.

    Returns
    -------
    DimHandler
        A function that takes a tensor and its current dims and aligns it to
        the desired dims.


    Examples
    --------
    Create a dim handler to align to ("channel", "group").

    .. code-block:: python

        import numpy as np

        from pymc_extras.prior import create_dim_handler

        dim_handler = create_dim_handler(("channel", "group"))

        result = dim_handler(np.array([1, 2, 3]), dims="channel")


    """

    def func(x: pt.TensorLike, dims: Dims) -> pt.TensorVariable:
        return handle_dims(x, dims, desired_dims)

    return func


def _dims_to_str(obj: tuple[str, ...]) -> str:
    if len(obj) == 1:
        return f'"{obj[0]}"'

    return "(" + ", ".join(f'"{i}"' if isinstance(i, str) else str(i) for i in obj) + ")"


def _get_pymc_distribution(name: str) -> type[pm.Distribution]:
    try:
        return getattr(pm, name)
    except AttributeError:
        raise UnsupportedDistributionError(f"PyMC doesn't have a distribution of name {name!r}")


def _get_pymc_dim_distribution(name: str) -> type[DimDistribution]:
    import pymc.dims as pmd

    try:
        return getattr(pmd, name)
    except AttributeError:
        raise UnsupportedDistributionError(
            f"PyMC.dims doesn't have a distribution of name {name!r}"
        )


Transform = Callable[[pt.TensorLike], pt.TensorLike]

CUSTOM_TRANSFORMS: dict[str, Transform] = {}


def register_tensor_transform(name: str, transform: Transform) -> None:
    """Register a tensor transform function to be used in the `Prior` class.

    Parameters
    ----------
    name : str
        The name of the transform.
    func : Callable[[pt.TensorLike], pt.TensorLike]
        The function to apply to the tensor.

    Examples
    --------
    Register a custom transform function.

    .. code-block:: python

        from pymc_extras.prior import (
            Prior,
            register_tensor_transform,
        )


        def custom_transform(x):
            return x**2


        register_tensor_transform("square", custom_transform)

        custom_distribution = Prior("Normal", transform="square")

    """
    CUSTOM_TRANSFORMS[name] = transform


def _get_transform(name: str, xdist: bool = False) -> Transform:
    if name in CUSTOM_TRANSFORMS:
        return CUSTOM_TRANSFORMS[name]

    if xdist:
        import pytensor.xtensor as ptx

        for module in (ptx.math, ptx.linalg, ptx):
            try:
                return getattr(module, name)
            except AttributeError:
                continue
        raise UnknownTransformError(
            f"Function {name!r} not present in pytensor.xtensor or its submodules. "
            "If this is a custom function, register it with `pymc_extras.prior.register_tensor_transform` first."
        )
    else:
        for module in (pt, pm.math):
            try:
                return getattr(module, name)
            except AttributeError:
                continue
        raise UnknownTransformError(
            f"Function {name!r} not present in pytensor.tensor or pymc.math. "
            "If this is a custom function, register it with `pymc_extras.prior.register_tensor_transform` first."
        )


def _get_pymc_parameters(distribution: pm.Distribution) -> set[str]:
    return set(signature(distribution.dist).parameters.keys()) - {"kwargs", "args"}


@runtime_checkable
class VariableFactory(Protocol):
    '''Protocol for something that works like a Prior class.

    Sample with :func:`sample_prior`.

    Examples
    --------
    Create a custom variable factory.

    .. code-block:: python

        import pymc as pm

        import pytensor.tensor as pt

        from pymc_extras.prior import sample_prior, VariableFactory


        class PowerSumDistribution:
            """Create a distribution that is the sum of powers of a base distribution."""

            def __init__(self, distribution: VariableFactory, n: int):
                self.distribution = distribution
                self.n = n

            @property
            def dims(self):
                return self.distribution.dims

            def create_variable(self, name: str) -> "TensorVariable":
                raw = self.distribution.create_variable(f"{name}_raw")
                return pm.Deterministic(
                    name,
                    pt.sum([raw**n for n in range(1, self.n + 1)], axis=0),
                    dims=self.dims,
                )


        cubic = PowerSumDistribution(Prior("Normal"), n=3)
        samples = sample_prior(cubic)

    '''

    dims: tuple[str, ...] | None
    """The dimensions of the variable to create."""

    def create_variable(self, name: str, xdist: bool = False) -> TensorVariable | XTensorVariable:
        """Create a variable."""


def sample_prior(
    factory: VariableFactory,
    coords=None,
    name: str = "variable",
    wrap: bool = False,
    xdist: bool = False,
    **sample_prior_predictive_kwargs,
) -> xr.Dataset:
    """Sample the prior for an arbitrary VariableFactory.

    Parameters
    ----------
    factory : VariableFactory
        The factory to sample from.
    coords : dict[str, list[str]], optional
        The coordinates for the variable, by default None.
        Only required if the dims are specified.
    name : str, optional
        The name of the variable, by default "variable".
    wrap : bool, optional
        Whether to wrap the variable in a `pm.Deterministic` node, by default False.
    sample_prior_predictive_kwargs : dict
        Additional arguments to pass to `pm.sample_prior_predictive`.
    xdist: bool, default False
        Whether to create a pymc.dims variable or a regular pymc variable

    Returns
    -------
    xr.Dataset
        The dataset of the prior samples.

    Example
    -------
    Sample from an arbitrary variable factory.

    .. code-block:: python

        import pymc as pm

        import pytensor.tensor as pt

        from pymc_extras.prior import sample_prior


        class CustomVariableDefinition:
            def __init__(self, dims, n: int):
                self.dims = dims
                self.n = n

            def create_variable(self, name: str) -> "TensorVariable":
                x = pm.Normal(f"{name}_x", mu=0, sigma=1, dims=self.dims)
                return pt.sum([x**n for n in range(1, self.n + 1)], axis=0)


        cubic = CustomVariableDefinition(dims=("channel",), n=3)
        coords = {"channel": ["C1", "C2", "C3"]}
        # Doesn't include the return value
        prior = sample_prior(cubic, coords=coords)

        prior_with = sample_prior(cubic, coords=coords, wrap=True)

    """
    coords = coords or {}

    dims = factory.dims
    if dims is not None:
        if isinstance(factory.dims, str):
            dims = (factory.dims,)
        else:
            dims = factory.dims

        if missing_keys := set(dims) - set(coords.keys()):
            raise KeyError(f"Coords are missing the following dims: {missing_keys}")

    with pm.Model(coords=coords) as model:
        if xdist:
            var = factory.create_variable(name, xdist=True)
        else:
            # Backwards compatibility when the xdist kwarg didn't exist
            var = factory.create_variable(name)
        if wrap:
            if xdist:
                from pymc.dims import Deterministic

                det_class = Deterministic
            else:
                det_class = pm.Deterministic

            det_class(name, var, dims=dims)

    return pm.sample_prior_predictive(
        model=model,
        **sample_prior_predictive_kwargs,
    ).prior


class Prior:
    """A class to represent a prior distribution.

    Make use of the various helper methods to understand the distributions
    better.

    - `preliz` attribute to get the equivalent distribution in `preliz`
    - `sample_prior` method to sample from the prior
    - `to_graph` get a dummy model graph with the distribution
    - `constrain` to shift the distribution to a different range

    Parameters
    ----------
    distribution : str
        The name of PyMC distribution.
    dims : Dims, optional
        The dimensions of the variable, by default None
    centered : bool, optional
        Whether the variable is centered or not, by default True.
        Only allowed for Normal distribution.
    transform : str, optional
        The name of the transform to apply to the variable after it is
        created, by default None or no transform. The transformation must
        be registered with `register_tensor_transform` function or
        be available in either `pytensor.tensor` or `pymc.math`.

    Examples
    --------
    Create a normal prior.

    .. code-block:: python

        from pymc_extras.prior import Prior

        normal = Prior("Normal")

    Create a hierarchical normal prior by using distributions for the parameters
    and specifying the dims.

    .. code-block:: python

        hierarchical_normal = Prior(
            "Normal",
            mu=Prior("Normal"),
            sigma=Prior("HalfNormal"),
            dims="channel",
        )

    Create a non-centered hierarchical normal prior with the `centered` parameter.

    .. code-block:: python

        non_centered_hierarchical_normal = Prior(
            "Normal",
            mu=Prior("Normal"),
            sigma=Prior("HalfNormal"),
            dims="channel",
            # Only change needed to make it non-centered
            centered=False,
        )

    Create a hierarchical beta prior by using Beta distribution, distributions for
    the parameters, and specifying the dims.

    .. code-block:: python

        hierarchical_beta = Prior(
            "Beta",
            alpha=Prior("HalfNormal"),
            beta=Prior("HalfNormal"),
            dims="channel",
        )

    Create a transformed hierarchical normal prior by using the `transform`
    parameter. Here the "sigmoid" transformation comes from `pm.math`.

    .. code-block:: python

        transformed_hierarchical_normal = Prior(
            "Normal",
            mu=Prior("Normal"),
            sigma=Prior("HalfNormal"),
            transform="sigmoid",
            dims="channel",
        )

    Create a prior with a custom transform function by registering it with
    :func:`register_tensor_transform`.

    .. code-block:: python

        from pymc_extras.prior import register_tensor_transform


        def custom_transform(x):
            return x**2


        register_tensor_transform("square", custom_transform)

        custom_distribution = Prior("Normal", transform="square")

    """

    # Taken from https://en.wikipedia.org/wiki/Location%E2%80%93scale_family
    non_centered_distributions: dict[str, dict[str, float]] = {
        "Normal": {"mu": 0, "sigma": 1},
        "StudentT": {"mu": 0, "sigma": 1},
        "ZeroSumNormal": {"sigma": 1},
    }
    """Available non-centered distributions and their default parameters."""

    pymc_distribution: type[pm.Distribution]
    """The PyMC distribution class."""

    @validate_call
    def __init__(
        self,
        distribution: str,
        *,
        dims: Dims | None = None,
        centered: bool = True,
        transform: str | None = None,
        **parameters,
    ) -> None:
        self.distribution = distribution
        self.parameters = parameters
        self.dims = dims
        self.centered = centered
        self.transform = transform

        self._checks()

    @property
    def distribution(self) -> str:
        """The name of the PyMC distribution."""
        return self._distribution

    @distribution.setter
    def distribution(self, distribution: str) -> None:
        if hasattr(self, "_distribution"):
            raise AttributeError("Can't change the distribution")

        self._distribution = distribution
        self.pymc_distribution = _get_pymc_distribution(distribution)

    @property
    def transform(self) -> str | None:
        """The name of the transform to apply to the variable after it is created."""
        return self._transform

    @transform.setter
    def transform(self, transform: str | None) -> None:
        self._transform = transform
        if transform is not None:
            # Validate transform exists
            _get_transform(transform)

    @property
    def dims(self) -> Dims | None:
        """The dimensions of the variable."""
        return self._dims

    @dims.setter
    def dims(self, dims) -> None:
        if dims is None:
            self._dims = None
            return

        if isinstance(dims, str):
            dims = (dims,)
        elif not isinstance(dims, tuple):
            dims = tuple(dims)

        self._dims = dims
        self._param_dims_work()
        self._unique_dims()

    def __getitem__(self, key: str) -> Prior | Any:
        """Return the parameter of the prior."""
        return self.parameters[key]

    def _checks(self) -> None:
        if not self.centered:
            self._correct_non_centered_distribution()

        self._parameters_are_at_least_subset_of_pymc()
        self._convert_lists_to_numpy()
        self._parameters_are_correct_type()

    def _parameters_are_at_least_subset_of_pymc(self) -> None:
        pymc_params = _get_pymc_parameters(self.pymc_distribution)
        if not set(self.parameters.keys()).issubset(pymc_params):
            msg = (
                f"Parameters {set(self.parameters.keys())} "
                "are not a subset of the pymc distribution "
                f"parameters {set(pymc_params)}"
            )
            raise ValueError(msg)

    def _convert_lists_to_numpy(self) -> None:
        def convert(x):
            if not isinstance(x, list):
                return x

            return np.array(x)

        self.parameters = {key: convert(value) for key, value in self.parameters.items()}

    def _parameters_are_correct_type(self) -> None:
        supported_types = (
            Variable,
            Prior,
            int,
            float,
            np.ndarray,
            xr.DataArray,
            VariableFactory,
        )

        incorrect_types = {
            param: type(value)
            for param, value in self.parameters.items()
            if not isinstance(value, supported_types)
        }
        if incorrect_types:
            msg = (
                "Parameters must be one of the following types: "
                f"(int, float, np.array, Prior, pt.TensorVariable). Incorrect parameters: {incorrect_types}"
            )
            raise ValueError(msg)

    def _correct_non_centered_distribution(self) -> None:
        if not self.centered and self.distribution not in self.non_centered_distributions:
            raise UnsupportedParameterizationError(
                f"{self.distribution!r} is not supported for non-centered parameterization. "
                f"Choose from {list(self.non_centered_distributions.keys())}"
            )

        required_parameters = set(self.non_centered_distributions[self.distribution].keys())

        if set(self.parameters.keys()) < required_parameters:
            msg = " and ".join([f"{param!r}" for param in required_parameters])
            raise ValueError(
                f"Must have at least {msg} parameter for non-centered for {self.distribution!r}"
            )

    def _unique_dims(self) -> None:
        if not self.dims:
            return

        if len(self.dims) != len(set(self.dims)):
            raise ValueError("Dims must be unique")

    def _param_dims_work(self) -> None:
        if self.dims is None:
            return

        other_dims_set = set()
        for value in self.parameters.values():
            if (other_dims := getattr(value, "dims", None)) is not None:
                other_dims_set.update(other_dims)

        if not other_dims_set.issubset(self.dims):
            raise UnsupportedShapeError(
                f"Parameter dims {other_dims} are not a subset of the prior dims {self.dims}"
            )

    def __str__(self) -> str:
        """Return a string representation of the prior."""
        param_str = ", ".join([f"{param}={value}" for param, value in self.parameters.items()])
        param_str = "" if not param_str else f", {param_str}"

        dim_str = f", dims={_dims_to_str(self.dims)}" if self.dims is not None else ""
        centered_str = f", centered={self.centered}" if not self.centered else ""
        transform_str = f', transform="{self.transform}"' if self.transform else ""
        return f'Prior("{self.distribution}"{param_str}{dim_str}{centered_str}{transform_str})'

    def __repr__(self) -> str:
        """Return a string representation of the prior."""
        return f"{self}"

    def _create_parameter(self, param, value, name, xdist: bool = False):
        if not hasattr(value, "create_variable"):
            return value

        child_name = f"{name}_{param}"
        if xdist:
            return value.create_variable(child_name, xdist=True)
        else:
            return self.dim_handler(value.create_variable(child_name), value.dims or ())

    def _create_centered_variable(self, name: str, xdist: bool = False):
        parameters = {
            param: self._create_parameter(param, value, name, xdist=xdist)
            for param, value in self.parameters.items()
        }
        if xdist:
            pymc_distribution = _get_pymc_dim_distribution(self.distribution)
        else:
            pymc_distribution = self.pymc_distribution

        return pymc_distribution(name, **parameters, dims=self.dims)

    def _create_non_centered_variable(
        self, name: str, xdist: bool = False
    ) -> TensorVariable | XTensorVariable:
        def handle_variable(var_name: str):
            parameter = self.parameters[var_name]
            if not hasattr(parameter, "create_variable"):
                return parameter

            if xdist:
                return parameter.create_variable(f"{name}_{var_name}", xdist=True)
            else:
                return self.dim_handler(
                    parameter.create_variable(f"{name}_{var_name}"),
                    parameter.dims,
                )

        defaults = self.non_centered_distributions[self.distribution]
        other_parameters = {
            param: handle_variable(param)
            for param in self.parameters.keys()
            if param not in defaults
        }
        if xdist:
            pymc_distribution = _get_pymc_dim_distribution(self.distribution)
        else:
            pymc_distribution = self.pymc_distribution

        offset = pymc_distribution(
            f"{name}_offset",
            **defaults,
            **other_parameters,
            dims=self.dims,
        )

        if "mu" in self.parameters:
            mu = (
                handle_variable("mu")
                if isinstance(self.parameters["mu"], Prior)
                else self.parameters["mu"]
            )
        else:
            mu = 0

        sigma = (
            handle_variable("sigma")
            if isinstance(self.parameters["sigma"], Prior)
            else self.parameters["sigma"]
        )

        if xdist:
            from pymc.dims import Deterministic

            det_class = Deterministic

        else:
            det_class = pm.Deterministic

        return det_class(
            name,
            mu + sigma * offset,
            dims=self.dims,
        )

    def create_variable(self, name: str, xdist: bool = False) -> TensorVariable | XTensorVariable:
        """Create a PyMC variable from the prior.

        Must be used in a PyMC model context.

        Parameters
        ----------
        name : str
            The name of the variable.
        xdist: bool, default False
            Whether to create a variable from pymc.dims or regular pymc distributions

        Returns
        -------
        TensorVariable | XTensorVariable
            The PyMC variable.

        Examples
        --------
        Create a hierarchical normal variable in larger PyMC model.

        .. code-block:: python

            dist = Prior(
                "Normal",
                mu=Prior("Normal"),
                sigma=Prior("HalfNormal"),
                dims="channel",
            )

            coords = {"channel": ["C1", "C2", "C3"]}
            with pm.Model(coords=coords):
                var = dist.create_variable("var")

        """
        # FIXME: We shouldn't mutate self when creating variables
        self.dim_handler = create_dim_handler(self.dims or ())

        if self.transform:
            var_name = f"{name}_raw"
            pytensor_transform = _get_transform(self.transform, xdist=xdist)

            def transform(var):
                if xdist:
                    from pymc.dims import Deterministic

                    det_class = Deterministic
                else:
                    det_class = pm.Deterministic

                return det_class(name, pytensor_transform(var), dims=self.dims)

        else:
            var_name = name

            def transform(var):
                return var

        create_variable = (
            self._create_centered_variable if self.centered else self._create_non_centered_variable
        )
        var = create_variable(name=var_name, xdist=xdist)
        return transform(var)

    @property
    def preliz(self):
        """Create an equivalent preliz distribution.

        Helpful to visualize a distribution when it is univariate.

        Returns
        -------
        preliz.distributions.Distribution

        Examples
        --------
        Create a preliz distribution from a prior.

        .. code-block:: python

            from pymc_extras.prior import Prior

            dist = Prior("Gamma", alpha=5, beta=1)
            dist.preliz.plot_pdf()

        """
        import preliz as pz

        return getattr(pz, self.distribution)(**self.parameters)

    def to_dict(self) -> dict[str, Any]:
        """Convert the prior to dictionary format.

        Returns
        -------
        dict[str, Any]
            The dictionary format of the prior.

        Examples
        --------
        Convert a prior to the dictionary format.

        .. code-block:: python

            from pymc_extras.prior import Prior

            dist = Prior("Normal", mu=0, sigma=1)

            dist.to_dict()

        Convert a hierarchical prior to the dictionary format.

        .. code-block:: python

            dist = Prior(
                "Normal",
                mu=Prior("Normal"),
                sigma=Prior("HalfNormal"),
                dims="channel",
            )

            dist.to_dict()

        """
        data: dict[str, Any] = {
            "dist": self.distribution,
        }
        if self.parameters:

            def handle_value(value):
                if isinstance(value, Prior):
                    return value.to_dict()

                if isinstance(value, Variable):
                    if isinstance(value.type, pt.TensorType):
                        value = value.eval()

                    # Avoid XTensor import warnings, remove this when the warnings are gone
                    elif value.type.__class__.__name__.startswith("XTensor"):
                        value = xr.DataArray(value.eval(), dims=value.type.dims)

                    else:
                        raise ValueError(
                            f"Prior does not know how to serialize pytensor variable of type {value.type}"
                        )

                if isinstance(value, np.ndarray):
                    return value.tolist()

                if isinstance(value, xr.DataArray):
                    return {
                        "class": "DataArray",
                        "data": value.data.tolist(),
                        "dims": list(value.dims),
                    }

                if hasattr(value, "to_dict"):
                    return value.to_dict()

                return value

            data["kwargs"] = {
                param: handle_value(value) for param, value in self.parameters.items()
            }
        if not self.centered:
            data["centered"] = False

        if self.dims is not None:
            data["dims"] = self.dims

        if self.transform:
            data["transform"] = self.transform

        return data

    @classmethod
    def from_dict(cls, data) -> Prior:
        """Create a Prior from the dictionary format.

        Parameters
        ----------
        data : dict[str, Any]
            The dictionary format of the prior.

        Returns
        -------
        Prior
            The prior distribution.

        Examples
        --------
        Convert prior in the dictionary format to a Prior instance.

        .. code-block:: python

            from pymc_extras.prior import Prior

            data = {
                "dist": "Normal",
                "kwargs": {"mu": 0, "sigma": 1},
            }

            dist = Prior.from_dict(data)
            dist
            # Prior("Normal", mu=0, sigma=1)

        """
        if not isinstance(data, dict):
            msg = (
                "Must be a dictionary representation of a prior distribution. "
                f"Not of type: {type(data)}"
            )
            raise ValueError(msg)

        dist = data["dist"]
        kwargs = data.get("kwargs", {})

        def handle_value(value):
            if isinstance(value, dict):
                return deserialize(value)

            if isinstance(value, list):
                return np.array(value)

            return value

        kwargs = {param: handle_value(value) for param, value in kwargs.items()}
        centered = data.get("centered", True)
        dims = data.get("dims")
        if isinstance(dims, list):
            dims = tuple(dims)
        transform = data.get("transform")

        return cls(dist, dims=dims, centered=centered, transform=transform, **kwargs)

    def constrain(self, lower: float, upper: float, mass: float = 0.95, kwargs=None) -> Prior:
        """Create a new prior with a given mass constrained within the given bounds.

        Wrapper around `preliz.maxent`.

        Parameters
        ----------
        lower : float
            The lower bound.
        upper : float
            The upper bound.
        mass: float = 0.95
            The mass of the distribution to keep within the bounds.
        kwargs : dict
            Additional arguments to pass to `pz.maxent`.

        Returns
        -------
        Prior
            The maximum entropy prior with a mass constrained to the given bounds.

        Examples
        --------
        Create a Beta distribution that is constrained to have 95% of the mass
        between 0.5 and 0.8.

        .. code-block:: python

            dist = Prior(
                "Beta",
            ).constrain(lower=0.5, upper=0.8)

        Create a Beta distribution with mean 0.6, that is constrained to
        have 95% of the mass between 0.5 and 0.8.

        .. code-block:: python

            dist = Prior(
                "Beta",
                mu=0.6,
            ).constrain(lower=0.5, upper=0.8)

        """
        from preliz import maxent

        if self.transform:
            raise ValueError("Can't constrain a transformed variable")

        if kwargs is None:
            kwargs = {}
            kwargs.setdefault("plot", False)

        if kwargs["plot"]:
            new_parameters = maxent(self.preliz, lower, upper, mass, **kwargs)[0].params_dict
        else:
            new_parameters = maxent(self.preliz, lower, upper, mass, **kwargs).params_dict

        return Prior(
            self.distribution,
            dims=self.dims,
            transform=self.transform,
            centered=self.centered,
            **new_parameters,
        )

    def __eq__(self, other) -> bool:
        """Check if two priors are equal."""
        if not isinstance(other, Prior):
            return False

        try:
            np.testing.assert_equal(self.parameters, other.parameters)
        except AssertionError:
            return False

        return (
            self.distribution == other.distribution
            and self.dims == other.dims
            and self.centered == other.centered
            and self.transform == other.transform
        )

    def sample_prior(
        self,
        coords=None,
        name: str = "variable",
        xdist: bool = False,
        **sample_prior_predictive_kwargs,
    ) -> xr.Dataset:
        """Sample the prior distribution for the variable.

        Parameters
        ----------
        coords : dict[str, list[str]], optional
            The coordinates for the variable, by default None.
            Only required if the dims are specified.
        name : str, optional
            The name of the variable, by default "variable".
        sample_prior_predictive_kwargs : dict
            Additional arguments to pass to `pm.sample_prior_predictive`.

        Returns
        -------
        xr.Dataset
            The dataset of the prior samples.

        Example
        -------
        Sample from a hierarchical normal distribution.

        .. code-block:: python

            dist = Prior(
                "Normal",
                mu=Prior("Normal"),
                sigma=Prior("HalfNormal"),
                dims="channel",
            )

            coords = {"channel": ["C1", "C2", "C3"]}
            prior = dist.sample_prior(coords=coords)

        """
        return sample_prior(
            factory=self,
            coords=coords,
            name=name,
            xdist=xdist,
            **sample_prior_predictive_kwargs,
        )

    def __deepcopy__(self, memo) -> Prior:
        """Return a deep copy of the prior."""
        if id(self) in memo:
            return memo[id(self)]

        copy_obj = Prior(
            self.distribution,
            dims=copy.copy(self.dims),
            centered=self.centered,
            transform=self.transform,
            **copy.deepcopy(self.parameters),
        )
        memo[id(self)] = copy_obj
        return copy_obj

    def deepcopy(self) -> Prior:
        """Return a deep copy of the prior."""
        return copy.deepcopy(self)

    def to_graph(self):
        """Generate a graph of the variables.

        Examples
        --------
        Create the graph for a 2D transformed hierarchical distribution.

        .. code-block:: python

            from pymc_extras.prior import Prior

            mu = Prior(
                "Normal",
                mu=Prior("Normal"),
                sigma=Prior("HalfNormal"),
                dims="channel",
            )
            sigma = Prior("HalfNormal", dims="channel")
            dist = Prior(
                "Normal",
                mu=mu,
                sigma=sigma,
                dims=("channel", "geo"),
                centered=False,
                transform="sigmoid",
            )

            dist.to_graph()

        .. image:: /_static/example-graph.png
            :alt: Example graph

        """
        coords = {name: ["DUMMY"] for name in self.dims or ()}
        with pm.Model(coords=coords) as model:
            self.create_variable("var")

        return pm.model_to_graphviz(model)

    def create_likelihood_variable(
        self,
        name: str,
        mu: TensorLike | XTensorLike,
        observed: TensorLike | XTensorLike,
        xdist: bool = False,
    ) -> TensorVariable | XTensorVariable:
        """Create a likelihood variable from the prior.

        Will require that the distribution has a `mu` parameter
        and that it has not been set in the parameters.

        Parameters
        ----------
        name : str
            The name of the variable.
        mu : TensorLike or XTensorLike
            The mu parameter for the likelihood.
        observed : TensorLike or XTensorLike
            The observed data.
        xdist: bool, default False
            Whether to create a variable from pymc.dims or regular pymc distributions

        Returns
        -------
        TensorVariable or XTensorVariable
            The PyMC variable.

        Examples
        --------
        Create a likelihood variable in a larger PyMC model.

        .. code-block:: python

            import pymc as pm

            dist = Prior("Normal", sigma=Prior("HalfNormal"))

            with pm.Model():
                # Create the likelihood variable
                mu = pm.Normal("mu", mu=0, sigma=1)
                dist.create_likelihood_variable("y", mu=mu, observed=observed)

        """
        if "mu" not in _get_pymc_parameters(self.pymc_distribution):
            raise UnsupportedDistributionError(
                f"Likelihood distribution {self.distribution!r} is not supported."
            )

        if "mu" in self.parameters:
            raise MuAlreadyExistsError(self)

        distribution = self.deepcopy()
        distribution.parameters["mu"] = mu
        distribution.parameters["observed"] = observed
        return distribution.create_variable(name, xdist=xdist)


class VariableNotFound(Exception):
    """Variable is not found."""


def _remove_random_variable(var: pt.TensorVariable) -> None:
    # This is brittle, as it doesn't rely on any official model API.
    # Fix this by allowing `Prior.create_dist` instead
    model = pm.modelcontext(None)
    model.rvs_to_initial_values.pop(var)
    model.rvs_to_transforms.pop(var)
    model.rvs_to_values.pop(var)
    model.free_RVs.remove(var)
    model.named_vars.pop(var.name)


@dataclass
class Censored:
    """Create censored random variable.

    Examples
    --------
    Create a censored Normal distribution:

    .. code-block:: python

        from pymc_extras.prior import Prior, Censored

        normal = Prior("Normal")
        censored_normal = Censored(normal, lower=0)

    Create hierarchical censored Normal distribution:

    .. code-block:: python

        from pymc_extras.prior import Prior, Censored

        normal = Prior(
            "Normal",
            mu=Prior("Normal"),
            sigma=Prior("HalfNormal"),
            dims="channel",
        )
        censored_normal = Censored(normal, lower=0)

        coords = {"channel": range(3)}
        samples = censored_normal.sample_prior(coords=coords)

    """

    distribution: InstanceOf[Prior]
    lower: float | InstanceOf[pt.TensorVariable] = -np.inf
    upper: float | InstanceOf[pt.TensorVariable] = np.inf

    def __post_init__(self) -> None:
        """Check validity at initialization."""
        if not self.distribution.centered:
            raise ValueError(
                "Censored distribution must be centered so that .dist() API can be used on distribution."
            )

        if self.distribution.transform is not None:
            raise ValueError(
                "Censored distribution can't have a transform so that .dist() API can be used on distribution."
            )

    @property
    def dims(self) -> tuple[str, ...] | None:
        """The dims from the distribution to censor."""
        return self.distribution.dims

    @dims.setter
    def dims(self, dims) -> None:
        self.distribution.dims = dims

    def create_variable(self, name: str, xdist: bool = False) -> pt.TensorVariable:
        """Create censored random variable."""
        if xdist:
            raise NotImplementedError("Censored does not support xdist yet")

        dist = self.distribution.create_variable(name)
        _remove_random_variable(var=dist)

        return pm.Censored(name, dist, lower=self.lower, upper=self.upper, dims=self.dims)

    def to_dict(self) -> dict[str, Any]:
        """Convert the censored distribution to a dictionary."""

        def handle_value(value):
            if isinstance(value, pt.TensorVariable):
                return value.eval().tolist()

            return value

        return {
            "class": "Censored",
            "data": {
                "dist": self.distribution.to_dict(),
                "lower": handle_value(self.lower),
                "upper": handle_value(self.upper),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Censored:
        """Create a censored distribution from a dictionary."""
        data = data["data"]
        return cls(  # type: ignore
            distribution=deserialize(data["dist"]),
            lower=data["lower"],
            upper=data["upper"],
        )

    def sample_prior(
        self,
        coords=None,
        name: str = "variable",
        xdist: bool = False,
        **sample_prior_predictive_kwargs,
    ) -> xr.Dataset:
        """Sample the prior distribution for the variable.

        Parameters
        ----------
        coords : dict[str, list[str]], optional
            The coordinates for the variable, by default None.
            Only required if the dims are specified.
        name : str, optional
            The name of the variable, by default "var".
        sample_prior_predictive_kwargs : dict
            Additional arguments to pass to `pm.sample_prior_predictive`.

        Returns
        -------
        xr.Dataset
            The dataset of the prior samples.

        Example
        -------
        Sample from a censored Gamma distribution.

        .. code-block:: python

            gamma = Prior("Gamma", mu=1, sigma=1, dims="channel")
            dist = Censored(gamma, lower=0.5)

            coords = {"channel": ["C1", "C2", "C3"]}
            prior = dist.sample_prior(coords=coords)

        """
        return sample_prior(
            factory=self,
            coords=coords,
            name=name,
            xdist=xdist,
            **sample_prior_predictive_kwargs,
        )

    def to_graph(self):
        """Generate a graph of the variables.

        Examples
        --------
        Create graph for a censored Normal distribution

        .. code-block:: python

            from pymc_extras.prior import Prior, Censored

            normal = Prior("Normal")
            censored_normal = Censored(normal, lower=0)

            censored_normal.to_graph()

        """
        coords = {name: ["DUMMY"] for name in self.dims or ()}
        with pm.Model(coords=coords) as model:
            self.create_variable("var")

        return pm.model_to_graphviz(model)

    def create_likelihood_variable(
        self,
        name: str,
        mu: pt.TensorLike,
        observed: pt.TensorLike,
    ) -> pt.TensorVariable:
        """Create observed censored variable.

        Will require that the distribution has a `mu` parameter
        and that it has not been set in the parameters.

        Parameters
        ----------
        name : str
            The name of the variable.
        mu : pt.TensorLike
            The mu parameter for the likelihood.
        observed : pt.TensorLike
            The observed data.

        Returns
        -------
        pt.TensorVariable
            The PyMC variable.

        Examples
        --------
        Create a censored likelihood variable in a larger PyMC model.

        .. code-block:: python

            import pymc as pm
            from pymc_extras.prior import Prior, Censored

            normal = Prior("Normal", sigma=Prior("HalfNormal"))
            dist = Censored(normal, lower=0)

            observed = 1

            with pm.Model():
                # Create the likelihood variable
                mu = pm.HalfNormal("mu", sigma=1)
                dist.create_likelihood_variable("y", mu=mu, observed=observed)

        """
        if "mu" not in _get_pymc_parameters(self.distribution.pymc_distribution):
            raise UnsupportedDistributionError(
                f"Likelihood distribution {self.distribution.distribution!r} is not supported."
            )

        if "mu" in self.distribution.parameters:
            raise MuAlreadyExistsError(self.distribution)

        distribution = self.distribution.deepcopy()
        distribution.parameters["mu"] = mu

        dist = distribution.create_variable(name)
        _remove_random_variable(var=dist)

        return pm.Censored(
            name,
            dist,
            observed=observed,
            lower=self.lower,
            upper=self.upper,
            dims=self.dims,
        )


class Scaled:
    """Scaled distribution for numerical stability.

    This is the same as multiplying the variable by a constant factor.

    Parameters
    ----------
    dist : Prior
        The prior distribution to scale.
    factor : pt.TensorLike
        The scaling factor. This will have to be broadcastable to the
        dimensions of the distribution.

    Examples
    --------
    Create a scaled normal distribution.

    .. code-block:: python

        from pymc_extras.prior import Prior, Scaled

        normal = Prior("Normal", mu=0, sigma=1)
        # Same as Normal(mu=0, sigma=10)
        scaled_normal = Scaled(normal, factor=10)

    """

    def __init__(self, dist: Prior, factor: XTensorLike) -> None:
        self.dist = dist
        self.factor = factor

    @property
    def dims(self) -> Dims | None:
        """The dimensions of the scaled distribution."""
        return self.dist.dims

    def create_variable(self, name: str, xdist: bool = False) -> TensorVariable | XTensorVariable:
        """Create a scaled variable.

        Parameters
        ----------
        name : str
            The name of the variable.

        Returns
        -------
        pt.TensorVariable
            The scaled variable.
        """
        var = self.dist.create_variable(f"{name}_unscaled", xdist=xdist)
        if xdist:
            from pymc.dims import Deterministic

            det_class = Deterministic
        else:
            det_class = pm.Deterministic

        return det_class(name, var * self.factor, dims=self.dims)


def _is_prior_type(data: dict) -> bool:
    return "dist" in data


def _is_censored_type(data: dict) -> bool:
    return data.keys() == {"class", "data"} and data["class"] == "Censored"


def _is_data_array_type(data: dict) -> bool:
    return data.keys() == {"class", "data", "dims"} and data["class"] == "DataArray"


register_deserialization(is_type=_is_prior_type, deserialize=Prior.from_dict)
register_deserialization(is_type=_is_censored_type, deserialize=Censored.from_dict)
register_deserialization(is_type=_is_data_array_type, deserialize=xr.DataArray.from_dict)


def __getattr__(name: str):
    """Get Prior class through the module.

    Examples
    --------
    Create a normal distribution.

    .. code-block:: python

        from pymc_extras.prior import Normal

        dist = Normal(mu=1, sigma=2)

    Create a hierarchical normal distribution.

    .. code-block:: python

        import pymc_extras.prior as pr

        dist = pr.Normal(mu=pr.Normal(), sigma=pr.HalfNormal(), dims="channel")
        samples = dist.sample_prior(coords={"channel": ["C1", "C2", "C3"]})

    """
    # Ignore Python internal attributes needed for introspection
    if name.startswith("__"):
        raise AttributeError(name)

    _get_pymc_distribution(name)
    return partial(Prior, distribution=name)
