from copy import deepcopy

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import xarray as xr

from graphviz.graphs import Digraph
from preliz.distributions import distributions as preliz_distributions
from pydantic import ValidationError
from pymc.model_graph import fast_eval
from xarray import DataArray

import pymc_extras.prior as pr

from pymc_extras.deserialize import DESERIALIZERS, deserialize, register_deserialization
from pymc_extras.prior import (
    Censored,
    MuAlreadyExistsError,
    Prior,
    Scaled,
    UnknownTransformError,
    UnsupportedDistributionError,
    UnsupportedParameterizationError,
    UnsupportedShapeError,
    VariableFactory,
    handle_dims,
    register_tensor_transform,
    sample_prior,
)
from pymc_extras.utils.model_equivalence import equivalent_models


@pytest.mark.parametrize(
    "x, dims, desired_dims, expected_fn",
    [
        (np.arange(3), "channel", "channel", lambda x: x),
        (np.arange(3), "channel", ("geo", "channel"), lambda x: x),
        (np.arange(3), "channel", ("channel", "geo"), lambda x: x[:, None]),
        (np.arange(3), "channel", ("x", "y", "channel", "geo"), lambda x: x[:, None]),
        (
            np.arange(3 * 2).reshape(3, 2),
            ("channel", "geo"),
            ("geo", "x", "y", "channel"),
            lambda x: x.T[:, None, None, :],
        ),
        (
            np.arange(4 * 2 * 3).reshape(4, 2, 3),
            ("channel", "geo", "store"),
            ("geo", "x", "store", "channel"),
            lambda x: x.swapaxes(0, 2).swapaxes(0, 1)[:, None, :, :],
        ),
    ],
    ids=[
        "same_dims",
        "different_dims",
        "dim_padding",
        "just_enough_dims",
        "transpose_and_padding",
        "swaps_and_padding",
    ],
)
def test_handle_dims(x, dims, desired_dims, expected_fn) -> None:
    result = handle_dims(x, dims, desired_dims)
    if isinstance(result, pt.TensorVariable):
        result = fast_eval(result)

    np.testing.assert_array_equal(result, expected_fn(x))


@pytest.mark.parametrize(
    "x, dims, desired_dims",
    [
        (np.ones(3), "channel", "something_else"),
        (np.ones((3, 2)), ("a", "b"), ("a", "B")),
    ],
    ids=["no_incommon", "some_incommon"],
)
def test_handle_dims_with_impossible_dims(x, dims, desired_dims) -> None:
    match = " are not a subset of the desired dims "
    with pytest.raises(UnsupportedShapeError, match=match):
        handle_dims(x, dims, desired_dims)


def test_missing_transform() -> None:
    match = r"Function 'foo_bar' not present in pytensor.tensor or pymc.math"
    with pytest.raises(UnknownTransformError, match=match):
        Prior("Normal", transform="foo_bar")


def test_getattr() -> None:
    assert pr.Normal() == Prior("Normal")


def test_import_directly() -> None:
    try:
        from pymc_extras.prior import Normal
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

    assert Normal() == Prior("Normal")


def test_import_incorrect_directly() -> None:
    match = "PyMC doesn't have a distribution of name 'SomeIncorrectDistribution'"
    with pytest.raises(UnsupportedDistributionError, match=match):
        from pymc_extras.prior import SomeIncorrectDistribution  # noqa: F401


def test_get_item() -> None:
    var = Prior("Normal", mu=0, sigma=1)

    assert var["mu"] == 0
    assert var["sigma"] == 1


def test_noncentered_needs_params() -> None:
    with pytest.raises(ValueError):
        Prior(
            "Normal",
            centered=False,
        )


def test_different_than_pymc_params() -> None:
    with pytest.raises(ValueError):
        Prior("Normal", mu=0, b=1)


def test_non_unique_dims() -> None:
    with pytest.raises(ValueError):
        Prior("Normal", mu=0, sigma=1, dims=("channel", "channel"))


def test_doesnt_check_validity_parameterization() -> None:
    try:
        Prior("Normal", mu=0, sigma=1, tau=1)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


def test_doesnt_check_validity_values() -> None:
    try:
        Prior("Normal", mu=0, sigma=-1)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


def test_preliz() -> None:
    var = Prior("Normal", mu=0, sigma=1)
    dist = var.preliz
    assert isinstance(dist, preliz_distributions.Distribution)


@pytest.mark.parametrize(
    "var, expected",
    [
        (Prior("Normal", mu=0, sigma=1), 'Prior("Normal", mu=0, sigma=1)'),
        (
            Prior("Normal", mu=Prior("Normal"), sigma=Prior("HalfNormal")),
            'Prior("Normal", mu=Prior("Normal"), sigma=Prior("HalfNormal"))',
        ),
        (Prior("Normal", dims="channel"), 'Prior("Normal", dims="channel")'),
        (
            Prior("Normal", mu=0, sigma=1, transform="sigmoid"),
            'Prior("Normal", mu=0, sigma=1, transform="sigmoid")',
        ),
    ],
)
def test_str(var, expected) -> None:
    assert str(var) == expected


@pytest.mark.parametrize(
    "var",
    [
        Prior("Normal", mu=0, sigma=1),
        Prior("Normal", mu=Prior("Normal"), sigma=Prior("HalfNormal"), dims="channel"),
        Prior("Normal", dims=("geo", "channel")),
    ],
)
def test_repr(var) -> None:
    assert eval(repr(var)) == var


def test_invalid_distribution() -> None:
    with pytest.raises(UnsupportedDistributionError):
        Prior("Invalid")


def test_broadcast_doesnt_work():
    with pytest.raises(UnsupportedShapeError):
        Prior(
            "Normal",
            mu=0,
            sigma=Prior("HalfNormal", sigma=1, dims="x"),
            dims="y",
        )


def test_dim_workaround_flaw() -> None:
    distribution = Prior(
        "Normal",
        mu=Prior("Normal"),
        sigma=Prior("HalfNormal"),
        dims="y",
    )

    try:
        distribution["mu"].dims = "x"
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

    with pytest.raises(UnsupportedShapeError):
        distribution._param_dims_work()


def test_noncentered_error() -> None:
    with pytest.raises(UnsupportedParameterizationError):
        Prior(
            "Gamma",
            mu=0,
            sigma=1,
            dims="x",
            centered=False,
        )


def test_create_variable_multiple_times() -> None:
    mu = Prior(
        "Normal",
        mu=Prior("Normal"),
        sigma=Prior("HalfNormal"),
        dims="channel",
        centered=False,
    )

    coords = {
        "channel": ["a", "b", "c"],
    }
    with pm.Model(coords=coords) as model:
        mu.create_variable("mu")
        mu.create_variable("mu_2")

    suffixes = [
        "",
        "_offset",
        "_mu",
        "_sigma",
    ]
    dims = [(3,), (3,), (), ()]

    for prefix in ["mu", "mu_2"]:
        for suffix, dim in zip(suffixes, dims, strict=False):
            assert fast_eval(model[f"{prefix}{suffix}"]).shape == dim


def test_create_variable() -> None:
    large_var = Prior(
        "Normal",
        mu=Prior(
            "Normal",
            mu=Prior("Normal", mu=1),
            sigma=Prior("HalfNormal"),
            dims="channel",
            centered=False,
        ),
        sigma=Prior("HalfNormal", sigma=Prior("HalfNormal"), dims="geo"),
        dims=("geo", "channel"),
    )

    coords = {
        "channel": ["a", "b", "c"],
        "geo": ["x", "y"],
    }
    with pm.Model(coords=coords) as model:
        large_var.create_variable("var")

    var_names = [
        "var",
        "var_mu",
        "var_sigma",
        "var_mu_offset",
        "var_mu_mu",
        "var_mu_sigma",
        "var_sigma_sigma",
    ]
    assert set(var.name for var in model.unobserved_RVs) == set(var_names)
    dims = [
        (2, 3),
        (3,),
        (2,),
        (3,),
        (),
        (),
        (),
    ]
    for var_name, dim in zip(var_names, dims, strict=False):
        assert fast_eval(model[var_name]).shape == dim


def test_transform() -> None:
    var = Prior("Normal", mu=0, sigma=1, transform="sigmoid")

    with pm.Model() as model:
        var.create_variable("var")

    var_names = [
        "var",
        "var_raw",
    ]
    dims = [
        (),
        (),
    ]
    for var_name, dim in zip(var_names, dims, strict=False):
        assert fast_eval(model[var_name]).shape == dim


def test_to_dict() -> None:
    large_var = Prior(
        "Normal",
        mu=Prior(
            "Normal",
            mu=Prior("Normal", mu=1),
            sigma=Prior("HalfNormal"),
            dims="channel",
            centered=False,
        ),
        sigma=Prior("HalfNormal", sigma=Prior("HalfNormal"), dims="geo"),
        dims=("geo", "channel"),
    )

    data = large_var.to_dict()
    assert data == {
        "dist": "Normal",
        "kwargs": {
            "mu": {
                "dist": "Normal",
                "kwargs": {
                    "mu": {
                        "dist": "Normal",
                        "kwargs": {
                            "mu": 1,
                        },
                    },
                    "sigma": {
                        "dist": "HalfNormal",
                    },
                },
                "centered": False,
                "dims": ("channel",),
            },
            "sigma": {
                "dist": "HalfNormal",
                "kwargs": {
                    "sigma": {
                        "dist": "HalfNormal",
                    },
                },
                "dims": ("geo",),
            },
        },
        "dims": ("geo", "channel"),
    }

    assert Prior.from_dict(data) == large_var


def test_to_dict_numpy() -> None:
    var = Prior("Normal", mu=np.array([0, 10, 20]), dims="channel")
    assert var.to_dict() == {
        "dist": "Normal",
        "kwargs": {
            "mu": [0, 10, 20],
        },
        "dims": ("channel",),
    }


@pytest.mark.parametrize("dims", (None, (), ("city",)))
def test_dict_round_trip_dims(dims) -> None:
    prior = Prior("Normal", dims=dims)
    d = prior.to_dict()

    expected_d = {"dist": "Normal"}
    if dims is not None:
        expected_d["dims"] = dims
    assert d == expected_d

    prior_again = Prior.from_dict(d)
    assert prior_again.dims == dims
    assert prior_again == prior
    assert prior_again.to_dict() == expected_d


def test_constrain_with_transform_error() -> None:
    var = Prior("Normal", transform="sigmoid")

    with pytest.raises(ValueError):
        var.constrain(lower=0, upper=1)


def test_constrain() -> None:
    var = Prior("Normal")

    new_var = var.constrain(lower=0, upper=1, mass=0.9545)
    np.testing.assert_allclose(new_var.parameters["mu"], 0.5, rtol=1e-4)
    np.testing.assert_allclose(new_var.parameters["sigma"], 0.25, rtol=1e-4)


def test_dims_change() -> None:
    var = Prior("Normal", mu=0, sigma=1)
    var.dims = "channel"

    assert var.dims == ("channel",)


def test_dims_change_error() -> None:
    mu = Prior("Normal", dims="channel")
    var = Prior("Normal", mu=mu, dims="channel")

    with pytest.raises(UnsupportedShapeError):
        var.dims = "geo"


def test_deepcopy() -> None:
    priors = {
        "alpha": Prior("Beta", alpha=1, beta=1),
        "gamma": Prior("Normal", mu=0, sigma=1),
    }

    new_priors = deepcopy(priors)
    priors["alpha"].dims = "channel"

    assert new_priors["alpha"].dims is None


def test_backwards_compat() -> None:
    """Make sure functionality is compatible with use in PyMC-marketing, where Prior objects originated from."""
    mmm_default_model_config = {
        "intercept": {"dist": "Normal", "kwargs": {"mu": 0, "sigma": 2}},
        "likelihood": {
            "dist": "Normal",
            "kwargs": {
                "sigma": {"dist": "HalfNormal", "kwargs": {"sigma": 2}},
            },
        },
        "gamma_control": {
            "dist": "Normal",
            "kwargs": {"mu": 0, "sigma": 2},
            "dims": "control",
        },
        "gamma_fourier": {
            "dist": "Laplace",
            "kwargs": {"mu": 0, "b": 1},
            "dims": "fourier_mode",
        },
    }

    result = {param: Prior.from_dict(value) for param, value in mmm_default_model_config.items()}
    assert result == {
        "intercept": Prior("Normal", mu=0, sigma=2),
        "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=2)),
        "gamma_control": Prior("Normal", mu=0, sigma=2, dims="control"),
        "gamma_fourier": Prior("Laplace", mu=0, b=1, dims="fourier_mode"),
    }


def test_sample_prior() -> None:
    var = Prior(
        "Normal",
        mu=Prior("Normal"),
        sigma=Prior("HalfNormal"),
        dims="channel",
        transform="sigmoid",
    )

    coords = {"channel": ["A", "B", "C"]}
    prior = var.sample_prior(coords=coords, draws=25)

    assert isinstance(prior, xr.Dataset)
    assert prior.sizes == {"chain": 1, "draw": 25, "channel": 3}


def test_sample_prior_missing_coords() -> None:
    dist = Prior("Normal", dims="channel")

    with pytest.raises(KeyError, match="Coords"):
        dist.sample_prior()


def test_to_graph() -> None:
    hierarchical_distribution = Prior(
        "Normal",
        mu=Prior("Normal"),
        sigma=Prior("HalfNormal"),
        dims="channel",
    )

    G = hierarchical_distribution.to_graph()
    assert isinstance(G, Digraph)


def test_from_dict_list() -> None:
    data = {
        "dist": "Normal",
        "kwargs": {
            "mu": [0, 1, 2],
            "sigma": 1,
        },
        "dims": "channel",
    }

    var = Prior.from_dict(data)
    assert var.dims == ("channel",)
    assert isinstance(var["mu"], np.ndarray)
    np.testing.assert_array_equal(var["mu"], [0, 1, 2])


def test_from_dict_list_dims() -> None:
    data = {
        "dist": "Normal",
        "kwargs": {
            "mu": 0,
            "sigma": 1,
        },
        "dims": ["channel", "geo"],
    }

    var = Prior.from_dict(data)
    assert var.dims == ("channel", "geo")


def test_to_dict_transform() -> None:
    dist = Prior("Normal", transform="sigmoid")

    data = dist.to_dict()
    assert data == {
        "dist": "Normal",
        "transform": "sigmoid",
    }


def test_equality_non_prior() -> None:
    dist = Prior("Normal")

    assert dist != 1


def test_deepcopy_memo() -> None:
    memo = {}
    dist = Prior("Normal")
    memo[id(dist)] = dist
    deepcopy(dist, memo)
    assert len(memo) == 1
    deepcopy(dist, memo)

    assert len(memo) == 1


def test_create_likelihood_variable() -> None:
    distribution = Prior("Normal", sigma=Prior("HalfNormal"))

    with pm.Model() as model:
        mu = pm.Normal("mu")

        data = distribution.create_likelihood_variable("data", mu=mu, observed=10)

    assert model.observed_RVs == [data]
    assert "data_sigma" in model


def test_create_likelihood_variable_already_has_mu() -> None:
    distribution = Prior("Normal", mu=Prior("Normal"), sigma=Prior("HalfNormal"))

    with pm.Model():
        mu = pm.Normal("mu")

        with pytest.raises(MuAlreadyExistsError):
            distribution.create_likelihood_variable("data", mu=mu, observed=10)


def test_create_likelihood_non_mu_parameterized_distribution() -> None:
    distribution = Prior("Cauchy")

    with pm.Model():
        mu = pm.Normal("mu")
        with pytest.raises(UnsupportedDistributionError):
            distribution.create_likelihood_variable("data", mu=mu, observed=10)


def test_non_centered_student_t() -> None:
    try:
        Prior(
            "StudentT",
            mu=Prior("Normal"),
            sigma=Prior("HalfNormal"),
            nu=Prior("HalfNormal"),
            dims="channel",
            centered=False,
        )
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


def test_cant_reset_distribution() -> None:
    dist = Prior("Normal")
    with pytest.raises(AttributeError, match="Can't change the distribution"):
        dist.distribution = "Cauchy"


def test_nonstring_distribution() -> None:
    with pytest.raises(ValidationError, match=".*Input should be a valid string.*"):
        Prior(pm.Normal)


def test_change_the_transform() -> None:
    dist = Prior("Normal")
    dist.transform = "logit"
    assert dist.transform == "logit"


def test_nonstring_transform() -> None:
    with pytest.raises(ValidationError, match=".*Input should be a valid string.*"):
        Prior("Normal", transform=pm.math.log)


def test_checks_param_value_types() -> None:
    with pytest.raises(ValueError, match="Parameters must be one of the following types"):
        Prior("Normal", mu="str", sigma="str")


def test_check_equality_with_numpy() -> None:
    dist = Prior("Normal", mu=np.array([1, 2, 3]), sigma=1)
    assert dist == dist.deepcopy()


def clear_custom_transforms() -> None:
    global CUSTOM_TRANSFORMS
    CUSTOM_TRANSFORMS = {}


def test_custom_transform() -> None:
    new_transform_name = "foo_bar"
    with pytest.raises(UnknownTransformError):
        Prior("Normal", transform=new_transform_name)

    register_tensor_transform(new_transform_name, lambda x: x**2)

    dist = Prior("Normal", transform=new_transform_name)
    prior = dist.sample_prior(draws=10)
    df_prior = prior.to_dataframe()

    np.testing.assert_array_equal(
        df_prior.variable.to_numpy(),
        df_prior.variable_raw.to_numpy() ** 2,
    )


def test_custom_transform_comes_first() -> None:
    # function in pytensor.tensor
    register_tensor_transform("square", lambda x: 2 * x)

    dist = Prior("Normal", transform="square")
    prior = dist.sample_prior(draws=10)
    df_prior = prior.to_dataframe()

    np.testing.assert_array_equal(
        df_prior.variable.to_numpy(),
        2 * df_prior.variable_raw.to_numpy(),
    )

    clear_custom_transforms()


def test_serialize_with_pytensor() -> None:
    sigma = pt.arange(1, 4)
    dist = Prior("Normal", mu=0, sigma=sigma)

    assert dist.to_dict() == {
        "dist": "Normal",
        "kwargs": {
            "mu": 0,
            "sigma": [1, 2, 3],
        },
    }


def test_zsn_non_centered() -> None:
    try:
        Prior("ZeroSumNormal", sigma=1, centered=False)
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


class TestCustomClass:
    def test_sample_prior_arbitrary_no_name(self) -> None:
        class ArbitraryWithoutName:
            def __init__(self, dims: str | tuple[str, ...]) -> None:
                self.dims = dims

            def create_variable(self, name: str, xdist: bool = False):
                if xdist:
                    raise NotImplementedError

                with pm.Model(name=name):
                    location = pm.Normal("location", dims=self.dims)
                    scale = pm.HalfNormal("scale", dims=self.dims)

                    return pm.Normal("standard_normal") * scale + location

        var = ArbitraryWithoutName(dims="channel")

        prior = sample_prior(var, coords={"channel": ["A", "B", "C"]}, draws=25)

        assert isinstance(prior, xr.Dataset)
        assert "variable" not in prior

        prior_with = sample_prior(
            var,
            coords={"channel": ["A", "B", "C"]},
            draws=25,
            wrap=True,
        )

        assert isinstance(prior_with, xr.Dataset)
        assert "variable" in prior_with

    def test_arbitrary_class(self) -> None:
        class Arbitrary:
            def __init__(self, dims: str | tuple[str, ...]) -> None:
                self.dims = dims

            def create_variable(self, name: str):
                return pm.Normal(name, dims=self.dims)

        prior = Prior(
            "Normal",
            mu=Arbitrary(dims=("channel",)),
            sigma=1,
            dims=("channel", "geo"),
        )

        coords = {
            "channel": ["C1", "C2", "C3"],
            "geo": ["G1", "G2"],
        }
        with pm.Model(coords=coords) as model:
            prior.create_variable("var")

        assert "var_mu" in model
        var_mu = model["var_mu"]
        assert fast_eval(var_mu).shape == (len(coords["channel"]),)

    def test_arbitrary_class_sample_prior(self) -> None:
        class Arbitrary:
            def __init__(self, dims: str | tuple[str, ...]) -> None:
                self.dims = dims

            def create_variable(self, name: str):
                return pm.Normal(name, dims=self.dims)

        var = Arbitrary(dims="channel")
        prior = sample_prior(var, coords={"channel": ["A", "B", "C"]}, draws=25)
        assert isinstance(prior, xr.Dataset)

    def test_arbitrary_serialization(self) -> None:
        class ArbitrarySerializable:
            def __init__(self, dims: str | tuple[str, ...]) -> None:
                self.dims = dims

            def create_variable(self, name: str):
                return pm.Normal(name, dims=self.dims)

            def to_dict(self):
                return {"dims": self.dims}

        arbitrary_serialized_data = {"dims": ("channel",)}

        register_deserialization(
            lambda data: isinstance(data, dict) and data.keys() == {"dims"},
            lambda data: ArbitrarySerializable(**data),
        )

        dist = Prior(
            "Normal",
            mu=ArbitrarySerializable(dims=("channel",)),
            sigma=1,
            dims=("channel", "geo"),
        )

        data = {
            "dist": "Normal",
            "kwargs": {
                "mu": arbitrary_serialized_data,
                "sigma": 1,
            },
            "dims": ("channel", "geo"),
        }

        assert dist.to_dict() == data

        dist_again = deserialize(data)
        assert isinstance(dist_again["mu"], ArbitrarySerializable)
        assert dist_again["mu"].dims == ("channel",)

        DESERIALIZERS.pop()


class TestScaled:
    def test_scaled_initializes_correctly(self) -> None:
        """Test that the Scaled class initializes correctly."""
        normal = Prior("Normal", mu=0, sigma=1)
        scaled = Scaled(normal, factor=2.0)

        assert scaled.dist == normal
        assert scaled.factor == 2.0

    def test_scaled_dims_property(self) -> None:
        """Test that the dims property returns the dimensions of the underlying distribution."""
        normal = Prior("Normal", mu=0, sigma=1, dims="channel")
        scaled = Scaled(normal, factor=2.0)

        assert scaled.dims == ("channel",)

        # Test with multiple dimensions
        normal.dims = ("channel", "geo")
        assert scaled.dims == ("channel", "geo")

    def test_scaled_create_variable(self) -> None:
        """Test that the create_variable method properly scales the variable."""
        normal = Prior("Normal", mu=0, sigma=1)
        scaled = Scaled(normal, factor=2.0)

        with pm.Model() as model:
            scaled_var = scaled.create_variable("scaled_var")

        # Check that both the scaled and unscaled variables exist
        assert "scaled_var" in model
        assert "scaled_var_unscaled" in model

        # The deterministic node should be the scaled variable
        assert model["scaled_var"] == scaled_var

    def test_scaled_creates_correct_dimensions(self) -> None:
        """Test that the scaled variable has the correct dimensions."""
        normal = Prior("Normal", dims="channel")
        scaled = Scaled(normal, factor=2.0)

        coords = {"channel": ["A", "B", "C"]}
        with pm.Model(coords=coords):
            scaled_var = scaled.create_variable("scaled_var")

        # Check that the scaled variable has the correct dimensions
        assert fast_eval(scaled_var).shape == (3,)

    def test_scaled_applies_factor(self) -> None:
        """Test that the scaling factor is correctly applied."""
        normal = Prior("Normal", mu=0, sigma=1)
        factor = 3.5
        scaled = Scaled(normal, factor=factor)

        # Sample from prior to verify scaling
        prior = sample_prior(scaled, draws=10, name="scaled_var")
        df_prior = prior.to_dataframe()

        # Check that scaled values are original values times the factor
        unscaled_values = df_prior["scaled_var_unscaled"].to_numpy()
        scaled_values = df_prior["scaled_var"].to_numpy()
        np.testing.assert_allclose(scaled_values, unscaled_values * factor)

    def test_scaled_with_tensor_factor(self) -> None:
        """Test that the Scaled class works with a tensor factor."""
        normal = Prior("Normal", mu=0, sigma=1)
        factor = pt.as_tensor_variable(2.5)
        scaled = Scaled(normal, factor=factor)

        # Sample from prior to verify tensor scaling
        prior = sample_prior(scaled, draws=10, name="scaled_var")
        df_prior = prior.to_dataframe()

        # Check that scaled values are original values times the factor
        unscaled_values = df_prior["scaled_var_unscaled"].to_numpy()
        scaled_values = df_prior["scaled_var"].to_numpy()
        np.testing.assert_allclose(scaled_values, unscaled_values * 2.5)

    def test_scaled_with_hierarchical_prior(self) -> None:
        """Test that the Scaled class works with hierarchical priors."""
        normal = Prior("Normal", mu=Prior("Normal"), sigma=Prior("HalfNormal"), dims="channel")
        scaled = Scaled(normal, factor=2.0)

        coords = {"channel": ["A", "B", "C"]}
        with pm.Model(coords=coords) as model:
            scaled.create_variable("scaled_var")

        # Check that all necessary variables were created
        assert "scaled_var" in model
        assert "scaled_var_unscaled" in model
        assert "scaled_var_unscaled_mu" in model
        assert "scaled_var_unscaled_sigma" in model

    def test_scaled_sample_prior(self) -> None:
        """Test that sample_prior works with the Scaled class."""
        normal = Prior("Normal", dims="channel")
        scaled = Scaled(normal, factor=2.0)

        coords = {"channel": ["A", "B", "C"]}
        prior = sample_prior(scaled, coords=coords, draws=25, name="scaled_var")

        assert isinstance(prior, xr.Dataset)
        assert prior.sizes == {"chain": 1, "draw": 25, "channel": 3}
        assert "scaled_var" in prior
        assert "scaled_var_unscaled" in prior


class TestCensored:
    def test_censored_is_variable_factory(
        self,
    ) -> None:
        normal = Prior("Normal")
        censored_normal = Censored(normal, lower=0)

        assert isinstance(censored_normal, VariableFactory)

    def test_deserialize_censored(self) -> None:
        data = {
            "class": "Censored",
            "data": {
                "dist": {
                    "dist": "Normal",
                },
                "lower": 0,
                "upper": float("inf"),
            },
        }

        instance = deserialize(data)
        assert isinstance(instance, Censored)
        assert isinstance(instance.distribution, Prior)
        assert instance.lower == 0
        assert instance.upper == float("inf")

    @pytest.mark.parametrize(
        "dims, expected_dims",
        [
            ("channel", ("channel",)),
            (("channel", "geo"), ("channel", "geo")),
        ],
        ids=["string", "tuple"],
    )
    def test_censored_dims_from_distribution(self, dims, expected_dims) -> None:
        normal = Prior("Normal", dims=dims)
        censored_normal = Censored(normal, lower=0)

        assert censored_normal.dims == expected_dims

    def test_censored_variables_created(
        self,
    ) -> None:
        normal = Prior("Normal", mu=Prior("Normal"), dims="dim")
        censored_normal = Censored(normal, lower=0)

        coords = {"dim": range(3)}
        with pm.Model(coords=coords) as model:
            censored_normal.create_variable("var")

        var_names = ["var", "var_mu"]
        assert set(var.name for var in model.unobserved_RVs) == set(var_names)
        dims = [(3,), ()]
        for var_name, dim in zip(var_names, dims, strict=False):
            assert fast_eval(model[var_name]).shape == dim

    def test_censored_sample_prior(
        self,
    ) -> None:
        normal = Prior("Normal", dims="channel")
        censored_normal = Censored(normal, lower=0)

        coords = {"channel": ["A", "B", "C"]}
        prior = censored_normal.sample_prior(coords=coords, draws=25)

        assert isinstance(prior, xr.Dataset)
        assert prior.sizes == {"chain": 1, "draw": 25, "channel": 3}

    def test_censored_to_graph(
        self,
    ) -> None:
        normal = Prior("Normal", dims="channel")
        censored_normal = Censored(normal, lower=0)

        G = censored_normal.to_graph()
        assert isinstance(G, Digraph)

    def test_censored_likelihood_variable(
        self,
    ) -> None:
        normal = Prior("Normal", sigma=Prior("HalfNormal"), dims="channel")
        censored_normal = Censored(normal, lower=0)

        coords = {"channel": range(3)}
        with pm.Model(coords=coords) as model:
            mu = pm.Normal("mu")
            variable = censored_normal.create_likelihood_variable(
                name="likelihood",
                mu=mu,
                observed=[1, 2, 3],
            )

        assert isinstance(variable, pt.TensorVariable)
        assert model.observed_RVs == [variable]
        assert "likelihood_sigma" in model

    def test_censored_likelihood_unsupported_distribution(
        self,
    ) -> None:
        cauchy = Prior("Cauchy")
        censored_cauchy = Censored(cauchy, lower=0)

        with pm.Model():
            mu = pm.Normal("mu")
            with pytest.raises(UnsupportedDistributionError):
                censored_cauchy.create_likelihood_variable(
                    name="likelihood",
                    mu=mu,
                    observed=1,
                )

    def test_censored_likelihood_already_has_mu(
        self,
    ) -> None:
        normal = Prior("Normal", mu=Prior("Normal"), sigma=Prior("HalfNormal"))
        censored_normal = Censored(normal, lower=0)

        with pm.Model():
            mu = pm.Normal("mu")
            with pytest.raises(MuAlreadyExistsError):
                censored_normal.create_likelihood_variable(
                    name="likelihood",
                    mu=mu,
                    observed=1,
                )

    def test_censored_to_dict(
        self,
    ) -> None:
        normal = Prior("Normal", mu=0, sigma=1, dims="channel")
        censored_normal = Censored(normal, lower=0)

        data = censored_normal.to_dict()
        assert data == {
            "class": "Censored",
            "data": {"dist": normal.to_dict(), "lower": 0, "upper": float("inf")},
        }

    @pytest.mark.parametrize(
        "mu",
        [
            0,
            np.arange(10),
        ],
        ids=["scalar", "vector"],
    )
    def test_censored_logp(self, mu) -> None:
        n_points = 10
        observed = np.zeros(n_points)
        coords = {"idx": range(n_points)}
        with pm.Model(coords=coords) as model:
            normal = Prior("Normal", dims="idx")
            Censored(normal, lower=0).create_likelihood_variable(
                "censored_normal",
                observed=observed,
                mu=mu,
            )
        logp = model.compile_logp()

        with pm.Model() as expected_model:
            pm.Censored(
                "censored_normal",
                pm.Normal.dist(mu=mu, sigma=1, shape=n_points),
                lower=0,
                upper=np.inf,
                observed=observed,
            )
        expected_logp = expected_model.compile_logp()

        point = {}
        np.testing.assert_allclose(logp(point), expected_logp(point))

    def test_censored_with_tensor_variable(self) -> None:
        normal = Prior("Normal", dims="channel")
        lower = pt.as_tensor_variable([0, 1, 2])
        censored_normal = Censored(normal, lower=lower)

        assert censored_normal.to_dict() == {
            "class": "Censored",
            "data": {
                "dist": normal.to_dict(),
                "lower": [0, 1, 2],
                "upper": float("inf"),
            },
        }

    def test_censored_dims_setter(self) -> None:
        normal = Prior("Normal", dims="channel")
        censored_normal = Censored(normal, lower=0)
        censored_normal.dims = "date"
        assert normal.dims == ("date",)

    def test_censored_normal(self) -> None:
        coords = {"idx": range(5)}
        observed = np.arange(5, dtype=float)
        mu = np.pi

        with pm.Model(coords=coords) as model:
            sigma = Prior("HalfNormal")
            normal = Prior("Normal", sigma=sigma, dims="idx")
            Censored(normal, lower=0).create_likelihood_variable(
                "censored_normal",
                mu=mu,
                observed=observed,
            )

        with pm.Model(coords=coords) as expected_model:
            sigma = pm.HalfNormal("censored_normal_sigma")
            normal = pm.Normal.dist(mu=mu, sigma=sigma)
            pm.Censored(
                "censored_normal",
                normal,
                lower=0,
                upper=np.inf,
                observed=observed,
                dims="idx",
            )

        # This doesn't work because of no OpFromGraph equality impl
        # assert equivalent_models(model, expected_model)

        ip = model.initial_point()
        np.testing.assert_allclose(model.compile_logp()(ip), expected_model.compile_logp()(ip))

    def test_censored_with_alternative_class(self) -> None:
        def is_type(data):
            return isinstance(data, dict) and "distribution" in data

        def deserialize_func(data):
            return Prior(**data)

        register_deserialization(is_type=is_type, deserialize=deserialize_func)

        data = {
            "class": "Censored",
            "data": {
                "dist": {
                    "distribution": "Normal",
                },
                "lower": 0,
                "upper": 10,
            },
        }

        instance = deserialize(data)

        assert isinstance(instance, Censored)
        assert instance.lower == 0
        assert instance.upper == 10
        assert instance.distribution == Prior("Normal")

        DESERIALIZERS.pop()


@pytest.mark.filterwarnings(
    "ignore:The `pymc.dims` module is experimental and may contain critical bugs"
)
class TestXDist:
    def test_xdist_serialization(self):
        import pymc.dims as pmd

        mu = pmd.as_xtensor([1, 2, 3], dims=("city",))
        sigma = DataArray([4, 5], dims=("country",))
        dims = ("city", "batch", "country")

        prior = Prior(
            "Normal",
            mu=mu,
            sigma=sigma,
            dims=dims,
        )

        data = prior.to_dict()
        assert data == {
            "dims": ("city", "batch", "country"),
            "dist": "Normal",
            "kwargs": {
                "mu": {
                    "class": "DataArray",
                    "data": [1, 2, 3],
                    "dims": ["city"],
                },
                "sigma": {
                    "class": "DataArray",
                    "data": [4, 5],
                    "dims": ["country"],
                },
            },
        }

        prior_again = deserialize(data)
        # Commented out because Prior equality fails with PyTensor / Xarray variables in the parameters
        # assert prior_again == prior

        data_again = prior_again.to_dict()
        assert data_again == data

    @pytest.mark.parametrize("transform", (None, "exp"))
    def test_xdist_prior(self, transform):
        import pymc.dims as pmd

        mu = pmd.as_xtensor([1, 2, 3], dims=("city",))
        sigma = DataArray([4, 5], dims=("country",))
        dims = ("city", "batch", "country")
        coords = {
            "city": range(3),
            "country": range(2),
            "batch": range(5),
        }

        prior = Prior(
            "Normal",
            mu=mu,
            sigma=sigma,
            dims=dims,
            transform=transform,
        )

        res = prior.sample_prior(draws=7, coords=coords, xdist=True)
        assert res.sizes == {"chain": 1, "draw": 7, "city": 3, "batch": 5, "country": 2}

        with pm.Model(coords=coords) as prior_m:
            prior.create_variable("x", xdist=True)

        if transform is None:
            with pm.Model(coords=coords) as expected_prior_m:
                pmd.Normal("x", mu=mu, sigma=sigma, dims=dims)
        else:
            with pm.Model(coords=coords) as expected_prior_m:
                x_raw = pmd.Normal("x_raw", mu=mu, sigma=sigma, dims=dims)
                pmd.Deterministic("x", pmd.math.exp(x_raw))

        assert equivalent_models(prior_m, expected_prior_m)

    def test_xdist_likelihood(self):
        import pymc.dims as pmd

        mu = pmd.as_xtensor([1, 2, 3], dims=("city",))
        sigma = DataArray([4, 5], dims=("country",))
        dims = ("city", "batch", "country")
        coords = {
            "batch": range(5),
            "city": range(3),
            "country": range(2),
        }

        likelihood = Prior(
            "Normal",
            sigma=sigma,
            dims=dims,
        )
        observed = np.random.normal(size=(3, 5, 2))
        with pm.Model(coords=coords) as obs_m:
            x_obs = pmd.Data("x_obs", observed, dims=dims)
            likelihood.create_likelihood_variable("x", mu=mu, observed=x_obs.T, xdist=True)

        with pm.Model(coords=coords) as expected_obs_m:
            x_obs = pmd.Data("x_obs", observed, dims=dims)
            pmd.Normal("x", mu=mu, sigma=sigma, observed=x_obs.T, dims=dims)

        assert equivalent_models(obs_m, expected_obs_m)

    def test_dims(self) -> None:
        assert Prior("Normal").dims is None
        assert Prior("Normal", dims=()).dims == ()

        p = Prior("Normal", mu=Prior("Normal", dims=("city",)))
        assert p.dims is None

        coords = {"city": range(3)}
        with pm.Model(coords=coords) as m:
            with pytest.raises(UnsupportedShapeError):
                p.create_variable("x")

        with pm.Model(coords=coords) as m:
            # xdist can infer dims on its own
            p.create_variable("x", xdist=True)
            assert m.named_vars_to_dims["x"] == ("city",)

        p.dims = ("city",)
        with pm.Model(coords=coords) as m:
            p.create_variable("x")
            assert m.named_vars_to_dims["x"] == ("city",)

        # This is always invalid
        with pytest.raises(UnsupportedShapeError):
            p.dims = ()
