#   Copyright 2023 The PyMC Developers
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

import hashlib
import json
import sys
import tempfile

import numpy as np
import pandas as pd
import pymc as pm
import pytest

from pymc_extras.model_builder import ModelBuilder


@pytest.fixture(scope="module")
def toy_X():
    x = np.linspace(start=0, stop=1, num=100)
    X = pd.DataFrame({"input": x})
    return X


@pytest.fixture(scope="module")
def toy_y(toy_X):
    y = 5 * toy_X["input"] + 3
    y = y + np.random.normal(0, 1, size=len(toy_X))
    y = pd.Series(y, name="output")
    return y


def get_unfitted_model_instance(X, y):
    """Creates an unfitted model instance to which idata can be copied in
    and then used as a fitted model instance. That way a fitted model
    can be used multiple times without having to run `fit` multiple times."""
    sampler_config = {
        "draws": 20,
        "tune": 10,
        "chains": 2,
        "target_accept": 0.95,
    }
    model_config = {
        "a": {"loc": 0, "scale": 10, "dims": ("numbers",)},
        "b": {"loc": 0, "scale": 10},
        "obs_error": 2,
    }
    model = test_ModelBuilder(
        model_config=model_config, sampler_config=sampler_config, test_parameter="test_paramter"
    )
    # Do the things that `model.fit` does except sample to create idata.
    model._generate_and_preprocess_model_data(X, y.values.flatten())
    model.build_model(X, y)
    return model


@pytest.fixture(scope="module")
def fitted_model_instance_base(toy_X, toy_y):
    """Because fitting takes a relatively long time, this is intended to
    be used only once and then have new instances created and fit data patched in
    for tests that use a fitted model instance. Tests should use
    `fitted_model_instance` instead of this."""
    model = get_unfitted_model_instance(toy_X, toy_y)
    model.fit(toy_X, toy_y)
    return model


@pytest.fixture
def fitted_model_instance(toy_X, toy_y, fitted_model_instance_base):
    """Get a fitted model instance. A new instance is created and fit data is
    patched in, so tests using this fixture can modify the model object without
    affecting other tests."""
    model = get_unfitted_model_instance(toy_X, toy_y)
    model.idata = fitted_model_instance_base.idata.copy()
    return model


class test_ModelBuilder(ModelBuilder):
    def __init__(self, model_config=None, sampler_config=None, test_parameter=None):
        self.test_parameter = test_parameter
        super().__init__(model_config=model_config, sampler_config=sampler_config)

    _model_type = "test_model"
    version = "0.1"

    def build_model(self, X: pd.DataFrame, y: pd.Series, model_config=None):
        coords = {"numbers": np.arange(len(X))}
        self.generate_and_preprocess_model_data(X, y)
        with pm.Model(coords=coords) as self.model:
            if model_config is None:
                model_config = self.model_config
            x = pm.Data("x", self.X["input"].values)
            y_data = pm.Data("y_data", self.y)

            # prior parameters
            a_loc = model_config["a"]["loc"]
            a_scale = model_config["a"]["scale"]
            b_loc = model_config["b"]["loc"]
            b_scale = model_config["b"]["scale"]
            obs_error = model_config["obs_error"]

            # priors
            a = pm.Normal("a", a_loc, sigma=a_scale, dims=model_config["a"]["dims"])
            b = pm.Normal("b", b_loc, sigma=b_scale)
            obs_error = pm.HalfNormal("σ_model_fmc", obs_error)

            # observed data
            output = pm.Normal("output", a + b * x, obs_error, shape=x.shape, observed=y_data)

    def _save_input_params(self, idata):
        idata.attrs["test_paramter"] = json.dumps(self.test_parameter)

    @property
    def output_var(self):
        return "output"

    def _data_setter(self, x: pd.Series, y: pd.Series = None):
        with self.model:
            pm.set_data({"x": x.values})
            if y is not None:
                pm.set_data({"y_data": y.values})

    @property
    def _serializable_model_config(self):
        return self.model_config

    def generate_and_preprocess_model_data(self, X: pd.DataFrame, y: pd.Series):
        self.X = X
        self.y = y

    @staticmethod
    def get_default_model_config() -> dict:
        return {
            "a": {"loc": 0, "scale": 10, "dims": ("numbers",)},
            "b": {"loc": 0, "scale": 10},
            "obs_error": 2,
        }

    def _generate_and_preprocess_model_data(
        self, X: pd.DataFrame | pd.Series, y: pd.Series
    ) -> None:
        self.X = X
        self.y = y

    @staticmethod
    def get_default_sampler_config() -> dict:
        return {
            "draws": 10,
            "tune": 10,
            "chains": 3,
            "target_accept": 0.95,
        }


def test_save_input_params(fitted_model_instance):
    assert fitted_model_instance.idata.attrs["test_paramter"] == '"test_paramter"'


@pytest.mark.skipif(
    sys.platform == "win32", reason="Permissions for temp files not granted on windows CI."
)
def test_save_load(fitted_model_instance):
    temp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
    fitted_model_instance.save(temp.name)
    test_builder2 = test_ModelBuilder.load(temp.name)
    assert fitted_model_instance.idata.groups() == test_builder2.idata.groups()
    assert fitted_model_instance.id == test_builder2.id
    x_pred = np.random.uniform(low=0, high=1, size=100)
    prediction_data = pd.DataFrame({"input": x_pred})
    pred1 = fitted_model_instance.predict(prediction_data["input"])
    pred2 = test_builder2.predict(prediction_data["input"])
    assert pred1.shape == pred2.shape
    temp.close()


def test_initial_build_and_fit(fitted_model_instance, check_idata=True) -> ModelBuilder:
    if check_idata:
        assert fitted_model_instance.idata is not None
        assert "posterior" in fitted_model_instance.idata.groups()


def test_save_without_fit_raises_runtime_error():
    model_builder = test_ModelBuilder()
    with pytest.raises(RuntimeError):
        model_builder.save("saved_model")


def test_empty_sampler_config_fit(toy_X, toy_y):
    sampler_config = {}
    model_builder = test_ModelBuilder(sampler_config=sampler_config)
    model_builder.idata = model_builder.fit(X=toy_X, y=toy_y)
    assert model_builder.idata is not None
    assert "posterior" in model_builder.idata.groups()


def test_fit(fitted_model_instance):
    prediction_data = pd.DataFrame({"input": np.random.uniform(low=0, high=1, size=100)})
    pred = fitted_model_instance.predict(prediction_data["input"])
    post_pred = fitted_model_instance.sample_posterior_predictive(
        prediction_data["input"], extend_idata=True, combined=True
    )
    post_pred[fitted_model_instance.output_var].shape[0] == prediction_data.input.shape


def test_fit_no_y(toy_X):
    model_builder = test_ModelBuilder()
    model_builder.idata = model_builder.fit(X=toy_X, chains=1, tune=1, draws=1)
    assert model_builder.model is not None
    assert model_builder.idata is not None
    assert "posterior" in model_builder.idata.groups()


def test_predict(fitted_model_instance):
    x_pred = np.random.uniform(low=0, high=1, size=100)
    prediction_data = pd.DataFrame({"input": x_pred})
    pred = fitted_model_instance.predict(prediction_data["input"])
    # Perform elementwise comparison using numpy
    assert isinstance(pred, np.ndarray)
    assert len(pred) > 0


@pytest.mark.parametrize("combined", [True, False])
def test_sample_posterior_predictive(fitted_model_instance, combined):
    n_pred = 100
    x_pred = np.random.uniform(low=0, high=1, size=n_pred)
    prediction_data = pd.DataFrame({"input": x_pred})
    pred = fitted_model_instance.sample_posterior_predictive(
        prediction_data["input"], combined=combined, extend_idata=True
    )
    chains = fitted_model_instance.idata["sample_stats"].sizes["chain"]
    draws = fitted_model_instance.idata["sample_stats"].sizes["draw"]
    expected_shape = (n_pred, chains * draws) if combined else (chains, draws, n_pred)
    assert pred[fitted_model_instance.output_var].shape == expected_shape
    assert np.issubdtype(pred[fitted_model_instance.output_var].dtype, np.floating)


@pytest.mark.parametrize("group", ["prior_predictive", "posterior_predictive"])
@pytest.mark.parametrize("extend_idata", [True, False])
def test_sample_xxx_extend_idata_param(fitted_model_instance, group, extend_idata):
    output_var = fitted_model_instance.output_var
    idata_prev = fitted_model_instance.idata[group][output_var]

    # Since coordinates are provided, the dimension must match
    n_pred = 100  # Must match toy_x
    x_pred = np.random.uniform(0, 1, n_pred)

    prediction_data = pd.DataFrame({"input": x_pred})
    if group == "prior_predictive":
        prediction_method = fitted_model_instance.sample_prior_predictive
    else:  # group == "posterior_predictive":
        prediction_method = fitted_model_instance.sample_posterior_predictive

    pred = prediction_method(prediction_data["input"], combined=False, extend_idata=extend_idata)

    pred_unstacked = pred[output_var].values
    idata_now = fitted_model_instance.idata[group][output_var].values

    if extend_idata:
        # After sampling, data in the model should be the same as the predictions
        np.testing.assert_array_equal(idata_now, pred_unstacked)
        # Data in the model should NOT be the same as before
        if idata_now.shape == idata_prev.values.shape:
            assert np.sum(np.abs(idata_now - idata_prev.values) < 1e-5) <= 2
    else:
        # After sampling, data in the model should be the same as it was before
        np.testing.assert_array_equal(idata_now, idata_prev.values)
        # Data in the model should NOT be the same as the predictions
        if idata_now.shape == pred_unstacked.shape:
            assert np.sum(np.abs(idata_now - pred_unstacked) < 1e-5) <= 2


def test_model_config_formatting():
    model_config = {
        "a": {
            "loc": [0, 0],
            "scale": 10,
            "dims": [
                "x",
            ],
        },
    }
    model_builder = test_ModelBuilder()
    converted_model_config = model_builder._model_config_formatting(model_config)
    np.testing.assert_equal(converted_model_config["a"]["dims"], ("x",))
    np.testing.assert_equal(converted_model_config["a"]["loc"], np.array([0, 0]))


def test_id():
    model_builder = test_ModelBuilder()
    expected_id = hashlib.sha256(
        str(model_builder.model_config.values()).encode()
        + model_builder.version.encode()
        + model_builder._model_type.encode()
    ).hexdigest()[:16]

    assert model_builder.id == expected_id
