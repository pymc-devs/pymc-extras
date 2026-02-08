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
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pymc_extras.linearmodel import LinearModel

try:
    from sklearn import set_config
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    set_config(transform_output="pandas")
    sklearn_available = True
except ImportError:
    sklearn_available = False


@pytest.fixture(scope="module")
def toy_actual_params():
    return {
        "intercept": 3,
        "slope": 5,
        "obs_error": 0.5,
    }


@pytest.fixture(scope="module")
def toy_X():
    x = np.linspace(start=0, stop=1, num=100)
    X = pd.DataFrame({"input": x})
    return X


@pytest.fixture(scope="module")
def toy_y(toy_X, toy_actual_params):
    y = toy_actual_params["slope"] * toy_X["input"] + toy_actual_params["intercept"]
    rng = np.random.default_rng(427)
    y = y + rng.normal(0, toy_actual_params["obs_error"], size=len(toy_X))
    y = pd.Series(y, name="output")
    return y


@pytest.fixture(scope="module")
def fitted_linear_model_instance(toy_X, toy_y):
    sampler_config = {
        "draws": 50,
        "tune": 30,
        "chains": 2,
        "target_accept": 0.95,
    }
    model = LinearModel(sampler_config=sampler_config)
    model.fit(toy_X, toy_y, random_seed=312)
    return model


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Permissions for temp files not granted on windows CI.",
)
def test_save_load(fitted_linear_model_instance):
    model = fitted_linear_model_instance
    temp = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
    model.save(temp.name)
    model2 = LinearModel.load(temp.name)
    assert model.idata.groups() == model2.idata.groups()

    X_pred = pd.DataFrame({"input": np.random.uniform(low=0, high=1, size=100)})
    pred1 = model.predict(X_pred, random_seed=423)
    pred2 = model2.predict(X_pred, random_seed=423)
    # Predictions should be identical
    np.testing.assert_array_equal(pred1, pred2)
    temp.close()


def test_save_without_fit_raises_runtime_error(toy_X, toy_y):
    test_model = LinearModel()
    with pytest.raises(RuntimeError):
        test_model.save("saved_model")


def test_fit(fitted_linear_model_instance):
    model = fitted_linear_model_instance

    new_X_pred = pd.DataFrame({"input": np.random.uniform(low=0, high=1, size=100)})

    pred = model.predict(new_X_pred)
    assert len(new_X_pred) == len(pred)
    assert isinstance(pred, np.ndarray)
    post_pred = model.predict_posterior(new_X_pred)
    assert len(new_X_pred) == len(post_pred)
    assert isinstance(post_pred, xr.DataArray)


def test_parameter_fit(toy_X, toy_y, toy_actual_params):
    """Check that the fit model recovered the data-generating parameters."""
    # Fit the model with a sufficient number of samples
    sampler_config = {
        "draws": 500,
        "tune": 300,
        "chains": 2,
        "target_accept": 0.95,
    }
    model = LinearModel(sampler_config=sampler_config)
    model.fit(toy_X, toy_y, random_seed=312)
    fit_params = model.idata["posterior"].mean()
    np.testing.assert_allclose(fit_params["intercept"], toy_actual_params["intercept"], rtol=0.1)
    np.testing.assert_allclose(fit_params["slope"], toy_actual_params["slope"], rtol=0.1)
    np.testing.assert_allclose(fit_params["σ_model_fmc"], toy_actual_params["obs_error"], rtol=0.1)


def test_predict(fitted_linear_model_instance):
    model = fitted_linear_model_instance
    X_pred = pd.DataFrame({"input": np.random.uniform(low=0, high=1, size=100)})
    pred = model.predict(X_pred)
    assert len(X_pred) == len(pred)
    assert np.issubdtype(pred.dtype, np.floating)


@pytest.mark.parametrize("combined", [True, False])
def test_predict_posterior(fitted_linear_model_instance, combined):
    model = fitted_linear_model_instance
    n_pred = 150
    X_pred = pd.DataFrame({"input": np.random.uniform(low=0, high=1, size=n_pred)})
    pred = model.predict_posterior(X_pred, combined=combined)
    chains = model.idata["sample_stats"].sizes["chain"]
    draws = model.idata["sample_stats"].sizes["draw"]
    expected_shape = (n_pred, chains * draws) if combined else (chains, draws, n_pred)
    assert pred.shape == expected_shape
    assert np.issubdtype(pred.dtype, np.floating)
    # TODO: check that extend_idata has the expected effect


@pytest.mark.parametrize("samples", [None, 300])
@pytest.mark.parametrize("combined", [True, False])
def test_sample_prior_predictive(samples, combined, toy_X, toy_y):
    model = LinearModel()
    prior_pred = model.sample_prior_predictive(toy_X, toy_y, samples, combined=combined)[
        model.output_var
    ]
    draws = model.sampler_config["draws"] if samples is None else samples
    chains = 1
    expected_shape = (len(toy_X), chains * draws) if combined else (chains, draws, len(toy_X))
    assert prior_pred.shape == expected_shape
    # TODO: check that extend_idata has the expected effect


def test_id():
    model_config = {
        "intercept": {"loc": 0, "scale": 10},
        "slope": {"loc": 0, "scale": 10},
        "obs_error": 2,
    }
    sampler_config = {
        "draws": 1_000,
        "tune": 1_000,
        "chains": 3,
        "target_accept": 0.95,
    }
    model = LinearModel(model_config=model_config, sampler_config=sampler_config)

    expected_id = hashlib.sha256(
        str(model_config.values()).encode() + model.version.encode() + model._model_type.encode()
    ).hexdigest()[:16]

    assert model.id == expected_id


@pytest.mark.skipif(not sklearn_available, reason="scikit-learn package is not available.")
def test_pipeline_integration(toy_X, toy_y):
    model_config = {
        "intercept": {"loc": 0, "scale": 2},
        "slope": {"loc": 0, "scale": 2},
        "obs_error": 1,
        "default_output_var": "y_hat",
    }
    model = Pipeline(
        [
            ("input_scaling", StandardScaler()),
            (
                "linear_model",
                TransformedTargetRegressor(LinearModel(model_config), transformer=StandardScaler()),
            ),
        ]
    )
    model.fit(toy_X, toy_y)

    X_pred = pd.DataFrame({"input": np.random.uniform(low=0, high=1, size=100)})
    model.predict(X_pred)
