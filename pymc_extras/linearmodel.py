import numpy as np
import pandas as pd
import pymc as pm

from sklearn.base import BaseEstimator

from pymc_extras.model_builder import ModelBuilder


class LinearModel(ModelBuilder, BaseEstimator):
    def __init__(
        self, model_config: dict | None = None, sampler_config: dict | None = None, nsamples=100
    ):
        self.nsamples = nsamples
        super().__init__(model_config, sampler_config)

    """
    This class is an implementation of a single-input linear regression model in PYMC using the
    BayesianEstimator base class for interoperability with scikit-learn.
    """

    _model_type = "LinearModel"
    version = "0.1"

    @staticmethod
    def get_default_model_config():
        return {
            "intercept": {"loc": 0, "scale": 10},
            "slope": {"loc": 0, "scale": 10},
            "obs_error": 2,
        }

    @staticmethod
    def get_default_sampler_config():
        return {
            "draws": 1_000,
            "tune": 1_000,
            "chains": 3,
            "target_accept": 0.95,
        }

    @property
    def _serializable_model_config(self) -> dict:
        return self.model_config

    @property
    def output_var(self):
        return "y_hat"

    def build_model(self, X: pd.DataFrame, y: pd.Series):
        """
        Build the PyMC model.

        Returns
        -------
        None

        Examples
        --------
        >>> self.build_model()
        >>> assert self.model is not None
        >>> assert isinstance(self.model, pm.Model)
        >>> assert "intercept" in self.model.named_vars
        >>> assert "slope" in self.model.named_vars
        >>> assert "σ_model_fmc" in self.model.named_vars
        >>> assert "y_model" in self.model.named_vars
        >>> assert "y_hat" in self.model.named_vars
        >>> assert self.output_var == "y_hat"
        """
        cfg = self.model_config

        # Data array size can change but number of dimensions must stay the same.
        with pm.Model() as self.model:
            x = pm.Data("x", np.zeros((1,)), dims="observation")
            y_data = pm.Data("y_data", np.zeros((1,)), dims="observation")

            # priors
            intercept = pm.Normal(
                "intercept", cfg["intercept"]["loc"], sigma=cfg["intercept"]["scale"]
            )
            slope = pm.Normal("slope", cfg["slope"]["loc"], sigma=cfg["slope"]["scale"])
            obs_error = pm.HalfNormal("σ_model_fmc", cfg["obs_error"])

            # Model
            y_model = pm.Deterministic("y_model", intercept + slope * x, dims="observation")

            # observed data
            pm.Normal(
                "y_hat",
                y_model,
                sigma=obs_error,
                shape=x.shape,
                observed=y_data,
                dims="observation",
            )

        self._data_setter(X, y)

    def _data_setter(self, X: pd.DataFrame, y: pd.DataFrame | pd.Series | None = None):
        with self.model:
            pm.set_data({"x": X.squeeze()})
            if y is not None:
                pm.set_data({"y_data": y.squeeze()})

    def _generate_and_preprocess_model_data(
        self, X: pd.DataFrame | pd.Series, y: pd.Series
    ) -> None:
        """
        Generate model data for linear regression.

        Parameters
        ----------
        nsamples : int, optional
            The number of samples to generate. Default is 100.
        data : np.ndarray, optional
            An optional data array to add noise to.

        Returns
        -------
        tuple
            A tuple of two np.ndarrays representing the feature matrix and target vector, respectively.

        Examples
        --------
        >>> import numpy as np
        >>> x, y = cls.generate_model_data()
        >>> assert isinstance(x, np.ndarray)
        >>> assert isinstance(y, np.ndarray)
        >>> assert x.shape == (100, 1)
        >>> assert y.shape == (100,)
        """
        self.X, self.y = X, y
