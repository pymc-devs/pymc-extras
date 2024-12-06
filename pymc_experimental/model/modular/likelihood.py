from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal, get_args

import arviz as az
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from pymc.backends.arviz import apply_function_over_dataset
from pymc.model.fgraph import clone_model
from pymc.pytensorf import reseed_rngs
from pytensor.tensor.random.type import RandomType

from pymc_experimental.model.marginal.marginal_model import MarginalModel
from pymc_experimental.model.modular.utilities import ColumnType

LIKELIHOOD_TYPES = Literal["lognormal", "logt", "mixture", "unmarginalized-mixture"]
valid_likelihoods = get_args(LIKELIHOOD_TYPES)


class Likelihood(ABC):
    """Class to represent a likelihood function for a GLM component. Subclasses should implement the _get_model_class
    method to return the type of model used by the likelihood function, and should implement a `register_xx` method for
    each parameter unique to that likelihood function."""

    def __init__(self, target_col: ColumnType, data: pd.DataFrame):
        """
        Initialization logic common to all likelihoods.

        All subclasses should call super().__init__(y) to register data and create a model object. The subclass __init__
        method should then create a PyMC model inside the self.model context.

        Parameters
        ----------
        y: Series or DataFrame, optional
            Observed data. Must have a name attribute (if a Series), and an index with a name attribute.
        """

        if not isinstance(target_col, str):
            [target_col] = target_col
        self.target_col = target_col

        # TODO: Reconsider this (two sources of nearly the same info not good)
        X_df = data.drop(columns=[target_col])
        X_data = X_df.copy()
        self.column_labels = {}
        for col, dtype in X_data.dtypes.to_dict().items():
            if dtype.name.startswith("float"):
                pass
            elif dtype.name == "object":
                # TODO: We definitely need to save these if we want to factorize predict data
                col_array, labels = pd.factorize(X_data[col], sort=True)
                X_data[col] = col_array.astype("float64")
                self.column_labels[col] = {label: i for i, label in enumerate(labels.values)}
            elif dtype.name.startswith("int"):
                X_data[col] = X_data[col].astype("float64")
            else:
                raise NotImplementedError(
                    f"Haven't decided how to handle the following type: {dtype.name}"
                )

        self.obs_dim = data.index.name
        coords = {
            self.obs_dim: data.index.values,
            "feature": list(X_data.columns),
        }
        with self._get_model_class(coords) as self.model:
            self.model.X_df = X_df  # FIXME: Definitely not a solution
            pm.Data(f"{target_col}_observed", data[target_col], dims=self.obs_dim)
            pm.Data(
                "X_data",
                X_data,
                dims=(self.obs_dim, "feature"),
                shape=(None, len(coords["feature"])),
            )

        self._predict_fn = None  # We are caching function for faster predictions

    def sample(self, **sample_kwargs):
        with self.model:
            return pm.sample(**sample_kwargs)

    def predict(
        self,
        idata: az.InferenceData,
        predict_df: pd.DataFrame,
        random_seed=None,
        compile_kwargs=None,
    ):
        # Makes sure only features present during fitting are used and sorted during prediction
        X_data = predict_df[list(self.model.coords["feature"])].copy()
        for col, dtype in X_data.dtypes.to_dict().items():
            if dtype.name.startswith("float"):
                pass
            elif dtype.name == "object":
                X_data[col] = X_data[col].map(self.column_labels[col]).astype("float64")
            elif dtype.name.startswith("int"):
                X_data[col] = X_data[col].astype("float64")
            else:
                raise NotImplementedError(
                    f"Haven't decided how to handle the following type: {dtype.name}"
                )

        coords = {self.obs_dim: X_data.index.values}

        predict_fn = self._predict_fn

        if predict_fn is None:
            model_copy = clone_model(self.model)
            # TODO: Freeze everything that is not supposed to change, when PyMC allows it
            # dims = [dim for dim self.model.coords.keys() if dim != self.obs_dim]
            # model_copy = freeze_dims_and_data(model_copy, dims=dims, data=[])
            observed_RVs = model_copy.observed_RVs
            if compile_kwargs is None:
                compile_kwargs = {}
            predict_fn = model_copy.compile_fn(
                observed_RVs,
                inputs=model_copy.free_RVs,
                mode=compile_kwargs.pop("mode", None),
                on_unused_input="ignore",
                **compile_kwargs,
            )
            predict_fn.trust_input = True
            self._predict_fn = predict_fn

        [X_var] = [shared for shared in predict_fn.f.get_shared() if shared.name == "X_data"]
        if random_seed is not None:
            rngs = [
                shared
                for shared in predict_fn.f.get_shared()
                if isinstance(shared.type, RandomType)
            ]
            reseed_rngs(rngs, random_seed)
        X_var.set_value(X_data.values, borrow=True)

        return apply_function_over_dataset(
            fn=predict_fn,
            dataset=idata.posterior[[rv.name for rv in self.model.free_RVs]],
            output_var_names=[rv.name for rv in self.model.observed_RVs],
            dims={rv.name: [self.obs_dim] for rv in self.model.observed_RVs},
            coords=coords,
            progressbar=False,
        )

    @abstractmethod
    def _get_model_class(self, coords: dict[str, Sequence]) -> pm.Model | MarginalModel:
        """Return the type on model used by the likelihood function"""
        raise NotImplementedError

    def register_mu(
        self,
        *,
        df: pd.DataFrame,
        mu=None,
    ):
        with self.model:
            if mu is not None:
                return pm.Deterministic("mu", mu.build(df=df), dims=[self.obs_dim])
            return pm.Normal("mu", 0, 100)

    def register_sigma(
        self,
        *,
        df: pd.DataFrame,
        sigma=None,
    ):
        with self.model:
            if sigma is not None:
                return pm.Deterministic("sigma", pt.exp(sigma.build(df=df)), dims=[self.obs_dim])
            return pm.Exponential("sigma", lam=1)


class LogNormalLikelihood(Likelihood):
    """Class to represent a log-normal likelihood function for a GLM component."""

    def __init__(
        self,
        mu,
        sigma,
        target_col: ColumnType,
        data: pd.DataFrame,
    ):
        super().__init__(target_col=target_col, data=data)

        with self.model:
            self.register_data(data[target_col])
            mu = self.register_mu(mu)
            sigma = self.register_sigma(sigma)

            pm.LogNormal(
                target_col,
                mu=mu,
                sigma=sigma,
                observed=self.model[f"{target_col}_observed"],
                dims=[self.obs_dim],
            )

    def _get_model_class(self, coords: dict[str, Sequence]) -> pm.Model | MarginalModel:
        return pm.Model(coords=coords)


class LogTLikelihood(Likelihood):
    """
    Class to represent a log-t likelihood function for a GLM component.
    """

    def __init__(
        self,
        mu,
        *,
        sigma=None,
        nu=None,
        target_col: ColumnType,
        data: pd.DataFrame,
    ):
        def log_student_t(nu, mu, sigma, shape=None):
            return pm.math.exp(pm.StudentT.dist(mu=mu, sigma=sigma, nu=nu, shape=shape))

        super().__init__(target_col=target_col, data=data)

        with self.model:
            mu = self.register_mu(mu=mu, df=data)
            sigma = self.register_sigma(sigma=sigma, df=data)
            nu = self.register_nu(nu=nu, df=data)

            pm.CustomDist(
                target_col,
                nu,
                mu,
                sigma,
                observed=self.model[f"{target_col}_observed"],
                shape=mu.shape,
                dims=[self.obs_dim],
                dist=log_student_t,
                class_name="LogStudentT",
            )

    def register_nu(self, *, df, nu=None):
        with self.model:
            if nu is not None:
                return pm.Deterministic("nu", pt.exp(nu.build(df=df)), dims=[self.obs_dim])
            return pm.Uniform("nu", 2, 30)

    def _get_model_class(self, coords: dict[str, Sequence]) -> pm.Model | MarginalModel:
        return pm.Model(coords=coords)


class BaseMixtureLikelihood(Likelihood):
    """
    Base class for mixture likelihood functions to hold common methods for registering parameters.
    """

    def register_sigma(self, *, df, sigma=None):
        with self.model:
            if sigma is None:
                sigma_not_outlier = pm.Exponential("sigma_not_outlier", lam=1)
            else:
                sigma_not_outlier = pm.Deterministic(
                    "sigma_not_outlier", pt.exp(sigma.build(df=df)), dims=[self.obs_dim]
                )
            sigma_outlier_offset = pm.Gamma("sigma_outlier_offset", mu=0.2, sigma=0.5)
            sigma = pm.Deterministic(
                "sigma",
                pt.as_tensor([sigma_not_outlier, sigma_not_outlier * (1 + sigma_outlier_offset)]),
                dims=["outlier"],
            )

            return sigma

    def register_p_outlier(self, *, df, p_outlier=None, **param_kwargs):
        mean_p = param_kwargs.get("mean_p", 0.1)
        concentration = param_kwargs.get("concentration", 50)

        with self.model:
            if p_outlier is not None:
                return pm.Deterministic(
                    "p_outlier", pt.sigmoid(p_outlier.build(df=df)), dims=[self.obs_dim]
                )
            return pm.Beta("p_outlier", mean_p * concentration, (1 - mean_p) * concentration)

    def _get_model_class(self, coords: dict[str, Sequence]) -> pm.Model | MarginalModel:
        coords["outlier"] = [False, True]
        return MarginalModel(coords=coords)


class MixtureLikelihood(BaseMixtureLikelihood):
    """
    Class to represent a mixture likelihood function for a GLM component. The mixture is implemented using pm.Mixture,
    and does not allow for automatic marginalization of components.
    """

    def __init__(
        self,
        mu,
        sigma,
        p_outlier,
        target_col: ColumnType,
        data: pd.DataFrame,
    ):
        super().__init__(target_col=target_col, data=data)

        with self.model:
            mu = self.register_mu(mu)
            sigma = self.register_sigma(sigma)
            p_outlier = self.register_p_outlier(p_outlier)

            pm.Mixture(
                target_col,
                w=[1 - p_outlier, p_outlier],
                comp_dists=pm.LogNormal.dist(mu[..., None], sigma=sigma.T),
                shape=mu.shape,
                observed=self.model[f"{target_col}_observed"],
                dims=[self.obs_dim],
            )


class UnmarginalizedMixtureLikelihood(BaseMixtureLikelihood):
    """
    Class to represent an unmarginalized mixture likelihood function for a GLM component. The mixture is implemented using
    a MarginalModel, and allows for automatic marginalization of components.
    """

    def __init__(
        self,
        mu,
        sigma,
        p_outlier,
        target_col: ColumnType,
        data: pd.DataFrame,
    ):
        super().__init__(target_col=target_col, data=data)

        with self.model:
            mu = self.register_mu(mu)
            sigma = self.register_sigma(sigma)
            p_outlier = self.register_p_outlier(p_outlier)

            is_outlier = pm.Bernoulli(
                "is_outlier",
                p_outlier,
                dims=["cusip"],
                # shape=X_pt.shape[0],  # Uncomment after https://github.com/pymc-devs/pymc-experimental/pull/304
            )

            pm.LogNormal(
                target_col,
                mu=mu,
                sigma=pm.math.switch(is_outlier, sigma[1], sigma[0]),
                observed=self.model[f"{target_col}_observed"],
                shape=mu.shape,
                dims=[data.index.name],
            )

        self.model.marginalize(["is_outlier"])

    def _get_model_class(self, coords: dict[str, Sequence]) -> pm.Model | MarginalModel:
        coords["outlier"] = [False, True]
        return MarginalModel(coords=coords)
