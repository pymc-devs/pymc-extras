from abc import ABC, abstractmethod
from collections.abc import Sequence
from io import StringIO
from typing import Literal, get_args

import arviz as az
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import rich

from pymc.backends.arviz import apply_function_over_dataset
from pymc.model.fgraph import clone_model
from pymc.pytensorf import reseed_rngs
from pytensor.tensor.random.type import RandomType

from pymc_experimental.model.marginal.marginal_model import MarginalModel
from pymc_experimental.model.modular.utilities import ColumnType, encode_categoricals
from pymc_experimental.printing import model_table

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

        X_df = data.drop(columns=[target_col])

        self.obs_dim = data.index.name if data.index.name is not None else "obs_idx"
        self.coords = {
            self.obs_dim: data.index.values,
        }

        X_df, self.coords = encode_categoricals(X_df, self.coords)

        numeric_cols = [
            col for col, dtype in X_df.dtypes.to_dict().items() if dtype.name.startswith("float")
        ]
        self.coords["feature"] = numeric_cols

        with self._get_model_class(self.coords) as self.model:
            pm.Data(f"{target_col}_observed", data[target_col], dims=self.obs_dim)
            pm.Data(
                "X_data",
                X_df,
                dims=(self.obs_dim, "feature"),
                shape=(None, len(self.coords["feature"])),
            )

        self._predict_fn = None  # We are caching function for faster predictions

    def sample(self, **sample_kwargs):
        with self.model:
            return pm.sample(**sample_kwargs)

    def sample_prior_predictive(self, **sample_kwargs):
        with self.model:
            return pm.sample_prior_predictive(**sample_kwargs)

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

    def register_mu(self, mu=None):
        with self.model:
            if mu is not None:
                return pm.Deterministic("mu", mu.build(self.model), dims=[self.obs_dim])
            return pm.Normal("mu", 0, 100)

    def register_sigma(self, sigma=None):
        with self.model:
            if sigma is not None:
                return pm.Deterministic(
                    "sigma", pt.exp(sigma.build(self.model)), dims=[self.obs_dim]
                )
            return pm.Exponential("sigma", lam=1)

    def __repr__(self):
        table = model_table(self.model)
        buffer = StringIO()
        rich.print(table, file=buffer)

        return buffer.getvalue()

    def to_graphviz(self):
        return self.model.to_graphviz()

    # def _repr_html_(self):
    #     return model_table(self.model)


class NormalLikelihood(Likelihood):
    """
    A model with normally distributed errors
    """

    def __init__(self, mu, sigma, target_col: ColumnType, data: pd.DataFrame):
        super().__init__(target_col=target_col, data=data)

        with self.model:
            mu = self.register_mu(mu)
            sigma = self.register_sigma(sigma)

            pm.Normal(
                target_col,
                mu=mu,
                sigma=sigma,
                observed=self.model[f"{target_col}_observed"],
                dims=[self.obs_dim],
            )

    def _get_model_class(self, coords: dict[str, Sequence]) -> pm.Model | MarginalModel:
        return pm.Model(coords=coords)
