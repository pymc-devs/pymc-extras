from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal, get_args

import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from model.modular.utilities import ColumnType, hierarchical_prior_to_requested_depth
from patsy import dmatrix

POOLING_TYPES = Literal["none", "complete", "partial"]
valid_pooling = get_args(POOLING_TYPES)


def _validate_pooling_params(pooling_columns: ColumnType, pooling: POOLING_TYPES):
    """
    Helper function to validate inputs to a GLM component.

    Parameters
    ----------
    index_data: Series or DataFrame
        Index data used to build hierarchical priors

    pooling: str
        Type of pooling to use in the component

    Returns
    -------
    None
    """
    if pooling_columns is not None and pooling == "complete":
        raise ValueError("Index data provided but complete pooling was requested")
    if pooling_columns is None and pooling != "complete":
        raise ValueError(
            "Index data must be provided for partial pooling (pooling = 'partial') or no pooling "
            "(pooling = 'none')"
        )


def _get_x_cols(
    cols: str | Sequence[str],
    model: pm.Model | None = None,
) -> pt.TensorVariable:
    model = pm.modelcontext(model)
    # Don't upcast a single column to a colum matrix
    if isinstance(cols, str):
        [cols_idx] = [i for i, col in enumerate(model.coords["feature"]) if col == cols]
    else:
        cols_idx = [i for i, col in enumerate(model.coords["feature"]) if col is cols]
    return model["X_data"][:, cols_idx]


class GLMModel(ABC):
    """Base class for GLM components. Subclasses should implement the build method to construct the component."""

    def __init__(self):
        self.model = None
        self.compiled = False

    @abstractmethod
    def build(self, model=None):
        pass

    def __add__(self, other):
        return AdditiveGLMComponent(self, other)

    def __mul__(self, other):
        return MultiplicativeGLMComponent(self, other)


class AdditiveGLMComponent(GLMModel):
    """Class to represent an additive combination of GLM components"""

    def __init__(self, left, right):
        self.left = left
        self.right = right
        super().__init__()

    def build(self, *args, **kwargs):
        return self.left.build(*args, **kwargs) + self.right.build(*args, **kwargs)


class MultiplicativeGLMComponent(GLMModel):
    """Class to represent a multiplicative combination of GLM components"""

    def __init__(self, left, right):
        self.left = left
        self.right = right
        super().__init__()

    def build(self, *args, **kwargs):
        return self.left.build(*args, **kwargs) * self.right.build(*args, **kwargs)


class Intercept(GLMModel):
    def __init__(
        self,
        name: str | None = None,
        *,
        pooling_cols: ColumnType = None,
        pooling: POOLING_TYPES = "complete",
        hierarchical_params: dict | None = None,
        prior: str = "Normal",
        prior_params: dict | None = None,
    ):
        """
        TODO: Update signature docs
        Class to represent an intercept term in a GLM model.

        By intercept, it is meant any constant term in the model that is not a function of any input data. This can be
        a simple constant term, or a hierarchical prior that creates fixed effects across level of one or more
        categorical variables.

        Parameters
        ----------
        name: str, optional
            Name of the intercept term. If None, a default name is generated based on the index_data.
        index_data: Series or DataFrame, optional
            Index data used to build hierarchical priors. If there are multiple columns, the columns are treated as
            levels of a "telescoping" hierarchy, with the leftmost column representing the top level of the hierarchy,
            and depth increasing to the right.

            The index of the index_data must match the index of the observed data.
        prior: str, optional
            Name of the PyMC distribution to use for the intercept term. Default is "Normal".
        pooling: str, one of ["none", "complete", "partial"], default "complete"
            Type of pooling to use for the intercept term. If "none", no pooling is applied, and each group in the
            index_data is treated as independent. If "complete", complete pooling is applied, and all data are treated
            as coming from the same group. If "partial", a hierarchical prior is constructed that shares information
            across groups in the index_data.
        prior_params: dict, optional
            Additional keyword arguments to pass to the PyMC distribution specified by the prior argument.
        hierarchical_params: dict, optional
            Additional keyword arguments to configure priors in the hierarchical_prior_to_requested_depth function.
            Options include:
                sigma_dist: str
                    Name of the distribution to use for the standard deviation of the hierarchy. Default is "Gamma"
                sigma_kwargs: dict
                    Additional keyword arguments to pass to the sigma distribution specified by the sigma_dist argument.
                    Default is {"alpha": 2, "beta": 1}
                offset_dist: str, one of ["zerosum", "normal", "laplace"]
                    Name of the distribution to use for the offset distribution. Default is "zerosum"
        """
        _validate_pooling_params(pooling_cols, pooling)

        self.pooling_cols = pooling_cols
        self.hierarchical_params = hierarchical_params if hierarchical_params is not None else {}
        self.pooling = pooling if pooling_cols is not None else "complete"

        self.prior = prior
        self.prior_params = prior_params if prior_params is not None else {}

        if pooling_cols is None:
            pooling_cols = []
        elif isinstance(pooling_cols, str):
            pooling_cols = [pooling_cols]

        data_name = ", ".join(pooling_cols)
        self.name = name or f"Constant(pooling_cols={data_name})"
        super().__init__()

    def build(self, model=None):
        model = pm.modelcontext(model)
        with model:
            if self.pooling == "complete":
                intercept = getattr(pm, self.prior)(f"{self.name}", **self.prior_params)
                return intercept

            [i for i, col in enumerate(model.coords["feature"]) if col in self.pooling_cols]

            intercept = hierarchical_prior_to_requested_depth(
                self.name,
                model.X_df[self.pooling_cols],  # TODO: Reconsider this
                model=model,
                dims=None,
                no_pooling=self.pooling == "none",
                **self.hierarchical_params,
            )
        return intercept


class Regression(GLMModel):
    def __init__(
        self,
        name: str,
        X: pd.DataFrame,
        prior: str = "Normal",
        index_data: pd.Series = None,
        pooling: POOLING_TYPES = "complete",
        **prior_params,
    ):
        """
        Class to represent a regression component in a GLM model.

        A regression component is a linear combination of input data and a set of parameters. The input data should be
        a DataFrame with the same index as the observed data. Parameteres can be made hierarchical by providing
        an index_data Series or DataFrame (which should have the same index as the observed data).

        Parameters
        ----------
        name: str, optional
            Name of the intercept term. If None, a default name is generated based on the index_data.
        X: DataFrame
            Exogenous data used to build the regression component. Each column of the DataFrame represents a feature
            used in the regression. Index of the DataFrame should match the index of the observed data.
        index_data: Series or DataFrame, optional
            Index data used to build hierarchical priors. If there are multiple columns, the columns are treated as
            levels of a "telescoping" hierarchy, with the leftmost column representing the top level of the hierarchy,
            and depth increasing to the right.

            The index of the index_data must match the index of the observed data.
        prior: str, optional
            Name of the PyMC distribution to use for the intercept term. Default is "Normal".
        pooling: str, one of ["none", "complete", "partial"], default "complete"
            Type of pooling to use for the intercept term. If "none", no pooling is applied, and each group in the
            index_data is treated as independent. If "complete", complete pooling is applied, and all data are treated
            as coming from the same group. If "partial", a hierarchical prior is constructed that shares information
            across groups in the index_data.
        curve_type: str, one of ["log", "abc", "ns", "nss", "box-cox"]
            Type of curve to build. For details, see the build_curve function.
        prior_params: dict, optional
            Additional keyword arguments to pass to the PyMC distribution specified by the prior argument.
        hierarchical_params: dict, optional
            Additional keyword arguments to configure priors in the hierarchical_prior_to_requested_depth function.
            Options include:
                sigma_dist: str
                    Name of the distribution to use for the standard deviation of the hierarchy. Default is "Gamma"
                sigma_kwargs: dict
                    Additional keyword arguments to pass to the sigma distribution specified by the sigma_dist argument.
                    Default is {"alpha": 2, "beta": 1}
                offset_dist: str, one of ["zerosum", "normal", "laplace"]
                    Name of the distribution to use for the offset distribution. Default is "zerosum"
        """
        _validate_pooling_params(index_data, pooling)

        self.name = name
        self.X = X
        self.index_data = index_data
        self.pooling = pooling

        self.prior = prior
        self.prior_params = prior_params

        super().__init__()

    def build(self, model=None):
        model = pm.modelcontext(model)
        feature_dim = f"{self.name}_features"
        obs_dim = self.X.index.name

        if feature_dim not in model.coords:
            model.add_coord(feature_dim, self.X.columns)

        with model:
            X_pt = pm.Data(f"{self.name}_data", self.X.values, dims=[obs_dim, feature_dim])
            if self.pooling == "complete":
                beta = getattr(pm, self.prior)(
                    f"{self.name}", **self.prior_params, dims=[feature_dim]
                )
                return X_pt @ beta

            beta = hierarchical_prior_to_requested_depth(
                self.name,
                self.index_data,
                model=model,
                dims=[feature_dim],
                no_pooling=self.pooling == "none",
            )

            regression_effect = (X_pt * beta.T).sum(axis=-1)
            return regression_effect


class Spline(Regression):
    def __init__(
        self,
        name: str,
        n_knots: int = 10,
        spline_data: pd.Series | pd.DataFrame | None = None,
        prior: str = "Normal",
        index_data: pd.Series | None = None,
        pooling: POOLING_TYPES = "complete",
        **prior_params,
    ):
        """
        Class to represent a spline component in a GLM model.

        A spline component is a linear combination of basis functions that are piecewise polynomial. The basis functions
        are constructed using the `bs` function from the patsy library. The spline_data should be a Series with the same
        index as the observed data.

        The weights of the spline components can be made hierarchical by providing an index_data Series or DataFrame
        (which should have the same index as the observed data).

        Parameters
        ----------
        name: str, optional
            Name of the intercept term. If None, a default name is generated based on the index_data.
        n_knots: int, default 10
            Number of knots to use in the spline basis.
        spline_data: Series or DataFrame
            Exogenous data to be interpolated using basis splines. If Series, must have a name attribute. If dataframe,
            must have exactly one column. In either case, the index of the data should match the index of the observed
            data.
        index_data: Series or DataFrame, optional
            Index data used to build hierarchical priors. If there are multiple columns, the columns are treated as
            levels of a "telescoping" hierarchy, with the leftmost column representing the top level of the hierarchy,
            and depth increasing to the right.

            The index of the index_data must match the index of the observed data.
        prior: str, optional
            Name of the PyMC distribution to use for the intercept term. Default is "Normal".
        pooling: str, one of ["none", "complete", "partial"], default "complete"
            Type of pooling to use for the intercept term. If "none", no pooling is applied, and each group in the
            index_data is treated as independent. If "complete", complete pooling is applied, and all data are treated
            as coming from the same group. If "partial", a hierarchical prior is constructed that shares information
            across groups in the index_data.
        curve_type: str, one of ["log", "abc", "ns", "nss", "box-cox"]
            Type of curve to build. For details, see the build_curve function.
        prior_params: dict, optional
            Additional keyword arguments to pass to the PyMC distribution specified by the prior argument.
        hierarchical_params: dict, optional
            Additional keyword arguments to configure priors in the hierarchical_prior_to_requested_depth function.
            Options include:
                sigma_dist: str
                    Name of the distribution to use for the standard deviation of the hierarchy. Default is "Gamma"
                sigma_kwargs: dict
                    Additional keyword arguments to pass to the sigma distribution specified by the sigma_dist argument.
                    Default is {"alpha": 2, "beta": 1}
                offset_dist: str, one of ["zerosum", "normal", "laplace"]
                    Name of the distribution to use for the offset distribution. Default is "zerosum"
        """
        _validate_pooling_params(index_data, pooling)

        spline_features = dmatrix(
            f"bs(maturity_years, df={n_knots}, degree=3) - 1",
            {"maturity_years": spline_data},
        )
        X = pd.DataFrame(
            spline_features,
            index=spline_data.index,
            columns=[f"Spline_{i}" for i in range(n_knots)],
        )

        super().__init__(
            name=name, X=X, prior=prior, index_data=index_data, pooling=pooling, **prior_params
        )
