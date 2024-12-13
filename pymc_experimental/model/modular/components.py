from abc import ABC, abstractmethod

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from patsy import dmatrix
from pytensor.graph import Apply, Op

from pymc_experimental.model.modular.utilities import (
    PRIOR_DEFAULT_KWARGS,
    ColumnType,
    PoolingType,
    at_least_list,
    get_X_data,
    make_hierarchical_prior,
    select_data_columns,
)


class GLMModel(ABC):
    """Base class for GLM components. Subclasses should implement the build method to construct the component."""

    def __init__(self, name=None):
        self.model = None
        self.compiled = False
        self.name = name

    @abstractmethod
    def build(self, model=None):
        pass

    def __add__(self, other):
        return AdditiveGLMComponent(self, other)

    def __mul__(self, other):
        return MultiplicativeGLMComponent(self, other)

    def __str__(self):
        return self.name


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
        pooling_columns: ColumnType = None,
        pooling: PoolingType = "complete",
        hierarchical_params: dict | None = None,
        prior: str = "Normal",
        prior_params: dict | None = None,
    ):
        """
        Class to represent an intercept term in a GLM model.

        By intercept, it is meant any constant term in the model that is not a function of any input data. This can be
        a simple constant term, or a hierarchical prior that creates fixed effects across level of one or more
        categorical variables.

        Parameters
        ----------
        name: str, optional
            Name of the intercept term. If None, a default name is generated based on the index_data.
        pooling_columns: str or list of str, optional
            Columns of the independent data to use as labels for pooling. These columns will be treated as categorical.
            If None, no pooling is applied. If a list is provided, a "telescoping" hierarchy is constructed from left
            to right, with the mean of each subsequent level centered on the mean of the previous level.
        pooling: str, one of ["none", "complete", "partial"], default "complete"
            Type of pooling to use for the intercept term. If "none", no pooling is applied, and each group in the
            index_data is treated as independent. If "complete", complete pooling is applied, and all data are treated
            as coming from the same group. If "partial", a hierarchical prior is constructed that shares information
            across groups in the index_data.
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
        prior: str, optional
            Name of the PyMC distribution to use for the intercept term. Default is "Normal".
        prior_params: dict, optional
            Additional keyword arguments to pass to the PyMC distribution specified by the prior argument.

        """
        self.hierarchical_params = hierarchical_params if hierarchical_params is not None else {}
        self.pooling = pooling

        self.prior = prior
        self.prior_params = prior_params if prior_params is not None else {}
        self.pooling_columns = pooling_columns

        name = name or f"Intercept(pooling_cols={pooling_columns})"

        super().__init__(name=name)

    def build(self, model: pm.Model | None = None):
        model = pm.modelcontext(model)
        with model:
            if self.pooling == "complete":
                prior_params = PRIOR_DEFAULT_KWARGS[self.prior].copy()
                prior_params.update(self.prior_params)

                intercept = getattr(pm, self.prior)(f"{self.name}", **prior_params)
                return intercept

            intercept = make_hierarchical_prior(
                self.name,
                X=get_X_data(model),
                model=model,
                pooling_columns=self.pooling_columns,
                dims=None,
                pooling=self.pooling,
                prior=self.prior,
                prior_kwargs=self.prior_params,
                **self.hierarchical_params,
            )

        return intercept


class Regression(GLMModel):
    def __init__(
        self,
        name: str | None = None,
        *,
        feature_columns: ColumnType | None = None,
        prior: str = "Normal",
        pooling: PoolingType = "complete",
        pooling_columns: ColumnType | None = None,
        hierarchical_params: dict | None = None,
        prior_params: dict | None = None,
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
        feature_columns: str or list of str
            Columns of the independent data to use in the regression.
        prior: str, optional
            Name of the PyMC distribution to use for the intercept term. Default is "Normal".
        pooling: str, one of ["none", "complete", "partial"], default "complete"
            Type of pooling to use for the intercept term. If "none", no pooling is applied, and each group in the
            index_data is treated as independent. If "complete", complete pooling is applied, and all data are treated
            as coming from the same group. If "partial", a hierarchical prior is constructed that shares information
            across groups in the index_data.
        pooling_columns: str or list of str, optional
            Columns of the independent data to use as labels for pooling. These columns will be treated as categorical.
            If None, no pooling is applied. If a list is provided, a "telescoping" hierarchy is constructed from left
            to right, with the mean of each subsequent level centered on the mean of the previous level.
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
        prior_params:
            Additional keyword arguments to pass to the PyMC distribution specified by the prior argument.
        """
        self.feature_columns = feature_columns
        self.pooling = pooling
        self.pooling_columns = pooling_columns

        self.prior = prior
        self.prior_params = {} if prior_params is None else prior_params
        self.hierarchical_params = {} if hierarchical_params is None else hierarchical_params

        name = name if name else f"Regression({feature_columns})"

        super().__init__(name=name)

    def build(self, model=None):
        model = pm.modelcontext(model)
        feature_dim = f"{self.name}_features"

        if feature_dim not in model.coords:
            model.add_coord(feature_dim, at_least_list(self.feature_columns))

        with model:
            full_X = get_X_data(model)
            X = select_data_columns(self.feature_columns, model, squeeze=False)

            if self.pooling == "complete":
                prior_params = PRIOR_DEFAULT_KWARGS[self.prior].copy()
                prior_params.update(self.prior_params)

                beta = getattr(pm, self.prior)(f"{self.name}", **prior_params, dims=[feature_dim])
                return X @ beta

            beta = make_hierarchical_prior(
                name=self.name,
                X=full_X,
                pooling=self.pooling,
                pooling_columns=self.pooling_columns,
                model=model,
                dims=[feature_dim],
                **self.hierarchical_params,
            )

            regression_effect = (X * beta.T).sum(axis=-1)
            return regression_effect


class SplineTensor(Op):
    def __init__(self, name, df=10, degree=3):
        """
        Thin wrapper around patsy dmatrix, allowing for the creation of spline basis functions given a symbolic input.

        Parameters
        ----------
        name: str, optional
            Name of the spline basis function.
        df: int
            Number of basis functions to generate
        degree: int
            Degree of the spline basis
        """
        self.name = name if name else ""
        self.df = df
        self.degree = degree

    def make_node(self, x):
        inputs = [pt.as_tensor(x)]
        outputs = [pt.dmatrix(f"{self.name}_spline_basis")]

        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        [x] = inputs

        outputs[0][0] = np.asarray(
            dmatrix(f"bs({self.name}, df={self.df}, degree={self.degree}) - 1", data={self.name: x})
        )


def pt_spline(x, name=None, df=10, degree=3) -> pt.Variable:
    return SplineTensor(name=name, df=df, degree=degree)(x)


class Spline(GLMModel):
    def __init__(
        self,
        name: str,
        *,
        feature_column: str | None = None,
        n_knots: int = 10,
        spline_degree: int = 3,
        pooling: PoolingType = "complete",
        pooling_columns: ColumnType | None = None,
        prior: str = "Normal",
        prior_params: dict | None = None,
        hierarchical_params: dict | None = None,
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
        feature_column: str
            Column of the independent data to use in the spline.
        n_knots: int, default 10
            Number of knots to use in the spline basis.
        spline_degree: int, default 3
            Degree of the spline basis.
        pooling: str, one of ["none", "complete", "partial"], default "complete"
            Type of pooling to use for the intercept term. If "none", no pooling is applied, and each group in the
            index_data is treated as independent. If "complete", complete pooling is applied, and all data are treated
            as coming from the same group. If "partial", a hierarchical prior is constructed that shares information
            across groups in the index_data.
        pooling_columns: str or list of str, optional
            Columns of the independent data to use as labels for pooling. These columns will be treated as categorical.
            If None, no pooling is applied. If a list is provided, a "telescoping" hierarchy is constructed from left
            to right, with the mean of each subsequent level centered on the mean of the previous level.
        prior: str, optional
            Name of the PyMC distribution to use for the intercept term. Default is "Normal".
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
        self.feature_column = feature_column
        self.n_knots = n_knots
        self.spline_degree = spline_degree

        self.prior = prior
        self.prior_params = {} if prior_params is None else prior_params
        self.hierarchical_params = {} if hierarchical_params is None else hierarchical_params

        self.pooling = pooling
        self.pooling_columns = pooling_columns

        name = name if name else f"Spline({feature_column}, df={n_knots}, degree={spline_degree})"
        super().__init__(name=name)

    def build(self, model: pm.Model | None = None):
        model = pm.modelcontext(model)
        spline_dim = f"{self.name}_knots"
        model.add_coord(spline_dim, range(self.n_knots))

        with model:
            X_spline = pt_spline(
                select_data_columns(self.feature_column, model),
                name=self.feature_column,
                df=self.n_knots,
                degree=self.spline_degree,
            )

            if self.pooling == "complete":
                prior_params = PRIOR_DEFAULT_KWARGS[self.prior].copy()
                prior_params.update(self.prior_params)

                beta = getattr(pm, self.prior)(f"{self.name}", **prior_params, dims=[spline_dim])
                return X_spline @ beta

            elif self.pooling_columns is not None:
                beta = make_hierarchical_prior(
                    name=self.name,
                    X=get_X_data(model),
                    pooling=self.pooling,
                    pooling_columns=self.pooling_columns,
                    model=model,
                    dims=[spline_dim],
                    **self.hierarchical_params,
                )

            spline_effect = (X_spline * beta.T).sum(axis=-1)
            return spline_effect
