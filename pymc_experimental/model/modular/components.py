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

CURVE_TYPES = Literal["log", "abc", "ns", "nss", "box-cox"]
valid_curves = get_args(CURVE_TYPES)


FEATURE_DICT = {
    "log": ["slope"],
    "box-cox": ["lambda", "slope", "intercept"],
    "nss": ["tau", "beta0", "beta1", "beta2"],
    "abc": ["a", "b", "c"],
}


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


def build_curve(
    time_pt: pt.TensorVariable,
    beta: pt.TensorVariable,
    curve_type: Literal["log", "abc", "ns", "nss", "box-cox"],
):
    """
    Build a curve based on the time data and parameters beta.

    In this context, a "curve" is a deterministic function that maps time to a value. The curve should (in general) be
    strictly increasing with time (df(t)/dt > 0), and should (in general) exhibit diminishing marginal growth with time
    (d^2f(t)/dt^2 < 0). These properties are not strictly necessary; some curve functions (such as nss) allow for
    local reversals.

    Parameters
    ----------
    time_pt: TensorVariable
        A pytensor variable representing the time data to build the curve from.
    beta: TensorVariable
        A pytensor variable representing the parameters of the curve. The number of parameters and their meaning depend
        on the curve_type.

        .. warning::
            Currently no checks are in place to ensure that the number of parameters in beta matches the expected number
            for the curve_type.

    curve_type: str, one of ["log", "abc", "ns", "nss", "box-cox"]
        Type of curve to build. Options are:

            - "log":
                A simple log-linear curve. The curve is defined as:

                .. math::

                    \beta \\log(t)

            - "abc":
                A curve parameterized by "a", "b", and "c", such that the minimum value of the curve is "a", the
                maximum value is "a + b", and the inflection point is "a + b / c". "C" thus controls the speed of change
                from the minimum to the maximum value. The curve is defined as:

                .. math::

                    \frac{a + bc t}{1 + ct}

            - "ns":
                The Nelson-Siegel yield curve model. The curve is parameterized by three parameters: :math:`\tau`,
                :math:`\beta_1`, and :math:`\beta_2`. :math:`\tau` is the decay rate of the exponential term, and
                :math:`\beta_1` and :math:`\beta_2` control the slope and curvature of the curve. The curve is defined as:

                .. math::

                    \begin{align}
                    x_t &= \beta_1 \\phi(t) + \beta_2 \\left (\\phi(t) - \\exp(-t/\tau) \right ) \\
                    \\phi(t) &= \frac{1 - \\exp(-t/\tau)}{t/\tau}
                    \\end{align}

            - "nss":
                The Nelson-Siegel-Svensson yield curve model. The curve is parameterized by four parameters:
                :math:`\tau_1`, :math:`\tau_2`, :math:`\beta_1`, and :math:`\beta_2`. :math:`\beta_3`

                Where :math:`\tau_1` and :math:`\tau_2` are the decay rates of the two exponential terms, :math:`\beta_1`
                controls the slope of the curve, and :math:`\beta_2` and :math:`\beta_3` control the curvature of the curve.
                To ensure that short-term rates are strictly postitive, one typically restrices :math:`\beta_1 + \beta_2 > 0`.

                The curve is defined as:

                .. math::
                    \begin{align}
                    x_t & = \beta_1 \\phi_1(t) + \beta_2 \\left (\\phi_1(t) - \\exp(-t/\tau_1) \right) + \beta_3 \\left (\\phi_2(t) - \\exp(-t/\tau_2) \right) \\
                    \\phi_1(t) &= \frac{1 - \\exp(-t/\tau_1)}{t/\tau_1} \\
                    \\phi_2(t) &= \frac{1 - \\exp(-t/\tau_2)}{t/\tau_2}
                    \\end{align}

                Note that this definition omits the constant term that is typically included in the Nelson-Siegel-Svensson;
                you are assumed to have already accounted for this with another component in the model.

            - "box-cox":
                A curve that applies a box-cox transformation to the time data. The curve is parameterized by two
                parameters: :math:`\\lambda` and :math:`\beta`, where :math:`\\lambda` is the box-cox parameter that
                interpolates between the log and linear transformations, and :math:`\beta` is the slope of the curve.

                The curve is defined as:

                .. math::

                    \beta \\left ( \frac{t^{\\lambda} - 1}{\\lambda} \right )

    Returns
    -------
    TensorVariable
        A pytensor variable representing the curve.
    """
    if curve_type == "box-cox":
        lam = beta[0] + 1e-12
        time_scaled = (time_pt**lam - 1) / lam
        curve = beta[1] * time_scaled

    elif curve_type == "log":
        time_scaled = pt.log(time_pt)
        curve = beta[0] * time_scaled

    elif curve_type == "ns":
        tau = pt.exp(beta[0])
        t_over_tau = time_pt / tau
        time_scaled = (1 - pt.exp(-t_over_tau)) / t_over_tau
        curve = beta[1] * time_scaled + beta[2] * (time_scaled - pt.exp(-t_over_tau))

    elif curve_type == "nss":
        tau = pt.exp(beta[:2])
        beta = beta[2:]
        t_over_tau_1 = time_pt / tau[0]
        t_over_tau_2 = time_pt / tau[1]
        time_scaled_1 = (1 - pt.exp(t_over_tau_1)) / t_over_tau_1
        time_scaled_2 = (1 - pt.exp(t_over_tau_2)) / t_over_tau_2
        curve = (
            beta[0] * time_scaled_1
            + beta[1] * (time_scaled_1 - pt.exp(-t_over_tau_1))
            + beta[2] * (time_scaled_2 - pt.exp(-t_over_tau_2))
        )

    elif curve_type == "abc":
        curve = (beta[0] + beta[1] * beta[2] * time_pt) / (1 + beta[2] * time_pt)

    else:
        raise ValueError(f"Unknown curve type: {curve_type}")

    return curve


class Curve(GLMModel):
    def __init__(
        self,
        name: str,
        t: pd.Series | pd.DataFrame,
        prior: str = "Normal",
        index_data: pd.Series | pd.DataFrame | None = None,
        pooling: POOLING_TYPES = "complete",
        curve_type: CURVE_TYPES = "log",
        prior_params: dict | None = None,
        hierarchical_params: dict | None = None,
    ):
        """
        Class to represent a curve in a GLM model.

        A curve is a deterministic function that transforms time data via a non-linear function. Currently, the following
        curve types are supported:
            - "log": A simple log-linear curve.
            - "abc": A curve defined by a minimum value (a), maximum value (b), and inflection point ((a + b) / c).
            - "ns": The Nelson-Siegel yield curve model.
            - "nss": The Nelson-Siegel-Svensson yield curve model.
            - "box-cox": A curve that applies a box-cox transformation to the time data.

        Parameters
        ----------
        name: str, optional
            Name of the intercept term. If None, a default name is generated based on the index_data.
        t: Series
            Time data used to build the curve. If Series, must have a name attribute. If dataframe, must have exactly
            one column.
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
        self.t = t if isinstance(t, pd.Series) else t.iloc[:, 0]
        self.curve_type = curve_type

        self.index_data = index_data
        self.pooling = pooling

        self.prior = prior
        self.prior_params = prior_params if prior_params is not None else {}
        self.hierarchical_params = hierarchical_params if hierarchical_params is not None else {}

        super().__init__()

    def build(self, model=None):
        model = pm.modelcontext(model)
        obs_dim = self.t.index.name
        feature_dim = f"{self.name}_features"
        if feature_dim not in model.coords:
            model.add_coord(feature_dim, FEATURE_DICT[self.curve_type])

        with model:
            t_pt = pm.Data("t", self.t.values, dims=[obs_dim])
            if self.pooling == "complete":
                beta = getattr(pm, self.prior)(
                    f"{self.name}_beta", **self.prior_params, dims=[feature_dim]
                )
                curve = build_curve(t_pt, beta, self.curve_type)
                return pm.Deterministic(f"{self.name}", curve, dims=[obs_dim])

            beta = hierarchical_prior_to_requested_depth(
                self.name,
                self.index_data,
                model=model,
                dims=[feature_dim],
                no_pooling=self.pooling == "none",
            )

            curve = build_curve(t_pt, beta, self.curve_type)
            return pm.Deterministic(f"{self.name}", curve, dims=[obs_dim])


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
