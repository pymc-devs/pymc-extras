import itertools

from collections.abc import Sequence
from copy import deepcopy
from typing import Literal, get_args

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from pytensor.compile import SharedVariable

ColumnType = str | list[str]

PoolingType = Literal["none", "complete", "partial", None]
valid_pooling = get_args(PoolingType)

# Dictionary to define offset distributions for hierarchical models
OFFSET_DIST_FACTORY = {
    "zerosum": lambda name, offset_dims: pm.ZeroSumNormal(f"{name}_offset", dims=offset_dims),
    "normal": lambda name, offset_dims: pm.Normal(f"{name}_offset", dims=offset_dims),
    "laplace": lambda name, offset_dims: pm.Laplace(f"{name}_offset", mu=0, b=1, dims=offset_dims),
}

# Default kwargs for sigma distributions
SIGMA_DEFAULT_KWARGS = {
    "Gamma": {"alpha": 2, "beta": 1},
    "Exponential": {"lam": 1},
    "HalfNormal": {"sigma": 1},
    "HalfCauchy": {"beta": 1},
}

PRIOR_DEFAULT_KWARGS = {
    "Normal": {"mu": 0, "sigma": 1},
    "Laplace": {"mu": 0, "b": 1},
    "StudentT": {"nu": 3, "mu": 0, "sigma": 1},
}


def select_data_columns(
    cols: str | Sequence[str] | None,
    model: pm.Model | None = None,
    data_name: str = "X_data",
    squeeze=True,
) -> pt.TensorVariable | None:
    """
    Create a tensor variable representing a subset of independent data columns.

    Parameters
    ----------
    cols: str or list of str
        Column names to select from the independent data
    model: Model, optional
        PyMC model object. If None, the model is taken from the context.

    Returns
    -------
        A tensor variable representing the selected columns of the independent data
    """
    model = pm.modelcontext(model)
    if cols is None:
        return

    cols = at_least_list(cols)

    missing_cols = [col for col in cols if col not in model.coords["feature"]]
    if missing_cols:
        raise ValueError(f"Columns {missing_cols} not found in the model")

    cols_idx = [model.coords["feature"].index(col) for col in cols]

    # Single columns are returned as 1d arrays
    if len(cols_idx) == 1 and squeeze:
        cols_idx = cols_idx[0]

    return get_X_data(model, data_name=data_name)[:, cols_idx]


def get_X_data(model, data_name="X_data") -> SharedVariable:
    return model[data_name]


def encode_categoricals(df, coords):
    df = df.copy()
    coords = deepcopy(coords)

    for col, dtype in df.dtypes.items():
        if dtype.name.startswith("float"):
            pass

        elif dtype.name == "object":
            col_array, labels = pd.factorize(df[col], sort=True)
            df[col] = col_array.astype("float64")
            coords[col] = labels

        elif dtype.name.startswith("int"):
            _data = df[col].copy()
            df[col] = df[col].astype("float64")
            assert np.all(
                _data == df[col].astype("int")
            ), "Information was lost in conversion to float"

        else:
            raise NotImplementedError(
                f"Haven't decided how to handle the following type: {dtype.name}"
            )

    return df, coords


def at_least_list(columns: ColumnType):
    if columns is None:
        columns = []
    elif isinstance(columns, str):
        columns = [columns]
    return columns


def make_level_maps(X: SharedVariable, coords: dict[str, tuple | None], ordered_levels: list[str]):
    r"""
    For each row of data, create a mapping between levels of a arbitrary set of levels defined by `ordered_levels`.

    Consider a set of levels (A, B, C) with members A: [A], B: [B1, B2], C: [C1, C2, C3, C4] arraged in a tree, like:
                A
             /     \
            B1     B2
           /  \   /   \
         C1   C2  C3   C4

    A "deep hierarchy" will have the following priors:
        A ~ F(...)
        B1, B2 ~ F(A, ...)
        C1, C2 ~ F(B1, ...)
        C3, C4 ~ F(B2, ...)

    Noting that there could be multiple such trees in a dataset, to create these priors in a memory efficient way we need 2 mappings: B to A, and C to B. These
    need to be generated at inference time, and also re-generated for out of sample prediction.

    Parameters
    ----------
    X: pt.TensorVariable
        It's data OK?

    coords: dict[str, list[str]]
        Dictionary of levels and their members. In the above example,
        ``coords = {'A': ['A'], 'B': ['B1', 'B2'], 'C': ['C1', 'C2', 'C3', 'C4']}``

    ordered_levels: list[str]
        Sequence of level names, ordered from highest to lowest. In the above example,
        ordered_levels = ['A', 'B', 'C']

    Returns
    -------
    mappings: list[pt.TensorVariable]
        `len(ordered_levels) - 1` list of arrays indexing each previous level to the next level.
         The i-th array in the list has shape len(df[ordered_levels[i+1]].unique())
    """
    level_idxs = [coords["feature"].index(level) for level in ordered_levels]
    mappings = [None]

    for level_im1, level_i in itertools.pairwise(level_idxs):
        edges = pt.unique(X[:, [level_im1, level_i]], axis=0)
        sorted_idx = pt.argsort(edges[:, 1])
        mappings.append(edges[sorted_idx, 0].astype(int))

    mappings.append(X[:, level_idxs[-1]].astype(int))
    return mappings


def make_sigma(name, sigma_dist, sigma_kwargs, sigma_dims):
    d_sigma = getattr(pm, sigma_dist)

    if sigma_kwargs is None:
        if sigma_dist not in SIGMA_DEFAULT_KWARGS:
            raise NotImplementedError(
                f"No defaults implemented for {sigma_dist}. Pass sigma_kwargs explictly."
            )
        sigma_kwargs = SIGMA_DEFAULT_KWARGS[sigma_dist]

    return d_sigma(f"{name}_sigma", **sigma_kwargs, dims=sigma_dims)


def make_next_level_hierarchy_variable(
    name: str,
    mu,
    sigma_dist: str = "Gamma",
    sigma_kwargs: dict | None = None,
    mapping=None,
    sigma_dims=None,
    offset_dims=None,
    offset_dist="Normal",
):
    sigma_ = make_sigma(name, sigma_dist, sigma_kwargs, sigma_dims)
    offset_dist = offset_dist.lower()
    if offset_dist not in OFFSET_DIST_FACTORY:
        raise NotImplementedError()

    offset = OFFSET_DIST_FACTORY[offset_dist](name, offset_dims)

    if mapping is None:
        return pm.Deterministic(
            f"{name}", mu[..., None] + sigma_[..., None] * offset, dims=offset_dims
        )
    else:
        return pm.Deterministic(
            f"{name}", mu[..., mapping] + sigma_[..., mapping] * offset, dims=offset_dims
        )


def make_partial_pooled_hierarchy(
    name: str,
    X: SharedVariable,
    model: pm.Model,
    pooling_columns: ColumnType = None,
    prior: str = "Normal",
    prior_kwargs: dict = {},
    dims: list[str] | None = None,
    **hierarchy_kwargs,
) -> pt.TensorVariable:
    r"""
    Given a dataframe of categorical data, construct a hierarchical prior that pools data telescopically, moving from
    left to right across the columns of the dataframe.

    At its simplest, this function can be used to construct a simple hierarchical prior for a single categorical
    variable. Consider the following example:

    .. code-block:: python

        df = pd.DataFrame(['Apple', 'Apple', 'Banana', 'Banana'], columns=['fruit'])
        coords = {'fruit': ['Apple', 'Banana']}
        with pm.Model(coords=coords) as model:
            fruit_effect = hierarchical_prior_to_requested_depth('fruit_effect', df)

    This will construct a simple, non-centered hierarchical intercept corresponding to the 'fruit' feature of the data.
    The power of the function comes from its ability to handle multiple categorical variables, and to construct a
    hierarchical prior that pools data across multiple levels of a hierarchy. Consider the following example:

    .. code-block:: python
        df = pd.DataFrame({'fruit': ['Apple', 'Apple', 'Banana', 'Banana'],
                             'color': ['Red', 'Green', 'Yellow', 'Brown']})
        coords = {'fruit': ['Apple', 'Banana'], 'color': ['Red', 'Green', 'Yellow', 'Brown']}
        with pm.Model(coords=coords) as model:
            fruit_effect = hierarchical_prior_to_requested_depth('fruit_effect', df[['fruit', 'color']])

    This will construct a two-level hierarchy. The first level will pool all rows of data with the same 'fruit' value,
    and the second level will pool color values within each fruit. The structure of the hierarchy will be:

    .. code-block::

                 Apple             Banana
                /     \            /    \
            Red       Green    Yellow   Brown


    That is, estimates for each of "red" and "green" will be centered on the estimate of "apple", and estimates for
    "yellow" and "brown" will be centered on the estimate of "banana".

    .. warning::
        Currently, the structure of the data **must** be bijective with respect to the levels of the hierarchy. That is,
        each child must map to exactly one parent. In the above example, we could not consider green bananas, for example,
        because the "green" level would not uniquely map to "apple". This is a limitation of the current implementation.


    Parameters
    ----------
    name: str
        Name of the variable to construct
    X: SharedVariable
        Feature data associated with the GLM model. Encoded categorical features used to form the hierarchical prior
        are expected to be columns in this data.
    pooling_columns: str or list of str
        Columns of the dataframe to use as the index of the hierarchy. If a list is provided, the hierarchy will be
        constructed from left to right across the columns.
    model: pm.Model, optional
        PyMC model to add the variable to. If None, the model on the current context stack is used.
    dims: list of str, optional
        Additional dimensions to add to the variable. These are treated as batch dimensions, and are added to the
        left of the hierarchy dimensions. For example, if dims=['feature'], and df has one column named "country",
         the returned variables will have dimensions ['feature', 'country']
    hierarchy_kwargs: dict
        Additional keyword arguments to pass to the underlying PyMC distribution. Options include:
            sigma_dist: str
                Name of the distribution to use for the standard deviation of the hierarchy. Default is "Gamma"
            sigma_kwargs: dict
                Additional keyword arguments to pass to the sigma distribution specified by the sigma_dist argument.
                Default is {"alpha": 2, "beta": 1}
            offset_dist: str, one of ["zerosum", "normal", "laplace"]
                Name of the distribution to use for the offset distribution. Default is "zerosum"

    Returns
    -------
    pm.Distribution
        PyMC distribution representing the hierarchical prior. The shape of the distribution will be
        (n_obs, *dims, df.loc[:, -1].nunique())
    """
    coords = model.coords

    if X.ndim == 1:
        X = X[:, None]

    sigma_dist = hierarchy_kwargs.pop("sigma_dist", "Gamma")
    sigma_kwargs = hierarchy_kwargs.pop("sigma_kwargs", {"alpha": 2, "beta": 1})
    offset_dist = hierarchy_kwargs.pop("offset_dist", "zerosum")

    idx_maps = make_level_maps(X, coords, pooling_columns)
    deepest_map = idx_maps[-1]

    Prior = getattr(pm, prior)
    prior_params = deepcopy(PRIOR_DEFAULT_KWARGS.get(prior))
    prior_params.update(prior_kwargs)

    with model:
        beta = Prior(f"{name}", **prior_params, dims=dims)

        for i, (last_level, level) in enumerate(itertools.pairwise([None, *pooling_columns])):
            if i == 0:
                sigma_dims = dims
            else:
                sigma_dims = [*dims, last_level] if dims is not None else [last_level]
            offset_dims = [*dims, level] if dims is not None else [level]

            # TODO: Need a better way to handle different priors at each level.
            if "beta" in sigma_kwargs:
                sigma_kwargs["beta"] = sigma_kwargs["beta"] ** (i + 1)

            beta = make_next_level_hierarchy_variable(
                f"{name}_{level}_effect",
                mu=beta,
                sigma_dist=sigma_dist,
                sigma_kwargs=sigma_kwargs,
                mapping=idx_maps[i],
                sigma_dims=sigma_dims,
                offset_dims=offset_dims,
                offset_dist=offset_dist,
            )

    return beta[..., deepest_map]


def make_unpooled_hierarchy(
    name: str,
    X: SharedVariable,
    model: pm.Model = None,
    prior: str = "Normal",
    levels: str | list[str] | None = None,
    dims: list[str] | None = None,
    **hierarchy_kwargs,
):
    coords = model.coords

    if X.ndim == 1:
        X = X[:, None]

    idx_maps = make_level_maps(X, coords, levels)
    deepest_map = idx_maps[-1]

    Prior = getattr(pm, prior)
    prior_kwargs = deepcopy(PRIOR_DEFAULT_KWARGS.get(prior))
    prior_kwargs.update(prior_kwargs)

    with model:
        beta = Prior(f"{name}_mu", **prior_kwargs, dims=dims)

        for i, (last_level, level) in enumerate(itertools.pairwise([None, *levels])):
            beta_dims = [*dims, level] if dims is not None else [level]
            prior_kwargs["mu"] = beta[..., idx_maps[i]]

            beta = Prior(f"{name}_{level}_effect", **prior_kwargs, dims=beta_dims)

    return beta[..., deepest_map]


def make_completely_pooled_hierarchy(
    name: str,
    model: pm.Model,
    dims: list[str] | None = None,
):
    with model:
        beta = pm.Normal(f"{name}_mu", 0, 1, dims=dims)

    return beta


def make_hierarchical_prior(
    name: str,
    X: SharedVariable,
    pooling: PoolingType,
    prior: str = "Normal",
    pooling_columns: ColumnType = None,
    model: pm.Model = None,
    dims: list[str] | None = None,
    **hierarchy_kwargs,
):
    model = pm.modelcontext(model)
    pooling_columns = at_least_list(pooling_columns)

    if pooling == "none":
        return make_unpooled_hierarchy(
            name=name, X=X, model=model, levels=pooling_columns, prior=prior, dims=dims
        )
    elif pooling == "partial":
        return make_partial_pooled_hierarchy(
            name=name,
            X=X,
            pooling_columns=pooling_columns,
            model=model,
            prior=prior,
            dims=dims,
            **hierarchy_kwargs,
        )
    else:
        raise NotImplementedError(f"{pooling} pooling not yet implemented")
