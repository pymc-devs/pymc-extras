import itertools

from collections.abc import Sequence

import pandas as pd
import pymc as pm
import pytensor.tensor as pt

ColumnType = str | Sequence[str] | None

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


def make_level_maps(df: pd.DataFrame, ordered_levels: list[str]):
    """
    For each row of data, create a mapping between levels of a arbitrary set of levels defined by `ordered_levels`.

    Consider a set of levels (A, B, C) with members A: [A], B: [B1, B2], C: [C1, C2, C3, C4] arraged in a tree, like:
                A
             /      \
            B1      B2
           /  \\    /   \
         C1   C2  C3    C4

    A "deep hierarchy" will have the following priors:
        A ~ F(...)
        B1, B2 ~ F(A, ...)
        C1, C2 ~ F(B1, ...)
        C3, C4 ~ F(B2, ...)

    Noting that there could be multiple such trees in a dataset, to create these priors in a memory efficient way we need 2 mappings: B to A, and C to B. These
    need to be generated at inference time, and also re-generated for out of sample prediction.

    Parameters
    ----------
    df: pd.DataFrame
        It's data OK?

    ordered_levels: list[str]
        Sequence of level names, ordered from highest to lowest. In the above example, ordered_levels = ['A', 'B', 'C']

    Returns
    -------
    labels: list[pd.Index]
        Unique labels generated for each level, sorted alphabetically. Ordering corresponds to the integers in the corresponding mapping, à la pd.factorize

    mappings: list[np.ndarray]
        `len(ordered_levels) - 1` list of arrays indexing each previous level to the next level. The i-th array in the list has shape len(df[ordered_levels[i+1]].unique())
    """
    # TODO: Raise an error if there are one-to-many mappings between levels?
    if not all([level in df for level in ordered_levels]):
        missing = set(ordered_levels) - set(df.columns)
        raise ValueError(f'Requested levels were not in provided dataframe: {", ".join(missing)}')

    level_pairs = itertools.pairwise(ordered_levels)
    mappings = []
    labels = []
    for pair in level_pairs:
        _, level_labels = pd.factorize(df[pair[0]], sort=True)
        edges = df[list(pair)].drop_duplicates().set_index(pair[1])[pair[0]].sort_index()
        idx = edges.map({k: i for i, k in enumerate(level_labels)}).values
        labels.append(level_labels)
        mappings.append(idx)

    last_map, last_labels = pd.factorize(df[ordered_levels[-1]], sort=True)
    labels.append(last_labels)
    mappings.append(last_map)

    return labels, mappings


def make_next_level_hierarchy_variable(
    name: str,
    mu,
    sigma_dist: str = "Gamma",
    sigma_kwargs: dict | None = None,
    mapping=None,
    sigma_dims=None,
    offset_dims=None,
    offset_dist="Normal",
    no_pooling=False,
):
    if no_pooling:
        if mapping is None:
            return pm.Deterministic(f"{name}", mu[..., None], dims=offset_dims)
        else:
            return pm.Deterministic(f"{name}", mu[..., mapping], dims=offset_dims)

    d_sigma = getattr(pm, sigma_dist)

    if sigma_kwargs is None:
        if sigma_dist not in SIGMA_DEFAULT_KWARGS:
            raise NotImplementedError(
                f"No defaults implemented for {sigma_dist}. Pass sigma_kwargs explictly."
            )
        sigma_kwargs = SIGMA_DEFAULT_KWARGS[sigma_dist]

    sigma_ = d_sigma(f"{name}_sigma", **sigma_kwargs, dims=sigma_dims)

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


def hierarchical_prior_to_requested_depth(
    name: str,
    df: pd.DataFrame,
    model: pm.Model = None,
    dims: list[str] | None = None,
    no_pooling: bool = False,
    **hierarchy_kwargs,
):
    """
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

                 Apple             Banana
                /     \\           /    \
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
    df: DataFrame
        DataFrame of categorical data. Each column represents a level of the hierarchy, with the leftmost column
        representing the top level of the hierarchy, with depth increasing to the right.
    model: pm.Model, optional
        PyMC model to add the variable to. If None, the model on the current context stack is used.
    dims: list of str, optional
        Additional dimensions to add to the variable. These are treated as batch dimensions, and are added to the
        left of the hierarchy dimensions. For example, if dims=['feature'], and df has one column named "country",
         the returned variables will have dimensions ['feature', 'country']
    no_pooling: bool, optional
        If True, no pooling is applied to the variable. Each level of the hierarchy is treated as independent, with no
        informaton shared across level members of a given level.
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

    if isinstance(df, pd.Series):
        df = df.to_frame()

    model = pm.modelcontext(model)
    sigma_dist = hierarchy_kwargs.pop("sigma_dist", "Gamma")
    sigma_kwargs = hierarchy_kwargs.pop("sigma_kwargs", {"alpha": 2, "beta": 1})
    offset_dist = hierarchy_kwargs.pop("offset_dist", "zerosum")

    levels = [None, *df.columns.tolist()]
    n_levels = len(levels) - 1
    idx_maps = None
    if n_levels > 1:
        labels, idx_maps = make_level_maps(df, levels[1:])

    if idx_maps:
        idx_maps = [None, *idx_maps]
    else:
        idx_maps = [None]

    for level_dim in levels[1:]:
        _, labels = pd.factorize(df[level_dim], sort=True)
        if level_dim not in model.coords:
            model.add_coord(level_dim, labels)

    # Danger zone, this assumes we factorized the same way here and in X_data
    deepest_map = _get_x_cols(df.columns[-1]).astype("int")

    with model:
        beta = pm.Normal(f"{name}_effect", 0, 1, dims=dims)
        for i, (level, last_level) in enumerate(zip(levels[1:], levels[:-1])):
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
                no_pooling=no_pooling,
            )

    return beta[..., deepest_map]
