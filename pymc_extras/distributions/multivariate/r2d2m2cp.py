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


from collections import namedtuple
from collections.abc import Sequence

import numpy as np
import pymc as pm
import pytensor.tensor as pt

__all__ = ["R2D2M2CP"]


def _psivar2musigma(
    psi: pt.TensorVariable,
    explained_var: pt.TensorVariable,
    psi_mask: pt.TensorLike | None,
) -> tuple[pt.TensorVariable, pt.TensorVariable]:
    sign = pt.sign(psi - 0.5)
    if psi_mask is not None:
        # any computation might be ignored for ~psi_mask
        # sign and explained_var are used
        psi = pt.where(psi_mask, psi, 0.5)
    pi = pt.erfinv(2 * psi - 1)
    f = (1 / (2 * pi**2 + 1)) ** 0.5
    sigma = explained_var**0.5 * f
    mu = sigma * pi * 2**0.5
    if psi_mask is not None:
        return (
            pt.where(psi_mask, mu, sign * explained_var**0.5),
            pt.where(psi_mask, sigma, 0),
        )
    else:
        return mu, sigma


def _R2D2M2CP_beta(
    name: str,
    output_sigma: pt.TensorVariable,
    input_sigma: pt.TensorVariable,
    r2: pt.TensorVariable,
    phi: pt.TensorVariable,
    psi: pt.TensorVariable,
    *,
    psi_mask,
    dims: str | Sequence[str],
    centered=False,
) -> pt.TensorVariable:
    """R2D2M2CP beta prior.

    Parameters
    ----------
    name: str
        Name for the distribution
    output_sigma: tensor
        standard deviation of the outcome
    input_sigma: tensor
        standard deviation of the explanatory variables
    r2: tensor
        expected R2 for the linear regression
    phi: tensor
        variance weights that sums up to 1
    psi: tensor
        probability of a coefficients to be positive
    """
    explained_variance = phi * pt.expand_dims(r2 * output_sigma**2, (-1,))
    mu_param, std_param = _psivar2musigma(psi, explained_variance, psi_mask=psi_mask)
    if not centered:
        with pm.Model(name):
            if psi_mask is not None and psi_mask.any():
                # limit case where some probs are not 1 or 0
                # setsubtensor is required
                r_idx = psi_mask.nonzero()
                with pm.Model("raw"):
                    raw = pm.Normal("masked", shape=len(r_idx[0]))
                    raw = pt.set_subtensor(pt.zeros_like(mu_param)[r_idx], raw)
                raw = pm.Deterministic("raw", raw, dims=dims)
            elif psi_mask is not None:
                # all variables are deterministic
                raw = pt.zeros_like(mu_param)
            else:
                raw = pm.Normal("raw", dims=dims)
        beta = pm.Deterministic(
            name, (raw * std_param + mu_param) / input_sigma, dims=dims
        )
    else:
        if psi_mask is not None and psi_mask.any():
            # limit case where some probs are not 1 or 0
            # setsubtensor is required
            r_idx = psi_mask.nonzero()
            with pm.Model(name):
                mean = (mu_param / input_sigma)[r_idx]
                sigma = (std_param / input_sigma)[r_idx]
                masked = pm.Normal(
                    "masked",
                    mean,
                    sigma,
                    shape=len(r_idx[0]),
                )
                beta = pt.set_subtensor(mean, masked)
            beta = pm.Deterministic(name, beta, dims=dims)
        elif psi_mask is not None:
            # all variables are deterministic
            beta = pm.Deterministic(name, (mu_param / input_sigma), dims=dims)
        else:
            beta = pm.Normal(
                name, mu_param / input_sigma, std_param / input_sigma, dims=dims
            )
    return beta


def _broadcast_as_dims(
    *values: np.ndarray,
    dims: Sequence[str],
) -> tuple[np.ndarray, ...] | np.ndarray:
    model = pm.modelcontext(None)
    shape = [len(model.coords[d]) for d in dims]
    ret = tuple(np.broadcast_to(v, shape) for v in values)
    # strip output
    if len(values) == 1:
        ret = ret[0]
    return ret


def _psi_masked(
    positive_probs: pt.TensorLike,
    positive_probs_std: pt.TensorLike,
    *,
    dims: Sequence[str],
) -> tuple[pt.TensorLike | None, pt.TensorVariable]:
    if not (
        isinstance(positive_probs, pt.Constant)
        and isinstance(positive_probs_std, pt.Constant)
    ):
        raise TypeError(
            "Only constant values for positive_probs and positive_probs_std are accepted"
        )
    positive_probs, positive_probs_std = _broadcast_as_dims(
        positive_probs.data, positive_probs_std.data, dims=dims
    )
    mask = ~np.bitwise_or(positive_probs == 1, positive_probs == 0)
    if np.bitwise_and(~mask, positive_probs_std != 0).any():
        raise ValueError(
            "Can't have both positive_probs == '1 or 0' and positive_probs_std != 0"
        )
    if (~mask).any() and mask.any():
        # limit case where some probs are not 1 or 0
        # setsubtensor is required
        r_idx = mask.nonzero()
        with pm.Model("psi"):
            psi = pm.Beta(
                "masked",
                mu=positive_probs[r_idx],
                sigma=positive_probs_std[r_idx],
                shape=len(r_idx[0]),
            )
        psi = pt.set_subtensor(pt.as_tensor(positive_probs)[r_idx], psi)
        psi = pm.Deterministic("psi", psi, dims=dims)
    elif (~mask).all():
        # limit case where all the probs are limit case
        psi = pt.as_tensor(positive_probs)
    else:
        psi = pm.Beta("psi", mu=positive_probs, sigma=positive_probs_std, dims=dims)
        mask = None
    return mask, psi


def _psi(
    positive_probs: pt.TensorLike,
    positive_probs_std: pt.TensorLike | None,
    *,
    dims: Sequence[str],
) -> tuple[pt.TensorLike | None, pt.TensorVariable]:
    if positive_probs_std is not None:
        mask, psi = _psi_masked(
            positive_probs=pt.as_tensor(positive_probs),
            positive_probs_std=pt.as_tensor(positive_probs_std),
            dims=dims,
        )
    else:
        positive_probs = pt.as_tensor(positive_probs)
        if not isinstance(positive_probs, pt.Constant):
            raise TypeError("Only constant values for positive_probs are allowed")
        psi = _broadcast_as_dims(positive_probs.data, dims=dims)
        mask = np.atleast_1d(~np.bitwise_or(psi == 1, psi == 0))
        if mask.all():
            mask = None
    return mask, psi


def _phi(
    variables_importance: pt.TensorLike | None,
    variance_explained: pt.TensorLike | None,
    importance_concentration: pt.TensorLike | None,
    *,
    dims: Sequence[str],
) -> pt.TensorVariable:
    *broadcast_dims, dim = dims
    model = pm.modelcontext(None)
    if variables_importance is not None:
        if variance_explained is not None:
            raise TypeError("Can't use variable importance with variance explained")
        if len(model.coords[dim]) <= 1:
            raise TypeError(
                "Can't use variable importance with less than two variables"
            )
        variables_importance = pt.as_tensor(variables_importance)
        if importance_concentration is not None:
            variables_importance *= importance_concentration
        return pm.Dirichlet("phi", variables_importance, dims=[*broadcast_dims, dim])
    elif variance_explained is not None:
        if len(model.coords[dim]) <= 1:
            raise TypeError("Can't use variance explained with less than two variables")
        phi = pt.as_tensor(variance_explained)
    else:
        phi = _broadcast_as_dims(1.0, dims=dims)
    if importance_concentration is not None:
        return pm.Dirichlet(
            "phi", importance_concentration * phi, dims=[*broadcast_dims, dim]
        )
    else:
        return phi


R2D2M2CPOut = namedtuple("R2D2M2CPOut", ["eps", "beta"])


def R2D2M2CP(
    name: str,
    output_sigma: pt.TensorLike,
    input_sigma: pt.TensorLike,
    *,
    dims: Sequence[str],
    r2: pt.TensorLike,
    variables_importance: pt.TensorLike | None = None,
    variance_explained: pt.TensorLike | None = None,
    importance_concentration: pt.TensorLike | None = None,
    r2_std: pt.TensorLike | None = None,
    positive_probs: pt.TensorLike | None = 0.5,
    positive_probs_std: pt.TensorLike | None = None,
    centered: bool = False,
) -> R2D2M2CPOut:
    """R2D2M2CP Prior.

    Parameters
    ----------
    name : str
        Name for the distribution
    output_sigma : Tensor
        Output standard deviation
    input_sigma : Tensor
        Input standard deviation
    dims : Union[str, Sequence[str]]
        Dims for the distribution
    r2 : Tensor
        :math:`R^2` estimate
    variables_importance : Tensor, optional
        Optional estimate for variables importance, positive, by default None
    variance_explained : Tensor, optional
        Alternative estimate for variables importance which is point estimate of
        variance explained, should sum up to one, by default None
    importance_concentration : Tensor, optional
        Confidence around variance explained or variable importance estimate
    r2_std : Tensor, optional
        Optional uncertainty over :math:`R^2`, by default None
    positive_probs : Tensor, optional
        Optional probability of variables contribution to be positive, by default 0.5
    positive_probs_std : Tensor, optional
        Optional uncertainty over effect direction probability, by default None
    centered : bool, optional
        Centered or Non-Centered parametrization of the distribution, by default Non-Centered. Advised to check both

    Returns
    -------
    residual_sigma, coefficients
        Output variance (sigma squared) is split in residual variance and explained variance.

    Raises
    ------
    TypeError
        If parametrization is wrong.

    Notes
    -----
    The R2D2M2CP prior is a modification of R2D2M2 prior.

    - ``(R2D2M2)`` CP is taken from https://arxiv.org/abs/2208.07132
    - R2D2M2 ``(CP)``, (Correlation Probability) is proposed and implemented by Max Kochurov (@ferrine)

    Examples
    --------
    Here are arguments explained in a synthetic example

    .. warning::

        To use the prior in a linear regression

        - make sure :math:`X` is centered around zero
        - intercept represents prior predictive mean when :math:`X` is centered
        - setting named dims is required

    .. code-block:: python

        import pymc_extras as pmx
        import pymc as pm
        import numpy as np

        X = np.random.randn(10, 3)
        b = np.random.randn(3)
        y = X @ b + np.random.randn(10) * 0.04 + 5
        with pm.Model(coords=dict(variables=["a", "b", "c"])) as model:
            eps, beta = pmx.distributions.R2D2M2CP(
                "beta",
                y.std(),
                X.std(0),
                dims="variables",
                # NOTE: global shrinkage
                r2=0.8,
                # NOTE: if you are unsure about r2
                r2_std=0.2,
                # NOTE: if you know where a variable should go
                # if you do not know, leave as 0.5
                positive_probs=[0.8, 0.5, 0.1],
                # NOTE: if you have different opinions about
                # where a variable should go.
                # NOTE: if you put 0.5 previously,
                # just put 0.1 there, but other
                # sigmas should work fine too
                positive_probs_std=[0.3, 0.1, 0.2],
                # NOTE: variable importances are relative to each other,
                # but larget numbers put "more" weight in the relation
                # use
                # * 1-10 for small confidence
                # * 10-30 for moderate confidence
                # * 30+ for high confidence
                # EXAMPLE:
                # "a" - is likely to be useful
                # "b" - no idea if it is useful
                # "c" - a must have in the relation
                variables_importance=[10, 1, 34],
                # NOTE: try both
                centered=True,
            )
            # intercept prior centering should be around prior predictive mean
            intercept = y.mean()
            # regressors should be centered around zero
            Xc = X - X.mean(0)
            obs = pm.Normal("obs", intercept + Xc @ beta, eps, observed=y)

    There can be special cases by choosing specific set of arguments

    Here the prior distribution of beta is ``Normal(0, y.std() * r2 ** .5)``

    .. code-block:: python

        with pm.Model(coords=dict(variables=["a", "b", "c"])) as model:
            eps, beta = pmx.distributions.R2D2M2CP(
                "beta",
                y.std(),
                X.std(0),
                dims="variables",
                # NOTE: global shrinkage
                r2=0.8,
                # NOTE: if you are unsure about r2
                r2_std=0.2,
                # NOTE: if you know where a variable should go
                # if you do not know, leave as 0.5
                centered=False,
            )
            # intercept prior centering should be around prior predictive mean
            intercept = y.mean()
            # regressors should be centered around zero
            Xc = X - X.mean(0)
            obs = pm.Normal("obs", intercept + Xc @ beta, eps, observed=y)


    It is fine to leave some of the ``_std`` arguments unspecified.
    You can also specify only ``positive_probs``, and all
    the variables are assumed to explain same amount of variance (same importance)

    .. code-block:: python

        with pm.Model(coords=dict(variables=["a", "b", "c"])) as model:
            eps, beta = pmx.distributions.R2D2M2CP(
                "beta",
                y.std(),
                X.std(0),
                dims="variables",
                # NOTE: global shrinkage
                r2=0.8,
                # NOTE: if you are unsure about r2
                r2_std=0.2,
                # NOTE: if you know where a variable should go
                # if you do not know, leave as 0.5
                positive_probs=[0.8, 0.5, 0.1],
                # NOTE: try both
                centered=True,
            )
            intercept = y.mean()
            obs = pm.Normal("obs", intercept + X @ beta, eps, observed=y)

    Notes
    -----
    To reference R2D2M2CP implementation, you can use the following bibtex entry:

    .. code-block::

        @misc{pymc-extras-r2d2m2cp,
            title = {pymc-devs/pymc-extras: {P}ull {R}equest 137, {R2D2M2CP}},
            url = {https://github.com/pymc-devs/pymc-extras/pull/137},
            author = {Max Kochurov},
            howpublished = {GitHub},
            year = {2023}
        }
    """
    if not isinstance(dims, list | tuple):
        dims = (dims,)
    *broadcast_dims, dim = dims
    input_sigma = pt.as_tensor(input_sigma)
    output_sigma = pt.as_tensor(output_sigma)
    with pm.Model(name):
        if r2_std is not None:
            r2 = pm.Beta("r2", mu=r2, sigma=r2_std, dims=broadcast_dims)
        phi = _phi(
            variables_importance=variables_importance,
            variance_explained=variance_explained,
            importance_concentration=importance_concentration,
            dims=dims,
        )
        mask, psi = _psi(
            positive_probs=positive_probs,
            positive_probs_std=positive_probs_std,
            dims=dims,
        )

    beta = _R2D2M2CP_beta(
        name,
        output_sigma,
        input_sigma,
        r2,
        phi,
        psi,
        dims=[*broadcast_dims, dim],
        centered=centered,
        psi_mask=mask,
    )
    resid_sigma = (1 - r2) ** 0.5 * output_sigma
    return R2D2M2CPOut(resid_sigma, beta)
