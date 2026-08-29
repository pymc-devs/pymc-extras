from dataclasses import dataclass, field

import numpy as np
import pytensor.tensor as pt

from pymc.blocking import DictToArrayBijection
from pymc.distributions import Normal
from pymc.logprob.basic import conditional_logp
from pymc.model.core import Deterministic, Model
from pymc.pytensorf import resolve_shapes
from pytensor.assumptions import assume
from pytensor.gradient import disconnected_grad
from pytensor.graph import ancestors
from pytensor.graph.basic import Variable, equal_computations
from pytensor.graph.replace import graph_replace
from pytensor.tensor.linalg import cholesky, solve_triangular


def get_value_shapes_and_dims(
    model: Model,
) -> tuple[dict[Variable, Variable], dict[Variable, tuple | None]]:
    """Return the symbolic shapes and model dims of the value variables of the model's free RVs.

    Guides parameterize the model in the space of its value variables (the unconstrained
    space), where a transformed RV may have a different shape than the RV itself
    (e.g., a Dirichlet of size n has a simplex-transformed value variable of size n - 1).

    Model dims describe each RV in its constrained space. They carry over to the value
    variable only when the transform preserves the variable's shape: there is no
    transform, the transform is elemwise (``ndim_supp == 0``), or the symbolic shapes
    of the RV and its value variable can be shown to be equal.
    """
    free_rvs = model.free_RVs
    transformed_rvs = [
        rv
        if (transform := model.rvs_to_transforms[rv]) is None
        else transform.forward(rv, *rv.owner.inputs)
        for rv in free_rvs
    ]
    shapes = resolve_shapes([var.shape for var in (*transformed_rvs, *free_rvs)])
    value_shapes, rv_shapes = shapes[: len(free_rvs)], shapes[len(free_rvs) :]
    if overlap := (set(free_rvs) & set(ancestors(value_shapes))):
        raise ValueError(f"value shapes still depend on the following rvs {overlap}")

    value_dims = {}
    for rv, value_shape, rv_shape in zip(free_rvs, value_shapes, rv_shapes):
        dims = model.named_vars_to_dims.get(rv.name, None)
        if dims is not None:
            transform = model.rvs_to_transforms[rv]
            preserves_shape = (
                transform is None
                or transform.ndim_supp == 0
                or equal_computations([value_shape], [rv_shape])
            )
            if not preserves_shape:
                dims = None
        value_dims[rv] = dims

    return dict(zip(free_rvs, value_shapes)), value_dims


@dataclass(frozen=True)
class AutoGuideModel:
    model: Model
    params_init_values: dict[Variable, np.ndarray]
    name_to_param: dict[str, Variable] = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "name_to_param",
            {x.name: x for x in self.params_init_values.keys()},
        )

    @property
    def params(self) -> tuple[Variable, ...]:
        return tuple(self.params_init_values.keys())

    def __getitem__(self, name: str) -> Variable:
        return self.name_to_param[name]

    @property
    def latent(self) -> Variable:
        """The whole unconstrained draw, before it is split into per-variable values.

        Only the multivariate guides build such a variable; accessing this on a guide
        that does not (e.g. the mean-field :func:`AutoDiagonalNormal`) raises a KeyError.
        """
        return self.model["latent"]

    def fit_quantities(self, params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Describe the fitted approximation by its mean and its covariance parameterization.

        Every guide reports ``mean_vector``; the rest is whatever that family stores, so the
        family is identifiable from which entries are present. The mean-field guide keeps a
        marginal standard deviation per parameter, and reports it exponentiated rather than in
        the unconstrained space its optimizer works in.

        Parameters
        ----------
        params : dict of str to ndarray
            Guide parameter values, keyed as in :attr:`params_init_values`.

        Returns
        -------
        quantities : dict of str to ndarray
            ``mean_vector`` and this family's covariance entries, flattened in the order the
            model's free RVs appear in.
        """
        locs = [params[p.name].ravel() for p in self.params if p.name.endswith("_loc")]
        scales = [params[p.name].ravel() for p in self.params if p.name.endswith("_scale")]
        return {
            "mean_vector": np.concatenate(locs),
            "standard_deviation": np.exp(np.concatenate(scales)),
        }

    def stochastic_logq(self, path_derivative_gradient: bool = True) -> pt.TensorVariable:
        """Returns a graph representing the logp of the guide model, evaluated under draws from its random variables.

        When ``path_derivative_gradient`` is True, the variational parameters are detached
        from the density evaluation, so that gradients flow only through the random draws.
        This yields the lower-variance path-derivative gradient estimator of _[1], also
        known as "sticking the landing".

        References
        ----------
        .. [1] Geoffrey Roeder, Yuhuai Wu, and David Duvenaud. Sticking the Landing: Simple,
               Lower-Variance Gradient Estimators for Variational Inference. NeurIPS, 2017.
        """
        logp_terms = conditional_logp(
            {rv: rv for rv in self.model.deterministics},
            warn_rvs=False,
        )
        logq = pt.sum([logp_term.sum() for logp_term in logp_terms.values()])

        if path_derivative_gradient:
            # Detach variational parameters from the density evaluation. The remaining
            # gradient flows through the random draws (the path derivative); the dropped
            # score-function term has zero expectation.
            repl = {p: disconnected_grad(p) for p in self.params}
            logq = graph_replace(logq, repl)

        return logq


def _check_continuous_rvs(model: Model, free_rvs: list[Variable]) -> None:
    if discrete_rvs := [
        rv.name for rv in free_rvs if not model.rvs_to_values[rv].type.dtype.startswith("float")
    ]:
        raise ValueError(
            f"ADVI requires continuous free RVs, but {discrete_rvs} are discrete. "
            "Marginalize them out or use another inference method."
        )


def AutoDiagonalNormal(model: Model, random_seed=None) -> AutoGuideModel:
    """
    Create a guide model for ADVI with a mean-field normal distribution.

    A guide model is a variational distribution that approximates the posterior distribution of the model's free
    random variables. In this case, we use a mean-field normal distribution, which assumes that the free random
    variables are independent and normally distributed. For details, see _[1].

    The guide is parameterized in the space of the model's value variables (the unconstrained space), as in _[1].
    For each free random variable in the model, we create a corresponding random variable in the guide model with a
    normal distribution over its value variable. The mean and standard deviation of each normal distribution are
    parameterized by learnable parameters (loc and scale), which are initialized to small random values.

    Parameters
    ----------
    model : Model
        The probabilistic model for which to create the guide.
    random_seed : optional
        Seed passed to ``model.initial_point``, which is only used when a variable's
        initial value strategy is random (e.g. a prior draw).

    Returns
    -------
    guide_model : AutoGuideModel
        An AutoGuideModel containing the guide model and the initial values for its parameters.

    References
    ----------
    .. [1] Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, and David M. Blei. Automatic Differentiation
           Variational Inference. Journal of Machine Learning Research, 18(14):1–45, 2017.
    """
    coords = model.coords
    free_rvs = model.free_RVs
    _check_continuous_rvs(model, free_rvs)

    value_shapes, value_dims = get_value_shapes_and_dims(model)
    # Initialize the guide means at the model initial point (already in the unconstrained
    # space), so the first guide draws start in a region of finite model logp
    initial_point = model.initial_point(random_seed=random_seed)
    params_init_values = {}

    # model=None detaches the guide from any model context the user may be inside,
    # which would otherwise register the guide as a nested submodel
    with Model(coords=coords, model=None) as guide_model:
        for rv in free_rvs:
            value_var = model.rvs_to_values[rv]
            loc = pt.tensor(f"{rv.name}_loc", shape=value_var.type.shape)
            scale = pt.tensor(f"{rv.name}_scale", shape=value_var.type.shape)

            loc_init = initial_point[value_var.name]
            params_init_values[loc] = loc_init
            # TODO: Make the scale init customizable
            params_init_values[scale] = np.full_like(loc_init, 0.1)

            z = Normal(
                f"{rv.name}_z",
                mu=0.0,
                sigma=1.0,
                shape=value_shapes[rv],
            )
            Deterministic(
                rv.name,
                loc + pt.exp(scale) * z,
                dims=value_dims[rv],
            )

    return AutoGuideModel(guide_model, params_init_values)


@dataclass(frozen=True)
class AutoFullRankGuideModel(AutoGuideModel):
    """Guide model for a full-rank (multivariate normal) ADVI approximation."""

    def fit_quantities(self, params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Report the mean and the lower-triangular Cholesky factor of the covariance."""
        loc = params["loc"]
        n_dim = loc.size
        cholesky = np.zeros((n_dim, n_dim), dtype=loc.dtype)
        cholesky[np.tril_indices(n_dim)] = params["L_packed"]
        np.fill_diagonal(cholesky, np.exp(np.diagonal(cholesky)))
        return {"mean_vector": loc, "cholesky_lower": cholesky}

    def stochastic_logq(self, path_derivative_gradient: bool = True) -> pt.TensorVariable:
        """Joint logq of the full-rank guide, derived by logprob inference.

        ``latent`` is the whole unconstrained draw ``loc + L @ z`` *before* it is split into
        per-variable values. ``conditional_logp`` inverts the affine matmul (``MeasurableMatMul``)
        and recovers the joint ``MvNormal`` density in a single term. Valuing the per-variable
        slices instead would hit the non-measurable reshape/split, so we value the un-split draw.
        """
        latent = self.latent
        [logq] = conditional_logp({latent: latent}, warn_rvs=False).values()
        logq = logq.sum()

        if path_derivative_gradient:
            logq = graph_replace(logq, {p: disconnected_grad(p) for p in self.params})

        return logq


def AutoMultivariateNormal(model: Model, random_seed=None) -> AutoGuideModel:
    """Create a guide model for ADVI with a full-rank multivariate normal distribution.

    Unlike the mean-field :func:`AutoDiagonalNormal`, this guide captures posterior correlations:
    the joint of all free RVs (in the unconstrained value space) is approximated by a single
    ``MvNormal`` reparameterized as ``loc + L @ z`` with ``z`` standard normal and ``L`` a learnable
    lower-triangular Cholesky factor.

    Parameters
    ----------
    model : Model
        The probabilistic model for which to create the guide.
    random_seed : optional
        Seed passed to ``model.initial_point``.

    Returns
    -------
    guide_model : AutoGuideModel
        An AutoGuideModel whose logq is the joint multivariate normal density.
    """
    free_rvs = model.free_RVs
    _check_continuous_rvs(model, free_rvs)
    value_shapes, value_dims = get_value_shapes_and_dims(model)

    # DictToArrayBijection flattens the unconstrained initial point into the guide mean init;
    # point_map_info gives the per-RV layout (thus the order) of that flat vector.
    loc_init, point_map_info = DictToArrayBijection.map(
        model.initial_point(random_seed=random_seed)
    )
    n_dim = loc_init.size
    value_to_rv = {model.rvs_to_values[rv].name: rv for rv in free_rvs}
    ordered_rvs = [value_to_rv[name] for name, *_ in point_map_info]

    # Initialize the diagonal params at 0.1 (off-diagonal at 0): the full-rank guide starts as
    # the mean-field guide (diagonal scale exp(0.1)) and grows correlation structure.
    # Matches AutoDiagonalNormal's scale init.
    rows, cols = np.tril_indices(n_dim)
    L_packed_init = np.zeros(rows.size, dtype=loc_init.dtype)
    L_packed_init[rows == cols] = 0.1

    with Model(coords=model.coords, model=None) as guide_model:
        loc = pt.tensor("loc", shape=(None,))
        L_packed = pt.tensor("L_packed", shape=(None,))
        n = loc.shape[0]
        idx = pt.arange(n)

        L = pt.zeros((n, n))[pt.tril_indices(n)].set(L_packed)
        L = L[idx, idx].set(pt.exp(pt.diagonal(L)))  # positive diagonal
        # Promise the structure so the MeasurableMatMul logq's solve(L, .) / slogdet(L) lower
        # to the triangular routines instead of a general LU.
        L = assume(L, lower_triangular=True)

        z = Normal("z", mu=0.0, sigma=1.0, shape=(n,))
        latent = loc + L @ z
        latent.name = "latent"
        guide_model.add_named_variable(latent)
        parts = pt.unpack(latent, packed_shapes=[value_shapes[rv] for rv in ordered_rvs])
        for rv, part in zip(ordered_rvs, parts):
            Deterministic(rv.name, part, dims=value_dims[rv])

    return AutoFullRankGuideModel(guide_model, {loc: loc_init, L_packed: L_packed_init})


@dataclass(frozen=True)
class AutoLowRankGuideModel(AutoGuideModel):
    """Guide model for a low-rank-plus-diagonal multivariate normal ADVI approximation."""

    def fit_quantities(self, params: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        r"""Report the mean and the two terms of the covariance.

        The covariance is :math:`W W^{T} + \mathrm{diag}(d^{2})`, so ``cov_factor`` is
        :math:`W` and ``diagonal_standard_deviation`` is :math:`d` -- a standard deviation,
        squared before it enters the covariance.
        """
        return {
            "mean_vector": params["loc"],
            "cov_factor": params["cov_factor"],
            "diagonal_standard_deviation": np.exp(params["cov_diag_unconstrained"]),
        }

    def stochastic_logq(self, path_derivative_gradient: bool = True) -> pt.TensorVariable:
        """Joint logq of the low-rank guide, evaluated in closed form.

        The draw ``loc + W @ eps_k + d * eps_d`` mixes two independent noise sources (a
        convolution), which logprob inference cannot invert. We instead evaluate the
        ``MvNormal`` density of ``cov = W @ W.T + diag(d ** 2)`` directly, via the Woodbury
        identity and the matrix-determinant lemma, so the cost is ``O(D * K ** 2)`` rather than
        ``O(D ** 3)``.
        """
        u = self.latent
        loc = self["loc"]
        W = self["cov_factor"]  # shape (D, K)
        d = pt.exp(self["cov_diag_unconstrained"])  # shape (D,), positive
        n_dim = u.shape[0]
        rank = W.shape[1]

        delta = u - loc
        d_inv = 1.0 / d**2
        # Woodbury capacitance matrix I_K + W.T @ D_inv @ W (K x K, symmetric PD)
        capacitance = pt.eye(rank) + (W * d_inv[:, None]).T @ W
        # cholesky outputs are auto-recognized as lower-triangular by pytensor's linalg rewrites
        chol_cap = cholesky(capacitance, lower=True)
        rhs = W.T @ (delta * d_inv)
        sol = solve_triangular(chol_cap, rhs, lower=True)

        quad = (delta**2 * d_inv).sum() - (sol**2).sum()
        logdet = 2.0 * pt.log(d).sum() + 2.0 * pt.log(pt.diagonal(chol_cap)).sum()
        logq = -0.5 * quad - 0.5 * logdet - 0.5 * n_dim * np.log(2 * np.pi)

        if path_derivative_gradient:
            logq = graph_replace(logq, {p: disconnected_grad(p) for p in self.params})

        return logq


def AutoLowRankMultivariateNormal(
    model: Model, rank: int | None = None, random_seed=None
) -> AutoGuideModel:
    """Create a guide model for ADVI with a low-rank-plus-diagonal multivariate normal.

    Approximates the joint posterior (in the unconstrained value space) by a ``MvNormal`` whose
    covariance is ``W @ W.T + diag(d ** 2)``, with ``W`` of shape ``(D, rank)``. This captures the
    leading ``rank`` correlation directions with ``O(D * rank)`` parameters, scaling to large ``D``
    where the full-rank guide's ``O(D ** 2)`` is prohibitive. The draw is reparameterized as
    ``loc + W @ eps_k + d * eps_d``; its logq is evaluated in closed form via Woodbury (see
    :meth:`AutoLowRankGuideModel.stochastic_logq`).

    Parameters
    ----------
    model : Model
        The probabilistic model for which to create the guide.
    rank : int, optional
        Rank of the low-rank covariance factor. Defaults to ``round(sqrt(D))`` (clamped to
        ``[1, D]``), where ``D`` is the total unconstrained dimension. This matches the default
        rank used by NumPyro's and Pyro's low-rank guides.
    random_seed : optional
        Seed passed to ``model.initial_point``.

    Returns
    -------
    guide_model : AutoGuideModel
        An AutoGuideModel whose logq is the low-rank multivariate normal density.
    """
    free_rvs = model.free_RVs
    _check_continuous_rvs(model, free_rvs)
    value_shapes, value_dims = get_value_shapes_and_dims(model)

    loc_init, point_map_info = DictToArrayBijection.map(
        model.initial_point(random_seed=random_seed)
    )
    n_dim = loc_init.size
    value_to_rv = {model.rvs_to_values[rv].name: rv for rv in free_rvs}
    ordered_rvs = [value_to_rv[name] for name, *_ in point_map_info]

    if rank is None:
        rank = round(n_dim**0.5)
    rank = max(1, min(rank, n_dim))

    # W starts at 0 (no correlation) and d at exp(0.1): the guide starts as the mean-field guide.
    W_init = np.zeros((n_dim, rank), dtype=loc_init.dtype)
    d_unconstrained_init = np.full(n_dim, 0.1, dtype=loc_init.dtype)

    with Model(coords=model.coords, model=None) as guide_model:
        loc = pt.tensor("loc", shape=(None,))
        W = pt.tensor("cov_factor", shape=(None, rank))
        d_unconstrained = pt.tensor("cov_diag_unconstrained", shape=(None,))
        d = pt.exp(d_unconstrained)
        n = loc.shape[0]

        eps_k = Normal("eps_k", mu=0.0, sigma=1.0, shape=(rank,))
        eps_d = Normal("eps_d", mu=0.0, sigma=1.0, shape=(n,))
        latent = loc + W @ eps_k + d * eps_d
        latent.name = "latent"
        guide_model.add_named_variable(latent)
        parts = pt.unpack(latent, packed_shapes=[value_shapes[rv] for rv in ordered_rvs])
        for rv, part in zip(ordered_rvs, parts):
            Deterministic(rv.name, part, dims=value_dims[rv])

    return AutoLowRankGuideModel(
        guide_model, {loc: loc_init, W: W_init, d_unconstrained: d_unconstrained_init}
    )
