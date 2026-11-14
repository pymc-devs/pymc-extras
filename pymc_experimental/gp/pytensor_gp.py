from collections.abc import Sequence

import pymc as pm
import pytensor.tensor as pt

from pymc.distributions.distribution import Continuous
from pymc.model.fgraph import fgraph_from_model, model_free_rv, model_from_fgraph
from pytensor import Variable
from pytensor.compile.builders import OpFromGraph


class GPCovariance(OpFromGraph):
    """OFG representing a GP covariance"""

    @staticmethod
    def square_dist(X, Xs, ls):
        assert X.ndim == 2, "Complain to Bill about it"
        assert Xs.ndim == 2, "Complain to Bill about it"

        X = X / ls
        Xs = Xs / ls

        X2 = pt.sum(pt.square(X), axis=-1)
        Xs2 = pt.sum(pt.square(Xs), axis=-1)

        sqd = -2.0 * X @ Xs.mT + (X2[..., :, None] + Xs2[..., None, :])
        return pt.clip(sqd, 0, pt.inf)


class ExpQuadCov(GPCovariance):
    """
    ExpQuad covariance function
    """

    @classmethod
    def exp_quad_full(cls, X, Xs, ls):
        return pt.exp(-0.5 * cls.square_dist(X, Xs, ls))

    @classmethod
    def build_covariance(cls, X, Xs=None, *, ls):
        X = pt.as_tensor(X)
        if Xs is None:
            Xs = X
        else:
            Xs = pt.as_tensor(Xs)
        ls = pt.as_tensor(ls)

        out = cls.exp_quad_full(X, Xs, ls)
        if Xs is X:
            return cls(inputs=[X, ls], outputs=[out])(X, ls)
        else:
            return cls(inputs=[X, Xs, ls], outputs=[out])(X, Xs, ls)


def ExpQuad(X, X_new=None, *, ls=1.0):
    return ExpQuadCov.build_covariance(X, X_new, ls=ls)


class GP_RV(pm.MvNormal.rv_type):
    name = "gaussian_process"
    signature = "(n),(n,n)->(n)"
    dtype = "floatX"
    _print_name = ("GP", "\\operatorname{GP}")


class GP(Continuous):
    rv_type = GP_RV
    rv_op = GP_RV()

    @classmethod
    def dist(cls, cov, **kwargs):
        cov = pt.as_tensor(cov)
        mu = pt.zeros(cov.shape[-1])
        return super().dist([mu, cov], **kwargs)


def conditional_gp(
    model,
    gp: Variable | str,
    Xnew,
    *,
    jitter=1e-6,
    dims: Sequence[str] = (),
    inline: bool = False,
):
    """
    Condition a GP on new data.

    Parameters
    ----------
    model: Model
    gp: Variable | str
        The GP to condition on.
    Xnew: Tensor-like
        New data to condition the GP on.
    jitter: float, default=1e-6
        Jitter to add to the new GP covariance matrix.
    dims: Sequence[str], default=()
        Dimensions of the new GP.
    inline: bool, default=False
        Whether to inline the new GP in place of the old one. This is not always a safe operation.
        If True, any variables that depend on the GP will be updated to depend on the new GP.

    Returns
    -------
    Conditional model: Model
        A new model with a GP free RV named f"{gp.name}_star" conditioned on the new data.

    """

    def _build_conditional(Xnew, f, cov, jitter):
        if not isinstance(cov.owner.op, GPCovariance):
            # TODO: Look for xx kernels in the ancestors of f
            raise NotImplementedError(f"Cannot build conditional of {cov.owner.op} operation")

        X, ls = cov.owner.inputs

        Kxx = cov
        # Kxs = toposort_replace(cov, tuple(zip(xx_kernels, xs_kernels)), rebuild=True)
        Kxs = cov.owner.op.build_covariance(X, Xnew, ls=ls)
        # Kss = toposort_replace(cov, tuple(zip(xx_kernels, ss_kernels)), rebuild=True)
        Kss = cov.owner.op.build_covariance(Xnew, ls=ls)

        L = pt.linalg.cholesky(Kxx + pt.eye(X.shape[0]) * jitter)
        # TODO: Use cho_solve
        A = pt.linalg.solve_triangular(L, Kxs, lower=True)
        v = pt.linalg.solve_triangular(L, f, lower=True)

        mu = (A.mT @ v).T  # Vector?
        cov = Kss - (A.mT @ A)

        return mu, cov

    if isinstance(gp, Variable):
        assert model[gp.name] is gp
    else:
        gp = model[gp.name]

    fgraph, memo = fgraph_from_model(model)
    gp_model_var = memo[gp]
    gp_rv = gp_model_var.owner.inputs[0]

    if isinstance(gp_rv.owner.op, pm.MvNormal.rv_type):
        _, cov = gp_rv.owner.op.dist_params(gp.owner)
    else:
        raise NotImplementedError("Can only condition on pure GPs")

    mu_star, cov_star = _build_conditional(Xnew, gp_model_var, cov, jitter)
    gp_rv_star = pm.MvNormal.dist(mu_star, cov_star, name=f"{gp.name}_star")

    value = gp_rv_star.clone()
    transform = None
    gp_model_var_star = model_free_rv(gp_rv_star, value, transform, *dims)

    if inline:
        fgraph.replace(gp_model_var, gp_model_var_star, import_missing=True)
    else:
        fgraph.add_output(gp_model_var_star, import_missing=True)

    return model_from_fgraph(fgraph, mutate_fgraph=True)
