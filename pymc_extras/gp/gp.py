"""A GP prior is an MvNormal over stacked inputs.

The design commitment: `GP` is a thin constructor, not a new distribution and
not an object that owns inference. Everything downstream (marginalization,
conditionals, posterior predictive) is the generic linear-Gaussian machinery in
`pymc_extras.model.marginal`, which knows nothing about GPs.

Partitioning uses stock `pt.pack` / `pt.unpack`: define the prior jointly over
every input set you care about (training points, prediction points, inducing
points), then "observed at the training points" is just `f_train`, a slice.
Slicing is affine, which is all the marginalization needs.

`packed_shapes` are symbolic graphs, not plain sizes, and when a part is a
`pm.Data` they reference that shared variable. Every pymc model transform goes
through `clone_model`, which clones shared variables, so shapes taken from the
original model keep reading the original model's data. Re-derive them against
the transformed model -- `pt.pack(X, cond_m["X_pred"], keep_axes=-1)` -- before
unpacking anything that came out of `marginalize` or `conditional`.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from pymc_extras.model.marginal import conditional, marginalize  # noqa: F401

# importing `marginal` above registers the linear-Gaussian conjugacy rewrite


def _as_2d(X):
    X = pt.as_tensor(X)
    return X[:, None] if X.ndim == 1 else X


def GP(name, X, cov, mean=0.0, jitter=1e-6, dims=None, model=None):
    """Register a GP prior over the inputs `X`.

    Parameters
    ----------
    X : array-like or tensor
        Input locations, typically the first output of `pack`.
    cov : Covariance
        Kernel object (see `pymc_extras.gp.kernels`), called as ``cov(X)``.
    mean : scalar, tensor, or callable
        Constant, vector, or a mean function called as ``mean(X)``.
    """
    model = pm.modelcontext(model)

    X = _as_2d(X)
    n = X.shape[0]
    K = cov(X) + jitter * pt.eye(n)
    mu = mean(X) if callable(mean) else pt.broadcast_to(pt.as_tensor(mean), (n,))

    rv = pm.MvNormal(name, mu=mu, cov=K, dims=dims, model=model)
    rv.X = X
    rv.cov = cov
    return rv


def conditional_moments(model, name="gp"):
    """``(mu, cov)`` of a recovered variable's conditional, as symbolic tensors.

    `conditional` hands back a *model* in which `name` is a free RV whose
    distribution is the conditional. Its parameters are the predictive
    moments, still symbolic in the hyperparameters.
    """
    rv = model[name]
    mu, cov = rv.owner.op.dist_params(rv.owner)
    return mu, cov


def predictive_fn(model, outs, name="gp"):
    """Compile ``outs`` into a callable taking a posterior point.

    Two wrinkles this papers over: the outputs must have RVs swapped for value
    variables before compiling, and the recovered variable is itself a free RV
    of `model`, so its value var is an unused input.
    """
    outs = model.replace_rvs_by_values(list(outs))
    fn = model.compile_fn(outs, inputs=model.value_vars, point_fn=True, on_unused_input="ignore")
    dummy = {model.rvs_to_values[model[name]].name: np.zeros(model[name].type.shape or ())}

    def call(point):
        return fn({**dummy, **point})

    return call


def project(gp, X_new, jitter=1e-6):
    """Noise-free projection of a GP onto new inputs: ``A @ gp``.

    ``A = K(X_new, Z) K(Z, Z)^-1`` where ``Z`` are `gp`'s own inputs. This is
    the sparse-GP building block: put a GP on a small set of inducing inputs
    and push it onto the data. The result is an *affine* function of `gp`,
    which is exactly what the linear-Gaussian marginalization handles, so
    sparse approximations need no approximation-specific machinery.
    """
    Z, cov = gp.X, gp.cov
    Kzz = cov(Z) + jitter * pt.eye(Z.shape[0])
    Kxz = cov(_as_2d(X_new), Z)

    L = pt.linalg.cholesky(Kzz)
    A = pt.linalg.solve_triangular(
        L.T, pt.linalg.solve_triangular(L, Kxz.T, lower=True), lower=False
    ).T
    return A @ gp


def prior_variance_correction(gp, X_new, jitter=1e-6):
    """``diag(K(X_new) - Q(X_new))``, the variance `project` throws away.

    Adding this to the observation noise turns DTC into FITC.
    """
    Z, cov = gp.X, gp.cov
    X_new = _as_2d(X_new)

    Kzz = cov(Z) + jitter * pt.eye(Z.shape[0])
    L = pt.linalg.cholesky(Kzz)
    V = pt.linalg.solve_triangular(L, cov(X_new, Z).T, lower=True)

    return pt.clip(pt.diag(cov(X_new)) - pt.sum(V**2, axis=0), 0.0, pt.inf)
