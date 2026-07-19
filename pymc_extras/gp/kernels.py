"""Composable covariance functions.

A kernel is a callable object, not a matrix: `k(X)` gives the full covariance,
`k(X, Xs)` the cross-covariance. Keeping the *function* rather than freezing a
matrix is what lets a model be re-evaluated at new inputs later.

Kernels compose with `+` and `*`, and scale by scalars (including model RVs).
`ls` accepts a vector for ARD; `input_dim` is inferred from `X`.
"""

import pytensor.tensor as pt


class Covariance:
    def __call__(self, X, Xs=None):
        raise NotImplementedError

    def __add__(self, other):
        return Add(self, other)

    __radd__ = __add__

    def __mul__(self, other):
        return Prod(self, other)

    __rmul__ = __mul__


def _as_cov(x):
    return x if isinstance(x, Covariance) else Constant(x)


class Add(Covariance):
    def __init__(self, a, b):
        self.a, self.b = _as_cov(a), _as_cov(b)

    def __call__(self, X, Xs=None):
        return self.a(X, Xs) + self.b(X, Xs)


class Prod(Covariance):
    def __init__(self, a, b):
        self.a, self.b = _as_cov(a), _as_cov(b)

    def __call__(self, X, Xs=None):
        return self.a(X, Xs) * self.b(X, Xs)


class Constant(Covariance):
    """Constant kernel; also how a scalar amplitude enters a product."""

    def __init__(self, c):
        self.c = c

    def __call__(self, X, Xs=None):
        X = pt.as_tensor(X)
        Xs = X if Xs is None else pt.as_tensor(Xs)
        return pt.full((X.shape[0], Xs.shape[0]), self.c)


class WhiteNoise(Covariance):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X, Xs=None):
        X = pt.as_tensor(X)
        if Xs is not None:
            # white noise has no cross-covariance between distinct input sets
            return pt.zeros((X.shape[0], pt.as_tensor(Xs).shape[0]))
        return pt.eye(X.shape[0]) * self.sigma**2


class Stationary(Covariance):
    """Base for kernels that depend on X only through scaled distances."""

    def __init__(self, ls, active_dims=None):
        self.ls = ls
        self.active_dims = active_dims

    def _slice(self, X):
        X = pt.as_tensor(X)
        if X.ndim == 1:
            X = X[:, None]
        if self.active_dims is not None:
            X = X[:, self.active_dims]
        return X

    def _euclidean(self, X, Xs):
        X = self._slice(X) / self.ls
        Xs = X if Xs is None else self._slice(Xs) / self.ls
        d2 = pt.sum(X**2, axis=1)[:, None] + pt.sum(Xs**2, axis=1)[None, :] - 2 * X @ Xs.T
        return pt.sqrt(pt.clip(d2, 1e-12, pt.inf))

    def __call__(self, X, Xs=None):
        return self._k(self._euclidean(X, Xs))

    def _k(self, r):
        raise NotImplementedError


class ExpQuad(Stationary):
    def _k(self, r):
        return pt.exp(-0.5 * r**2)


class Matern32(Stationary):
    def _k(self, r):
        s = pt.sqrt(3.0) * r
        return (1.0 + s) * pt.exp(-s)


class Matern52(Stationary):
    def _k(self, r):
        s = pt.sqrt(5.0) * r
        return (1.0 + s + 5.0 / 3.0 * r**2) * pt.exp(-s)
