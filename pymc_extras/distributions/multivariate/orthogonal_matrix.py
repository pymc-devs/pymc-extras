#   Copyright 2025 The PyMC Developers
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

import pytensor.tensor as pt

from pytensor.tensor import TensorVariable

from pymc_extras.distributions.multivariate.spherical import Spherical

__all__ = ["SemiOrthogonalMatrix"]


class SemiOrthogonalMatrix:
    def __new__(cls, name, D, Q, **kwargs):
        dof = D * Q - Q * (Q - 1) // 2  # Total degrees of freedom

        vs, pos = pt.zeros(dof), 0
        for q in range(Q):
            vq = Spherical(f"{name}_v{q}", D - q)
            vs = pt.set_subtensor(vs[pos : pos + D - q], vq)
            pos += D - q

        return cls.orth_from_vs(vs, D, Q)

    # Create a householder matrix from a vector
    @classmethod
    def _householder_matrix(cls, v: TensorVariable, D: int) -> TensorVariable:
        Q = v.shape[0]
        H = pt.eye(D)
        sgn = 1.0  # Original paper recommends sign(v[0]) but that causes divergences
        u = pt.inc_subtensor(v[0], sgn * pt.linalg.norm(v))
        H = pt.set_subtensor(
            H[-Q:, -Q:], -sgn * (pt.eye(Q, Q) - 2 * u[:, None] * u[None, :] / (pt.dot(u, u) + 1e-6))
        )
        return H

    # Construct an orthogonal matrix from a vector of normally distributed values
    # as a cumulative product of householder matrices
    @classmethod
    def orth_from_vs(cls, vs: TensorVariable, D: int, Q: int) -> TensorVariable:
        """Construct an orthogonal matrix from a set of direction vectors v"""
        H_p = pt.eye(D)
        pos, q = 0, 0
        dof = D * Q - Q * (Q - 1) // 2
        while pos < dof:
            v = vs[pos : pos + D - q]
            H = cls._householder_matrix(v, D)
            H_p = H @ H_p
            pos += D - q
            q += 1
        return H_p[:q, :]

    @classmethod
    def vs_from_orth(cls, U: TensorVariable, D: int, Q: int) -> TensorVariable:
        """Get the vs values that would lead to orthogonal matrix U. Inverse of orth_from_vs"""
        vs = []
        vl = D * Q - Q * (Q - 1) // 2
        vs, pos = pt.zeros(vl), 0
        for q in range(Q):
            v = U[q:, q]  # Top row of the remaining submatrix

            vs = pt.set_subtensor(vs[pos : pos + D - q], v)
            H = cls._householder_matrix(v, D)
            U = H.dot(U)
            pos += D - q
        return vs
