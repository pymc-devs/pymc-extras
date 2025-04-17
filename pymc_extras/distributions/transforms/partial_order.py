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
import numpy as np
import pytensor.tensor as pt

from pymc.logprob.transforms import Transform

__all__ = ["PartialOrder"]


def dtype_minval(dtype):
    """Find the minimum value for a given dtype"""
    return np.iinfo(dtype).min if np.issubdtype(dtype, np.integer) else np.finfo(dtype).min


def padded_where(x, to_len, padval=-1):
    """A padded version of np.where"""
    w = np.where(x)
    return np.concatenate([w[0], np.full(to_len - len(w[0]), padval)])


class PartialOrder(Transform):
    """Create a PartialOrder transform

    A more flexible version of the pymc ordered transform that
    allows specifying a (strict) partial order on the elements.

    Examples
    --------
    .. code:: python

        import numpy as np
        import pymc as pm
        import pymc_extras as pmx

        # Define two partial orders on 4 elements
        # am[i,j] = 1 means i < j
        adj_mats = np.array([
            # 0 < {1, 2} < 3
            [[0, 1, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0]],

            # 1 < 0 < 3 < 2
            [[0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0]],
        ])

        # Create the partial order from the adjacency matrices
        po = pmx.PartialOrder(adj_mats)

        with pm.Model() as model:
            # Generate 3 samples from both partial orders
            pm.Normal("po_vals", shape=(3,2,4), transform=po,
                        initval=po.initvals((3,2,4)))

            idata = pm.sample()

        # Verify that for first po, the zeroth element is always the smallest
        assert (idata.posterior['po_vals'][:,:,:,0,0] <
                idata.posterior['po_vals'][:,:,:,0,1:]).all()

        # Verify that for second po, the second element is always the largest
        assert (idata.posterior['po_vals'][:,:,:,1,2] >=
                idata.posterior['po_vals'][:,:,:,1,:]).all()

    Technical notes
    ----------------
    Partial order needs to be strict, i.e. without equalities.
    A DAG defining the partial order is sufficient, as transitive closure is automatically computed.
    Code works in O(N*D) in runtime, but takes O(N^3) in initialization,
    where N is the number of nodes in the dag and D is the maximum
    in-degree of a node in the transitive reduction.
    """

    name = "partial_order"

    def __init__(self, adj_mat):
        """
        Initialize the PartialOrder transform

        Parameters
        ----------
        adj_mat: ndarray
            adjacency matrix for the DAG that generates the partial order,
            where ``adj_mat[i][j] = 1`` denotes ``i < j``.
            Note this also accepts multiple DAGs if RV is multidimensional
        """

        # Basic input checks
        if adj_mat.ndim < 2:
            raise ValueError("Adjacency matrix must have at least 2 dimensions")
        if adj_mat.shape[-2] != adj_mat.shape[-1]:
            raise ValueError("Adjacency matrix is not square")
        if adj_mat.min() != 0 or adj_mat.max() != 1:
            raise ValueError("Adjacency matrix must contain only 0s and 1s")

        # Create index over the first ellipsis dimensions
        idx = np.ix_(*[np.arange(s) for s in adj_mat.shape[:-2]])

        # Transitive closure using Floyd-Warshall
        tc = adj_mat.astype(bool)
        for k in range(tc.shape[-1]):
            tc |= np.logical_and(tc[..., :, k, None], tc[..., None, k, :])

        # Check if the dag is acyclic
        if np.any(tc.diagonal(axis1=-2, axis2=-1)):
            raise ValueError("Partial order contains equalities")

        # Transitive reduction using the closure
        # This gives the minimum description of the partial order
        # This is to minmax the input degree
        adj_mat = tc * (1 - np.matmul(tc, tc))

        # Find the maximum in-degree of the reduced dag
        dag_idim = adj_mat.sum(axis=-2).max()

        # Topological sort
        ts_inds = np.zeros(adj_mat.shape[:-1], dtype=int)
        dm = adj_mat.copy()
        for i in range(adj_mat.shape[1]):
            assert dm.sum(axis=-2).min() == 0  # DAG is acyclic
            nind = np.argmin(dm.sum(axis=-2), axis=-1)
            dm[(*idx, slice(None), nind)] = 1  # Make nind not show up again
            dm[(*idx, nind, slice(None))] = 0  # Allow it's children to show
            ts_inds[(*idx, i)] = nind
        self.ts_inds = ts_inds

        # Change the dag to adjacency lists (with -1 for NA)
        dag_T = np.apply_along_axis(padded_where, axis=-2, arr=adj_mat, padval=-1, to_len=dag_idim)
        self.dag = np.swapaxes(dag_T, -2, -1)
        self.is_start = np.all(self.dag[..., :, :] == -1, axis=-1)

    def initvals(self, shape=None, lower=-1, upper=1):
        """
        Create a set of appropriate initial values for the variable.
        NB! It is important that proper initial values are used,
        as only properly ordered values are in the range of the transform.

        Parameters
        ----------
        shape: tuple, default None
            shape of the initial values. If None, adj_mat[:-1] is used
        lower: float, default -1
            lower bound for the initial values
        upper: float, default 1
            upper bound for the initial values

        Returns
        -------
        vals: ndarray
            initial values for the transformed variable
        """

        if shape is None:
            shape = self.dag.shape[:-1]

        if shape[-len(self.dag.shape[:-1]) :] != self.dag.shape[:-1]:
            raise ValueError("Shape must match the shape of the adjacency matrix")

        # Create the initial values
        vals = np.linspace(lower, upper, self.dag.shape[-2])
        inds = np.argsort(self.ts_inds, axis=-1)
        ivals = vals[inds]

        # Expand the initial values to the extra dimensions
        extra_dims = shape[: -len(self.dag.shape[:-1])]
        ivals = np.tile(ivals, extra_dims + tuple([1] * len(self.dag.shape[:-1])))

        return ivals

    def backward(self, value, *inputs):
        minv = dtype_minval(value.dtype)
        x = pt.concatenate(
            [pt.zeros_like(value), pt.full(value.shape[:-1], minv)[..., None]], axis=-1
        )

        # Indices to allow broadcasting the max over the last dimension
        idx = np.ix_(*[np.arange(s) for s in self.dag.shape[:-2]])
        idx2 = tuple(np.tile(i[:, None], self.dag.shape[-1]) for i in idx)

        # Has to be done stepwise as next steps depend on previous values
        # Also has to be done in topological order, hence the ts_inds
        for i in range(self.dag.shape[-2]):
            tsi = self.ts_inds[..., i]
            if len(tsi.shape) == 0:
                tsi = int(tsi)  # if shape 0, it's a scalar
            ni = (*idx, tsi)  # i-th node in topological order
            eni = (Ellipsis, *ni)
            ist = self.is_start[ni]

            mval = pt.max(x[(Ellipsis, *idx2, self.dag[ni])], axis=-1)
            x = pt.set_subtensor(x[eni], ist * value[eni] + (1 - ist) * (mval + pt.exp(value[eni])))
        return x[..., :-1]

    def forward(self, value, *inputs):
        y = pt.zeros_like(value)

        minv = dtype_minval(value.dtype)
        vx = pt.concatenate([value, pt.full(value.shape[:-1], minv)[..., None]], axis=-1)

        # Indices to allow broadcasting the max over the last dimension
        idx = np.ix_(*[np.arange(s) for s in self.dag.shape[:-2]])
        idx = tuple(np.tile(i[:, None, None], self.dag.shape[-2:]) for i in idx)

        y = self.is_start * value + (1 - self.is_start) * (
            pt.log(value - pt.max(vx[(Ellipsis, *idx, self.dag[..., :])], axis=-1))
        )

        return y

    def log_jac_det(self, value, *inputs):
        return pt.sum(value * (1 - self.is_start), axis=-1)
