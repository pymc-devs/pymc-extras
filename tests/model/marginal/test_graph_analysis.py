import pytensor.tensor as pt
import pytest

from pymc.distributions import CustomDist
from pymc.variational.minibatch_rv import create_minibatch_rv
from pytensor.tensor.type_other import NoneTypeT

from pymc_extras.model.marginal.graph_analysis import (
    is_conditional_dependent,
    subgraph_batch_dim_connection,
)


def test_is_conditional_dependent_static_shape():
    """Test that we don't consider dependencies through "constant" shape Ops"""
    x1 = pt.matrix("x1", shape=(None, 5))
    y1 = pt.random.normal(size=pt.shape(x1))
    assert is_conditional_dependent(y1, x1, [x1, y1])

    x2 = pt.matrix("x2", shape=(9, 5))
    y2 = pt.random.normal(size=pt.shape(x2))
    assert not is_conditional_dependent(y2, x2, [x2, y2])


class TestSubgraphBatchDimConnection:
    def test_dimshuffle(self):
        inp = pt.tensor(shape=(5, 1, 4, 3))
        out1 = pt.matrix_transpose(inp)
        out2 = pt.expand_dims(inp, 1)
        out3 = pt.squeeze(inp)
        [dims1, dims2, dims3] = subgraph_batch_dim_connection(inp, [out1, out2, out3])
        assert dims1 == (0, 1, 3, 2)
        assert dims2 == (0, None, 1, 2, 3)
        assert dims3 == (0, 2, 3)

    def test_careduce(self):
        inp = pt.tensor(shape=(4, 3, 2))

        out = pt.sum(inp[:, None], axis=(1,))
        [dims] = subgraph_batch_dim_connection(inp, [out])
        assert dims == (0, 1, 2)

        invalid_out = pt.sum(inp, axis=(1,))
        with pytest.raises(ValueError, match="Use of known dimensions"):
            subgraph_batch_dim_connection(inp, [invalid_out])

    def test_subtensor(self):
        inp = pt.tensor(shape=(4, 3, 2))

        invalid_out = inp[0, :1]
        with pytest.raises(
            ValueError,
            match="Partial slicing or indexing of known dimensions not supported",
        ):
            subgraph_batch_dim_connection(inp, [invalid_out])

        # If we are selecting dummy / unknown dimensions that's fine
        valid_out = pt.expand_dims(inp, (0, 1))[0, :1]
        [dims] = subgraph_batch_dim_connection(inp, [valid_out])
        assert dims == (None, 0, 1, 2)

    def test_advanced_subtensor_value(self):
        inp = pt.tensor(shape=(2, 4))
        intermediate_out = inp[:, None, :, None] + pt.zeros((2, 3, 4, 5))

        # Index on an unlabled dim introduced by broadcasting with zeros
        out = intermediate_out[:, [0, 0, 1, 2]]
        [dims] = subgraph_batch_dim_connection(inp, [out])
        assert dims == (0, None, 1, None)

        # Indexing that introduces more dimensions
        out = intermediate_out[:, [[0, 0], [1, 2]], :]
        [dims] = subgraph_batch_dim_connection(inp, [out])
        assert dims == (0, None, None, 1, None)

        # Special case where advanced dims are moved to the front of the output
        out = intermediate_out[:, [0, 0, 1, 2], :, 0]
        [dims] = subgraph_batch_dim_connection(inp, [out])
        assert dims == (None, 0, 1)

        # Indexing on a labeled dim fails
        out = intermediate_out[:, :, [0, 0, 1, 2]]
        with pytest.raises(ValueError, match="Partial slicing or advanced integer indexing"):
            subgraph_batch_dim_connection(inp, [out])

    def test_advanced_subtensor_key(self):
        inp = pt.tensor(shape=(5, 5), dtype=int)
        base = pt.zeros((2, 3, 4))

        out = base[inp]
        [dims] = subgraph_batch_dim_connection(inp, [out])
        assert dims == (0, 1, None, None)

        out = base[:, :, inp]
        [dims] = subgraph_batch_dim_connection(inp, [out])
        assert dims == (
            None,
            None,
            0,
            1,
        )

        out = base[1:, 0, inp]
        [dims] = subgraph_batch_dim_connection(inp, [out])
        assert dims == (None, 0, 1)

        # Special case where advanced dims are moved to the front of the output
        out = base[0, :, inp]
        [dims] = subgraph_batch_dim_connection(inp, [out])
        assert dims == (0, 1, None)

        # Mix keys dimensions
        out = base[:, inp, inp.T]
        with pytest.raises(ValueError, match="Different known dimensions mixed via broadcasting"):
            subgraph_batch_dim_connection(inp, [out])

    def test_elemwise(self):
        inp = pt.tensor(shape=(5, 5))

        out = inp + inp
        [dims] = subgraph_batch_dim_connection(inp, [out])
        assert dims == (0, 1)

        out = inp + inp.T
        with pytest.raises(ValueError, match="Different known dimensions mixed via broadcasting"):
            subgraph_batch_dim_connection(inp, [out])

        out = inp[None, :, None, :] + inp[:, None, :, None]
        with pytest.raises(
            ValueError, match="Same known dimension used in different axis after broadcasting"
        ):
            subgraph_batch_dim_connection(inp, [out])

    def test_blockwise(self):
        inp = pt.tensor(shape=(5, 4))

        invalid_out = inp @ pt.ones((4, 3))
        with pytest.raises(ValueError, match="Use of known dimensions"):
            subgraph_batch_dim_connection(inp, [invalid_out])

        out = (inp[:, :, None, None] + pt.zeros((2, 3))) @ pt.ones((2, 3))
        [dims] = subgraph_batch_dim_connection(inp, [out])
        assert dims == (0, 1, None, None)

    def test_random_variable(self):
        inp = pt.tensor(shape=(5, 4, 3))

        out1 = pt.random.normal(loc=inp)
        out2 = pt.random.categorical(p=inp[..., None])
        out3 = pt.random.multivariate_normal(mean=inp[..., None], cov=pt.eye(1))
        [dims1, dims2, dims3] = subgraph_batch_dim_connection(inp, [out1, out2, out3])
        assert dims1 == (0, 1, 2)
        assert dims2 == (0, 1, 2)
        assert dims3 == (0, 1, 2, None)

        invalid_out = pt.random.categorical(p=inp)
        with pytest.raises(ValueError, match="Use of known dimensions"):
            subgraph_batch_dim_connection(inp, [invalid_out])

        invalid_out = pt.random.multivariate_normal(mean=inp, cov=pt.eye(3))
        with pytest.raises(ValueError, match="Use of known dimensions"):
            subgraph_batch_dim_connection(inp, [invalid_out])

    def test_minibatched_random_variable(self):
        inp = pt.tensor(shape=(4, 3, 2))
        out1 = pt.random.normal(loc=inp)
        out2 = create_minibatch_rv(out1, total_size=(10, 10, 10))
        [dims1] = subgraph_batch_dim_connection(inp, [out2])
        assert dims1 == (0, 1, 2)

    def test_symbolic_random_variable(self):
        inp = pt.tensor(shape=(4, 3, 2))

        # Test univariate
        out = CustomDist.dist(
            inp,
            dist=lambda mu, size: pt.random.normal(loc=mu, size=size),
        )
        [dims] = subgraph_batch_dim_connection(inp, [out])
        assert dims == (0, 1, 2)

        # Test multivariate
        def dist(mu, size):
            if isinstance(size.type, NoneTypeT):
                size = mu.shape
            return pt.random.normal(loc=mu[..., None], size=(*size, 2))

        out = CustomDist.dist(inp, dist=dist, size=(4, 3, 2), signature="()->(2)")
        [dims] = subgraph_batch_dim_connection(inp, [out])
        assert dims == (0, 1, 2, None)
