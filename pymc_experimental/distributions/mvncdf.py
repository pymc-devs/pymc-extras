import aesara
import aesara.tensor as at
from aesara.graph.basic import Apply
from aesara.graph.op import Op
from aesara.tensor.type import TensorType
from aesara.tensor.random.basic import MvNormalRV
from aeppl.logprob import _logcdf
from scipy.stats import multivariate_normal

# Op calculates mvncdf


def conditional_covariance(Sigma, mu, conditioned, conditioned_values):

    """
    Sigma: d x d covariance matrix
    mu: mean, array length d
    conditioned: array of dimensions to condition on
    conditioned_values: array of conditional values equal in length to `conditioned`
    """

    target = (1 - conditioned).astype(bool)

    Sigma_22 = Sigma[conditioned][:, conditioned]
    Sigma_21 = Sigma[conditioned][:, target]
    Sigma_12 = Sigma[target][:, conditioned]
    Sigma_11 = Sigma[target][:, target]

    inv = pm.math.matrix_inverse(Sigma_22)
    Sigma_cond = Sigma_11 - pm.math.matrix_dot(Sigma_12, inv, Sigma_21)
    mean_cond = pm.math.matrix_dot(Sigma_12, inv, conditioned_values)

    return Sigma_cond, mean_cond


class Mvncdf(Op):

    def make_node(self, value, mu, cov):
        value = at.as_tensor_variable(value)
        mu = at.as_tensor_variable(mu)
        cov = at.as_tensor_variable(cov)

        return Apply(self, [value, mu, cov], [value[0].type()])

    def perform(self, node, inputs, output_storage):
        ## input and output should be np arrays
        value, mu, cov = inputs
        output_storage[0][0] = multivariate_normal.logcdf(value, mu, cov)

        

    #def infer_shape(self, fgraph, node, i0_shapes):
    #    return i0_shapes[0]

mvncdf = Mvncdf()

@_logcdf.register(MvNormalRV)
def mvnormal_logcdf(op, value, rng, size, dtype, mu, cov, **kwargs):
    return mvncdf(value, mu, cov)