import pytensor.tensor as pt


def bmv(A, x):
    return pt.matmul(A, x[..., None])[..., 0]
