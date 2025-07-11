import numpy as np

from pytensor import tensor as pt


def order_to_mask(order):
    if isinstance(order, int):
        return np.ones(order).astype(bool)
    else:
        return np.array(order).astype(bool)


def _frequency_transition_block(s, j):
    lam = 2 * np.pi * j / s

    return pt.stack([[pt.cos(lam), pt.sin(lam)], [-pt.sin(lam), pt.cos(lam)]])
