import numpy as np
import pytest

from pytensor import function
from pytensor import tensor as pt

from pymc_extras.statespace.models.utilities import (
    add_tensors_by_dim_labels,
    join_tensors_by_dim_labels,
    reorder_from_labels,
)


def test_reorder_from_labels():
    x = pt.tensor("x", shape=(None, None))
    labels = ["A", "B", "D"]
    combined_labels = ["A", "D", "B"]

    x_sorted = reorder_from_labels(x, labels, combined_labels, labeled_axis=0)
    fn = function([x], x_sorted)

    test_val = np.eye(3) * np.arange(1, 4)
    idx = np.array([0, 2, 1])
    out = fn(test_val)
    np.testing.assert_allclose(out, test_val[idx, :])

    x_sorted = reorder_from_labels(x, labels, combined_labels, labeled_axis=1)
    fn = function([x], x_sorted)

    out = fn(test_val)
    np.testing.assert_allclose(out, test_val[:, idx])

    x_sorted = reorder_from_labels(x, labels, combined_labels, labeled_axis=(0, 1))
    fn = function([x], x_sorted)

    out = fn(test_val)
    np.testing.assert_allclose(out, test_val[np.ix_(idx, idx)])


def make_zeros(x):
    if x.ndim == 1:
        zeros = np.zeros(
            1,
        )
    else:
        zeros = np.zeros((x.shape[0], 1))
    return zeros


def add(left, right):
    return left + right


def same_but_mixed(left, right):
    return left + right[..., np.array([1, 2, 0])]


def concat(left, right):
    return np.concatenate([left, right], axis=-1)


def pad_and_add_left(left, right):
    left = np.concatenate([left, make_zeros(left)], axis=-1)
    return left + right


def pad_and_add_right(left, right):
    right = np.concatenate([right, make_zeros(right)], axis=-1)
    return left + right


def mixed_and_padded(left, right):
    left = np.concatenate([left, make_zeros(left)], axis=-1)
    right = right[..., np.array([2, 1, 0])]
    return left + right


@pytest.mark.parametrize(
    "left_names, right_names, expected_computation",
    [
        (["data"], ["data"], add),
        (["A", "C", "B"], ["B", "A", "C"], same_but_mixed),
        (["data"], ["different_data"], concat),
        (["data"], ["data", "different_data"], pad_and_add_left),
        (["data", "more_data"], ["data"], pad_and_add_right),
        (["A", "B"], ["D", "B", "A"], mixed_and_padded),
    ],
    ids=[
        "same_names",
        "same_but_mixed",
        "different_names",
        "overlap_right",
        "overlap_left",
        "pad_and_mix",
    ],
)
@pytest.mark.parametrize("ndim", [1, 2], ids=["vector", "matrix"])
def test_add_matrices_by_observed_state_names(left_names, right_names, expected_computation, ndim):
    rng = np.random.default_rng()
    n_left = len(left_names)
    n_right = len(right_names)

    left = pt.tensor("left", shape=(None,) * ndim)
    right = pt.tensor("right", shape=(None,) * ndim)

    result = add_tensors_by_dim_labels(left, right, left_names, right_names)
    fn = function([left, right], result)

    left_value = rng.normal(size=(n_left,) if ndim == 1 else (10, n_left))
    right_value = rng.normal(size=(n_right,) if ndim == 1 else (10, n_right))

    np.testing.assert_allclose(
        fn(left_value, right_value), expected_computation(left_value, right_value)
    )


class TestAddCovarianceMatrices:
    def _setup_H(self, states_1, states_2):
        n_1 = len(states_1)
        n_2 = len(states_2)

        H_1 = pt.tensor("H_1", shape=(n_1, n_1))
        H_2 = pt.tensor("H_2", shape=(n_2, n_2))

        return H_1, H_2

    @pytest.mark.parametrize("n_states", [1, 3], ids=["1x1", "3x3"])
    def test_add_fully_overlapping_covariance_matrices(self, n_states):
        rng = np.random.default_rng()
        states = list("ABCD")

        observed_states_1 = states[:n_states]
        observed_states_2 = states[:n_states]

        H_1, H_2 = self._setup_H(observed_states_1, observed_states_2)
        res = add_tensors_by_dim_labels(
            H_1, H_2, observed_states_1, observed_states_2, labeled_axis=(0, 1)
        )

        fn = function([H_1, H_2], res)

        H_1_val = rng.normal(size=(n_states, n_states))
        H_2_val = rng.normal(size=(n_states, n_states))

        np.testing.assert_allclose(fn(H_1_val, H_2_val), H_1_val + H_2_val)

    def test_add_fully_overlapping_mixed_covariance_matrices(self):
        rng = np.random.default_rng()

        observed_states_1 = ["A", "B", "C", "D"]
        observed_states_2 = ["A", "B", "C", "D"]
        rng.shuffle(observed_states_2)

        H_1, H_2 = self._setup_H(observed_states_1, observed_states_2)

        res = add_tensors_by_dim_labels(
            H_1, H_2, observed_states_1, observed_states_2, labeled_axis=(0, 1)
        )

        H_1_val = rng.normal(size=(4, 4))
        H_2_val = rng.normal(size=(4, 4))

        fn = function([H_1, H_2], res)

        state_to_idx = {name: idx for idx, name in enumerate(observed_states_1)}
        idx = np.argsort([state_to_idx[state] for state in observed_states_2])

        np.testing.assert_allclose(fn(H_1_val, H_2_val), H_1_val + H_2_val[np.ix_(idx, idx)])

    def test_add_non_overlapping_covaraince_matrices(self):
        rng = np.random.default_rng()

        observed_states_1 = ["A", "B"]
        observed_states_2 = ["C", "D"]

        H_1, H_2 = self._setup_H(observed_states_1, observed_states_2)

        res = add_tensors_by_dim_labels(
            H_1, H_2, observed_states_1, observed_states_2, labeled_axis=(0, 1)
        )

        H_1_val = rng.normal(size=(2, 2))
        H_2_val = rng.normal(size=(2, 2))
        zeros = np.zeros_like(H_1_val)

        fn = function([H_1, H_2], res)

        np.testing.assert_allclose(
            fn(H_1_val, H_2_val), np.block([[H_1_val, zeros], [zeros, H_2_val]])
        )

    def test_add_partially_overlapping_covaraince_matrices(self):
        rng = np.random.default_rng()
        observed_states_1 = ["A", "B"]
        observed_states_2 = ["B", "C", "D", "A"]
        H_1, H_2 = self._setup_H(observed_states_1, observed_states_2)

        res = add_tensors_by_dim_labels(
            H_1, H_2, observed_states_1, observed_states_2, labeled_axis=(-2, -1)
        )

        fn = function([H_1, H_2], res)
        H_1_val = rng.normal(size=(2, 2))
        H_2_val = rng.normal(size=(4, 4))

        upper = np.zeros((4, 4))
        upper_idx = np.ix_([0, 1], [0, 1])
        upper[upper_idx] = H_1_val
        expected_value = upper + H_2_val[np.ix_([3, 0, 1, 2], [3, 0, 1, 2])]

        np.testing.assert_allclose(fn(H_1_val, H_2_val), expected_value)


class TestJoinDesignMatrices:
    def _setup_Z(self, states_1, states_2, k_endog=2):
        Z_1 = pt.tensor("Z_1", shape=(len(states_1), k_endog))
        Z_2 = pt.tensor("Z_2", shape=(len(states_2), k_endog))

        return Z_1, Z_2

    def test_join_fully_overlapping_design_matrices(self):
        observed_states_1 = ["A"]
        observed_states_2 = ["A"]

        Z_1, Z_2 = self._setup_Z(observed_states_1, observed_states_2)
        res = join_tensors_by_dim_labels(
            Z_1, Z_2, observed_states_1, observed_states_2, labeled_axis=0, join_axis=1
        )

        fn = function([Z_1, Z_2], res)

        Z_1_val = np.array([[1.0, 0.0]])
        Z_2_val = np.array([[0.0, 1.0]])

        np.testing.assert_allclose(fn(Z_1_val, Z_2_val), np.array([[1.0, 0.0, 0.0, 1.0]]))

    def test_join_fully_overlapping_mixed_design_matrices(self):
        observed_states_1 = ["A", "B", "C"]
        observed_states_2 = ["C", "B", "A"]

        Z_1, Z_2 = self._setup_Z(observed_states_1, observed_states_2, k_endog=3)
        res = join_tensors_by_dim_labels(
            Z_1, Z_2, observed_states_1, observed_states_2, labeled_axis=0, join_axis=1
        )

        fn = function([Z_1, Z_2], res)

        Z_1_val = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        Z_2_val = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

        # Rows 0 and 2 should be swapped in the output, because the ordering A, B, C becomes canonical as it was passed
        # in first, and because we said the labeled dim was axis=0. After reordering, the matrices should be
        # concatenated on axis = 1 (again, as requested).
        np.testing.assert_allclose(
            fn(Z_1_val, Z_2_val),
            np.array(
                [
                    [1.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                    [1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
                ]
            ),
        )

    def test_join_non_overlapping_design_matrices(self):
        observed_states_1 = ["A"]
        observed_states_2 = ["B"]

        Z_1, Z_2 = self._setup_Z(observed_states_1, observed_states_2)
        fn = function(
            [Z_1, Z_2], join_tensors_by_dim_labels(Z_1, Z_2, observed_states_1, observed_states_2)
        )

        Z_1_val = np.array([[1.0, 0.0]])
        Z_2_val = np.array([[1.0, 0.0]])
        out = fn(Z_1_val, Z_2_val)

        np.testing.assert_allclose(out, [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])

    def test_join_partially_overlapping_design_matrices(self):
        observed_states_1 = ["A"]
        observed_states_2 = ["A", "B", "C"]

        Z_1, Z_2 = self._setup_Z(observed_states_1, observed_states_2)
        res = join_tensors_by_dim_labels(
            Z_1, Z_2, observed_states_1, observed_states_2, labeled_axis=0, join_axis=1
        )
        fn = function([Z_1, Z_2], res)

        Z_1_val = np.array([[1.0, 0.0]])
        Z_2_val = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0]])

        # Z_1 should be zero padded with the missing observed states, then concatenated along axis = -1
        expected_output = np.array(
            [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        )

        np.testing.assert_allclose(fn(Z_1_val, Z_2_val), expected_output)
