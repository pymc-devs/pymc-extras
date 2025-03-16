import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest

from numpy.testing import assert_allclose

from pymc_extras.statespace.filters import (
    KalmanSmoother,
    SquareRootFilter,
    StandardFilter,
    UnivariateFilter,
)
from pymc_extras.statespace.filters.batched_kalman_filter import (
    BatchedSquareRootFilter,
    BatchedStandardFilter,
    BatchedUnivariateFilter,
)
from tests.statespace.utilities.shared_fixtures import rng
from tests.statespace.utilities.test_helpers import (
    get_expected_shape,
    initialize_filter,
    make_test_inputs,
)

floatX = pytensor.config.floatX

# Tolerances
ATOL = 1e-6 if floatX.endswith("64") else 1e-3
RTOL = 1e-6 if floatX.endswith("64") else 1e-3

# Output names for tests
output_names = [
    "filtered_states",
    "predicted_states",
    "smoothed_states",
    "filtered_covs",
    "predicted_covs",
    "smoothed_covs",
    "log_likelihood",
    "ll_obs",
]


def initialize_batched_filter(filter_cls, p=1, m=1, r=1, n=10, batch_size=2):
    """Initialize the batched filter with dummy data."""
    # Create a batch of test inputs
    batch_inputs = []
    for i in range(batch_size):
        inputs = make_test_inputs(p, m, r, n, rng)
        batch_inputs.append(inputs)

    # Stack inputs along a new batch dimension
    batched_inputs = []
    for i in range(len(batch_inputs[0])):
        stacked = np.stack([batch_inputs[j][i] for j in range(batch_size)], axis=0)
        batched_inputs.append(stacked)

    # Create symbolic variables for inputs
    data = pt.tensor(name="data", dtype=floatX, shape=(None, None, None))
    a0 = pt.tensor(name="a0", dtype=floatX, shape=(None, None))
    P0 = pt.tensor(name="P0", dtype=floatX, shape=(None, None, None))
    c = pt.tensor(name="c", dtype=floatX, shape=(None, None))
    d = pt.tensor(name="d", dtype=floatX, shape=(None, None))
    T = pt.tensor(name="T", dtype=floatX, shape=(None, None, None))
    Z = pt.tensor(name="Z", dtype=floatX, shape=(None, None, None))
    R = pt.tensor(name="R", dtype=floatX, shape=(None, None, None))
    H = pt.tensor(name="H", dtype=floatX, shape=(None, None, None))
    Q = pt.tensor(name="Q", dtype=floatX, shape=(None, None, None))

    # Create batched filter
    batched_filter = filter_cls()

    # Build graph for filter
    filter_outputs = batched_filter.build_graph(data, a0, P0, c, d, T, Z, R, H, Q)

    # Create function
    inputs = [data, a0, P0, c, d, T, Z, R, H, Q]
    f = pytensor.function(inputs, filter_outputs, on_unused_input="ignore")

    return f, batched_inputs


def test_batched_standard_filter_output_shapes():
    """Test the output shapes for BatchedStandardFilter."""
    batch_size = 3
    p, m, r, n = 1, 5, 1, 10

    f, batched_inputs = initialize_batched_filter(
        BatchedStandardFilter, p=p, m=m, r=r, n=n, batch_size=batch_size
    )

    # Run the filter
    outputs = f(*batched_inputs)

    # Check that outputs have the correct batch dimension
    for output_idx, name in enumerate(output_names):
        assert outputs[output_idx].shape[0] == batch_size, f"Batch dimension of {name} is incorrect"

        # Expected shape without batch dimension
        expected_output = get_expected_shape(name, p, m, r, n)

        # Full expected shape with batch dimension
        full_expected_shape = (batch_size,) + expected_output

        assert outputs[output_idx].shape == full_expected_shape, (
            f"Shape of {name} does not match expected"
        )


def test_batched_filter_consistent_with_non_batched():
    """Test that batched filter produces same results as non-batched filter."""
    batch_size = 2
    p, m, r, n = 1, 3, 1, 10

    # Create non-batched filter
    standard_inout = initialize_filter(StandardFilter())
    f_standard = pytensor.function(*standard_inout, on_unused_input="ignore")

    # Create batched filter
    f_batched, batched_inputs = initialize_batched_filter(
        BatchedStandardFilter, p=p, m=m, r=r, n=n, batch_size=batch_size
    )

    # Extract individual batches
    individual_inputs = []
    for i in range(batch_size):
        single_inputs = [inp[i] for inp in batched_inputs]
        individual_inputs.append(single_inputs)

    # Run non-batched filter on each batch
    non_batched_outputs = []
    for inputs in individual_inputs:
        outputs = f_standard(*inputs)
        non_batched_outputs.append(outputs)

    # Run batched filter
    batched_outputs = f_batched(*batched_inputs)

    # Compare results
    for output_idx, name in enumerate(output_names):
        for batch_idx in range(batch_size):
            assert_allclose(
                batched_outputs[output_idx][batch_idx],
                non_batched_outputs[batch_idx][output_idx],
                rtol=RTOL,
                atol=ATOL,
                err_msg=f"Batch {batch_idx} of {name} does not match non-batched output",
            )


@pytest.mark.parametrize(
    "filter_cls",
    [BatchedStandardFilter, BatchedSquareRootFilter, BatchedUnivariateFilter],
    ids=["Standard", "SquareRoot", "Univariate"],
)
def test_all_batched_filters(filter_cls):
    """Test that all batched filter types can be initialized and run."""
    batch_size = 2
    p, m, r, n = 1, 3, 1, 10

    f, batched_inputs = initialize_batched_filter(
        filter_cls, p=p, m=m, r=r, n=n, batch_size=batch_size
    )

    # Run the filter
    outputs = f(*batched_inputs)

    # Check that outputs have the correct batch dimension
    for output_idx, name in enumerate(output_names):
        assert outputs[output_idx].shape[0] == batch_size, f"Batch dimension of {name} is incorrect"
