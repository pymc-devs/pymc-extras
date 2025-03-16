"""
Batched Kalman Filter Implementation

This module extends the basic Kalman Filter implementation to support
batch dimensions, allowing for parallel filtering of multiple time series.
"""

import pytensor
import pytensor.tensor as pt

from pytensor.graph import vectorize_graph

from pymc_extras.statespace.filters.kalman_filter import (
    BaseFilter,
    SquareRootFilter,
    StandardFilter,
    UnivariateFilter,
)


class BatchedFilter:
    """
    Wrapper for Kalman filters that adds support for batch dimensions.

    This class wraps any existing Kalman filter implementation and adds
    support for batch dimensions, allowing for parallel filtering of
    multiple time series.
    """

    def __init__(self, filter_cls, mode=None):
        """
        Initialize a batched Kalman filter.

        Parameters
        ----------
        filter_cls : BaseFilter or subclass
            The filter class to be batched. Must be a subclass of BaseFilter.
        mode : str, optional
            The mode used for PyTensor compilation.
        """
        if not issubclass(filter_cls, BaseFilter):
            raise TypeError("filter_cls must be a subclass of BaseFilter")

        self.base_filter = filter_cls(mode=mode)
        self.mode = mode

    def build_graph(
        self,
        data,
        a0,
        P0,
        c,
        d,
        T,
        Z,
        R,
        H,
        Q,
        mode=None,
        return_updates=False,
        missing_fill_value=None,
        cov_jitter=None,
    ):
        """
        Construct the computation graph for the batched Kalman filter.

        This method wraps the base filter's build_graph method and adds
        support for batch dimensions.

        Parameters
        ----------
        data : TensorVariable
            Data to be filtered. Should have shape (batch, time, obs).
        a0 : TensorVariable
            Initial state vector. Should have shape (batch, state).
        P0 : TensorVariable
            Initial state covariance. Should have shape (batch, state, state).
        c : TensorVariable
            State bias term. Should have shape (batch, state).
        d : TensorVariable
            Observation bias term. Should have shape (batch, obs).
        T : TensorVariable
            State transition matrix. Should have shape (batch, state, state).
        Z : TensorVariable
            Observation matrix. Should have shape (batch, obs, state).
        R : TensorVariable
            State noise selection matrix. Should have shape (batch, state, posdef).
        H : TensorVariable
            Observation noise covariance. Should have shape (batch, obs, obs).
        Q : TensorVariable
            State noise covariance. Should have shape (batch, posdef, posdef).

        Returns
        -------
        list[TensorVariable]
            List of filter outputs with batch dimensions.
        """
        if mode is None:
            mode = self.mode

        # Get the base filter's build_graph function
        base_build_graph = getattr(self.base_filter, "build_graph")

        # Create a vectorized version of the base filter's build_graph
        def base_build_graph_wrapper(
            data_item, a0_item, P0_item, c_item, d_item, T_item, Z_item, R_item, H_item, Q_item
        ):
            return base_build_graph(
                data_item,
                a0_item,
                P0_item,
                c_item,
                d_item,
                T_item,
                Z_item,
                R_item,
                H_item,
                Q_item,
                mode=mode,
                return_updates=return_updates,
                missing_fill_value=missing_fill_value,
                cov_jitter=cov_jitter,
            )

        # Create symbolic variables for the vectorized function inputs
        inputs = [
            pt.tensor(dtype=var.dtype, shape=var.type.shape[1:], name=f"{var.name}_item")
            for var in [data, a0, P0, c, d, T, Z, R, H, Q]
        ]

        # Get the vectorized outputs
        outputs = base_build_graph_wrapper(*inputs)

        # Create the vectorized function using pytensor's vectorize_graph
        axis = 0  # The batch dimension is always the first dimension

        try:
            # Try using vectorize_graph directly
            vectorized_outputs = []

            for i, output in enumerate(outputs):
                # For each output, create a vectorized version
                replace_map = {
                    inputs[0]: data,
                    inputs[1]: a0,
                    inputs[2]: P0,
                    inputs[3]: c,
                    inputs[4]: d,
                    inputs[5]: T,
                    inputs[6]: Z,
                    inputs[7]: R,
                    inputs[8]: H,
                    inputs[9]: Q,
                }

                vectorized_output = vectorize_graph(output, replace=replace_map, axis=axis)
                vectorized_outputs.append(vectorized_output)

            return vectorized_outputs

        except Exception as e:
            # If vectorize_graph fails, use scan as a fallback
            print(f"Vectorization failed: {e}. Falling back to scan.")

            def process_batch_item(
                data_item, a0_item, P0_item, c_item, d_item, T_item, Z_item, R_item, H_item, Q_item
            ):
                return base_build_graph(
                    data_item,
                    a0_item,
                    P0_item,
                    c_item,
                    d_item,
                    T_item,
                    Z_item,
                    R_item,
                    H_item,
                    Q_item,
                    mode=mode,
                    return_updates=return_updates,
                    missing_fill_value=missing_fill_value,
                    cov_jitter=cov_jitter,
                )

            # Use scan to iterate over batch dimension
            results, updates = pytensor.scan(
                fn=process_batch_item,
                sequences=[data, a0, P0, c, d, T, Z, R, H, Q],
                name="batched_kalman_filter",
            )

            # Return the scan results
            if return_updates:
                return results, updates

            return results


class BatchedStandardFilter(BatchedFilter):
    """Batched version of StandardFilter."""

    def __init__(self, mode=None):
        super().__init__(StandardFilter, mode=mode)


class BatchedSquareRootFilter(BatchedFilter):
    """Batched version of SquareRootFilter."""

    def __init__(self, mode=None):
        super().__init__(SquareRootFilter, mode=mode)


class BatchedUnivariateFilter(BatchedFilter):
    """Batched version of UnivariateFilter."""

    def __init__(self, mode=None):
        super().__init__(UnivariateFilter, mode=mode)
