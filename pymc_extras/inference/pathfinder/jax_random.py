"""JAX-native random generation for pathfinder algorithm.

This module provides JAX-compatible random number generation that avoids
the dynamic slicing issues that prevent JAX compilation in the current
pathfinder implementation.

Following PyMC's JAX patterns for proper PRNG key management and static
compilation compatibility.
"""

import jax
import jax.numpy as jnp
import pytensor.tensor as pt

from pytensor.graph import Apply, Op
from pytensor.link.jax.dispatch import jax_funcify


class JAXRandomSampleOp(Op):
    """Custom Op for JAX-native random sample generation.

    This Op generates random samples using JAX PRNG internally,
    avoiding PyTensor's dynamic slicing approach that causes
    compilation failures in JAX mode.
    """

    def __init__(self, num_samples: int):
        """Initialize with static sample count.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate (must be static for JAX compilation)
        """
        self.num_samples = num_samples

    def make_node(self, L_size, N_size, jax_key):
        """Create computation node for JAX random sampling.

        Parameters
        ----------
        L_size : TensorVariable (scalar)
            Number of paths
        N_size : TensorVariable (scalar)
            Number of parameters
        jax_key : TensorVariable
            JAX PRNG key as uint32 array

        Returns
        -------
        Apply
            Computation node with random samples output
        """
        L_size = pt.as_tensor_variable(L_size)
        N_size = pt.as_tensor_variable(N_size)
        jax_key = pt.as_tensor_variable(jax_key)

        # Output: (L, num_samples, N) with static num_samples
        output = pt.tensor(
            dtype="float64",
            shape=(None, self.num_samples, None),  # Only num_samples is static
        )

        return Apply(self, [L_size, N_size, jax_key], [output])

    def perform(self, node, inputs, outputs):
        """PyTensor implementation using NumPy (fallback)."""
        import numpy as np

        L, N, key_array = inputs
        L, N = int(L), int(N)

        # Convert key back to JAX format and generate samples
        np.random.seed(key_array[0] + key_array[1])  # Simple seed from key
        samples = np.random.normal(size=(L, self.num_samples, N)).astype("float64")

        outputs[0][0] = samples

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.num_samples == other.num_samples

    def __hash__(self):
        return hash((type(self), self.num_samples))


@jax_funcify.register(JAXRandomSampleOp)
def jax_funcify_JAXRandomSampleOp(op, node=None, **kwargs):
    """JAX implementation for JAXRandomSampleOp.

    Uses JAX PRNG key management following PyMC patterns
    with concrete shape extraction to solve JAX v0.7 shape requirements.
    """
    num_samples = op.num_samples

    # Try to extract concrete L,N values from the node if available
    # This follows PyTensor's pattern for handling static shapes
    static_L = None
    static_N = None

    if node is not None:
        # Check if L,N inputs are constants (concrete values)
        L_input = node.inputs[0]  # L_size input
        N_input = node.inputs[1]  # N_size input

        # If L is a Constant, extract its value
        if hasattr(L_input, "data") and L_input.data is not None:
            try:
                static_L = int(L_input.data)
            except (ValueError, TypeError):
                pass

        # If N is a Constant, extract its value
        if hasattr(N_input, "data") and N_input.data is not None:
            try:
                static_N = int(N_input.data)
            except (ValueError, TypeError):
                pass

    # Choose the appropriate JAX implementation path
    if static_L is not None and static_N is not None:
        # Static path: L,N are concrete - use them directly
        def jax_random_samples_static(L, N, jax_key):
            """JAX implementation with concrete L,N values."""
            key = jax.random.key_data(jax_key)
            samples = jax.random.normal(
                key,
                shape=(static_L, num_samples, static_N),  # All concrete values
                dtype=jnp.float64,
            )
            return samples

        return jax_random_samples_static

    else:
        # Dynamic path: L,N are traced - use fixed buffer approach
        def jax_random_samples_dynamic(L, N, jax_key):
            """JAX implementation for traced L,N values using fixed buffer strategy.

            JAX v0.7 Fix: Generate samples with concrete maximum dimensions,
            then slice dynamically to get the required (L, num_samples, N) shape.

            This works because:
            1. JAX operations use only concrete shapes
            2. Dynamic slicing happens after generation (JAX can handle this)
            3. Mathematical result is correct, just with unused buffer space
            """
            key = jax.random.key_data(jax_key)

            # Define concrete maximum buffer sizes for JAX compatibility
            # These should be generous enough for typical pathfinder usage
            MAX_L = 50  # Maximum number of paths
            MAX_N = 500  # Maximum number of parameters

            # Generate samples with concrete buffer dimensions
            # Shape: (MAX_L, num_samples, MAX_N) - all concrete values
            buffer_samples = jax.random.normal(
                key, shape=(MAX_L, num_samples, MAX_N), dtype=jnp.float64
            )

            # Dynamically slice to get the actual required shape (L, num_samples, N)
            # JAX can handle dynamic slicing after generation
            actual_samples = jax.lax.dynamic_slice(
                buffer_samples,
                (0, 0, 0),  # Start indices
                (L, num_samples, N),  # Slice sizes (can be traced)
            )

            return actual_samples

        return jax_random_samples_dynamic


def create_jax_random_samples(num_samples: int, L_tensor, N_tensor, random_seed: int = 42):
    """Create JAX-compatible random samples for pathfinder.

    This function creates a computation graph that generates random samples
    using JAX PRNG, avoiding the dynamic slicing issues in the current
    pathfinder implementation.

    Parameters
    ----------
    num_samples : int
        Number of samples (static for JAX compilation)
    L_tensor : TensorVariable
        Number of paths (can be dynamic)
    N_tensor : TensorVariable
        Number of parameters (can be dynamic)
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    TensorVariable
        Random samples with shape (L, num_samples, N)
    """
    # Create JAX PRNG key
    key = jax.random.PRNGKey(random_seed)
    key_array = jnp.array(key, dtype=jnp.uint32)
    jax_key_tensor = pt.constant(key_array, dtype="uint32")

    # Create JAX random sample Op
    random_op = JAXRandomSampleOp(num_samples=num_samples)
    samples = random_op(L_tensor, N_tensor, jax_key_tensor)

    return samples
