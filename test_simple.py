#!/usr/bin/env python3

import pymc as pm
import pytensor.tensor as pt

from pymc_extras.distributions import GrassiaIIGeometric


def test_basic_functionality():
    """Test basic functionality of GrassiaIIGeometric distribution"""
    print("Testing basic GrassiaIIGeometric functionality...")

    # Test 1: Create distribution with None time_covariate_vector
    try:
        dist = GrassiaIIGeometric.dist(r=2.0, alpha=1.0, time_covariate_vector=None)
        print("✓ Distribution created successfully with None time_covariate_vector")

        # Test sampling
        samples = dist.eval()
        print(f"✓ Direct sampling successful: {samples}")

    except Exception as e:
        print(f"✗ Failed to create distribution with None time_covariate_vector: {e}")
        return False

    # Test 2: Create distribution with scalar time_covariate_vector
    try:
        dist = GrassiaIIGeometric.dist(r=2.0, alpha=1.0, time_covariate_vector=0.5)
        print("✓ Distribution created successfully with scalar time_covariate_vector")

        # Test sampling
        samples = dist.eval()
        print(f"✓ Direct sampling successful: {samples}")

    except Exception as e:
        print(f"✗ Failed to create distribution with scalar time_covariate_vector: {e}")
        return False

    # Test 3: Test logp function
    try:
        r = pt.scalar("r")
        alpha = pt.scalar("alpha")
        time_covariate_vector = pt.scalar("time_covariate_vector")
        value = pt.scalar("value", dtype="int64")

        logp = pm.logp(GrassiaIIGeometric.dist(r, alpha, time_covariate_vector), value)
        logp_fn = pm.compile_fn([value, r, alpha, time_covariate_vector], logp)

        result = logp_fn(2, 1.0, 1.0, 0.0)
        print(f"✓ Logp function works: {result}")

    except Exception as e:
        print(f"✗ Failed to test logp function: {e}")
        return False

    print("✓ All basic functionality tests passed!")
    return True


if __name__ == "__main__":
    test_basic_functionality()
