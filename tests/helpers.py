"""Test helpers for step method testing."""

import numpy as np
import pymc as pm

from pymc.step_methods.compound import Competence


class StepMethodTester:
    """Base class for testing step methods."""

    def step_continuous(self, step_fn, draws):
        """Test step method on continuous variables."""
        with pm.Model() as model:
            x = pm.Normal("x", mu=0, sigma=1, shape=2)
            y = pm.Normal("y", mu=x, sigma=1, shape=2)

            # Create covariance matrix for testing
            C = np.array([[1, 0.5], [0.5, 1]])
            step = step_fn(C, model)

            trace = pm.sample(
                draws=draws,
                tune=100,
                chains=1,
                step=step,
                return_inferencedata=False,
                progressbar=False,
                compute_convergence_checks=False,
            )

            # Basic checks
            assert len(trace) == draws
            assert "x" in trace.varnames
            assert "y" in trace.varnames


class RVsAssignmentStepsTester:
    """Test random variable assignment for step methods."""

    def continuous_steps(self, step_class, step_kwargs):
        """Test step method assignment for continuous variables."""
        with pm.Model() as model:
            x = pm.Normal("x", mu=0, sigma=1)
            y = pm.Normal("y", mu=x, sigma=1)

            # Test that step method can be created
            step = step_class(**step_kwargs)

            # Test competence
            if hasattr(step_class, "competence"):
                # Mock variable for competence testing
                class MockVar:
                    dtype = "float64"

                var = MockVar()
                competence = step_class.competence(var, has_grad=True)
                assert competence in [Competence.COMPATIBLE, Competence.PREFERRED]
