import pandas as pd
import numpy as np
import pytest
import pymc as pm
# Import the actual class which is modified
from pymc_extras.statespace.core.statespace import PyMCStateSpace

def verify_fix():
    # 1. Setup a dummy class that inherits from modified class
    class DummySS(PyMCStateSpace):
        def __init__(self, names):
            self.observed_state_names = names
            # Mock internal requirements so build_statespace_graph doesn't crash early
            self.ssm = type('obj', (object,), {'k_endog': len(names)})
            self.mode = None
        
        # Override abstract/internal methods to do nothing
        def _insert_random_variables(self): pass
        def _save_exogenous_data_info(self): pass
        def _insert_data_variables(self): pass

    # 2. Initialize with specific names
    tester = DummySS(names=["A", "B"])

    # --- TEST CASE 1: SWAPPED COLUMNS ---
    df_swapped = pd.DataFrame(np.random.randn(5, 2), columns=["B", "A"])
    print("Testing Case 1 (Swapped Columns)...")
    try:
        with pm.Model():
            tester.build_statespace_graph(df_swapped)
        print("FAILED: Model accepted swapped columns.")
    except ValueError as e:
        print(f"✅ PASSED: Caught expected error: {e}")

    # --- TEST CASE 2: CORRECT COLUMNS ---
    df_correct = pd.DataFrame(np.random.randn(5, 2), columns=["A", "B"])
    print("\nTesting Case 2 (Correct Order)...")
    try:
        with pm.Model():
            tester.build_statespace_graph(df_correct)
    except AttributeError as e:
        # If we hit an AttributeError about 'kalman_filter', 
        # it means we successfully got PAST your validation!
        if "kalman_filter" in str(e):
            print(" PASSED: Valid columns passed your check and reached the filter step.")
        else:
            print(f" FAILED: Unexpected error: {e}")
    except ValueError as e:
        print(f" FAILED: Valid columns triggered a mismatch error: {e}")

if __name__ == "__main__":
    verify_fix()
