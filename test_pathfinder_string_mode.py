
import pymc as pm
import numpy as np
from pymc_extras.inference.pathfinder.pathfinder import fit_pathfinder

def test_string_mode():
    with pm.Model() as model:
        x = pm.Normal("x", 0, 1)
        y = pm.Normal("y", x, 1, observed=0)
    
    print("Fitting with mode='FAST_COMPILE'...")
    idata = fit_pathfinder(model=model, compile_kwargs={"mode": "FAST_COMPILE"}, num_paths=1)
    print("Done.")

if __name__ == "__main__":
    test_string_mode()
