
import pymc as pm
import numpy as np
from pymc_extras.inference.pathfinder.pathfinder import fit_pathfinder
import pytensor

def test_fast_compile():
    with pm.Model() as model:
        x = pm.Normal("x", 0, 1)
        y = pm.Normal("y", x, 1, observed=0)
    
    # Try with fast_compile
    print("Fitting with FAST_COMPILE...")
    idata = fit_pathfinder(model=model, compile_kwargs={"mode": "FAST_COMPILE"}, num_paths=1)
    print("Done.")

if __name__ == "__main__":
    test_fast_compile()
