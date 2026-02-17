
import pymc as pm
import numpy as np
from pymc_extras.inference.pathfinder.pathfinder import fit_pathfinder
from pytensor.compile.mode import Mode

def test_no_opt_py_linker():
    with pm.Model() as model:
        x = pm.Normal("x", 0, 1)
        y = pm.Normal("y", x, 1, observed=0)
    
    print("Fitting with Mode(linker='py', optimizer=None)...")
    idata = fit_pathfinder(model=model, compile_kwargs={"mode": Mode(linker='py', optimizer=None)}, num_paths=1)
    print("Done.")

if __name__ == "__main__":
    test_no_opt_py_linker()
