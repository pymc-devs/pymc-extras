# Welcome to `pymc-extras`
<a href="https://gitpod.io/#https://github.com/pymc-devs/pymc-extras">
  <img
    src="https://img.shields.io/badge/Contribute%20with-Gitpod-908a85?logo=gitpod"
    alt="Contribute with Gitpod"
  />
</a>
<img
  src="https://codecov.io/gh/pymc-devs/pymc-extras/branch/main/graph/badge.svg"
  alt="Codecov Badge"
/>

As PyMC continues to mature and expand its functionality to accommodate more domains of application, we increasingly see cutting-edge methodologies, highly specialized statistical distributions, and complex models appear.
While this adds to the functionality and relevance of the project, it can also introduce instability and impose a burden on testing and quality control.
To reduce the burden on the main `pymc` repository, this `pymc-extras` repository can become the aggregator and testing ground for new additions to PyMC.
This may include unusual probability distributions, advanced model fitting algorithms, innovative yet not fully tested methods, or niche functionality that might not fit in the main PyMC repository, but still may be of interest to users.

The `pymc-extras` repository can be understood as the first step in the PyMC development pipeline, where all novel code is introduced until it is obvious that it belongs in the main repository.
We hope that this organization improves the stability and streamlines the testing overhead of the `pymc` repository, while allowing users and developers to test and evaluate cutting-edge methods and not yet fully mature features.

`pymc-extras` would be designed to mirror the namespaces in `pymc` to make usage and migration as easy as possible.
For example, a `ParabolicFractal` distribution could be used analogously to those in `pymc`:

```python
import pymc as pm
import pymc_extras as pmx

with pm.Model():
  alpha = pmx.ParabolicFractal('alpha', b=1, c=1)

  ...

```

## Questions

### What belongs in `pymc-extras`?

- newly-implemented statistical methods, for example step methods or model construction helpers
- distributions that are tricky to sample from or test
- infrequently-used fitting methods or distributions
- any code that requires additional optimization before it can be used in practice


### What does not belong in `pymc-extras`?
- Case studies
- Implementations that cannot be applied generically, for example because they are tied to variables from a toy example


### Should there be more than one add-on repository?

Since there is a lot of code that we may not want in the main repository, does it make sense to have more than one additional repository?
For example, `pymc-extras` may just include methods that are not fully developed, tested and trusted, while code that is known to work well and has adequate test coverage, but is still too specialized to become part of `pymc` could reside in a `pymc-extras` (or similar) repository.


### Unanswered questions & ToDos
This project is still young and many things have not been answered or implemented.
Please get involved!

* What are guidelines for organizing submodules?
  * Proposal: No default imports of WIP/unstable submodules. By importing manually we can avoid breaking the package if a submodule breaks, for example because of an updated dependency.
