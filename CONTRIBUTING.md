# Contributing guide

Page in construction, for now go to https://github.com/pymc-devs/pymc-extras#questions.

## Building the documentation

To build the documentation locally, you need to install the necessary
dependencies and then use `make` to build the HTML files.

First, install the package with the optional documentation dependencies:

```bash
pip install ".[docs]"
```

Then, navigate to the `docs` directory and run `make html`:

```bash
cd docs
make html
```

The generated HTML files will be in the `docs/_build/html` directory. You can
open the `index.html` file in that directory to view the documentation.
