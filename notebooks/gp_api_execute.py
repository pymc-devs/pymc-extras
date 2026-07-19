"""Execute gp_api.ipynb in place, embedding outputs."""

import pathlib
import sys

import nbformat

from nbclient import NotebookClient

here = pathlib.Path(__file__).parent
path = here / "gp_api.ipynb"

nb = nbformat.read(path, as_version=4)
_, nb = nbformat.validator.normalize(nb)  # cell ids; missing ones are a future hard error
client = NotebookClient(
    nb,
    timeout=3600,
    kernel_name="python3",
    resources={"metadata": {"path": str(here)}},
    allow_errors=False,
)

try:
    client.execute()
except Exception as exc:
    nbformat.write(nb, path)
    print(f"EXECUTION FAILED: {type(exc).__name__}: {str(exc)[:2000]}")
    sys.exit(1)

nbformat.write(nb, path)

n_code = sum(1 for c in nb.cells if c.cell_type == "code")
n_out = sum(1 for c in nb.cells if c.cell_type == "code" and c.outputs)
n_img = sum(
    1
    for c in nb.cells
    if c.cell_type == "code"
    for o in c.outputs
    if "image/png" in getattr(o, "get", lambda *_: {})("data", {})
)
print(f"OK: {len(nb.cells)} cells, {n_code} code cells, {n_out} produced output, {n_img} figures")
