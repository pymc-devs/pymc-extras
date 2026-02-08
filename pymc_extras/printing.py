import numpy as np

from pymc import Model
from pymc.printing import str_for_dist, str_for_potential_or_deterministic
from pytensor import Mode
from pytensor.compile.sharedvalue import SharedVariable
from pytensor.graph.type import Constant, Variable
from rich.box import SIMPLE_HEAD
from rich.table import Table


def variable_expression(
    model: Model,
    var: Variable,
    truncate_deterministic: int | None,
) -> str:
    """Get the expression of a variable in a human-readable format."""
    if var in model.data_vars:
        var_expr = "Data"
    elif var in model.deterministics:
        str_repr = str_for_potential_or_deterministic(var, dist_name="")
        _, var_expr = str_repr.split(" ~ ")
        var_expr = var_expr[1:-1]  # Remove outer parentheses (f(...))
        if truncate_deterministic is not None and len(var_expr) > truncate_deterministic:
            contents = var_expr[2:-1].split(", ")
            str_len = 0
            for show_n, content in enumerate(contents):
                str_len += len(content) + 2
                if str_len > truncate_deterministic:
                    break
            var_expr = f"f({', '.join(contents[:show_n])}, ...)"
    elif var in model.potentials:
        var_expr = str_for_potential_or_deterministic(var, dist_name="Potential").split(" ~ ")[1]
    else:  # basic_RVs
        var_expr = str_for_dist(var).split(" ~ ")[1]
    return var_expr


def _extract_dim_value(var: SharedVariable | Constant) -> np.ndarray:
    if isinstance(var, SharedVariable):
        return var.get_value(borrow=True)
    else:
        return var.data


def dims_expression(model: Model, var: Variable) -> str:
    """Get the dimensions of a variable in a human-readable format."""
    if (dims := model.named_vars_to_dims.get(var.name)) is not None:
        dim_sizes = {dim: _extract_dim_value(model.dim_lengths[dim]) for dim in dims}
        return " × ".join(f"{dim}[{dim_size}]" for dim, dim_size in dim_sizes.items())
    else:
        dim_sizes = list(var.shape.eval(mode=Mode(linker="py", optimizer="fast_compile")))
        return f"[{', '.join(map(str, dim_sizes))}]" if dim_sizes else ""


def model_parameter_count(model: Model) -> int:
    """Count the number of parameters in the model."""
    rv_shapes = model.eval_rv_shapes()  # Includes transformed variables
    return np.sum([np.prod(rv_shapes[free_rv.name]).astype(int) for free_rv in model.free_RVs])


def model_table(
    model: Model,
    *,
    split_groups: bool = True,
    truncate_deterministic: int | None = None,
    parameter_count: bool = True,
) -> Table:
    """Create a rich table with a summary of the model's variables and their expressions.

    Parameters
    ----------
    model : Model
        The PyMC model to summarize.
    split_groups : bool
        If True, each group of variables (data, free_RVs, deterministics, potentials, observed_RVs)
        will be separated by a section.
    truncate_deterministic : int | None
        If not None, truncate the expression of deterministic variables that go beyond this length.
    empty_dims : bool
        If True, show the dimensions of scalar variables as an empty list.
    parameter_count : bool
        If True, add a row with the total number of parameters in the model.

    Returns
    -------
    Table
        A rich table with the model's variables, their expressions and dims.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        import pymc as pm

        from pymc_extras.printing import model_table

        coords = {"subject": range(20), "param": ["a", "b"]}
        with pm.Model(coords=coords) as m:
            x = pm.Data("x", np.random.normal(size=(20, 2)), dims=("subject", "param"))
            y = pm.Data("y", np.random.normal(size=(20,)), dims="subject")

            beta = pm.Normal("beta", mu=0, sigma=1, dims="param")
            mu = pm.Deterministic("mu", pm.math.dot(x, beta), dims="subject")
            sigma = pm.HalfNormal("sigma", sigma=1)

            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y, dims="subject")

        table = model_table(m)
        table  # Displays the following table in an interactive environment
        '''
         Variable  Expression         Dimensions
        ─────────────────────────────────────────────────────
              x =  Data               subject[20] × param[2]
              y =  Data               subject[20]

           beta ~  Normal(0, 1)       param[2]
          sigma ~  HalfNormal(0, 1)
                                      Parameter count = 3

             mu =  f(beta)            subject[20]

          y_obs ~  Normal(mu, sigma)  subject[20]
        '''

    Output can be explicitly rendered in a rich console or exported to text, html or svg.

    .. code-block:: python

        from rich.console import Console

        console = Console(record=True)
        console.print(table)
        text_export = console.export_text()
        html_export = console.export_html()
        svg_export = console.export_svg()

    """
    table = Table(
        show_header=True,
        show_edge=False,
        box=SIMPLE_HEAD,
        highlight=False,
        collapse_padding=True,
    )
    table.add_column("Variable", justify="right")
    table.add_column("Expression", justify="left")
    table.add_column("Dimensions")

    if split_groups:
        groups = (
            model.data_vars,
            model.free_RVs,
            model.deterministics,
            model.potentials,
            model.observed_RVs,
        )
    else:
        # Show variables in the order they were defined
        groups = (model.named_vars.values(),)

    for group in groups:
        if not group:
            continue

        for var in group:
            var_name = var.name
            sep = f"[b]{' ~' if (var in model.basic_RVs) else ' ='}[/b]"
            var_expr = variable_expression(model, var, truncate_deterministic)
            dims_expr = dims_expression(model, var)
            if dims_expr == "[]":
                dims_expr = ""
            table.add_row(var_name + sep, var_expr, dims_expr)

        if parameter_count and (not split_groups or group == model.free_RVs):
            n_parameters = model_parameter_count(model)
            table.add_row("", "", f"[i]Parameter count = {n_parameters}[/i]")

        table.add_section()

    return table
