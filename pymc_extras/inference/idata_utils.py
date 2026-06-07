from itertools import product

import numpy as np
import pymc as pm


def make_default_labels(name: str, shape: tuple[int, ...]) -> list:
    if len(shape) == 0:
        return [name]

    return [list(range(dim)) for dim in shape]


def make_unpacked_variable_names(
    names: list[str], model: pm.Model, var_name_to_model_var: dict[str, str] | None = None
) -> list[str]:
    """Expand raveled variable names into one coordinate-aware label per scalar element.

    Each name in ``names`` is unpacked over its shape, labelling each element with the model's
    coords/dims where available (e.g. ``beta[Intercept]``) and falling back to integer indices
    otherwise. The output order matches the C-order ravel of the parameter vector.
    """
    coords = model.coords
    initial_point = model.initial_point()

    if var_name_to_model_var is None:
        var_name_to_model_var = {}

    value_to_dim = {
        value.name: model.named_vars_to_dims.get(model.values_to_rvs[value].name, None)
        for value in model.value_vars
    }
    value_to_dim = {k: v for k, v in value_to_dim.items() if v is not None}

    rv_to_dim = model.named_vars_to_dims
    dims_dict = rv_to_dim | value_to_dim

    unpacked_variable_names = []
    for name in names:
        name = var_name_to_model_var.get(name, name)
        shape = initial_point[name].shape
        if shape:
            dims = dims_dict.get(name)
            if dims:
                labels_by_dim = [
                    coords[dim] if shape[i] == len(coords[dim]) else np.arange(shape[i])
                    for i, dim in enumerate(dims)
                ]
            else:
                labels_by_dim = make_default_labels(name, shape)
            labels = product(*labels_by_dim)
            unpacked_variable_names.extend(
                [f"{name}[{','.join(map(str, label))}]" for label in labels]
            )
        else:
            unpacked_variable_names.extend([name])
    return unpacked_variable_names
