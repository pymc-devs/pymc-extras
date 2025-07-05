import numpy as np

from pytensor import tensor as pt

from pymc_extras.statespace.models.structural.core import Component
from pymc_extras.statespace.utils.constants import TIME_DIM


class RegressionComponent(Component):
    def __init__(
        self,
        k_exog: int | None = None,
        name: str | None = "Exogenous",
        state_names: list[str] | None = None,
        observed_state_names: list[str] | None = None,
        innovations=False,
    ):
        if observed_state_names is None:
            observed_state_names = ["data"]

        self.innovations = innovations
        k_exog = self._handle_input_data(k_exog, state_names, name)

        k_states = k_exog
        k_endog = len(observed_state_names)
        k_posdef = k_exog

        super().__init__(
            name=name,
            k_endog=k_endog,
            k_states=k_states * k_endog,
            k_posdef=k_posdef * k_endog,
            state_names=self.state_names,
            observed_state_names=observed_state_names,
            measurement_error=False,
            combine_hidden_states=False,
            exog_names=[f"data_{name}"],
            obs_state_idxs=np.ones(k_states),
        )

    @staticmethod
    def _get_state_names(k_exog: int | None, state_names: list[str] | None, name: str):
        if k_exog is None and state_names is None:
            raise ValueError("Must specify at least one of k_exog or state_names")
        if state_names is not None and k_exog is not None:
            if len(state_names) != k_exog:
                raise ValueError(f"Expected {k_exog} state names, found {len(state_names)}")
        elif k_exog is None:
            k_exog = len(state_names)
        else:
            state_names = [f"{name}_{i + 1}" for i in range(k_exog)]

        return k_exog, state_names

    def _handle_input_data(self, k_exog: int, state_names: list[str] | None, name) -> int:
        k_exog, state_names = self._get_state_names(k_exog, state_names, name)
        self.state_names = state_names

        return k_exog

    def make_symbolic_graph(self) -> None:
        k_endog = self.k_endog
        k_states = self.k_states // k_endog
        self.k_posdef // k_endog

        betas = self.make_and_register_variable(f"beta_{self.name}", shape=(k_endog, k_states))
        regression_data = self.make_and_register_data(f"data_{self.name}", shape=(None, k_states))

        self.ssm["initial_state", :] = betas.reshape((1, -1)).squeeze()
        T = np.eye(k_states)
        self.ssm["transition", :, :] = pt.linalg.block_diag(*[T for _ in range(k_endog)])
        self.ssm["selection", :, :] = np.eye(self.k_states)
        Z = pt.linalg.block_diag(*[pt.expand_dims(regression_data, 1) for _ in range(k_endog)])
        self.ssm["design"] = pt.specify_shape(
            Z, (None, k_endog, regression_data.type.shape[1] * k_endog)
        )

        if self.innovations:
            sigma_beta = self.make_and_register_variable(
                f"sigma_beta_{self.name}", (self.k_states,)
            )
            row_idx, col_idx = np.diag_indices(self.k_states)
            self.ssm["state_cov", row_idx, col_idx] = sigma_beta**2

    def populate_component_properties(self) -> None:
        k_endog = self.k_endog
        k_states = self.k_states // k_endog
        self.k_posdef // k_endog

        self.shock_names = self.state_names

        self.param_names = [f"beta_{self.name}"]
        self.data_names = [f"data_{self.name}"]
        self.param_dims = {
            f"beta_{self.name}": ("exog_endog", "exog_state"),
        }

        base_names = self.state_names
        self.state_names = [
            f"{name}[{obs_name}]" for obs_name in self.observed_state_names for name in base_names
        ]

        self.param_info = {
            f"beta_{self.name}": {
                "shape": (k_endog, k_states),
                "constraints": None,
                "dims": ("exog_endog", "exog_state"),
            },
        }

        self.data_info = {
            f"data_{self.name}": {
                "shape": (None, k_states),
                "dims": (TIME_DIM, "exog_state"),
            },
        }
        self.coords = {"exog_state": base_names, "exog_endog": self.observed_state_names}

        if self.innovations:
            self.param_names += [f"sigma_beta_{self.name}"]
            self.param_dims[f"sigma_beta_{self.name}"] = "exog_state"
            self.param_info[f"sigma_beta_{self.name}"] = {
                "shape": (),
                "constraints": "Positive",
                "dims": ("exog_state",),
            }
