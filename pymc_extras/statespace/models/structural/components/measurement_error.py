import numpy as np

from pymc_extras.statespace.models.structural.core import Component


class MeasurementError(Component):
    r"""
    Measurement error term for a structural timeseries model

    Parameters
    ----------
    name: str, optional

        Name of the observed data. Default is "obs".

    Notes
    -----
    This component should only be used in combination with other components, because it has no states. It's only use
    is to add a variance parameter to the model, associated with the observation noise matrix H.

    Examples
    --------
    Create and estimate a deterministic linear trend with measurement error

    .. code:: python

        from pymc_extras.statespace import structural as st
        import pymc as pm
        import pytensor.tensor as pt

        trend = st.LevelTrendComponent(order=2, innovations_order=0)
        error = st.MeasurementError()
        ss_mod = (trend + error).build()

        with pm.Model(coords=ss_mod.coords) as model:
            P0 = pm.Deterministic('P0', pt.eye(ss_mod.k_states) * 10, dims=ss_mod.param_dims['P0'])
            intitial_trend = pm.Normal('initial_trend', sigma=10, dims=ss_mod.param_dims['initial_trend'])
            sigma_obs = pm.Exponential('sigma_obs', 1, dims=ss_mod.param_dims['sigma_obs'])

            ss_mod.build_statespace_graph(data)
            idata = pm.sample(nuts_sampler='numpyro')
    """

    def __init__(
        self, name: str = "MeasurementError", observed_state_names: list[str] | None = None
    ):
        if observed_state_names is None:
            observed_state_names = ["data"]

        k_endog = len(observed_state_names)
        k_states = 0
        k_posdef = 0

        super().__init__(
            name,
            k_endog,
            k_states,
            k_posdef,
            measurement_error=True,
            combine_hidden_states=False,
            observed_state_names=observed_state_names,
        )

    def populate_component_properties(self):
        self.param_names = [f"sigma_{self.name}"]
        self.param_dims = {}
        self.param_info = {
            f"sigma_{self.name}": {
                "shape": (),
                "constraints": "Positive",
                "dims": None,
            }
        }

    def make_symbolic_graph(self) -> None:
        sigma_shape = ()
        error_sigma = self.make_and_register_variable(f"sigma_{self.name}", shape=sigma_shape)
        diag_idx = np.diag_indices(self.k_endog)
        idx = np.s_["obs_cov", diag_idx[0], diag_idx[1]]
        self.ssm[idx] = error_sigma**2
