import numpy as np

from pymc_extras.statespace.models.structural.core import Component
from pymc_extras.statespace.models.structural.utils import order_to_mask
from pymc_extras.statespace.utils.constants import AR_PARAM_DIM


class AutoregressiveComponent(Component):
    r"""
    Autoregressive timeseries component

    Parameters
    ----------
    order: int or sequence of int

        If int, the number of lags to include in the model.
        If a sequence, an array-like of zeros and ones indicating which lags to include in the model.

    Notes
    -----
    An autoregressive component can be thought of as a way o introducing serially correlated errors into the model.
    The process is modeled:

    .. math::
        x_t = \sum_{i=1}^p \rho_i x_{t-i}

    Where ``p``, the number of autoregressive terms to model, is the order of the process. By default, all lags up to
    ``p`` are included in the model. To disable lags, pass a list of zeros and ones to the ``order`` argumnet. For
    example, ``order=[1, 1, 0, 1]`` would become:

    .. math::
        x_t = \rho_1 x_{t-1} + \rho_2 x_{t-1} + \rho_4 x_{t-1}

    The coefficient :math:`\rho_3` has been constrained to zero.

    .. warning:: This class is meant to be used as a component in a structural time series model. For modeling of
              stationary processes with ARIMA, use ``statespace.BayesianSARIMA``.

    Examples
    --------
    Model a timeseries as an AR(2) process with non-zero mean:

    .. code:: python

        from pymc_extras.statespace import structural as st
        import pymc as pm
        import pytensor.tensor as pt

        trend = st.LevelTrendComponent(order=1, innovations_order=0)
        ar = st.AutoregressiveComponent(2)
        ss_mod = (trend + ar).build()

        with pm.Model(coords=ss_mod.coords) as model:
            P0 = pm.Deterministic('P0', pt.eye(ss_mod.k_states) * 10, dims=ss_mod.param_dims['P0'])
            intitial_trend = pm.Normal('initial_trend', sigma=10, dims=ss_mod.param_dims['initial_trend'])
            ar_params = pm.Normal('ar_params', dims=ss_mod.param_dims['ar_params'])
            sigma_ar = pm.Exponential('sigma_ar', 1, dims=ss_mod.param_dims['sigma_ar'])

            ss_mod.build_statespace_graph(data)
            idata = pm.sample(nuts_sampler='numpyro')

    """

    def __init__(
        self,
        order: int = 1,
        name: str = "AutoRegressive",
        observed_state_names: list[str] | None = None,
    ):
        if observed_state_names is None:
            observed_state_names = ["data"]

        order = order_to_mask(order)
        ar_lags = np.flatnonzero(order).ravel().astype(int) + 1
        k_states = len(order)
        k_posdef = k_endog = len(observed_state_names)

        self.order = order
        self.ar_lags = ar_lags

        super().__init__(
            name=name,
            k_endog=k_endog,
            k_states=k_states,
            k_posdef=k_posdef,
            measurement_error=True,
            combine_hidden_states=True,
            observed_state_names=observed_state_names,
            obs_state_idxs=np.r_[[1.0], np.zeros(k_states - 1)],
        )

    def populate_component_properties(self):
        self.state_names = [f"L{i + 1}.data" for i in range(self.k_states)]
        self.shock_names = [f"{self.name}_innovation"]
        self.param_names = ["ar_params", "sigma_ar"]
        self.param_dims = {"ar_params": (AR_PARAM_DIM,)}
        self.coords = {AR_PARAM_DIM: self.ar_lags.tolist()}

        self.param_info = {
            "ar_params": {
                "shape": (self.k_states,),
                "constraints": None,
                "dims": (AR_PARAM_DIM,),
            },
            "sigma_ar": {"shape": (), "constraints": "Positive", "dims": None},
        }

    def make_symbolic_graph(self) -> None:
        k_nonzero = int(sum(self.order))
        ar_params = self.make_and_register_variable("ar_params", shape=(k_nonzero,))
        sigma_ar = self.make_and_register_variable("sigma_ar", shape=())

        T = np.eye(self.k_states, k=-1)
        self.ssm["transition", :, :] = T
        self.ssm["selection", 0, 0] = 1
        self.ssm["design", 0, 0] = 1

        ar_idx = ("transition", np.zeros(k_nonzero, dtype="int"), np.nonzero(self.order)[0])
        self.ssm[ar_idx] = ar_params

        cov_idx = ("state_cov", *np.diag_indices(1))
        self.ssm[cov_idx] = sigma_ar**2
