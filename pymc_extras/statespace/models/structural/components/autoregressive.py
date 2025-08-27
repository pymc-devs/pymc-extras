import numpy as np
import pytensor.tensor as pt

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

    name: str, default "auto_regressive"
        A name for this autoregressive component. Used to label dimensions and coordinates.

    observed_state_names: list[str] | None, default None
        List of strings for observed state labels. If None, defaults to ["data"].

    share_states: bool, default False
        Whether latent states are shared across the observed states. If True, there will be only one set of latent
        states, which are observed by all observed states. If False, each observed state has its own set of
        latent states. This argument has no effect if `k_endog` is 1.

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
              stationary processes with ARIMA, use ``statespace.BayesianSARIMAX``.

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
        name: str = "auto_regressive",
        observed_state_names: list[str] | None = None,
        share_states: bool = False,
    ):
        if observed_state_names is None:
            observed_state_names = ["data"]

        k_endog = len(observed_state_names)
        k_endog_effective = k_posdef = 1 if share_states else k_endog

        order = order_to_mask(order)
        ar_lags = np.flatnonzero(order).ravel().astype(int) + 1
        k_states = len(order)

        self.share_states = share_states
        self.order = order
        self.ar_lags = ar_lags

        super().__init__(
            name=name,
            k_endog=k_endog,
            k_states=k_states * k_endog_effective,
            k_posdef=k_posdef,
            measurement_error=True,
            combine_hidden_states=True,
            observed_state_names=observed_state_names,
            obs_state_idxs=np.tile(np.r_[[1.0], np.zeros(k_states - 1)], k_endog_effective),
            share_states=share_states,
        )

    def populate_component_properties(self):
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog

        k_states = self.k_states // k_endog_effective  # this is also the number of AR lags
        base_names = [f"L{i + 1}_{self.name}" for i in range(k_states)]

        if self.share_states:
            self.state_names = [f"{name}[shared]" for name in base_names]
            self.shock_names = [f"{self.name}[shared]"]
        else:
            self.state_names = [
                f"{name}[{state_name}]"
                for state_name in self.observed_state_names
                for name in base_names
            ]
            self.shock_names = [
                f"{self.name}[{obs_name}]" for obs_name in self.observed_state_names
            ]

        self.param_names = [f"params_{self.name}", f"sigma_{self.name}"]
        self.param_dims = {f"params_{self.name}": (f"lag_{self.name}",)}
        self.coords = {f"lag_{self.name}": self.ar_lags.tolist()}

        if k_endog_effective > 1:
            self.param_dims[f"params_{self.name}"] = (
                f"endog_{self.name}",
                f"lag_{self.name}",
            )
            self.param_dims[f"sigma_{self.name}"] = (f"endog_{self.name}",)

            self.coords[f"endog_{self.name}"] = self.observed_state_names

        self.param_info = {
            f"params_{self.name}": {
                "shape": (k_states,) if self.k_endog == 1 else (self.k_endog, k_states),
                "constraints": None,
                "dims": (AR_PARAM_DIM,)
                if self.k_endog == 1
                else (
                    f"endog_{self.name}",
                    f"lag_{self.name}",
                ),
            },
            f"sigma_{self.name}": {
                "shape": () if self.k_endog == 1 else (self.k_endog,),
                "constraints": "Positive",
                "dims": None if self.k_endog == 1 else (f"endog_{self.name}",),
            },
        }

    def make_symbolic_graph(self) -> None:
        k_endog = self.k_endog
        k_endog_effective = 1 if self.share_states else k_endog

        k_states = self.k_states // k_endog_effective
        k_posdef = self.k_posdef

        k_nonzero = int(sum(self.order))
        ar_params = self.make_and_register_variable(
            f"params_{self.name}",
            shape=(k_nonzero,) if k_endog_effective == 1 else (k_endog_effective, k_nonzero),
        )
        sigma_ar = self.make_and_register_variable(
            f"sigma_{self.name}", shape=() if k_endog_effective == 1 else (k_endog_effective,)
        )

        if k_endog_effective == 1:
            T = pt.eye(k_states, k=-1)
            ar_idx = (np.zeros(k_nonzero, dtype="int"), np.nonzero(self.order)[0])
            T = T[ar_idx].set(ar_params)

        else:
            transition_matrices = []

            for i in range(k_endog_effective):
                T = pt.eye(k_states, k=-1)
                ar_idx = (np.zeros(k_nonzero, dtype="int"), np.nonzero(self.order)[0])
                T = T[ar_idx].set(ar_params[i])
                transition_matrices.append(T)
            T = pt.specify_shape(
                pt.linalg.block_diag(*transition_matrices), (self.k_states, self.k_states)
            )

        self.ssm["transition", :, :] = T

        R = np.eye(k_states)
        R_mask = np.full((k_states,), False)
        R_mask[0] = True
        R = R[:, R_mask]

        self.ssm["selection", :, :] = pt.specify_shape(
            pt.linalg.block_diag(*[R for _ in range(k_endog_effective)]), (self.k_states, k_posdef)
        )

        Zs = [pt.zeros((1, k_states))[0, 0].set(1.0) for _ in range(k_endog)]

        if self.share_states:
            Z = pt.join(0, *Zs)
        else:
            Z = pt.linalg.block_diag(*Zs)
        self.ssm["design", :, :] = pt.specify_shape(Z, (k_endog, self.k_states))

        cov_idx = ("state_cov", *np.diag_indices(k_posdef))
        self.ssm[cov_idx] = sigma_ar**2
