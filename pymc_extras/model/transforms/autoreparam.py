import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import singledispatch

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import scipy.special
from pymc.distributions import SymbolicRandomVariable
from pymc.logprob.transforms import Transform
from pymc.model.fgraph import (ModelDeterministic, ModelNamed,
                               fgraph_from_model, model_deterministic,
                               model_free_rv, model_from_fgraph, model_named)
from pymc.pytensorf import toposort_replace
from pytensor.graph.basic import Apply, Variable
from pytensor.tensor.random.op import RandomVariable

_log = logging.getLogger("pmx")


@dataclass
class VIP:
    r"""Helper to reparemetrize VIP model.

    Manipulation of :math:`\lambda` in the below equation is done using this helper class.

    .. math::

        \begin{align*}
        \eta_{k} &\sim \text{normal}(\lambda_{k} \cdot \mu, \sigma^{\lambda_{k}})\\
        \theta_{k} &= \mu + \sigma^{1 - \lambda_{k}} ( \eta_{k}  - \lambda_{k} \cdot \mu)
        \sim \text{normal}(\mu, \sigma).
        \end{align*}
    """

    _logit_lambda: dict[str, pytensor.tensor.sharedvar.TensorSharedVariable]

    @property
    def variational_parameters(
        self,
    ) -> list[pytensor.tensor.sharedvar.TensorSharedVariable]:
        r"""Return raw :math:`\operatorname{logit}(\lambda_k)` for custom optimization.

        Examples
        --------
        with model:
            # set all parameterizations to mix of centered and non-centered
            vip.set_all_lambda(0.5)

            pm.fit(more_obj_params=vip.variational_parameters, method="fullrank_advi")
        """
        return list(self._logit_lambda.values())

    def truncate_lambda(self, **kwargs: float):
        r"""Truncate :math:`\lambda_k` with :math:`\varepsilon`.

        .. math::

            \hat \lambda_k = \begin{cases}
                0, \quad &\lambda_k \le \varepsilon\\
                \lambda_k, \quad &\varepsilon \lt \lambda_k \lt 1-\varepsilon\\
                1, \quad &\lambda_k \ge 1-\varepsilon\\
            \end{cases}

        Parameters
        ----------
        kwargs : Dict[str, float]
            Variable to :math:`\varepsilon` mapping.
            If :math:`\lambda` (or :math:`1-\lambda`) is not passing
            the threshold of :math:`\varepsilon`, it will be clipped
            to 1 or zero if rounding is turned on.
        """
        lambdas = self.get_lambda()
        update = dict()
        for var, eps in kwargs.items():
            lam = lambdas[var]
            update[var] = np.piecewise(
                lam,
                [lam < eps, lam > (1 - eps)],
                [0, 1, lambda x: x],
            )
        self.set_lambda(**update)

    def truncate_all_lambda(self, value: float):
        r"""Truncate all :math:`\lambda_k` with :math:`\varepsilon`.

        .. math::

            \hat \lambda_k = \begin{cases}
                0, \quad &\lambda_k \le \varepsilon\\
                \lambda_k, \quad &\varepsilon \lt \lambda_k \lt 1-\varepsilon\\
                1, \quad &\lambda_k \ge 1-\varepsilon\\
            \end{cases}



        Parameters
        ----------
        value : float
            :math:`\varepsilon`
        """
        truncate = dict.fromkeys(
            self._logit_lambda.keys(),
            value,
        )
        self.truncate_lambda(**truncate)

    def get_lambda(self) -> dict[str, np.ndarray]:
        r"""Get :math:`\lambda_k` that are currently used by the model.

        Returns
        -------
        Dict[str, np.ndarray]
            Mapping from variable name to :math:`\lambda_k`.
        """
        return {
            name: scipy.special.expit(shared.get_value())
            for name, shared in self._logit_lambda.items()
        }

    def set_lambda(self, **kwargs: dict[str, np.ndarray | float]):
        r"""Set :math:`\lambda_k` per variable."""
        for key, value in kwargs.items():
            logit_lam = scipy.special.logit(value)
            shared = self._logit_lambda[key]
            fill = np.broadcast_to(
                logit_lam,
                shared.type.shape,
            )
            shared.set_value(fill)

    def set_all_lambda(self, value: np.ndarray | float):
        r"""Set :math:`\lambda_k` globally."""
        config = dict.fromkeys(
            self._logit_lambda.keys(),
            value,
        )
        self.set_lambda(**config)

    def fit(self, *args, **kwargs) -> pm.Approximation:
        r"""Set :math:`\lambda_k` using Variational Inference.

        Examples
        --------

        .. code-block:: python

            with model:
                # set all parameterizations to mix of centered and non-centered
                vip.set_all_lambda(0.5)

                # fit using ADVI
                mf = vip.fit(random_seed=42)
        """
        kwargs.setdefault("obj_optimizer", pm.adagrad_window(learning_rate=0.1))
        kwargs.setdefault("method", "advi")
        return pm.fit(
            *args,
            more_obj_params=self.variational_parameters,
            **kwargs,
        )


def vip_reparam_node(
    op: RandomVariable,
    node: Apply,
    name: str,
    dims: list[Variable],
    transform: Transform | None,
) -> tuple[ModelDeterministic, ModelNamed]:
    if not isinstance(node.op, RandomVariable | SymbolicRandomVariable):
        raise TypeError("Op should be RandomVariable type")
    # FIXME: This is wrong when size is None
    _, size, *_ = node.inputs
    eval_size = size.eval(mode="FAST_COMPILE")
    if eval_size is not None:
        rv_shape = tuple(eval_size)
    else:
        rv_shape = ()
    lam_name = f"{name}::lam_logit__"
    _log.debug(f"Creating {lam_name} with shape of {rv_shape}")
    logit_lam_ = pytensor.shared(
        np.zeros(rv_shape),
        shape=rv_shape,
        name=lam_name,
    )
    logit_lam = model_named(logit_lam_, *dims)
    lam = pt.sigmoid(logit_lam)
    return (
        _vip_reparam_node(
            op,
            node=node,
            name=name,
            dims=dims,
            transform=transform,
            lam=lam,
        ),
        logit_lam,
    )


@singledispatch
def _vip_reparam_node(
    op: RandomVariable,
    node: Apply,
    name: str,
    dims: list[Variable],
    transform: Transform | None,
    lam: pt.TensorVariable,
) -> ModelDeterministic:
    raise NotImplementedError


@_vip_reparam_node.register
def _(
    op: pm.Normal,
    node: Apply,
    name: str,
    dims: list[Variable],
    transform: Transform | None,
    lam: pt.TensorVariable,
) -> ModelDeterministic:
    rng, size, loc, scale = node.inputs
    if transform is not None:
        raise NotImplementedError(
            "Reparametrization of Normal with Transform is not implemented"
        )
    vip_rv_ = pm.Normal.dist(
        lam * loc,
        scale**lam,
        size=size,
        rng=rng,
    )
    vip_rv_.name = f"{name}::tau_"

    vip_rv = model_free_rv(
        vip_rv_,
        vip_rv_.clone(),
        None,
        *dims,
    )

    vip_rep_ = loc + scale ** (1 - lam) * (vip_rv - lam * loc)

    vip_rep_.name = name

    vip_rep = model_deterministic(vip_rep_, *dims)
    return vip_rep


@_vip_reparam_node.register
def _(
    op: pm.Exponential,
    node: Apply,
    name: str,
    dims: list[Variable],
    transform: Transform | None,
    lam: pt.TensorVariable,
) -> ModelDeterministic:
    rng, size, scale = node.inputs
    scale_centered = scale**lam
    scale_noncentered = scale ** (1 - lam)
    vip_rv_ = pm.Exponential.dist(
        scale=scale_centered,
        size=size,
        rng=rng,
    )
    vip_rv_value_ = vip_rv_.clone()
    vip_rv_.name = f"{name}::tau_"
    if transform is not None:
        vip_rv_value_.name = f"{vip_rv_.name}_{transform.name}__"
    else:
        vip_rv_value_.name = vip_rv_.name
    vip_rv = model_free_rv(
        vip_rv_,
        vip_rv_value_,
        transform,
        *dims,
    )

    vip_rep_ = scale_noncentered * vip_rv

    vip_rep_.name = name

    vip_rep = model_deterministic(vip_rep_, *dims)
    return vip_rep


def vip_reparametrize(
    model: pm.Model,
    var_names: Sequence[str],
) -> tuple[pm.Model, VIP]:
    r"""Repametrize Model using Variationally Informed Parametrization (VIP).

    .. math::

        \begin{align*}
        \eta_{k} &\sim \text{normal}(\lambda_{k} \cdot \mu, \sigma^{\lambda_{k}})\\
        \theta_{k} &= \mu + \sigma^{1 - \lambda_{k}} ( \eta_{k}  - \lambda_{k} \cdot \mu)
        \sim \text{normal}(\mu, \sigma).
        \end{align*}

    Parameters
    ----------
    model : Model
        Model with centered parameterizations for variables.
    var_names : Sequence[str]
        Target variables to reparemetrize.

    Returns
    -------
    Tuple[Model, VIP]
        Updated model and VIP helper to reparametrize or infer parametrization of the model.

    Examples
    --------
    The traditional eight schools.

    .. code-block:: python

        import pymc as pm
        import numpy as np

        J = 8
        y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
        sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

        with pm.Model() as Centered_eight:
            mu = pm.Normal("mu", mu=0, sigma=5)
            tau = pm.HalfCauchy("tau", beta=5)
            theta = pm.Normal("theta", mu=mu, sigma=tau, shape=J)
            obs = pm.Normal("obs", mu=theta, sigma=sigma, observed=y)

    The regular model definition with centered parametrization is sufficient to use VIP.
    To change the model parametrization use the following function.

    .. code-block:: python

        from pymc_extras.model.transforms.autoreparam import vip_reparametrize
        Reparam_eight, vip = vip_reparametrize(Centered_eight, ["theta"])

        with Reparam_eight:
            # set all parameterizations to cenered (not needed)
            vip.set_all_lambda(1)

            # set all parameterizations to non-cenered (desired)
            vip.set_all_lambda(0)

            # or per variable
            vip.set_lambda(theta=0)

            # just set non-centered parameterization
            trace = pm.sample()

    However, setting it manually is not always great experience, we can learn it.

    .. code-block:: python

        with Reparam_eight:
            # set all parameterizations to mix of centered and non-centered
            vip.set_all_lambda(0.5)

            # fit using ADVI
            mf = vip.fit(random_seed=42)

            # display lambdas
            print(vip.get_lambda())

            # {'theta': array([0.01473405, 0.02221006, 0.03656685, 0.03798879, 0.04876761,
            #    0.0300203 , 0.02733082, 0.01817754])}

    Now you can use sampling again:

    .. code-block:: python

        with Reparam_eight:
            trace = pm.sample()

    Sometimes it makes sense to enable clipping (that is off by default).
    The idea is to round :math:`\varepsilon` to the closest extremum (:math:`0` or :math:`0`)

    .. math::

        \hat \lambda_k = \begin{cases}
            0, \quad &\lambda_k \le \varepsilon\\
            \lambda_k, \quad &\varepsilon \lt \lambda_k \lt 1-\varepsilon\\
            1, \quad &\lambda_k \ge 1-\varepsilon
        \end{cases}

    .. code-block:: python

        vip.truncate_all_lambda(0.1)

    Sampling has to be performed again

    .. code-block:: python

        with Reparam_eight:
            trace = pm.sample()

    References
    ----------
    - Automatic Reparameterisation of Probabilistic Programs,
        Maria I. Gorinova, Dave Moore, Matthew D. Hoffman (2019)
    """
    fmodel, memo = fgraph_from_model(model)
    lambda_names = []
    replacements = []
    for name in var_names:
        old = memo[model.named_vars[name]]
        rv, _, *dims = old.owner.inputs
        new, lam = vip_reparam_node(
            rv.owner.op,
            rv.owner,
            name=rv.name,
            dims=dims,
            transform=old.owner.op.transform,
        )
        replacements.append((old, new))
        lambda_names.append(lam.name)
    toposort_replace(fmodel, replacements, reverse=True)
    reparam_model = model_from_fgraph(fmodel)
    model_lambdas = {
        var_name: reparam_model[lambda_name]
        for lambda_name, var_name in zip(lambda_names, var_names)
    }
    vip = VIP(model_lambdas)
    return reparam_model, vip
