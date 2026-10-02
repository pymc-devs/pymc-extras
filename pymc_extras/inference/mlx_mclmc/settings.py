from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from pytensor_ml.optim import Transform


class AdaptationSettings(NamedTuple):
    r"""
    Everything :func:`warmup` adapts, and how.

    Attributes
    ----------
    mass_matrix : str
        ``"variance"`` for blackjax's per-coordinate posterior variance, ``"gradient"`` for
        nuts-rs's :math:`\sqrt{\mathrm{Var}[x] / \mathrm{Var}[\nabla \log p]}`, or
        ``"low_rank"`` to fit a low-rank correction on top of the latter. Only ``"low_rank"`` can
        precondition a posterior whose ridges are not axis-aligned.
    diagonal_preconditioning : bool
        Whether to install the estimated metric at all.
    early_switch_freq, switch_freq : int
        Moment-estimator window lengths, before and after ``early_end``.
    early_end : int
        Phase-2 step at which the short early windows give way to the growing ones. 0 disables the
        early phase.
    window_growth : float
        Factor by which each main-phase window exceeds the last.
    low_rank_window : int
        Trailing phase-2 draws retained for a ``"low_rank"`` fit.
    low_rank_refits : int
        Refits within phase 2 before the final one. The first is always the gradient diagonal,
        which seeds the later low-rank fits with draws taken under something better than identity.
    desired_energy_var : float
        Target energy variance per dimension for the step-size controller.
    trust_in_estimate : float
        Width of the controller's Gaussian weighting. Larger values give more weight to single-step
        estimates far from the target.
    num_effective_samples : float
        Sets the controller's exponential decay rate.
    frac_tune1, frac_tune2, frac_tune3 : float
        Fractions of the step budget given to each adaptation phase.
    l_factor : float
        Multiplier on the autocorrelation-derived ``L`` in phase 3.
    advi_steps : int
        ADVI steps fitted before adaptation. 0, the default, skips ADVI. When set, the adapting
        chains start from draws of the fitted approximation and the warmup metric starts at its
        covariance, in place of the ``initial_jitter`` scatter and the identity.
    advi_guide : str
        ``"mean_field"`` for a diagonal Gaussian, or ``"low_rank"`` for a covariance of
        :math:`W W^\top + \mathrm{diag}(d^2)`, which becomes a diagonal metric with a low-rank
        correction. Phase 2 refits the metric, so the correction only lasts past it under
        ``mass_matrix="low_rank"``.
    advi_rank : int, optional
        Rank of the ``"low_rank"`` guide. Defaults to the guide's own, the square root of the
        dimension.
    advi_optimizer : Transform, optional
        A :mod:`pytensor_ml.optim` optimizer for the ADVI fit, such as
        ``apply_if_finite(adam(cosine_schedule(1e-2, total_steps=advi_steps)))``. Defaults to
        :func:`~pymc_extras.inference.advi.default_optimizer`. One given here replaces that whole
        chain, including its guard against non-finite steps.
    initial_jitter : float
        Half-width of the uniform scatter of the adapting chains around the initial point, in the
        unconstrained space. 0 starts every chain at the initial point; 1 is pymc's scatter, which
        displaces a tall model too far for the tuning budget to recover from. Unused when
        ``advi_steps`` is set.
    """

    mass_matrix: str = "gradient"
    diagonal_preconditioning: bool = True
    early_switch_freq: int = 10
    switch_freq: int = 80
    early_end: int = 0
    window_growth: float = 1.5
    low_rank_window: int = 400
    low_rank_refits: int = 2
    desired_energy_var: float = 5e-4
    trust_in_estimate: float = 1.5
    num_effective_samples: float = 150
    frac_tune1: float = 0.1
    frac_tune2: float = 0.1
    frac_tune3: float = 0.1
    l_factor: float = 0.4
    advi_steps: int = 0
    advi_guide: str = "mean_field"
    advi_rank: int | None = None
    advi_optimizer: "Transform | None" = None
    initial_jitter: float = 0.3
