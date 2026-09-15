from typing import NamedTuple


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
    settings.diagonal_preconditioning : bool
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
    settings.desired_energy_var : float
        Target energy variance per dimension for the step-size controller.
    settings.trust_in_estimate : float
        Width of the controller's Gaussian weighting. Larger values give more weight to single-step
        estimates far from the target.
    settings.num_effective_samples : float
        Sets the controller's exponential decay rate.
    settings.frac_tune1, settings.frac_tune2, settings.frac_tune3 : float
        Fractions of the step budget given to each adaptation phase.
    settings.l_factor : float
        Multiplier on the autocorrelation-derived ``L`` in phase 3.
    settings.optimize_steps : int
        Maximum Adam steps taken toward the mode before adaptation. Pass 0 for a log-density
        unbounded above, such as a centered hierarchical model.
    settings.optimize_learning_rate : float
        Adam learning rate for that ascent.
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
    optimize_steps: int = 200
    optimize_learning_rate: float = 0.05
