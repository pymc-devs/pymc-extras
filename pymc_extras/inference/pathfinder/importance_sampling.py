import logging
import warnings

from dataclasses import dataclass, field
from typing import Literal

import arviz_stats.accessors  # noqa: F401 - registers .azstats accessor
import numpy as np
import xarray as xr

from numpy.typing import NDArray
from scipy.special import softmax

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImportanceSamplingResult:
    """container for importance sampling results"""

    samples: NDArray
    pareto_k: float | None = None
    warnings: list[str] = field(default_factory=list)
    method: Literal["psis", "psir", "identity"] | None = "psis"


def importance_sampling(
    samples: NDArray,
    logP: NDArray,
    logQ: NDArray,
    num_draws: int,
    method: Literal["psis", "psir", "identity"] | None,
    random_seed: int | None = None,
) -> ImportanceSamplingResult:
    """Pareto Smoothed Importance Resampling (PSIR).

    Implements the PSIR method from Algorithm 5 of Zhang et al. (2022), which follows the
    Algorithm 1 PSIS diagnostic of Yao et al. (2018). Before computing the importance ratio
    :math:`r_s`, logP and logQ are adjusted to account for the number of estimators (paths).
    Resampling then draws from the original samples with replacement, with probabilities
    proportional to the PSIS importance weights.

    Parameters
    ----------
    samples : NDArray
        Samples from the proposal distribution, shape (L, M, N).
    logP : NDArray
        Log-probability values of the target distribution, shape (L, M).
    logQ : NDArray
        Log-probability values of the proposal distribution, shape (L, M).
    num_draws : int
        Number of draws to return, where num_draws <= samples.shape[0].
    method : str or None, optional
        Method to apply based on log importance weights (logP - logQ):

        - "psis" : Pareto Smoothed Importance Sampling; usually most stable.
        - "psir" : Pareto Smoothed Importance Resampling; less stable than PSIS.
        - "identity" : apply log importance weights directly without resampling.
        - None : no importance sampling; return raw samples of shape
          (num_paths, num_draws_per_path, N). The other methods return shape (num_draws, N).
    random_seed : int, optional
        Random seed for the resampling RNG. Default None.

    Returns
    -------
    ImportanceSamplingResult
        Importance-sampled draws and related diagnostics for the chosen method.

    Notes
    -----
    Possible future extensions: the three sampling approaches and five weighting functions of
    Elvira et al. (2019), and the Algorithm 2 VSBC marginal diagnostics of Yao et al. (2018).

    References
    ----------
    Elvira, V., Martino, L., Luengo, D., & Bugallo, M. F. (2019). Generalized Multiple
    Importance Sampling. Statistical Science, 34(1), 129-155. https://doi.org/10.1214/18-STS668

    Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Yes, but Did It Work?: Evaluating
    Variational Inference. arXiv:1802.02538 [Stat]. http://arxiv.org/abs/1802.02538

    Zhang, L., Carpenter, B., Gelman, A., & Vehtari, A. (2022). Pathfinder: Parallel
    quasi-Newton variational inference. Journal of Machine Learning Research, 23(306), 1-49.
    """

    result_warnings = []
    num_paths, _, N = samples.shape

    if method is None:
        result_warnings.append(
            "Importance sampling is disabled. The samples are returned as is which may "
            "include samples from failed paths with non-finite logP or logQ values. It is "
            "recommended to use importance_sampling='psis' for better stability."
        )
        return ImportanceSamplingResult(samples=samples, warnings=result_warnings, method=method)
    else:
        samples = samples.reshape(-1, N)

        # Adjust log densities for the num_paths-way mixture. Subtract on a fresh array (``ravel``
        # returns a view) so the caller's logP/logQ are not mutated in place.
        log_I = np.log(num_paths)
        logP = logP.ravel() - log_I
        logQ = logQ.ravel() - log_I
        logiw = logP - logQ

        # Filter failed or degenerate paths out of the Pareto fit and give them zero resampling weight
        finite = np.isfinite(logiw)
        if not finite.any():
            raise ValueError(
                "All Pathfinder importance weights are non-finite; the approximation is "
                "degenerate. Consider decreasing the jitter or reparameterizing the model."
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="overflow encountered in exp"
            )
            match method:
                case "psis" | "psir":
                    replace = method == "psir"
                    smoothed = np.full(logiw.shape, -np.inf)
                    logiw_da = xr.DataArray(-logiw[finite], dims=["sample"])
                    logiw_da, pareto_k = logiw_da.azstats.psislw(dim="sample")
                    smoothed[finite] = logiw_da.values
                    logiw, pareto_k = smoothed, float(pareto_k)
                case "identity":
                    replace = False
                    pareto_k = None
                    logiw = np.where(finite, logiw, -np.inf)

    # NOTE: Pareto k is normally bad for Pathfinder even when the posterior is close to the NUTS
    # posterior, or closer to NUTS than ADVI, so it may not be a good diagnostic here.
    # TODO: Find replacement diagnostics for Pathfinder.

    p = softmax(logiw)
    p = np.where(np.isfinite(p), p, 0.0)  # guard against any residual nan/inf from the smoothing
    if not (p_sum := p.sum()) > 0:
        raise ValueError(
            "All Pathfinder importance weights collapsed to zero; the approximation is "
            "degenerate."
        )
    p = p / p_sum
    rng = np.random.default_rng(random_seed)

    try:
        resampled = rng.choice(samples, size=num_draws, replace=replace, p=p, shuffle=False, axis=0)
        return ImportanceSamplingResult(
            samples=resampled, pareto_k=pareto_k, warnings=result_warnings, method=method
        )
    except ValueError as e1:
        msg = str(e1)
        # Both messages mean we cannot draw num_draws distinct samples under replace=False (too few
        # non-zero weights, or num_draws exceeds the pooled population); retry with replacement.
        cannot_draw_distinct = (
            "Fewer non-zero entries in p than size" in msg or "larger sample than population" in msg
        )
        if cannot_draw_distinct:
            num_nonzero = np.count_nonzero(p)
            result_warnings.append(
                f"Not enough valid samples: {num_nonzero} available out of {num_draws} "
                "requested. Switching to psir importance sampling."
            )
            try:
                resampled = rng.choice(
                    samples, size=num_draws, replace=True, p=p, shuffle=False, axis=0
                )
                return ImportanceSamplingResult(
                    samples=resampled, pareto_k=pareto_k, warnings=result_warnings, method=method
                )
            except ValueError as e2:
                logger.error(
                    "Importance sampling failed even with psir importance sampling. "
                    "This might indicate invalid probability weights or insufficient valid samples."
                )
                raise ValueError(
                    "Importance sampling failed for both with and without replacement"
                ) from e2
        raise
