import logging
import warnings as _warnings

from dataclasses import dataclass, field
from typing import Literal

import arviz as az
import numpy as np

from numpy.typing import NDArray
from scipy.special import logsumexp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ImportanceSamplingResult:
    """container for importance sampling results"""

    samples: NDArray
    pareto_k: float | None = None
    warnings: list[str] = field(default_factory=list)
    method: str = "psis"


def importance_sampling(
    samples: NDArray,
    logP: NDArray,
    logQ: NDArray,
    num_draws: int,
    method: Literal["psis", "psir", "identity"] | None,
    random_seed: int | None = None,
) -> ImportanceSamplingResult:
    """Pareto Smoothed Importance Resampling (PSIR)
    This implements the Pareto Smooth Importance Resampling (PSIR) method, as described in Algorithm 5 of Zhang et al. (2022). The PSIR follows a similar approach to Algorithm 1 PSIS diagnostic from Yao et al., (2018). However, before computing the the importance ratio r_s, the logP and logQ are adjusted to account for the number multiple estimators (or paths). The process involves resampling from the original sample with replacement, with probabilities proportional to the computed importance weights from PSIS.

    Parameters
    ----------
    samples : NDArray
        samples from proposal distribution, shape (L, M, N)
    logP : NDArray
        log probability values of target distribution, shape (L, M)
    logQ : NDArray
        log probability values of proposal distribution, shape (L, M)
    num_draws : int
        number of draws to return where num_draws <= samples.shape[0]
    method : str, None, optional
        Method to apply sampling based on log importance weights (logP - logQ).
        Options are:
        "psis" : Pareto Smoothed Importance Sampling (default)
                Recommended for more stable results.
        "psir" : Pareto Smoothed Importance Resampling
                Less stable than PSIS.
        "identity" : Applies log importance weights directly without resampling.
        None : No importance sampling weights. Returns raw samples of size (num_paths, num_draws_per_path, N) where N is number of model parameters. Other methods return samples of size (num_draws, N).
    random_seed : int | None

    Returns
    -------
    ImportanceSamplingResult
        importance sampled draws and other info based on the specified method

    Future work!
    ----------
    - Implement the 3 sampling approaches and 5 weighting functions from Elvira et al. (2019)
    - Implement Algorithm 2 VSBC marginal diagnostics from Yao et al. (2018)
    - Incorporate these various diagnostics, sampling approaches and weighting functions into VI algorithms.

    References
    ----------
    Elvira, V., Martino, L., Luengo, D., & Bugallo, M. F. (2019). Generalized Multiple Importance Sampling. Statistical Science, 34(1), 129-155. https://doi.org/10.1214/18-STS668

    Yao, Y., Vehtari, A., Simpson, D., & Gelman, A. (2018). Yes, but Did It Work?: Evaluating Variational Inference. arXiv:1802.02538 [Stat]. http://arxiv.org/abs/1802.02538

    Zhang, L., Carpenter, B., Gelman, A., & Vehtari, A. (2022). Pathfinder: Parallel quasi-Newton variational inference. Journal of Machine Learning Research, 23(306), 1-49.
    """

    warnings = []
    num_paths, _, N = samples.shape

    if method is None:
        warnings.append(
            "Importance sampling is disabled. The samples are returned as is which may include samples from failed paths with non-finite logP or logQ values. It is recommended to use importance_sampling='psis' for better stability."
        )
        return ImportanceSamplingResult(samples=samples, warnings=warnings, method=method)
    else:
        samples = samples.reshape(-1, N)
        logP = logP.ravel()
        logQ = logQ.ravel()

        # adjust log densities
        log_I = np.log(num_paths)
        logP -= log_I
        logQ -= log_I
        logiw = logP - logQ

        with _warnings.catch_warnings():
            _warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message="overflow encountered in exp"
            )
            match method:
                case "psis":
                    replace = False
                    logiw, pareto_k = az.psislw(logiw)
                case "psir":
                    replace = True
                    logiw, pareto_k = az.psislw(logiw)
                case "identity":
                    replace = False
                    pareto_k = None

    # NOTE: Pareto k is normally bad for Pathfinder even when the posterior is close to the NUTS posterior or closer to NUTS than ADVI.
    # Pareto k may not be a good diagnostic for Pathfinder.
    # TODO: Find replacement diagnostics for Pathfinder.

    p = np.exp(logiw - logsumexp(logiw))
    rng = np.random.default_rng(random_seed)

    try:
        resampled = rng.choice(samples, size=num_draws, replace=replace, p=p, shuffle=False, axis=0)
        return ImportanceSamplingResult(
            samples=resampled, pareto_k=pareto_k, warnings=warnings, method=method
        )
    except ValueError as e1:
        if "Fewer non-zero entries in p than size" in str(e1):
            num_nonzero = np.where(np.nonzero(p)[0], 1, 0).sum()
            warnings.append(
                f"Not enough valid samples: {num_nonzero} available out of {num_draws} requested. Switching to psir importance sampling."
            )
            try:
                resampled = rng.choice(
                    samples, size=num_draws, replace=True, p=p, shuffle=False, axis=0
                )
                return ImportanceSamplingResult(
                    samples=resampled, pareto_k=pareto_k, warnings=warnings, method=method
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
