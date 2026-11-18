"""Compare sparse count logp/gradient kernels on a common observed-zero grid.

Run from an environment with this checkout installed::

    python scripts/benchmark_sparse_distributions.py --mode NUMBA

Every distribution uses the same fixed observed count tensor at each grid point.
Masks are nested: increasing sparsity removes entries without changing the
remaining positive counts. All timings include gradients through learned
probabilities or means and, where applicable, a learned dispersion/concentration.
Compilation and the first call are reported separately. Multinomial, binomial
and Poisson sparse expressions are benchmark-only controls, not APIs.
"""

import argparse
import json
import time

from statistics import median

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pymc.distributions.dist_math import binomln, check_parameters, factln, logpow

from pymc_extras import SparseBetaBinomial, SparseDirichletMultinomial, SparseNegativeBinomial
from pymc_extras.distributions.multivariate.sparse import _sparse_sum_last_axis

DISTRIBUTIONS = [
    "DirichletMultinomial",
    "Multinomial",
    "NegativeBinomial",
    "BetaBinomial",
    "Binomial",
    "Poisson",
]


def logps(name, counts, theta, log_scale, *, trials):
    """Build the native reference and a sparse expression at the same parameters."""
    positive = np.nonzero(counts > 0)
    x = counts[positive]
    if name in {"DirichletMultinomial", "Multinomial"}:
        p = pt.special.softmax(theta, axis=-1)
        n = counts.sum(axis=-1)
        if name == "DirichletMultinomial":
            a = pt.exp(log_scale) * p
            return (
                pm.logp(pm.DirichletMultinomial.dist(n=n, a=a), counts),
                pm.logp(SparseDirichletMultinomial.dist(n=n, a=a), counts),
            )
        sparse = factln(n) + _sparse_sum_last_axis(
            pt.as_tensor(counts), p, lambda x, p: logpow(p, x) - factln(x)
        )
        return pm.logp(pm.Multinomial.dist(n=n, p=p), counts), check_parameters(
            sparse,
            0 <= p,
            p <= 1,
            pt.isclose(p.sum(axis=-1), 1),
            n >= 0,
            msg="0 <= p <= 1, sum(p) = 1, n >= 0",
        )
    mu = pt.exp(theta)
    if name == "NegativeBinomial":
        alpha = pt.exp(log_scale)
        return (
            pm.logp(pm.NegativeBinomial.dist(mu=mu, alpha=alpha), counts),
            pm.logp(SparseNegativeBinomial.dist(mu=mu, alpha=alpha), counts),
        )
    if name == "Poisson":
        ref = pm.logp(pm.Poisson.dist(mu=mu), counts)
        sparse = pt.inc_subtensor((-mu)[positive], logpow(mu[positive], x) - factln(x))
        return ref, check_parameters(sparse, mu >= 0, msg="mu >= 0")

    n = np.int64(trials)
    p = pt.sigmoid(theta)
    if name == "Binomial":
        ref = pm.logp(pm.Binomial.dist(n=n, p=p), counts)
        # Replace positive-count entries rather than subtracting the zero baseline:
        # this avoids infinity cancellation at boundary probabilities.
        nonzero_logp = binomln(n, x) + logpow(p[positive], x) + logpow(1 - p[positive], n - x)
        sparse = pt.set_subtensor(logpow(1 - p, n)[positive], nonzero_logp)
        return ref, check_parameters(sparse, n >= 0, 0 <= p, p <= 1, msg="n >= 0, 0 <= p <= 1")

    kappa = pt.exp(log_scale)
    a, b = kappa * p, kappa * (1 - p)
    return (
        pm.logp(pm.BetaBinomial.dist(n=n, alpha=a, beta=b), counts),
        pm.logp(SparseBetaBinomial.dist(n=n, alpha=a, beta=b), counts),
    )


def benchmark(name, counts, values, *, mode, repeats, trials):
    theta = pt.matrix("theta")
    log_scale = pt.scalar("log_scale")
    learned_scale = name in {"DirichletMultinomial", "NegativeBinomial", "BetaBinomial"}
    scale = {"DirichletMultinomial": 100.0, "NegativeBinomial": 2.0, "BetaBinomial": 10.0}
    inputs, point = [theta], [values]
    if learned_scale:
        inputs.append(log_scale)
        point.append(np.asarray(np.log(scale[name]), dtype=pytensor.config.floatX))
    fns, outputs, compile_times = [], [], []
    for expr in logps(name, counts, theta, log_scale, trials=trials):
        logp = expr.sum()
        start = time.perf_counter()
        fn = pytensor.function(
            inputs,
            [logp, *pt.grad(logp, inputs, disconnected_inputs="ignore")],
            mode=mode,
            on_unused_input="ignore",
        )
        outputs.append(fn(*point))
        compile_times.append(time.perf_counter() - start)
        fns.append(fn)

    tolerance = 1e-4 if pytensor.config.floatX == "float32" else 1e-8
    errors = []
    for dense, sparse in zip(outputs[0], outputs[1], strict=True):
        np.testing.assert_allclose(dense, sparse, rtol=tolerance, atol=tolerance)
        errors.append(float(np.max(np.abs(dense - sparse))))
    for fn in fns:
        for _ in range(3):
            fn(*point)
    samples = [[], []]
    for repeat in range(repeats):
        # Alternate order to reduce drift between the reference and sparse timings.
        for index in [0, 1] if repeat % 2 == 0 else [1, 0]:
            start = time.perf_counter()
            fns[index](*point)
            samples[index].append(time.perf_counter() - start)
    timings = [1000 * median(sample) for sample in samples]
    constant_likelihood = not counts.any() and name in {"DirichletMultinomial", "Multinomial"}
    return {
        "distribution": name,
        "prototype": name not in {"DirichletMultinomial", "NegativeBinomial", "BetaBinomial"},
        "zero_fraction": float(np.mean(counts == 0)),
        "shape": list(counts.shape),
        "dense_ms": timings[0],
        "sparse_ms": timings[1],
        "speedup": None if constant_likelihood else timings[0] / timings[1],
        "constant_likelihood": constant_likelihood,
        "max_abs_errors_logp_then_gradients": errors,
        "compile_and_first_call_seconds": compile_times,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--categories", type=int, default=128)
    parser.add_argument(
        "--zero-percentages", type=float, nargs="+", default=[0, 5, 25, 50, 75, 95, 100]
    )
    parser.add_argument("--distributions", choices=DISTRIBUTIONS, nargs="+", default=DISTRIBUTIONS)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--mode", choices=["NUMBA", "FAST_RUN"], default="NUMBA")
    args = parser.parse_args()
    if min(args.rows, args.categories, args.repeats) < 1:
        parser.error("rows, categories and repeats must be positive")
    if any(not 0 <= percentage <= 100 for percentage in args.zero_percentages):
        parser.error("zero percentages must lie in [0, 100]")
    print(
        json.dumps(
            {
                "pymc": pm.__version__,
                "pytensor": pytensor.__version__,
                "mode": args.mode,
                "dtype": pytensor.config.floatX,
                "repeats": args.repeats,
                "seed": 602,
                "positive_counts": "integers 1 through 4; identical across distributions and masks",
                "dm_concentration": 100.0,
                "nb_dispersion": 2.0,
                "bb_concentration": 10.0,
                "binomial_trials": 20,
                "dm_multinomial_trials": "row sums of observed counts",
            }
        ),
        flush=True,
    )
    rng = np.random.default_rng(602)
    shape = (args.rows, args.categories)
    positive_counts = rng.integers(1, 5, shape, dtype="int64")
    order = rng.permutation(positive_counts.size)
    values = rng.normal(0, 0.4, shape).astype(pytensor.config.floatX)
    for percentage in args.zero_percentages:
        counts = positive_counts.copy()
        counts.flat[order[: round(counts.size * percentage / 100)]] = 0
        for name in args.distributions:
            result = benchmark(
                name, counts, values, mode=args.mode, repeats=args.repeats, trials=20
            )
            result["target_zero_percentage"] = percentage
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
