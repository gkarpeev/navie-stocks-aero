"""Check of the width of the parameter distributions on independent DNS realizations.

Within one data set every information estimate treats the residuals as random
with local dependence. The spread of theta_hat over independent realizations of
the flow and of the noise is what the width should estimate, and on synthetic
data it can be measured directly. Realization 0 is the main synthetic data set
(scenario A). Every realization also goes through the decision rule, which
gives the frequency of each decision over independent data.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import config
from .estimation import LOSSES, METHODS, bootstrap
from .experiment import MAIN, NAMES, Design, decision_tests

BOOTSTRAP_REPLICATES = 300
Z95 = 1.959963984540054


def realization(index: int) -> dict:
    """Fits and decisions of one synthetic data set: DNS seed and noise seed shifted by index."""
    data = config.synthetic_data(
        dns_seed=config.DNS_SEED + index, noise_seed=config.NOISE_SEED + index
    )
    design = Design.build(data, config.SYNTHETIC)
    base = design.distributions()
    out = {"index": index, "re_lambda": data.meta["end"]["re_lambda"]}
    for loss in LOSSES:
        out[loss] = {"theta": base[loss, "fisher"].mean.tolist()}
        for method in METHODS:
            out[loss][method] = base[loss, method].std.tolist()
    replicates = bootstrap(design.X, design.y, design.clusters, BOOTSTRAP_REPLICATES, seed=index)
    out["l2"]["bootstrap"] = replicates.std(axis=0, ddof=1).tolist()
    tests = decision_tests(design, config.PLAN, key=1000 + index, base=base)
    out["keep"] = {e: {m: t["keep"] for m, t in tests[e]["by_method"].items()} for e in NAMES}
    out["kl_shift"] = {e: tests[e]["kl_shift"] for e in NAMES}
    return out


def summarise(results: list[dict]) -> dict:
    """Spread over realizations against the mean width of each estimate; decision frequencies."""
    summary = {}
    for loss in LOSSES:
        theta = np.array([r[loss]["theta"] for r in results])
        centre, spread = theta.mean(axis=0), theta.std(axis=0, ddof=1)
        methods = [m for m in (*METHODS, "bootstrap") if m in results[0][loss]]
        sigma = {m: np.array([r[loss][m] for r in results]) for m in methods}
        summary[loss] = {
            "theta_mean": centre.tolist(),
            "theta_spread": spread.tolist(),
            "sigma_mean": {m: s.mean(axis=0).tolist() for m, s in sigma.items()},
            "ratio": {m: (s.mean(axis=0) / spread).tolist() for m, s in sigma.items()},
            # Share of realizations whose 95% interval covers the mean over realizations.
            "coverage": {
                m: np.mean(np.abs(theta - centre) < Z95 * s, axis=0).tolist()
                for m, s in sigma.items()
            },
        }
    keys = results[0]["keep"]["A"].keys()
    summary["keep_share"] = {
        m: {e: float(np.mean([r["keep"][e][m] for r in results])) for e in NAMES} for m in keys
    }
    summary["main"] = f"{MAIN[0]}/{MAIN[1]}"
    return summary


def run(workers: int, count: int = config.MONTE_CARLO) -> dict:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(realization, range(count)))
    return {
        "dns_seeds": [config.DNS_SEED + i for i in range(count)],
        "noise_seeds": [config.NOISE_SEED + i for i in range(count)],
        "realizations": results,
        "summary": summarise(results),
    }
