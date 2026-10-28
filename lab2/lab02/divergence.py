"""Kullback-Leibler divergence between Gaussian parameter distributions (method 3.2).

For p = N(m_p, S_p) and q = N(m_q, S_q) in k dimensions (Kullback & Leibler 1951):

    KL(p || q) = 1/2 (m_p - m_q)' S_q^-1 (m_p - m_q)                       shift
               + 1/2 [tr(S_q^-1 S_p) - k - ln det(S_q^-1 S_p)]              shape

The shift part measures how far the centre moved, in units of the width of q;
the shape part how much the width changed. The intervention is compared with
the original data as KL(after || before), so the shift is measured in units of
the original distribution. For nearby distributions KL ~ 1/2 d' F d: the
Fisher matrix is the metric of KL (Amari 2016), which makes the triple
"Fisher - intervention - KL" consistent.
"""

from __future__ import annotations

import numpy as np

from .estimation import Gaussian

# Half the chi^2_1 quantile 3.841: the KL of a centre shifted by 1.96 sigma (5% level).
THRESHOLD = 0.5 * 3.841458820694124


def kl_parts(p: Gaussian, q: Gaussian) -> dict[str, float]:
    """Shift and shape parts of KL(p || q); their sum is the divergence."""
    if p.names != q.names:
        raise ValueError("Distributions must describe the same parameters")
    chol = np.linalg.cholesky(q.cov)
    delta = np.linalg.solve(chol, p.mean - q.mean)
    whitened = np.linalg.solve(chol, np.linalg.solve(chol, p.cov).T)
    k = len(p.names)
    logdet_q = 2 * np.log(np.diag(chol)).sum()
    logdet_p = np.linalg.slogdet(p.cov)[1]
    shift = 0.5 * float(delta @ delta)
    shape = 0.5 * float(np.trace(whitened) - k + logdet_q - logdet_p)
    return {"shift": shift, "shape": shape, "total": shift + shape}


def kl(p: Gaussian, q: Gaussian) -> float:
    return kl_parts(p, q)["total"]


def removal_kl(g: Gaussian, name: str) -> float:
    """S_e: KL from N(theta, S) to the nearest same-covariance Gaussian with theta_e = 0.

    The minimiser is the Optimal Brain Surgeon step (Hassibi & Stork 1993); the
    value theta_e^2 / (2 S_ee) is half the squared Wald statistic. It is the
    limit of the shift part of KL(after || before) when an intervention removes
    all information about theta_e.
    """
    i = g.names.index(name)
    return float(g.mean[i] ** 2 / (2 * g.cov[i, i]))
