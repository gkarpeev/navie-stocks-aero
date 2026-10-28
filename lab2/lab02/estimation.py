"""Parameter distributions of a model linear in its structural parameters (method 1.4).

The weak-form model is y = X theta + r. The error function S(theta) defines the
likelihood, and the Fisher information gives the Laplace (asymptotic normal)
distribution N(theta_hat, F^-1) (MacKay 1992; Kuznetsov, Tokmakova, Strijov
2016). The distribution of a subvector is the corresponding block of F^-1.

Error functions:

* ``l2``    - Gaussian likelihood, S = sum r^2;
* ``huber`` - robust M-estimator, quadratic near zero and linear in the tails (Huber 1964).

Information estimates:

* ``fisher``  - Fisher information for independent rows, F = X'X / sigma^2;
* ``godambe`` - Godambe (sandwich) information H J^-1 H, where J is the
  covariance of the score summed over clusters of overlapping test windows:
  the Fisher matrix generalised to dependent rows (Godambe 1960; Liang &
  Zeger 1986). For independent rows and the L2 error it equals ``fisher``.

The cluster bootstrap checks the Gaussian shape; the width is checked on
independent DNS realizations in ``montecarlo.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HUBER_C = 1.345  # 95% efficiency at the normal model
METHODS = ("fisher", "godambe")
LOSSES = ("l2", "huber")


@dataclass(frozen=True)
class Gaussian:
    """Multivariate normal distribution of named parameters."""

    names: tuple
    mean: np.ndarray
    cov: np.ndarray

    def __post_init__(self):
        k = len(self.names)
        if self.mean.shape != (k,) or self.cov.shape != (k, k):
            raise ValueError("Mean and covariance do not match the parameter names")
        if not np.allclose(self.cov, self.cov.T, rtol=1e-8, atol=0):
            raise ValueError("Covariance must be symmetric")
        if np.linalg.eigvalsh(self.cov).min() <= 0:
            raise ValueError("Covariance must be positive definite")

    def marginal(self, names) -> "Gaussian":
        """Distribution of a subvector: the corresponding block of the covariance."""
        names = tuple([names] if isinstance(names, str) else names)
        idx = [self.names.index(n) for n in names]
        return Gaussian(names, self.mean[idx], self.cov[np.ix_(idx, idx)])

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def to_dict(self) -> dict:
        return {"names": list(self.names), "mean": self.mean.tolist(), "cov": self.cov.tolist()}

    @classmethod
    def from_dict(cls, payload: dict) -> "Gaussian":
        return cls(tuple(payload["names"]), np.array(payload["mean"]), np.array(payload["cov"]))


@dataclass(frozen=True)
class Fit:
    """Point estimate with the quantities every information estimate needs."""

    loss: str
    names: tuple
    theta: np.ndarray
    residual: np.ndarray
    scale: float  # residual scale: sigma for l2, the fixed robust scale for huber
    psi: np.ndarray  # dS/dr per row (the score is X * psi)
    dpsi: np.ndarray  # derivative of psi (1 for l2)
    r2: float


def fit_l2(X: np.ndarray, y: np.ndarray, names) -> Fit:
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ theta
    n, p = X.shape
    sigma = float(np.sqrt(r @ r / (n - p)))
    return Fit("l2", tuple(names), theta, r, sigma, r.copy(), np.ones_like(r), _r2(y, r))


def fit_huber(
    X: np.ndarray, y: np.ndarray, names, c: float = HUBER_C, max_iter: int = 500, tol: float = 1e-10
) -> Fit:
    """IRLS with the scale fixed at the normalised MAD of the L2 residuals.

    A fixed preliminary scale keeps the estimator a plain M-estimator, for which
    the sandwich covariance is the textbook asymptotic result.
    """
    start = fit_l2(X, y, names)
    scale = 1.4826 * float(np.median(np.abs(start.residual - np.median(start.residual))))
    if scale <= 0:
        raise ValueError("Degenerate residual scale")
    theta = start.theta
    for _ in range(max_iter):
        r = y - X @ theta
        w = np.minimum(1.0, c * scale / np.maximum(np.abs(r), 1e-300))
        new = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
        converged = np.max(np.abs(new - theta)) <= tol * (1 + np.max(np.abs(theta)))
        theta = new
        if converged:
            break
    else:
        raise RuntimeError("Huber IRLS did not converge")
    r = y - X @ theta
    psi = np.clip(r, -c * scale, c * scale)
    dpsi = (np.abs(r) < c * scale).astype(float)
    if dpsi.sum() <= X.shape[1]:
        raise ValueError("Too few rows inside the Huber band to estimate the information")
    return Fit("huber", tuple(names), theta, r, scale, psi, dpsi, _r2(y, r))


def _r2(y: np.ndarray, r: np.ndarray) -> float:
    # Uncentred: the weak-form target has no intercept and zero mean by construction.
    return float(1 - (r @ r) / (y @ y))


def fit_model(X: np.ndarray, y: np.ndarray, names, loss: str) -> Fit:
    if loss == "l2":
        return fit_l2(X, y, names)
    if loss == "huber":
        return fit_huber(X, y, names)
    raise ValueError(f"Unknown error function {loss!r}")


def _gaussian(fit: Fit, cov: np.ndarray) -> Gaussian:
    return Gaussian(fit.names, fit.theta, 0.5 * (cov + cov.T))


def fisher(fit: Fit, X: np.ndarray) -> Gaussian:
    """Inverse Fisher information assuming independent rows.

    l2: sigma^2 (X'X)^-1. huber: E[psi^2] / E[psi']^2 (X'X)^-1 (Huber 1964).
    """
    xtx_inv = np.linalg.inv(X.T @ X)
    if fit.loss == "l2":
        factor = fit.scale**2
    else:
        factor = float(np.mean(fit.psi**2) / np.mean(fit.dpsi) ** 2)
    return _gaussian(fit, factor * xtx_inv)


def godambe(fit: Fit, X: np.ndarray, clusters: np.ndarray) -> Gaussian:
    """Inverse Godambe information H^-1 J H^-1 with cluster-summed scores.

    H = X' diag(psi') X is the observed information and J the covariance of the
    score summed within clusters; G/(G-1) is the usual small-sample factor.
    """
    bread = np.linalg.inv(X.T @ (fit.dpsi[:, None] * X))
    labels, inverse = np.unique(clusters, return_inverse=True)
    g = len(labels)
    if g < 10 * X.shape[1]:
        raise ValueError(f"Only {g} clusters for {X.shape[1]} parameters")
    scores = np.zeros((g, X.shape[1]))
    np.add.at(scores, inverse, X * fit.psi[:, None])
    return _gaussian(fit, g / (g - 1) * bread @ (scores.T @ scores) @ bread)


def information(fit: Fit, X: np.ndarray, clusters: np.ndarray, method: str) -> Gaussian:
    if method == "fisher":
        return fisher(fit, X)
    if method == "godambe":
        return godambe(fit, X, clusters)
    raise ValueError(f"Unknown information estimate {method!r}")


def distributions(X: np.ndarray, y: np.ndarray, names, clusters: np.ndarray) -> dict:
    """Every (error function, information estimate) distribution and the R^2 per error function."""
    out = {}
    for loss in LOSSES:
        fit = fit_model(X, y, names, loss)
        for method in METHODS:
            out[loss, method] = information(fit, X, clusters, method)
        out[loss, "r2"] = fit.r2
    return out


def bootstrap(
    X: np.ndarray, y: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int
) -> np.ndarray:
    """L2 cluster-bootstrap replicates of theta_hat, shape (n_boot, p).

    The normal equations are additive over clusters, so a replicate is the
    solution with resampled per-cluster sums of X'X and X'y.
    """
    rng = np.random.default_rng(seed)
    labels, inverse = np.unique(clusters, return_inverse=True)
    g, p = len(labels), X.shape[1]
    xtx = np.zeros((g, p, p))
    xty = np.zeros((g, p))
    np.add.at(xtx, inverse, X[:, :, None] * X[:, None, :])
    np.add.at(xty, inverse, X * y[:, None])
    out = np.empty((n_boot, p))
    for b in range(n_boot):
        pick = rng.integers(0, g, g)
        out[b] = np.linalg.solve(xtx[pick].sum(0), xty[pick].sum(0))
    return out
