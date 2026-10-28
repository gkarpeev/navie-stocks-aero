import numpy as np
from scipy.optimize import minimize

from lab02.divergence import THRESHOLD, kl, kl_parts, removal_kl
from lab02.estimation import Gaussian


def _random_gaussian(rng, k=3, names=("A", "V", "D")):
    a = rng.standard_normal((k, k))
    return Gaussian(names, rng.standard_normal(k), a @ a.T + 0.5 * np.eye(k))


def test_closed_form_matches_monte_carlo(rng):
    p, q = _random_gaussian(rng), _random_gaussian(rng)
    x = rng.multivariate_normal(p.mean, p.cov, size=400_000)

    def logpdf(g, x):
        d = x - g.mean
        inv = np.linalg.inv(g.cov)
        return -0.5 * (
            np.einsum("ij,jk,ik->i", d, inv, d) + np.linalg.slogdet(2 * np.pi * g.cov)[1]
        )

    estimate = np.mean(logpdf(p, x) - logpdf(q, x))
    assert abs(estimate - kl(p, q)) < 0.02 * max(1.0, kl(p, q))


def test_parts_identity_and_zero_self_divergence(rng):
    p, q = _random_gaussian(rng), _random_gaussian(rng)
    parts = kl_parts(p, q)
    assert np.isclose(parts["shift"] + parts["shape"], parts["total"])
    delta = q.mean - p.mean
    # The shift is measured with the covariance of the second argument.
    assert np.isclose(parts["shift"], 0.5 * delta @ np.linalg.solve(q.cov, delta))
    assert abs(kl(p, p)) < 1e-12 and parts["shape"] >= 0


def test_equal_covariances_give_the_fisher_metric(rng):
    p = _random_gaussian(rng)
    step = 1e-3 * rng.standard_normal(3)
    q = Gaussian(p.names, p.mean + step, p.cov)
    assert np.isclose(kl(q, p), 0.5 * step @ np.linalg.solve(p.cov, step), rtol=1e-12)


def test_removal_kl_is_the_optimal_brain_surgeon_saliency(rng):
    g = _random_gaussian(rng)
    inv = np.linalg.inv(g.cov)
    i = 1
    # Minimum of 1/2 d' S^-1 d subject to (theta + d)_i = 0.
    result = minimize(
        lambda d: 0.5 * d @ inv @ d,
        np.zeros(3),
        constraints={"type": "eq", "fun": lambda d: g.mean[i] + d[i]},
        method="SLSQP",
        options={"ftol": 1e-14},
    )
    assert np.isclose(result.fun, removal_kl(g, "V"), rtol=1e-6)
    # The 5% Wald threshold: |z| = 1.96 sits exactly on it.
    one = Gaussian(("A",), np.array([1.959963984540054]), np.array([[1.0]]))
    assert np.isclose(removal_kl(one, "A"), THRESHOLD)


def test_shift_of_an_attenuated_parameter_is_rho_squared_times_removal_kl(rng):
    # The identity behind the decision: theta_e -> (1 - rho) theta_e gives a
    # shift part rho^2 S_e whatever happens to the width, which only adds a
    # shape part on top.
    before = _random_gaussian(rng)
    scale = np.diag([1.0, 1.7, 1.0])
    widened = scale @ before.cov @ scale
    for rho in (0.1, 0.5, 1.0):
        mean = before.mean * np.array([1.0, 1.0 - rho, 1.0])
        after = Gaussian(before.names, mean, widened)
        parts = kl_parts(after.marginal("V"), before.marginal("V"))
        assert np.isclose(parts["shift"], rho**2 * removal_kl(before, "V"), rtol=1e-12)
        assert parts["shape"] > 0.1
