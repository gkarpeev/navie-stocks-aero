import numpy as np
import pytest

from lab02.estimation import (
    Gaussian,
    bootstrap,
    distributions,
    fisher,
    fit_huber,
    fit_l2,
    godambe,
)

NAMES = ("a", "b")


def _clustered(rng, n_clusters=200, size=20, rho=0.0, outliers=0.0):
    """y = X theta + e with exchangeable within-cluster correlation rho."""
    X = rng.standard_normal((n_clusters * size, 2)) + np.repeat(
        rng.standard_normal((n_clusters, 2)), size, axis=0
    )  # correlated regressors too
    shared = np.repeat(rng.standard_normal(n_clusters), size)
    e = np.sqrt(rho) * shared + np.sqrt(1 - rho) * rng.standard_normal(len(X))
    if outliers:
        hit = rng.random(len(X)) < outliers
        e = np.where(hit, 50.0 * np.sign(rng.standard_normal(len(X))), e)
    clusters = np.repeat(np.arange(n_clusters), size)
    return X, X @ np.array([1.0, -0.5]) + e, clusters


def test_l2_fisher_matches_the_closed_form(rng):
    X, y, _ = _clustered(rng)
    fit = fit_l2(X, y, NAMES)
    g = fisher(fit, X)
    sigma2 = fit.residual @ fit.residual / (len(y) - 2)
    np.testing.assert_allclose(g.cov, sigma2 * np.linalg.inv(X.T @ X), rtol=1e-12)
    np.testing.assert_allclose(g.mean, [1.0, -0.5], atol=0.1)


@pytest.mark.parametrize("rho", [0.0, 0.5])
def test_coverage_of_fisher_and_godambe(rho):
    # Independent rows: both cover ~95%. Correlated rows: only Godambe does.
    rng = np.random.default_rng(7)
    hits = {"fisher": 0, "godambe": 0}
    reps = 200
    for _ in range(reps):
        X, y, clusters = _clustered(rng, n_clusters=60, size=10, rho=rho)
        fit = fit_l2(X, y, NAMES)
        for name, g in (("fisher", fisher(fit, X)), ("godambe", godambe(fit, X, clusters))):
            hits[name] += abs(g.mean[0] - 1.0) < 1.96 * g.std[0]
    assert 0.88 < hits["godambe"] / reps < 0.99
    if rho == 0.0:
        assert 0.88 < hits["fisher"] / reps < 0.99
    else:
        assert hits["fisher"] / reps < 0.80


def test_huber_resists_gross_errors_that_bias_l2(rng):
    X, y, clusters = _clustered(rng, rho=0.0, outliers=0.05)
    y = y + 50.0 * (X[:, 0] > 2.0)  # outliers aligned with a regressor bias l2
    l2, hub = fit_l2(X, y, NAMES), fit_huber(X, y, NAMES)
    assert abs(hub.theta[0] - 1.0) < 0.3 * abs(l2.theta[0] - 1.0)
    assert np.all(np.linalg.eigvalsh(godambe(hub, X, clusters).cov) > 0)


def test_cluster_bootstrap_agrees_with_godambe(rng):
    X, y, clusters = _clustered(rng, rho=0.4)
    g = godambe(fit_l2(X, y, NAMES), X, clusters)
    draws = bootstrap(X, y, clusters, n_boot=600, seed=3)
    np.testing.assert_allclose(draws.std(axis=0), g.std, rtol=0.2)


def test_bootstrap_replicate_is_least_squares_on_resampled_clusters(rng):
    # The cluster-sum shortcut must equal a refit on the physically resampled rows.
    X, y, clusters = _clustered(rng, n_clusters=30, size=4)
    draw = bootstrap(X, y, clusters, n_boot=1, seed=5)[0]
    pick = np.random.default_rng(5).integers(0, 30, 30)
    rows = np.concatenate([np.flatnonzero(clusters == g) for g in pick])
    np.testing.assert_allclose(draw, fit_l2(X[rows], y[rows], NAMES).theta, rtol=1e-9)


def test_all_distributions_share_the_point_estimate_per_loss(rng):
    X, y, clusters = _clustered(rng)
    d = distributions(X, y, NAMES, clusters)
    for loss in ("l2", "huber"):
        np.testing.assert_array_equal(d[loss, "fisher"].mean, d[loss, "godambe"].mean)
        assert 0.5 < d[loss, "r2"] < 1


def test_gaussian_rejects_invalid_covariances():
    with pytest.raises(ValueError):
        Gaussian(("a",), np.zeros(1), np.array([[0.0]]))
    with pytest.raises(ValueError):
        Gaussian(("a", "b"), np.zeros(2), np.array([[1.0, 0.5], [0.4, 1.0]]))


def test_too_few_clusters_fail_loudly(rng):
    X, y, _ = _clustered(rng, n_clusters=20, size=50)
    with pytest.raises(ValueError):
        godambe(fit_l2(X, y, NAMES), X, np.repeat(np.arange(5), 200))
