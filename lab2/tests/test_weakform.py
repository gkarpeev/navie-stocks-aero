import numpy as np
import pytest

from lab02.estimation import fit_model
from lab02.weakform import WeakForm, branch_features, bump_kernels


def test_bump_derivatives_vanish_at_the_support_edges():
    kernels = bump_kernels(6, 0.1, power=7, max_order=3)
    assert np.allclose(kernels[:, [0, -1]], 0)
    with pytest.raises(ValueError):
        bump_kernels(6, 0.1, power=3, max_order=3)


def _ibp_error(m: int, order: int) -> float:
    """Relative error of <psi^(n), f> = (-1)^n <psi, f^(n)> on a support of length 0.8."""
    h = 0.4 / m
    s = np.arange(-m, m + 1) * h
    k = bump_kernels(m, h, power=7, max_order=3)
    f = np.sin(3 * s + 0.4)
    dnf = 3.0**order * np.sin(3 * s + 0.4 + order * np.pi / 2)
    exact = (-1) ** order * np.sum(k[0] * dnf)
    return abs(np.sum(k[order] * f) - exact) / np.sum(np.abs(k[0] * dnf))


def test_discrete_integration_by_parts_converges():
    # The identity behind every weak column; the third derivative (viscous
    # branch) is the least accurate and sets the minimum support width.
    assert _ibp_error(8, 1) < 1e-5
    assert _ibp_error(8, 3) < 1e-2
    assert _ibp_error(16, 3) < _ibp_error(8, 3) / 30


def _abc(n, frames, dt, drift, nu):
    """Translating, decaying ABC flow: an exact Navier-Stokes solution."""
    x = np.arange(n) * 2 * np.pi / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    out = []
    for i in range(frames):
        t = i * dt
        xs = X - drift * t
        amp = np.exp(-nu * t)
        out.append(
            np.stack(
                [
                    drift + amp * (np.sin(Z) + np.cos(Y)),
                    amp * (np.sin(xs) + np.cos(Z)),
                    amp * (np.sin(Y) + np.cos(xs)),
                ]
            )
        )
    return np.array(out)


def _abc_residual(n: int, mx: int):
    nu, dt, mt = 0.2, 0.02, 8
    u = _abc(n, 2 * mt + 17, dt, drift=1.5, nu=nu)
    form = WeakForm(u.shape, 2 * np.pi / n, dt, mx=mx, mt=mt, stride_x=2, stride_t=mt, power=7)
    f = branch_features(form, {"n01": u, "n03": u, "n13": u, "d": u}, nu)
    y = form.target(u)
    # ABC: Lap u = -u for the vortex part, so V = nu * D up to quadrature; the
    # pair (V, D) is collinear and the regression uses (A, V) only.
    theta = np.linalg.lstsq(np.stack([f["A"], f["V"]], 1), y, rcond=None)[0]
    return theta, np.linalg.norm(y - f["A"] - f["V"]) / np.linalg.norm(y)


def test_exact_solution_gives_unit_coefficients():
    # Same physical support, twice the resolution: the quadrature error of the
    # weak columns must collapse and theta converge to (1, 1).
    coarse_theta, coarse = _abc_residual(24, 6)
    fine_theta, fine = _abc_residual(48, 12)
    assert coarse < 3e-3 and fine < coarse / 30
    np.testing.assert_allclose(coarse_theta, [1.0, 1.0], atol=1.5e-2)
    np.testing.assert_allclose(fine_theta, [1.0, 1.0], atol=5e-4)


def test_pressure_column_is_exactly_zero(rng):
    shape = (21, 3, 15, 15, 15)
    form = WeakForm(shape, 0.1, 0.01, mx=4, mt=5, stride_x=3, stride_t=5, power=7)
    column = form.pressure(rng.standard_normal((21, 15, 15, 15)), rho=1.3)
    assert column.shape == (form.n_rows,) and np.all(column == 0.0)


def test_rows_and_clusters_are_consistent():
    shape = (31, 3, 20, 18, 16)
    form = WeakForm(shape, 0.1, 0.01, mx=4, mt=5, stride_x=2, stride_t=5, power=7)
    index = form.row_index()
    assert all(len(v) == form.n_rows for v in index.values())
    clusters = form.clusters()
    # Components of one centre share data, hence one cluster.
    for c in (1, 2):
        np.testing.assert_array_equal(
            clusters[index["component"] == 0], clusters[index["component"] == c]
        )


def test_support_larger_than_data_is_rejected():
    with pytest.raises(ValueError):
        WeakForm((10, 3, 20, 20, 20), 0.1, 0.01, mx=4, mt=5, stride_x=2, stride_t=5)


def test_dns_recovers_the_true_structure(small_dns):
    d = small_dns
    form = WeakForm(d.velocity.shape, d.h, d.dt, mx=5, mt=6, stride_x=2, stride_t=6, power=7)
    u = d.velocity
    f = branch_features(form, {"n01": u, "n03": u, "n13": u, "d": u}, d.nu)
    fit = fit_model(np.stack([f[k] for k in "AVD"], 1), form.target(u), ("A", "V", "D"), "l2")
    assert abs(fit.theta[0] - 1) < 2e-3
    assert abs(fit.theta[1] - 1) < 2e-2
    assert abs(fit.theta[2]) < 2e-2
    assert fit.r2 > 0.999


def test_damping_column_is_minus_u():
    # On ABC flow nu*Lap(u) = -nu*(u - drift), and a constant field has a zero
    # weak column (div of the curl test function); hence the viscous term is
    # the control -alpha*u with alpha = +nu, and a sign error would give -nu.
    nu, dt, mt, n = 0.2, 0.02, 8, 24
    u = _abc(n, 2 * mt + 17, dt, drift=1.5, nu=nu)
    form = WeakForm(u.shape, 2 * np.pi / n, dt, mx=6, mt=mt, stride_x=2, stride_t=mt, power=7)
    f = branch_features(form, {"n01": u, "n03": u, "n13": u, "d": u}, nu)
    theta = np.linalg.lstsq(np.stack([f["A"], f["D"]], 1), form.target(u), rcond=None)[0]
    np.testing.assert_allclose(theta, [1.0, nu], rtol=2e-3)


def test_windows_group_the_three_components_of_a_centre():
    shape = (31, 3, 20, 18, 16)
    form = WeakForm(shape, 0.1, 0.01, mx=4, mt=5, stride_x=2, stride_t=5, power=7)
    windows = form.windows()
    index = form.row_index()
    n = int(np.prod([len(c) for c in form.centers]))
    assert sorted(set(windows.tolist())) == list(range(n))
    for c in (1, 2):
        np.testing.assert_array_equal(
            windows[index["component"] == 0], windows[index["component"] == c]
        )
