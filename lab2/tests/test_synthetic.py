import numpy as np

from lab02.synthetic import SpectralNS, with_noise


def _beltrami(n):
    x = np.arange(n) * 2 * np.pi / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    return np.stack([np.sin(Z) + np.cos(Y), np.sin(X) + np.cos(Z), np.sin(Y) + np.cos(X)])


def test_beltrami_mode_decays_exactly():
    # ABC flow: the nonlinear term is a gradient, so u(t) = u(0) exp(-(nu + alpha) t).
    for alpha in (0.0, 0.3):
        n, nu, dt = 16, 0.1, 0.01
        model = SpectralNS(n, nu, alpha)
        u0 = _beltrami(n)
        uh = model.spectral(u0)
        for _ in range(50):
            uh = model.step(uh, dt)
        np.testing.assert_allclose(model.physical(uh), u0 * np.exp(-(nu + alpha) * 0.5), atol=1e-9)


def test_pressure_satisfies_the_poisson_equation():
    # Lap p = -d_i d_j (u_i u_j)  <=>  -k^2 p_hat = k_i k_j FT(u_i u_j).
    model = SpectralNS(16, 0.05)
    uh = model.random_field(k0=2.0, urms=1.0, seed=3)
    u = model.physical(uh)
    lhs = -model.k2 * model.spectral(model.pressure(uh))
    rhs = sum(
        model.k[i] * model.k[j] * model.spectral(u[i] * u[j]) for i in range(3) for j in range(3)
    )
    mask = model.dealias & (model.k2 > 0)
    np.testing.assert_allclose(lhs[mask], rhs[mask], rtol=0, atol=1e-10 * np.abs(rhs).max())


def test_simulation_is_solenoidal_and_decaying(small_dns):
    meta = small_dns.meta
    assert meta["start"]["max_divergence"] < 1e-9 and meta["end"]["max_divergence"] < 1e-9
    assert meta["end"]["urms"] < meta["start"]["urms"]
    assert meta["start"]["kmax_eta"] > 1.0  # resolved dissipation range
    mean_drift = small_dns.velocity.mean(axis=(0, 2, 3, 4))
    np.testing.assert_allclose(mean_drift, meta["drift"], atol=1e-10)
    assert meta["truth"] == {"A": 1.0, "V": 1.0, "D": 0.0}


def test_noise_level_is_relative_to_fluctuations(small_dns):
    noisy = with_noise(small_dns, 0.1, seed=1)
    fluct = small_dns.velocity - small_dns.velocity.mean(axis=(0, 2, 3, 4), keepdims=True)
    added = noisy.velocity - small_dns.velocity
    assert abs(added.std() / np.sqrt((fluct**2).mean()) - 0.1) < 0.002
