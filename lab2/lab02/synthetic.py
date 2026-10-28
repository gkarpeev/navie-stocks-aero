"""Synthetic 3D + time Navier-Stokes data with known structure.

Pseudo-spectral DNS of decaying turbulence in a 2*pi-periodic box (rotational
form, 2/3 dealiasing, classical RK4; Orszag & Patterson 1972). A uniform drift
is added by an exact Galilean transformation, so the lab-frame data resemble a
wake: mean advection plus turbulence. The data satisfy the LR1 equation with
theta_A = theta_V = 1 and unit density. An optional linear friction -alpha*u
acting on the co-moving velocity gives data in which the control element D of
the analysis is real (theta_D = alpha); without it theta_D = 0.
"""

from __future__ import annotations

import numpy as np
from numpy.fft import fftfreq, irfftn, rfftfreq, rfftn

from .dataset import FieldData


class SpectralNS:
    """Incompressible Navier-Stokes with optional linear friction, unit density, (2*pi)^3 box."""

    def __init__(self, n: int, nu: float, alpha: float = 0.0):
        if n % 2 or n < 8:
            raise ValueError("Use an even grid with at least 8 points")
        if alpha < 0:
            raise ValueError("Friction must dissipate: alpha >= 0")
        self.n, self.nu, self.alpha = n, nu, alpha
        k = fftfreq(n, 1.0 / n)
        kz = rfftfreq(n, 1.0 / n)
        self.k = np.stack(np.meshgrid(k, k, kz, indexing="ij"))
        self.k2 = (self.k**2).sum(axis=0)
        self.k2_safe = np.where(self.k2 == 0, 1.0, self.k2)
        self.dealias = np.all(np.abs(self.k) < n / 3, axis=0)

    def physical(self, uh: np.ndarray) -> np.ndarray:
        return irfftn(uh, s=(self.n,) * 3, axes=(-3, -2, -1))

    def spectral(self, u: np.ndarray) -> np.ndarray:
        return rfftn(u, axes=(-3, -2, -1))

    def project(self, uh: np.ndarray) -> np.ndarray:
        return uh - self.k * (self.k * uh).sum(axis=0) / self.k2_safe

    def curl(self, uh: np.ndarray) -> np.ndarray:
        k = self.k
        return 1j * np.stack(
            [
                k[1] * uh[2] - k[2] * uh[1],
                k[2] * uh[0] - k[0] * uh[2],
                k[0] * uh[1] - k[1] * uh[0],
            ]
        )

    def rhs(self, uh: np.ndarray) -> np.ndarray:
        """P[u x omega] - (nu k^2 + alpha) u: the Bernoulli head is removed by the projection."""
        u = self.physical(uh)
        omega = self.physical(self.curl(uh))
        nonlinear = self.spectral(np.cross(u, omega, axis=0)) * self.dealias
        return self.project(nonlinear) - (self.nu * self.k2 + self.alpha) * uh

    def pressure(self, uh: np.ndarray) -> np.ndarray:
        """Kinematic pressure from Lap p = -d_i d_j (u_i u_j)."""
        u = self.physical(uh)
        total = 0.0
        for i in range(3):
            for j in range(3):
                total = total + self.k[i] * self.k[j] * self.spectral(u[i] * u[j])
        ph = -total * self.dealias / self.k2_safe
        ph[0, 0, 0] = 0.0
        return self.physical(ph)

    def step(self, uh: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.rhs(uh)
        k2 = self.rhs(uh + 0.5 * dt * k1)
        k3 = self.rhs(uh + 0.5 * dt * k2)
        k4 = self.rhs(uh + dt * k3)
        return uh + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

    def random_field(self, k0: float, urms: float, seed: int) -> np.ndarray:
        """Solenoidal Gaussian field with E(k) ~ k^4 exp(-2 (k/k0)^2)."""
        rng = np.random.default_rng(seed)
        shape = (3,) + self.k2.shape
        noise = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        kk = np.sqrt(self.k2)
        amplitude = np.sqrt(kk**4 * np.exp(-2 * (kk / k0) ** 2) / (4 * np.pi * self.k2_safe))
        uh = self.project(noise * amplitude * self.dealias)
        uh[:, 0, 0, 0] = 0.0
        u = self.physical(uh)
        u *= urms / np.sqrt((u**2).mean())
        return self.spectral(u)

    def diagnostics(self, uh: np.ndarray) -> dict:
        u = self.physical(uh)
        omega = self.physical(self.curl(uh))
        urms = float(np.sqrt((u**2).mean()))
        epsilon = float(self.nu * (omega**2).sum(axis=0).mean())
        taylor = np.sqrt(15 * self.nu * urms**2 / epsilon)
        eta = (self.nu**3 / epsilon) ** 0.25
        return {
            "urms": urms,
            "dissipation": epsilon,
            "re_lambda": float(urms * taylor / self.nu),
            "kmax_eta": float(self.n / 3 * eta),
            "max_divergence": float(np.abs((self.k * uh).sum(axis=0)).max()),
        }


def simulate(
    n: int = 48,
    nu: float = 0.04,
    k0: float = 3.0,
    urms: float = 1.0,
    seed: int = 2026,
    spinup: float = 0.6,
    n_frames: int = 73,
    frame_every: int = 2,
    drift=(3.0, 0.0, 0.0),
    cfl: float = 0.25,
    alpha: float = 0.0,
) -> FieldData:
    """Noise-free lab-frame frames u(x, t) = U + u_co(x - U t, t) with pressure.

    The friction -alpha*u_co = -alpha*(u - U) is seen by the weak control column
    as -alpha*u: a uniform field integrates to zero against a curl test function.
    """
    model = SpectralNS(n, nu, alpha)
    uh = model.random_field(k0, urms, seed)
    h = 2 * np.pi / n
    dt = cfl * h / float(np.abs(model.physical(uh)).max())
    for _ in range(int(np.ceil(spinup / dt))):
        uh = model.step(uh, dt)
    start = model.diagnostics(uh)
    drift = np.asarray(drift, dtype=float)
    frames, pressures = [], []
    for index in range(n_frames):
        if index:
            for _ in range(frame_every):
                uh = model.step(uh, dt)
        # Translating the co-moving solution by U t is a phase shift in Fourier space.
        shift = np.exp(-1j * np.tensordot(drift, model.k, axes=1) * index * frame_every * dt)
        frames.append(model.physical(uh * shift) + drift[:, None, None, None])
        pressures.append(model.pressure(uh * shift))
    return FieldData(
        name="synthetic_dns",
        velocity=np.array(frames),
        h=h,
        dt=frame_every * dt,
        nu=nu,
        pressure=np.array(pressures),
        meta={
            "grid": n,
            "nu_true": nu,
            "alpha_true": alpha,
            "drift": drift.tolist(),
            "solver_dt": dt,
            "seed": seed,
            "spinup": spinup,
            "start": start,
            "end": model.diagnostics(uh),
            "truth": {"A": 1.0, "V": 1.0, "D": alpha},
        },
    )


def with_noise(data: FieldData, level: float, seed: int) -> FieldData:
    """Add i.i.d. Gaussian measurement noise with std = level * rms fluctuation."""
    fluct = data.velocity - data.velocity.mean(axis=(0, 2, 3, 4), keepdims=True)
    scale = level * float(np.sqrt((fluct**2).mean()))
    rng = np.random.default_rng(seed)
    noisy = data.velocity + scale * rng.standard_normal(data.velocity.shape)
    meta = dict(data.meta, noise_level=level, noise_std=scale, noise_seed=seed)
    return FieldData(data.name, noisy, data.h, data.dt, data.nu, meta, data.pressure)
