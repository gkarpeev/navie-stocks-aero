"""Weak (integral) form of the LR1 momentum equation with solenoidal test functions.

Each equation row tests the momentum balance with w = curl(psi * e_c), where
psi(x, y, z, t) is a smooth compactly supported bump and c is a component.
Integration by parts moves every derivative of the LR1 tree onto psi:

    <w, d_t u>          = -<d_t w, u>                                    (target)
    <w, -(grad a) b>    =  sum_ij <d_j w_i, a_i b_j> + <w_i, a_i div b>  (n01=a, n03=b)
    <w, -grad(q)/rho>   =  <div w, q>/rho = 0                            (n06=q, n08=rho)
    <w, nu Lap c>       =  nu <Lap w, c>                                 (n13=c, n12=nu)
    <w, -d>             = -<w, d>                                        (control, not in LR1)

The div-b term vanishes for solenoidal data; it is kept because interventions
make the advecting field non-solenoidal and the tree uses the advective form.
It is the only place where data are differentiated (by finite differences):
for a non-solenoidal b the advective form needs a derivative of a or of b.
Pressure drops out identically - div curl = 0 holds for the discrete kernels
too, because separable kernels commute - so velocity data carry no
information on the pressure branch scale or on rho (Reinbold et al. 2020, 2021).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.polynomial import Polynomial
from scipy.ndimage import correlate1d

# Levi-Civita symbol: w_i = sum_m EPS[i, m, c] d_m psi for w = curl(psi e_c).
EPS = np.zeros((3, 3, 3))
EPS[0, 1, 2] = EPS[1, 2, 0] = EPS[2, 0, 1] = 1.0
EPS[0, 2, 1] = EPS[2, 1, 0] = EPS[1, 0, 2] = -1.0
CURL_PAIRS = tuple(
    tuple((i, m, EPS[i, m, c]) for i in range(3) for m in range(3) if EPS[i, m, c])
    for c in range(3)
)


def bump_kernels(half_width: int, step: float, power: int, max_order: int) -> np.ndarray:
    """Samples of b(s) = (1 - (s/L)^2)^power and its derivatives on s = j*step.

    L = half_width*step. Derivatives up to power-1 vanish at |s| = L, so the
    boundary terms of integration by parts are zero and the rectangle rule is
    the trapezoidal rule on the support.
    """
    if power <= max_order:
        raise ValueError("power must exceed the highest derivative order")
    length = half_width * step
    x = np.arange(-half_width, half_width + 1) / half_width
    base = Polynomial([1.0, 0.0, -1.0]) ** power
    return np.array([base.deriv(k)(x) / length**k for k in range(max_order + 1)])


def _orders(*axes: int) -> tuple[int, int, int]:
    orders = [0, 0, 0]
    for axis in axes:
        orders[axis] += 1
    return tuple(orders)


@dataclass(frozen=True)
class WeakForm:
    """Test-function grid for fields shaped (T, 3, nx, ny, nz) on a uniform mesh."""

    shape: tuple
    h: float
    dt: float
    mx: int
    mt: int
    stride_x: int
    stride_t: int
    power: int = 5

    def __post_init__(self):
        frames, comps, *space = self.shape
        if comps != 3 or len(space) != 3:
            raise ValueError("Expected velocity shaped (T, 3, nx, ny, nz)")
        if frames <= 2 * self.mt or min(space) <= 2 * self.mx:
            raise ValueError("Test-function support exceeds the data")

    @cached_property
    def ks(self) -> np.ndarray:
        return bump_kernels(self.mx, self.h, self.power, 3)

    @cached_property
    def kt(self) -> np.ndarray:
        return bump_kernels(self.mt, self.dt, self.power, 1)

    @cached_property
    def centers(self) -> tuple[np.ndarray, ...]:
        frames, _, nx, ny, nz = self.shape
        ct = np.arange(self.mt, frames - self.mt, self.stride_t)
        return (ct,) + tuple(np.arange(self.mx, n - self.mx, self.stride_x) for n in (nx, ny, nz))

    @property
    def n_rows(self) -> int:
        return 3 * int(np.prod([len(c) for c in self.centers]))

    def in_time(self, field: np.ndarray, dt_order: int) -> np.ndarray:
        """First, most expensive stage: correlate along time, keep window centres."""
        return correlate1d(field, self.kt[dt_order], axis=0, mode="constant")[self.centers[0]]

    def in_space(self, timed: np.ndarray, orders) -> np.ndarray:
        """Second stage: separable spatial correlation, subsampled after each axis.

        The supports lie inside the array, so the zero padding never contributes.
        """
        _, cx, cy, cz = self.centers
        out = correlate1d(timed, self.ks[orders[0]], axis=1, mode="constant")[:, cx]
        out = correlate1d(out, self.ks[orders[1]], axis=2, mode="constant")[:, :, cy]
        out = correlate1d(out, self.ks[orders[2]], axis=3, mode="constant")[:, :, :, cz]
        return out * (self.h**3 * self.dt)

    @staticmethod
    def _rows(per_component) -> np.ndarray:
        return np.concatenate([per_component(c).ravel() for c in range(3)])

    def target(self, u: np.ndarray) -> np.ndarray:
        """-<d_t w, u>: the weak time derivative of the observed velocity."""
        timed = [self.in_time(u[:, i], 1) for i in range(3)]
        return self._rows(
            lambda c: sum(-e * self.in_space(timed[i], _orders(m)) for i, m, e in CURL_PAIRS[c])
        )

    def advection(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Weak form of -(grad a) b, the advection branch with leaves n01=a, n03=b."""
        div_b = sum(np.gradient(b[:, j], self.h, axis=j + 1) for j in range(3))
        products = {(i, j): self.in_time(a[:, i] * b[:, j], 0) for i in range(3) for j in range(3)}
        sources = {i: self.in_time(a[:, i] * div_b, 0) for i in range(3)}

        def component(c):
            total = 0.0
            for i, m, e in CURL_PAIRS[c]:
                for j in range(3):
                    total = total + e * self.in_space(products[i, j], _orders(m, j))
                total = total + e * self.in_space(sources[i], _orders(m))
            return total

        return self._rows(component)

    def viscous(self, c_field: np.ndarray, nu: float) -> np.ndarray:
        """Weak form of nu*Lap(c), the viscous branch with leaves n12=nu, n13=c."""
        timed = [self.in_time(c_field[:, i], 0) for i in range(3)]
        return self._rows(
            lambda c: sum(
                e * nu * sum(self.in_space(timed[i], _orders(m, k, k)) for k in range(3))
                for i, m, e in CURL_PAIRS[c]
            )
        )

    def pressure(self, q: np.ndarray, rho: float) -> np.ndarray:
        """Weak form of -grad(q)/rho = <div w, q>/rho; identically zero by symmetry."""
        timed = self.in_time(q, 0)
        return self._rows(
            lambda c: sum(
                e / rho * self.in_space(timed, _orders(i, m)) for i, m, e in CURL_PAIRS[c]
            )
        )

    def damping(self, d: np.ndarray) -> np.ndarray:
        """Weak form of -d: linear damping, a control element absent from LR1."""
        timed = [self.in_time(d[:, i], 0) for i in range(3)]
        return self._rows(
            lambda c: sum(-e * self.in_space(timed[i], _orders(m)) for i, m, e in CURL_PAIRS[c])
        )

    def row_index(self) -> dict[str, np.ndarray]:
        """Component, window and centre indices of every row, in feature order."""
        grids = np.meshgrid(*[np.arange(len(c)) for c in self.centers], indexing="ij")
        per = {name: g.ravel() for name, g in zip(("t", "x", "y", "z"), grids)}
        index = {name: np.tile(values, 3) for name, values in per.items()}
        index["component"] = np.repeat(np.arange(3), per["t"].size)
        return index

    def windows(self) -> np.ndarray:
        """Index of the test window (space-time centre) of every row; components share it."""
        index = self.row_index()
        grid = tuple(len(c) for c in self.centers)
        return np.ravel_multi_index((index["t"], index["x"], index["y"], index["z"]), grid)

    def clusters(self, block_cells: int | None = None, windows: int = 2) -> np.ndarray:
        """Cluster labels for the Godambe information.

        Rows sharing data are dependent: all three components of a window, and
        neighbouring windows whose supports overlap. A cluster joins ``windows``
        consecutive time windows and a spatial block of ``block_cells`` cells
        (default: one support width 2*mx); pairs straddling a block border are
        treated as independent.
        """
        block = block_cells or 2 * self.mx
        index = self.row_index()
        _, cx, cy, cz = self.centers
        keys = np.stack(
            [
                index["t"] // windows,
                cx[index["x"]] // block,
                cy[index["y"]] // block,
                cz[index["z"]] // block,
            ],
            axis=1,
        )
        _, labels = np.unique(keys, axis=0, return_inverse=True)
        return labels.ravel()


def branch_features(form: WeakForm, leaves: dict, nu: float) -> dict:
    """Weak-form columns of the identifiable branches and the control.

    ``leaves`` maps the LR1 leaf IDs n01, n03, n13 and 'd' (control input) to
    velocity fields; an intervention replaces one entry and keeps the others.
    """
    return {
        "A": form.advection(leaves["n01"], leaves["n03"]),
        "V": form.viscous(leaves["n13"], nu),
        "D": form.damping(leaves["d"]),
    }
