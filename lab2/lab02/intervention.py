"""Intervention 2.4: values moved beyond three sigma of their conditional distribution.

Window level (main). The observed variables of the weak-form regression are
window integrals: the target y and the input x_e of every element e, i.e. the
value of the branch that enters the output sum of the tree. The conditional
law of x_e given the inputs of the other elements is modelled per component c
of the test function as linear Gaussian,

    x_e | x_-e ~ N(mu, sigma_c^2),   mu = b_0 + sum_j b_j x_j,

and in a fraction q of the windows (all three component rows of a window) the
input is moved outwards on its own side:

    x_e' = mu + sign(x_e - mu) * k * sigma_c,   k >= 3.

By the Frisch-Waugh-Lovell theorem theta_e is identified only by x_e - mu, so
the conditional move destroys the information about theta_e while the part of
x_e shared with the other inputs stays. For any unimodal law the band of three
sigma holds at least 1 - 4/81 of the mass (Pukelsheim 1994), so the moved
values are outliers of the conditional law. The placebo draws the same windows
from N(mu, sigma_c^2), the conditional randomization of Candes et al. (2018).

Grid level (variant). The values of one leaf of the LR1 tree (u at n01, n03,
n13) at grid nodes are moved beyond three sigma of their conditional law given
the six face neighbours, a Gaussian Markov random field (Rue & Held 2005). The
weak form averages such point outliers over a window, which is why the main
intervention acts on windows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SQRT_2_OVER_PI = float(np.sqrt(2 / np.pi))


def _check_k(k: float) -> None:
    if k < 3:
        raise ValueError("The intervention moves values beyond three sigma: k >= 3")


def select_windows(n_windows: int, q: float, rng: np.random.Generator) -> np.ndarray:
    """Mask of exactly ceil(q * n_windows) windows drawn without replacement."""
    if not 0 < q <= 1:
        raise ValueError("Fraction q must lie in (0, 1]")
    mask = np.zeros(n_windows, dtype=bool)
    mask[rng.choice(n_windows, int(np.ceil(q * n_windows)), replace=False)] = True
    return mask


# -- window level -------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowConditional:
    """Linear Gaussian law of one input column given the other columns, per component."""

    element: int
    mean: np.ndarray  # (n_rows,) conditional means
    sigma: np.ndarray  # (3,) conditional standard deviations per component
    component: np.ndarray  # (n_rows,) component of the test function of each row
    coef: np.ndarray  # (3, p): intercept, then the other columns in order

    def z(self, column: np.ndarray) -> np.ndarray:
        """Standardized deviation (x - mu) / sigma of a column from the conditional law."""
        return (column - self.mean) / self.sigma[self.component]


def fit_window_conditional(X: np.ndarray, element: int, component: np.ndarray) -> WindowConditional:
    n, p = X.shape
    others = [j for j in range(p) if j != element]
    mean = np.empty(n)
    sigma = np.empty(3)
    coef = np.empty((3, p))
    for c in range(3):
        rows = component == c
        Z = np.column_stack([np.ones(rows.sum()), X[rows][:, others]])
        b, *_ = np.linalg.lstsq(Z, X[rows, element], rcond=None)
        mean[rows] = Z @ b
        sigma[c] = float(np.std(X[rows, element] - mean[rows], ddof=p))
        coef[c] = b
    if not np.all(sigma > 0):
        raise ValueError("Input is an exact combination of the other inputs: nothing to move")
    return WindowConditional(element, mean, sigma, component, coef)


def three_sigma_windows(
    X: np.ndarray,
    cond: WindowConditional,
    window: np.ndarray,
    k: float,
    q: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Move the input of ``cond.element`` to mu +- k sigma in a fraction q of windows."""
    _check_k(k)
    chosen = select_windows(int(window.max()) + 1, q, rng)[window]
    column = X[:, cond.element]
    side = np.where(column >= cond.mean, 1.0, -1.0)
    moved = cond.mean + side * k * cond.sigma[cond.component]
    out = X.copy()
    out[:, cond.element] = np.where(chosen, moved, column)
    return out


def placebo_windows(
    X: np.ndarray, cond: WindowConditional, window: np.ndarray, q: float, rng: np.random.Generator
) -> np.ndarray:
    """Redraw the input in a fraction q of windows from its conditional law."""
    chosen = select_windows(int(window.max()) + 1, q, rng)[window]
    drawn = cond.mean + cond.sigma[cond.component] * rng.standard_normal(len(cond.mean))
    out = X.copy()
    out[:, cond.element] = np.where(chosen, drawn, X[:, cond.element])
    return out


def attenuation(k: float, q: float) -> float:
    """Expected relative loss rho of a real parameter: theta_e -> (1 - rho) theta_e.

    Least squares on the input's own part r = x_e - mu: the untouched windows
    keep the slope theta_e with weight (1 - q) sigma^2; a moved window carries
    k sigma sign(r) and its covariance with the target is theta_e k sigma E|r| =
    theta_e k sigma^2 sqrt(2/pi) for Gaussian r. Hence
    1 - rho = (1 - q + q k sqrt(2/pi)) / (1 - q + q k^2).
    """
    return 1.0 - (1 - q + q * k * SQRT_2_OVER_PI) / (1 - q + q * k * k)


# -- grid level (variant) -----------------------------------------------------------------

# Interior slices of the six face neighbours: (-x, +x, -y, +y, -z, +z).
_CORE = (slice(None), slice(1, -1), slice(1, -1), slice(1, -1))
_NEIGHBOURS = (
    (slice(None), slice(0, -2), slice(1, -1), slice(1, -1)),
    (slice(None), slice(2, None), slice(1, -1), slice(1, -1)),
    (slice(None), slice(1, -1), slice(0, -2), slice(1, -1)),
    (slice(None), slice(1, -1), slice(2, None), slice(1, -1)),
    (slice(None), slice(1, -1), slice(1, -1), slice(0, -2)),
    (slice(None), slice(1, -1), slice(1, -1), slice(2, None)),
)


@dataclass(frozen=True)
class FieldConditional:
    """Neighbour-conditional Gaussian model of each velocity component on the grid."""

    coef: np.ndarray  # (3, 7): six neighbour weights and the intercept
    sigma: np.ndarray  # (3,) conditional standard deviations
    mean: np.ndarray  # conditional means, shaped like the velocity
    marginal_std: np.ndarray  # (3,) for reference: sigma << marginal std for smooth fields


def fit_field_conditional(velocity: np.ndarray) -> FieldConditional:
    """Least-squares conditional means per component; the boundary layer keeps mu = u."""
    comps = velocity.shape[1]
    coef = np.zeros((comps, 7))
    sigma = np.zeros(comps)
    mean = velocity.copy()
    for c in range(comps):
        field = velocity[:, c]
        # Normal equations from strided views: the stacked design would need
        # ~0.4 GB per component for the experimental block.
        columns = [field[s] for s in _NEIGHBOURS] + [np.ones(field[_CORE].shape)]
        gram = np.array([[np.vdot(a, b) for b in columns] for a in columns])
        rhs = np.array([np.vdot(a, field[_CORE]) for a in columns])
        coef[c] = np.linalg.solve(gram, rhs)
        prediction = sum(w * col for w, col in zip(coef[c], columns))
        sigma[c] = float(np.sqrt(np.mean((field[_CORE] - prediction) ** 2)))
        mean[:, c][_CORE] = prediction
    return FieldConditional(coef, sigma, mean, velocity.std(axis=(0, 2, 3, 4)))


def _interior_choice(shape, q: float, rng: np.random.Generator) -> np.ndarray:
    if not 0 < q <= 1:
        raise ValueError("Fraction q must lie in (0, 1]")
    interior = np.zeros(shape, dtype=bool)
    interior[:, 1:-1, 1:-1, 1:-1] = True
    return (rng.random(shape) < q) & interior


def three_sigma_field(
    velocity: np.ndarray, model: FieldConditional, k: float, q: float, rng: np.random.Generator
) -> np.ndarray:
    """Move a fraction q of interior grid values of every component to mu +- k sigma."""
    _check_k(k)
    out = velocity.copy()
    for c in range(velocity.shape[1]):
        chosen = _interior_choice(velocity[:, c].shape, q, rng)
        side = np.where(velocity[:, c] >= model.mean[:, c], 1.0, -1.0)
        moved = model.mean[:, c] + side * k * model.sigma[c]
        out[:, c] = np.where(chosen, moved, velocity[:, c])
    return out
