"""Common container for velocity data entering the weak-form analysis."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class FieldData:
    """Velocity frames on a uniform grid.

    ``velocity`` has shape (T, 3, nx, ny, nz). ``nu`` is the viscosity placed in
    the viscous column, so theta_V = nu_true/nu: the true value for synthetic
    data, the LR1 reference value for the experiment.
    """

    name: str
    velocity: np.ndarray
    h: float
    dt: float
    nu: float
    meta: dict = field(default_factory=dict)
    pressure: np.ndarray | None = None

    def __post_init__(self):
        v = self.velocity
        if v.ndim != 5 or v.shape[1] != 3:
            raise ValueError(f"Expected (T, 3, nx, ny, nz), got {v.shape}")
        if not np.isfinite(v).all():
            raise ValueError("Velocity must be finite; mask invalid points beforehand")
        if self.h <= 0 or self.dt <= 0 or self.nu <= 0:
            raise ValueError("Grid step, frame interval and viscosity must be positive")
        if self.pressure is not None and self.pressure.shape != (v.shape[0],) + v.shape[2:]:
            raise ValueError("Pressure must be shaped (T, nx, ny, nz)")
