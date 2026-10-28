"""Configuration of the computational experiment, shared by the scripts and the notebook."""

from __future__ import annotations

from pathlib import Path

from . import legacy, realdata, synthetic
from .dataset import FieldData
from .experiment import Plan, WeakSettings

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RAW = ROOT / "data" / "raw" / "turbine_1L"
LR1_CROP = legacy.UPSTREAM_ROOT / "data" / "turbine_1L" / "measurement_crop.npz"

# Test functions: half-widths mx (cells) and mt (frames), strides between centres.
SYNTHETIC = WeakSettings(mx=8, mt=8, stride_x=3, stride_t=8, power=7)
REAL = WeakSettings(mx=6, mt=10, stride_x=3, stride_t=10, power=7)
PLAN = Plan()
# Weak-form scales (mx, mt, stride_x, stride_t) for the robustness of the decisions on the
# experiment: the decisions must not depend on the size of the test windows.
REAL_SCALES = ((5, 10, 2, 10), (6, 10, 3, 10), (8, 10, 4, 10), (6, 15, 3, 15), (6, 20, 3, 20))

DNS_SEED = 2026
NOISE_LEVEL = 0.05  # measurement noise of the synthetic data, x rms fluctuation
NOISE_SEED = 7
# Scenario B: linear friction -alpha*u in the DNS, so the control element is real.
FRICTION = 0.01
MONTE_CARLO = 16  # independent DNS realizations for the check of the width

# Data sets: name -> (key of the random streams, title for figures and tables).
DATASETS = {
    "syn_a": (1, "синтетика А: Навье–Стокс"),
    "syn_b": (2, "синтетика Б: + трение α"),
    "real": (3, "эксперимент VIC#, случай 1L"),
}


def synthetic_data(
    alpha: float = 0.0,
    dns_seed: int = DNS_SEED,
    noise_seed: int = NOISE_SEED,
    level: float = NOISE_LEVEL,
) -> FieldData:
    """A synthetic data set of the experiment (deterministic: fixed seeds)."""
    clean = synthetic.simulate(seed=dns_seed, alpha=alpha)
    return synthetic.with_noise(clean, level, noise_seed) if level else clean


def real_data() -> FieldData:
    """The experimental block, checked against the LR1 crop."""
    return realdata.load_block(RAW, LR1_CROP)


def data(name: str) -> FieldData:
    if name == "syn_a":
        return synthetic_data()
    if name == "syn_b":
        return synthetic_data(alpha=FRICTION)
    if name == "real":
        return real_data()
    raise KeyError(name)


def settings(name: str) -> WeakSettings:
    return REAL if name == "real" else SYNTHETIC
