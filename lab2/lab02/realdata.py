"""Experimental data: VIC# velocity of the model wind-turbine wake, case 1L.

Hillestad et al., Phys. Rev. Fluids 11, 014701 (2026); data DataverseNO,
doi:10.18710/SX0ZQT (CC0). LR1 used every fifth frame of 2732..2817; the weak
time derivative needs the consecutive 500 Hz frames, so all 86 are fetched by
HTTP byte ranges and cropped with the unmodified LR1 code. The 18 LR1 frames
are reproduced bit for bit (checked in ``load_block``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from . import legacy
from .dataset import FieldData

FRAMES = tuple(range(2732, 2818))
FRAME_RATE_HZ = 500.0
# LR1 reference viscosity nu = U D / Re = 0.4 * 0.045 / 2e4 (water, Re_D ~ 2e4).
NU_REFERENCE = 0.4 * 0.045 / 20000
LR1_FRAMES = tuple(range(2732, 2818, 5))


def fetch(raw_directory: Path, frames=FRAMES, workers: int = 3) -> list[dict]:
    """Download (or re-verify) the raw text frames; returns provenance records."""
    return legacy.fetch_frames(frames, raw_directory, workers=workers)


def crop(raw_directory: Path) -> dict:
    """LR1 ``prepare_measurements``: crop to its bounds, cache measurement_crop.npz."""
    return legacy.prepare_measurements(raw_directory)


def load_block(raw_directory: Path, lr1_crop: Path | None = None) -> FieldData:
    """Cropped consecutive frames as FieldData, with structural checks.

    ``lr1_crop`` is the upstream measurement_crop.npz; if given, the frames
    shared with LR1 must coincide exactly, which ties this data set to LR1.
    """
    m = crop(raw_directory)
    frames = m["frame_ids"]
    if not np.array_equal(frames, np.arange(frames[0], frames[0] + len(frames))):
        raise ValueError("Frames must be consecutive for the weak time derivative")
    steps = [np.diff(m[a]) for a in "xyz"]
    h = float(steps[0][0])
    if not all(np.allclose(s, h, rtol=1e-6, atol=0) for s in steps):
        raise ValueError("Expected a uniform cubic grid")
    shared = {}
    if lr1_crop is not None:
        reference = dict(np.load(lr1_crop))
        index = [int(np.flatnonzero(frames == f)[0]) for f in reference["frame_ids"]]
        shared = {
            "lr1_frames": reference["frame_ids"].tolist(),
            "identical_to_lr1": bool(
                np.array_equal(m["velocity"][index], reference["velocity"])
                and all(np.array_equal(m[a], reference[a]) for a in "xyz")
            ),
        }
        if not shared["identical_to_lr1"]:
            raise ValueError("Cropped frames differ from the LR1 crop")
    velocity = np.ascontiguousarray(np.moveaxis(m["velocity"], -1, 1))
    meta = {
        "source": "Hillestad et al. (2026), DataverseNO doi:10.18710/SX0ZQT, case 1L",
        "frames": [int(frames[0]), int(frames[-1])],
        "frame_rate_hz": FRAME_RATE_HZ,
        "bounds_m": [list(b) for b in legacy.BOUNDS],
        "axes_m": {a: [float(m[a][0]), float(m[a][-1])] for a in "xyz"},
        "nu_reference": NU_REFERENCE,
        **shared,
    }
    return FieldData("turbine_1L", velocity, h, 1.0 / FRAME_RATE_HZ, NU_REFERENCE, meta)


def write_manifest(records: list[dict], raw_directory: Path, path: Path) -> dict:
    """Provenance of the raw frames and of the cropped cache."""
    cache = Path(raw_directory) / "measurement_crop.npz"
    manifest = {
        "dataset_doi": "10.18710/SX0ZQT",
        "case": "1L",
        "archive": "1WT_rand_100_2D.zip",
        "download": "HTTP byte ranges with ZIP CRC32 check (LR1 fetch_experiment.py)",
        "upstream": legacy.upstream_fingerprint(),
        "frames": sorted(
            ({k: r[k] for k in ("archive_entry", "bytes", "crc32", "sha256")} for r in records),
            key=lambda r: r["archive_entry"],
        ),
        "crop_sha256": hashlib.sha256(cache.read_bytes()).hexdigest() if cache.exists() else None,
    }
    Path(path).write_text(json.dumps(manifest, indent=1))
    return manifest
