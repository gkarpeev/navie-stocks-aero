"""Single coupling point to the LR1 code of another student.

Upstream: https://github.com/Chessmatus/navie-stocks-aero, cloned without
modifications into ``third_party/navie_stocks_aero`` at ``UPSTREAM_COMMIT``.
Everything this laboratory takes from it is imported here, so an upstream API
change breaks this module loudly instead of silently changing the results.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from third_party.navie_stocks_aero import fetch_experiment as _fetch
from third_party.navie_stocks_aero.lab01.experiment import BOUNDS, prepare_measurements
from third_party.navie_stocks_aero.lab01.graphs import (
    OperatorDAG,
    export_dag,
    navier_stokes_tree,
    plot_adjacency,
    plot_tree,
)

__all__ = [
    "BOUNDS",
    "OperatorDAG",
    "export_dag",
    "navier_stokes_tree",
    "plot_adjacency",
    "plot_tree",
    "prepare_measurements",
    "fetch_frames",
    "upstream_fingerprint",
]

UPSTREAM_URL = "https://github.com/Chessmatus/navie-stocks-aero"
UPSTREAM_COMMIT = "13e4e8faf8154baa677e40da4817a32c883ab3ed"
UPSTREAM_ROOT = Path(_fetch.__file__).resolve().parent
ARCHIVE_INDEX = UPSTREAM_ROOT / "data" / "turbine_1L" / "archive_index.json"
# Upstream files whose behaviour the results depend on; hashed into every report.
USED_FILES = (
    "lab01/graphs.py",
    "lab01/experiment.py",
    "fetch_experiment.py",
    "data/turbine_1L/archive_index.json",
)


def upstream_fingerprint() -> dict:
    """Commit and SHA-256 of the upstream files used, for result provenance."""
    return {
        "url": UPSTREAM_URL,
        "commit": UPSTREAM_COMMIT,
        "sha256": {
            name: hashlib.sha256((UPSTREAM_ROOT / name).read_bytes()).hexdigest()
            for name in USED_FILES
        },
    }


@contextmanager
def _download_directory(directory: Path):
    # fetch_experiment writes next to its own file through module globals; the
    # clone must stay pristine, so the globals are redirected for one call only.
    saved = (_fetch.ROOT, _fetch.DEST)
    _fetch.ROOT, _fetch.DEST = directory, directory
    try:
        yield
    finally:
        _fetch.ROOT, _fetch.DEST = saved


def fetch_frames(frames, directory, workers: int = 3) -> list[dict]:
    """Download VIC# frames by HTTP byte ranges with the upstream CRC32 checks.

    Returns the upstream provenance records (size, CRC32, SHA-256, byte offset).
    Frames already present are re-verified instead of re-downloaded.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    index = {Path(e["name"]).name: e for e in json.loads(ARCHIVE_INDEX.read_text())}
    names = [f"B{int(frame):05d}.txt" for frame in frames]
    missing = [name for name in names if name not in index]
    if missing:
        raise KeyError(f"Frames absent from the archive index: {missing[:5]}")
    with _download_directory(directory), ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_fetch.fetch_entry, [index[name] for name in names]))
