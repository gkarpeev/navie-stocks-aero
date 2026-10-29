"""Execute lab02_criteria.ipynb top to bottom in a fresh kernel and store the outputs.

    python scripts/run_notebook.py

The run is recorded in results/notebook_run.json (time, cells, kernel).
"""

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "lab02_criteria.ipynb"


def main():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    start = time.perf_counter()
    # No per-cell execution timestamps: they would change the file on every run; the
    # time of the run is recorded once, in results/notebook_run.json.
    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        record_timing=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    client.execute()
    seconds = time.perf_counter() - start
    nbformat.write(nb, NOTEBOOK)
    code = [c for c in nb.cells if c.cell_type == "code"]
    record = {
        "notebook": NOTEBOOK.name,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(seconds, 1),
        "code_cells": len(code),
        "executed_in_order": [c.execution_count for c in code] == list(range(1, len(code) + 1)),
        "python": platform.python_version(),
    }
    (ROOT / "results" / "notebook_run.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
