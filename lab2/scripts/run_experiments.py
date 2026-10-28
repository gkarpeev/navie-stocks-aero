"""Run the computational experiment and write every result to results/.

    python scripts/run_experiments.py [--workers 8]

Data sets: synthetic scenario A (Navier-Stokes), synthetic scenario B (with
linear friction, the control element is real) and the VIC# experiment. For
each: weak-form design, distributions, intervention sweeps, backward
elimination; the grid-level variant for A and the experiment; the Monte Carlo
check of the width over independent DNS realizations; the identifiability of
the 15 edge weights. About a minute on 8 processes.
"""

import argparse
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab02 import config, graph, legacy, montecarlo  # noqa: E402
from lab02.experiment import (  # noqa: E402
    MAIN,
    Design,
    WeakSettings,
    analyse,
    eliminate,
    final_keep,
    leaf_variant,
)

LEAF_DATASETS = ("syn_a", "real")


def _json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False, allow_nan=False))


def run_dataset(name: str) -> dict:
    """Analysis of one data set; arrays go to an .npz next to the JSON report."""
    start = time.perf_counter()
    key, title = config.DATASETS[name]
    data = config.data(name)
    design = Design.build(data, config.settings(name))
    truth = data.meta.get("truth")
    report = analyse(design, config.PLAN, key, truth)
    replicates = report.pop("_replicates")
    slice_, extent = snapshot(data)
    np.savez_compressed(
        config.RESULTS / f"{name}_design.npz",
        y=design.y,
        X=design.X,
        clusters=design.clusters,
        window=design.window,
        component=design.component,
        bootstrap=replicates,
        snapshot=slice_,
    )
    if name in LEAF_DATASETS:
        rows, info = leaf_variant(data, config.settings(name), design, config.PLAN, key)
        report["leaf_variant"] = {"rows": rows, **info}
    report["title"] = title
    report["data_meta"] = _plain(data.meta)
    report["field"] = {
        "shape": list(data.velocity.shape),
        "h": data.h,
        "dt": data.dt,
        "nu_column": data.nu,
        "mean_velocity": data.velocity.mean(axis=(0, 2, 3, 4)).tolist(),
        "snapshot_extent": extent,
        "fluctuation_rms": float(
            np.sqrt(
                ((data.velocity - data.velocity.mean(axis=(0, 2, 3, 4), keepdims=True)) ** 2).mean()
            )
        ),
    }
    if name == "syn_a":
        report["identifiability"] = identifiability(data, report)
    if name == "real":
        report["scales"] = scale_robustness(data, key)
    report["seconds"] = round(time.perf_counter() - start, 1)
    _json(config.RESULTS / f"{name}.json", report)
    return {"name": name, "seconds": report["seconds"], "keep": report["keep"]}


def snapshot(data) -> tuple[np.ndarray, list[float]]:
    """Streamwise velocity in the mid-plane z of the first frame and its extent (x0, x1, y0, y1).

    The experiment carries physical axes in its metadata; the DNS box starts at zero.
    """
    nx, ny, nz = data.velocity.shape[2:]
    axes = data.meta.get("axes_m")
    x0, y0 = (axes["x"][0], axes["y"][0]) if axes else (0.0, 0.0)
    extent = [x0, x0 + (nx - 1) * data.h, y0, y0 + (ny - 1) * data.h]
    return data.velocity[0, 0, :, :, nz // 2], extent


def scale_robustness(data, key: int) -> list[dict]:
    """Decisions on the experiment for several sizes of the test windows."""
    out = []
    for i, (mx, mt, sx, st) in enumerate(config.REAL_SCALES):
        settings = WeakSettings(mx=mx, mt=mt, stride_x=sx, stride_t=st)
        design = Design.build(data, settings)
        steps = eliminate(design, config.PLAN, key * 100 + i)
        first = steps[0]["tests"]
        out.append(
            {
                "settings": asdict(settings),
                "rows": design.meta["rows"],
                "clusters": design.meta["clusters"],
                "r2": steps[0]["r2"],
                "theta": {e: first[e]["theta"] for e in design.names},
                "sigma": {e: first[e]["sigma"] for e in design.names},
                "kl_shift": {e: first[e]["kl_shift"] for e in design.names},
                "keep": final_keep(steps),
            }
        )
    return out


def _plain(meta: dict) -> dict:
    return json.loads(
        json.dumps(meta, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    )


def identifiability(data, report) -> dict:
    """Edge-to-branch incidence on a frame with the true pressure and the edge Fisher matrix."""
    incidence = graph.incidence_from_synthetic(data.velocity[0], data.pressure[0], data.nu, data.h)
    base = report["base"][f"{MAIN[0]}/{MAIN[1]}"]
    cov = np.array(base["cov"])[:2, :2]  # (A, V) after profiling out the control
    edge_f = graph.edge_fisher(incidence, graph.observed_branch_fisher(np.linalg.inv(cov)))
    diag = np.diag(edge_f)
    # Rows of the pressure branch are zero up to round-off; normalizing them would
    # turn round-off into spurious correlations, so they are reported as "no information".
    zero = diag < 1e-9 * diag.max()
    safe = np.where(zero, 1.0, diag)
    corr = edge_f / np.sqrt(np.outer(safe, safe))
    corr[zero, :] = np.nan
    corr[:, zero] = np.nan
    eigen = np.linalg.eigvalsh(edge_f)
    return {
        "edges": [f"{s}→{t}" for s, t, _ in incidence.edges],
        "branches": list(incidence.branches),
        "incidence": incidence.matrix.round(9).tolist(),
        "incidence_residual": incidence.residual,
        "edge_fisher_correlation": [
            [None if np.isnan(v) else float(v) for v in row] for row in corr
        ],
        "edge_fisher_diag": diag.tolist(),
        "no_information": [bool(z) for z in zero],
        "eigenvalues": eigen.tolist(),
        "rank": int(np.sum(eigen > 1e-9 * eigen.max())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config.RESULTS.mkdir(exist_ok=True)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.workers, 3)) as pool:
        for done in pool.map(run_dataset, config.DATASETS):
            print(f"{done['name']}: {done['seconds']} s, keep {done['keep']}", flush=True)
    mc = montecarlo.run(args.workers)
    _json(config.RESULTS / "monte_carlo.json", mc)
    print(
        "Monte Carlo:", json.dumps(mc["summary"]["keep_share"][mc["summary"]["main"]]), flush=True
    )
    _json(
        config.RESULTS / "environment.json",
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "upstream": legacy.upstream_fingerprint(),
            "plan": asdict(config.PLAN),
            "weak_form": {"synthetic": asdict(config.SYNTHETIC), "real": asdict(config.REAL)},
            "seconds_total": round(time.perf_counter() - start, 1),
        },
    )
    print(f"total {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
