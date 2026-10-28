"""Stored results -> the figures and tables of the notebook and the slides, by name.

The notebook renders every figure through ``render`` and saves it to figures/;
the slides include the saved files under the same names, and
scripts/build_slides.py refuses to compile when one is missing or older than
the results. One recipe per figure keeps the notebook and the slides identical.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import config, plots
from .divergence import THRESHOLD
from .experiment import NAMES, rng_for
from .intervention import fit_window_conditional, three_sigma_windows

DATASET_NAMES = tuple(config.DATASETS)
TITLE = {name: title for name, (_, title) in config.DATASETS.items()}
SHORT = {"syn_a": "синтетика А", "syn_b": "синтетика Б", "real": "эксперимент"}
# What scripts/run_experiments.py writes and the figures are drawn from (not the notebook's run record).
RESULT_FILES = (
    *(f"{n}.json" for n in DATASET_NAMES),
    *(f"{n}_design.npz" for n in DATASET_NAMES),
    "monte_carlo.json",
    "environment.json",
)


@dataclass(frozen=True)
class Results:
    reports: dict  # data set -> JSON report of scripts/run_experiments.py
    arrays: dict  # data set -> arrays of its design (.npz)
    monte_carlo: dict
    environment: dict


def load(directory: Path = config.RESULTS) -> Results:
    reports = {n: json.loads((directory / f"{n}.json").read_text()) for n in DATASET_NAMES}
    arrays = {}
    for n in DATASET_NAMES:
        with np.load(directory / f"{n}_design.npz") as npz:
            arrays[n] = {key: npz[key] for key in npz.files}
    return Results(
        reports,
        arrays,
        json.loads((directory / "monte_carlo.json").read_text()),
        json.loads((directory / "environment.json").read_text()),
    )


# -- inputs of figures that are not stored as such ----------------------------------------


def intervention_example(
    results: Results,
    name: str = "real",
    element: str = "A",
    k: float = 10.0,
    q: float = 0.05,
    component: int = 0,
) -> dict:
    """One window-level intervention on the stored design, rows of one test-function component."""
    a = results.arrays[name]
    X, window, comp = a["X"], a["window"], a["component"]
    i = NAMES.index(element)
    cond = fit_window_conditional(X, i, comp)
    key = config.DATASETS[name][0]
    moved_X = three_sigma_windows(X, cond, window, k, q, rng_for(config.PLAN, key, 900, i))
    # A moved value lands exactly on mu +- k sigma, which a measured value hits with probability zero.
    moved = moved_X[:, i] != X[:, i]
    rows = comp == component
    return {
        "mu": cond.mean[rows],
        "x": X[rows, i],
        "x_after": moved_X[rows, i],
        "moved": moved[rows],
        "sigma": float(cond.sigma[component]),
        "k": k,
        "q": q,
        "element": element,
        "moved_windows": int(len(np.unique(window[moved]))),
        "windows": int(window.max() + 1),
    }


def snapshot_fields(results: Results) -> dict:
    """Synthetic A and the experiment; B is the same DNS with weak friction and looks the same."""
    out = {}
    for n in ("syn_a", "real"):
        slice_ = results.arrays[n]["snapshot"]
        extent = results.reports[n]["field"]["snapshot_extent"]
        if n == "real":
            out[TITLE[n]] = (slice_, [v * 1e3 for v in extent], ("x, мм", "y, мм", r"$u_x$, м/с"))
        else:
            out[TITLE[n]] = (slice_, extent, ("x", "y", r"$u_x$"))
    return out


def _pair(results: Results) -> dict:
    return {SHORT["syn_a"]: results.reports["syn_a"], SHORT["real"]: results.reports["real"]}


def _first_tests(report: dict) -> dict:
    return report["steps"][0]["tests"]


FIGURES = {
    "tree": lambda r: plots.plot_tree_elements(),
    "identifiability": lambda r: plots.plot_identifiability(r.reports["syn_a"]["identifiability"]),
    "snapshots": lambda r: plots.plot_snapshots(snapshot_fields(r)),
    "intervention": lambda r: plots.plot_intervention(intervention_example(r)),
    "monte_carlo": lambda r: plots.plot_monte_carlo(
        r.monte_carlo["summary"], len(r.monte_carlo["realizations"])
    ),
    "leaf": lambda r: plots.plot_leaf_vs_window(_pair(r)),
    "cross": lambda r: plots.plot_cross(_pair(r)),
    "scales": lambda r: plots.plot_scales(r.reports["real"]["scales"]),
    "kl_parts_real": lambda r: plots.plot_kl_parts(
        r.reports["real"], "Эксперимент: сдвиг и форма KL"
    ),
    "q_sensitivity_real": lambda r: plots.plot_q_sensitivity(
        r.reports["real"], "Эксперимент, адвекция A"
    ),
    "adjacency_real": lambda r: plots.plot_adjacency_diff(r.reports["real"]["keep"]),
}
# Per data set only what the notebook shows: distributions for one synthetic set and the
# experiment, the intervention response for all three, the tree after the decisions for the experiment.
for _n in DATASET_NAMES:
    FIGURES[f"pq_{_n}"] = lambda r, n=_n: plots.plot_pq(
        _first_tests(r.reports[n]), TITLE[n], r.reports[n]["truth"], config.PLAN.decision_point[0]
    )
    FIGURES[f"kl_{_n}"] = lambda r, n=_n: plots.plot_kl_sweeps(r.reports[n], TITLE[n])
    FIGURES[f"decision_{_n}"] = lambda r, n=_n: plots.plot_decision_tree(r.reports[n]["keep"])
for _n in ("syn_a", "real"):
    FIGURES[f"methods_{_n}"] = lambda r, n=_n: plots.plot_methods(
        r.reports[n]["base"], r.arrays[n]["bootstrap"], TITLE[n]
    )
    FIGURES[f"losses_{_n}"] = lambda r, n=_n: plots.plot_losses(r.reports[n]["base"], TITLE[n])
FIGURES["after_real"] = lambda r: plots.plot_decision_tree(
    r.reports["real"]["keep"], show_after=True
)


def render(name: str, results: Results, directory: Path | None = config.FIGURES):
    """Build one figure and save it as PDF (slides) and PNG; returns the Figure."""
    fig = FIGURES[name](results)
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"{name}.pdf")
        fig.savefig(directory / f"{name}.png")
    return fig


def render_all(results: Results, directory: Path = config.FIGURES) -> list[str]:
    for name in FIGURES:
        plt.close(render(name, results, directory))
    return list(FIGURES)


# -- tables (Markdown) -------------------------------------------------------------------


def markdown_table(header, rows) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def _pm(mean: float, std: float) -> str:
    return plots._fmt(mean, std)


def table_distributions(results: Results) -> str:
    """theta +- sigma per data set and estimate; the widths of all three methods side by side."""
    rows = []
    for n in DATASET_NAMES:
        rep = results.reports[n]
        G, F, H = (rep["base"][key] for key in ("l2/godambe", "l2/fisher", "huber/godambe"))
        boot = rep["bootstrap_std"]
        for i, e in enumerate(NAMES):
            gs, fs, hs = (np.sqrt(d["cov"][i][i]) for d in (G, F, H))
            truth = rep["truth"][e] if rep["truth"] else "—"
            rows.append(
                [
                    SHORT[n],
                    f"$\\theta_{e}$",
                    truth,
                    _pm(G["mean"][i], gs),
                    f"{fs / gs:.2f}",
                    f"{boot[i] / gs:.2f}",
                    _pm(H["mean"][i], hs),
                ]
            )
    header = [
        "данные",
        "параметр",
        "истина",
        "L2, Годамбе: θ ± σ",
        "σ Фишер / σ Годамбе",
        "σ бутстрэп / σ Годамбе",
        "Хьюбер, Годамбе: θ ± σ",
    ]
    return markdown_table(header, rows)


def table_decisions(report: dict) -> str:
    """First step of the elimination: the full-strength intervention on every element."""
    rows = []
    for e, t in _first_tests(report).items():
        verdict = "**оставить**" if t["keep"] else "удалить"
        why = []
        if not t["significant"]:
            why.append(f"$KL_{{сдв}} \\le {THRESHOLD:.2f}$")
        if t["significant"] and not t["admissible"]:
            why.append("знак недопустим")
        rows.append(
            [
                plots.ELEMENT_NAME[e],
                _pm(t["theta"], t["sigma"]),
                _pm(t["theta_after"], t["sigma_after"]),
                f"{t['kl_shift']:.3g}",
                f"{t['kl_shape']:.3g}",
                f"{t['removal_kl']:.3g}",
                verdict + (f" ({', '.join(why)})" if why else ""),
                "да" if t["unanimous"] else "нет",
            ]
        )
    header = [
        "элемент",
        "P: θ ± σ",
        "Q: θ ± σ",
        "KL сдвиг",
        "KL форма",
        "$S_e$",
        "решение",
        "4 оценки согласны",
    ]
    return markdown_table(header, rows)


def table_elimination(report: dict) -> str:
    rows = []
    for i, step in enumerate(report["steps"]):
        weakest = min(step["tests"].items(), key=lambda item: item[1]["kl_shift"])
        rows.append(
            [
                i + 1,
                " + ".join(step["model"]),
                f"{step['r2']:.6f}",
                f"{weakest[0]}: {weakest[1]['kl_shift']:.3g}",
                step["removed"] or "—",
            ]
        )
    return markdown_table(["шаг", "модель", "R²", "слабейший: KL сдвиг", "удалён"], rows)


def table_monte_carlo(mc: dict) -> str:
    s = mc["summary"]["l2"]
    rows = []
    for i, e in enumerate(NAMES):
        rows.append(
            [
                f"$\\theta_{e}$",
                f"{s['theta_mean'][i]:.5f}",
                f"{s['theta_spread'][i]:.2e}",
                *(
                    f"{s['ratio'][m][i]:.2f} / {s['coverage'][m][i]:.2f}"
                    for m in ("fisher", "godambe", "bootstrap")
                ),
            ]
        )
    header = [
        "параметр",
        "среднее θ̂",
        "разброс θ̂ (сд)",
        "Фишер: σ/разброс / покрытие",
        "Годамбе",
        "бутстрэп",
    ]
    return markdown_table(header, rows)


def table_scales(scales: list[dict]) -> str:
    rows = []
    for s in scales:
        st = s["settings"]
        rows.append(
            [
                f"{st['mx']}, {st['mt']}",
                f"{st['stride_x']}, {st['stride_t']}",
                s["rows"],
                s["clusters"],
                f"{s['r2']:.3f}",
                *(_pm(s["theta"][e], s["sigma"][e]) for e in NAMES),
                ", ".join(e for e in NAMES if s["keep"][e]),
            ]
        )
    header = ["mx, mt", "шаг x, t", "строк", "кластеров", "R²", "θ_A", "θ_V", "θ_D", "оставлено"]
    return markdown_table(header, rows)
