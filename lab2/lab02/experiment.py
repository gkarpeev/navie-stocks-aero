"""Computational experiment: distributions, interventions, KL distances, decisions.

A ``Design`` is the weak-form regression of one data set: the target y, the
inputs X of the elements A, V, D, cluster labels for the Godambe information
and the test window of every row. Everything after the design is cheap: an
intervention replaces one column of X and the model is refitted with both
error functions and both information estimates.

Decision rule. Element e stays in the graph if

* it is identifiable (its column is not identically zero), and
* the window-level intervention of full strength (k = 10, every window) moves
  the centre of the distribution of theta_e by more than the threshold,
  KL_shift(after || before) > 1.92 nats (the centre leaves the band of 1.96
  sigma of the original distribution), and
* the sign of theta_e is physically admissible.

Elements are removed one at a time, the weakest first, with a refit after
every removal. The shape part of KL is reported, never used: the moved values
are high-leverage points that narrow the distribution of any parameter,
whether the element is real or not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .dataset import FieldData
from .divergence import THRESHOLD, kl_parts, removal_kl
from .estimation import LOSSES, METHODS, Gaussian, bootstrap, distributions
from .graph import ADMISSIBLE_SIGN
from .intervention import (
    attenuation,
    fit_field_conditional,
    fit_window_conditional,
    placebo_windows,
    three_sigma_field,
    three_sigma_windows,
)
from .weakform import WeakForm, branch_features

NAMES = ("A", "V", "D")
MAIN = ("l2", "godambe")  # error function and information estimate of the decisions
KINDS = ("three_sigma", "placebo")
# Grid-level variant: which element each leaf of the tree feeds.
LEAVES = {"n01": "A", "n03": "A", "n13": "V", "d": "D"}


@dataclass(frozen=True)
class WeakSettings:
    """Test-function grid: half-widths and strides in grid cells (mx) and frames (mt)."""

    mx: int
    mt: int
    stride_x: int
    stride_t: int
    power: int = 7


@dataclass(frozen=True)
class Plan:
    """Hyperparameters of the intervention and the decision point."""

    ks: tuple = (3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0)
    k_sweep_q: float = 0.05
    qs: tuple = (1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
    q_sweep_k: float = 10.0
    seeds: int = 10
    placebo_seeds: int = 10
    n_boot: int = 1000
    seed: int = 20261028

    def __post_init__(self):
        if min(self.ks) < 3:
            raise ValueError("The intervention moves values beyond three sigma: k >= 3")
        if self.qs[-1] != 1.0 or list(self.qs) != sorted(self.qs):
            raise ValueError("The q-sweep must be increasing and end at the full strength q = 1")
        if self.k_sweep_q not in self.qs or self.q_sweep_k not in self.ks:
            raise ValueError("The two sweeps must cross at a point of both")

    @property
    def decision_point(self) -> tuple[float, float]:
        return self.q_sweep_k, 1.0


def rng_for(plan: Plan, *key: int) -> np.random.Generator:
    """Independent reproducible stream for every (data set, element, sweep, point, seed)."""
    return np.random.default_rng(np.random.SeedSequence([plan.seed, *key]))


@dataclass
class Design:
    name: str
    y: np.ndarray
    X: np.ndarray
    clusters: np.ndarray
    window: np.ndarray
    component: np.ndarray
    names: tuple = NAMES
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        n = len(self.y)
        if self.X.shape != (n, len(self.names)):
            raise ValueError("X must have one column per element name")
        if not all(len(a) == n for a in (self.clusters, self.window, self.component)):
            raise ValueError("Row annotations must match the rows of X")
        if np.any(np.all(self.X == 0, axis=0)):
            raise ValueError("A column is identically zero: the element is not identifiable")

    @classmethod
    def build(cls, data: FieldData, settings: WeakSettings) -> "Design":
        u = data.velocity
        form = weak_form(data, settings)
        features = branch_features(form, {"n01": u, "n03": u, "n13": u, "d": u}, data.nu)
        index = form.row_index()
        meta = {
            "dataset": data.name,
            "settings": asdict(settings),
            "rows": form.n_rows,
            "windows": int(np.prod([len(c) for c in form.centers])),
        }
        if data.pressure is not None:
            # The pressure branch is annihilated by the solenoidal test functions.
            meta["pressure_column_max"] = float(np.abs(form.pressure(data.pressure, 1.0)).max())
        design = cls(
            data.name,
            form.target(u),
            np.stack([features[n] for n in NAMES], axis=1),
            form.clusters(),
            form.windows(),
            index["component"],
            NAMES,
            meta,
        )
        design.meta["clusters"] = int(len(np.unique(design.clusters)))
        return design

    def columns(self, names) -> "Design":
        idx = [self.names.index(n) for n in names]
        return Design(
            self.name,
            self.y,
            self.X[:, idx],
            self.clusters,
            self.window,
            self.component,
            tuple(names),
            self.meta,
        )

    def distributions(self) -> dict:
        return distributions(self.X, self.y, self.names, self.clusters)

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            y=self.y,
            X=self.X,
            clusters=self.clusters,
            window=self.window,
            component=self.component,
        )


def weak_form(data: FieldData, settings: WeakSettings) -> WeakForm:
    s = settings
    return WeakForm(
        data.velocity.shape, data.h, data.dt, s.mx, s.mt, s.stride_x, s.stride_t, s.power
    )


# -- interventions ------------------------------------------------------------------------


def intervene(
    design: Design, element: str, kind: str, k: float | None, q: float, rng
) -> np.ndarray:
    """Design matrix after the window-level intervention on the input of ``element``."""
    e = design.names.index(element)
    cond = fit_window_conditional(design.X, e, design.component)
    if kind == "three_sigma":
        return three_sigma_windows(design.X, cond, design.window, k, q, rng)
    if kind == "placebo":
        return placebo_windows(design.X, cond, design.window, q, rng)
    raise ValueError(f"Unknown intervention {kind!r}")


@dataclass(frozen=True)
class Refit:
    """Distributions after one intervention: (loss, method) -> Gaussian."""

    element: str
    kind: str
    sweep: str  # "k" or "q"
    k: float | None  # None for the placebo
    q: float
    seed: int
    dists: dict

    def measures(self, base: dict, loss: str, method: str) -> dict:
        """KL(after || before) of the element's parameter and the shifts of all parameters."""
        P, Q = base[loss, method], self.dists[loss, method]
        marginal = kl_parts(Q.marginal(self.element), P.marginal(self.element))
        full = kl_parts(Q, P)
        i = P.names.index(self.element)
        return {
            "shift": marginal["shift"],
            "shape": marginal["shape"],
            "full_shift": full["shift"],
            "full_shape": full["shape"],
            "rho": float(1 - Q.mean[i] / P.mean[i]),
            "z": ((Q.mean - P.mean) / P.std).tolist(),
            "width_ratio": float(Q.std[i] / P.std[i]),
        }


def sweep(design: Design, plan: Plan, key: int) -> list[Refit]:
    """k-sweep at q = k_sweep_q, q-sweep at k = q_sweep_k, and placebos at every q."""
    out = []
    for ei, element in enumerate(design.names):
        points = [("three_sigma", "k", k, plan.k_sweep_q, plan.seeds) for k in plan.ks]
        points += [("three_sigma", "q", plan.q_sweep_k, q, plan.seeds) for q in plan.qs]
        points += [("placebo", "q", None, q, plan.placebo_seeds) for q in plan.qs]
        for pi, (kind, axis, k, q, seeds) in enumerate(points):
            for s in range(seeds):
                X = intervene(design, element, kind, k, q, rng_for(plan, key, ei, pi, s))
                dists = distributions(X, design.y, design.names, design.clusters)
                out.append(Refit(element, kind, axis, k, q, s, dists))
    return out


def sweep_table(base: dict, refits: list[Refit]) -> list[dict]:
    """One flat row per refit and (loss, method) with every KL measure."""
    rows = []
    for r in refits:
        for loss in LOSSES:
            for method in METHODS:
                rows.append(
                    {
                        "element": r.element,
                        "kind": r.kind,
                        "sweep": r.sweep,
                        "k": r.k,
                        "q": r.q,
                        "seed": r.seed,
                        "loss": loss,
                        "method": method,
                        **r.measures(base, loss, method),
                    }
                )
    return rows


def curve(
    rows: list[dict],
    element: str,
    sweep_axis: str,
    measure: str = "shift",
    kind: str = "three_sigma",
    loss: str = MAIN[0],
    method: str = MAIN[1],
) -> dict:
    """Median and range over seeds of one measure along one sweep."""
    sel = [
        r
        for r in rows
        if r["element"] == element
        and r["sweep"] == sweep_axis
        and r["kind"] == kind
        and r["loss"] == loss
        and r["method"] == method
    ]
    x_key = "k" if sweep_axis == "k" else "q"
    xs = sorted({r[x_key] for r in sel})
    values = [np.array([r[measure] for r in sel if r[x_key] == x]) for x in xs]
    return {
        "x": xs,
        "median": [float(np.median(v)) for v in values],
        "low": [float(v.min()) for v in values],
        "high": [float(v.max()) for v in values],
        "q95": [float(np.quantile(v, 0.95)) for v in values],
    }


def critical_fraction(c: dict, threshold: float = THRESHOLD) -> dict:
    """Smallest q at which the median shift reaches the threshold (log interpolation).

    If the first grid point is already above the threshold, the crossing lies
    below the grid and the grid edge is only an upper bound; the result says
    so rather than passing the edge off as the crossing.
    """
    xs, ys = np.array(c["x"]), np.array(c["median"])
    above = np.flatnonzero(ys >= threshold)
    if len(above) == 0:
        return {"value": None, "upper_bound": False}
    i = above[0]
    if i == 0:
        return {"value": float(xs[0]), "upper_bound": True}
    x0, x1, y0, y1 = np.log(xs[i - 1]), np.log(xs[i]), np.log(max(ys[i - 1], 1e-300)), np.log(ys[i])
    value = float(np.exp(x0 + (np.log(threshold) - y0) * (x1 - x0) / (y1 - y0)))
    return {"value": value, "upper_bound": False}


# -- decisions ----------------------------------------------------------------------------


def rule(element: str, theta: float, kl_shift: float) -> dict:
    """The decision for one element from the shift part of KL(after || before).

    The shape part is deliberately absent: the moved values narrow or widen
    the distribution of any parameter, real or not.
    """
    significant = kl_shift > THRESHOLD
    admissible = bool(np.sign(theta) == ADMISSIBLE_SIGN[element])
    return {
        "significant": bool(significant),
        "admissible": admissible,
        "keep": bool(significant and admissible),
    }


def decision_tests(design: Design, plan: Plan, key: int, base: dict | None = None) -> dict:
    """Full-strength intervention on every element of the design, all (loss, method) pairs."""
    base = design.distributions() if base is None else base
    k, q = plan.decision_point
    tests = {}
    for ei, element in enumerate(design.names):
        X = intervene(design, element, "three_sigma", k, q, rng_for(plan, key, 99, ei))
        after = distributions(X, design.y, design.names, design.clusters)
        per = {}
        for loss in LOSSES:
            for method in METHODS:
                P, Q = base[loss, method], after[loss, method]
                i = P.names.index(element)
                part = kl_parts(Q.marginal(element), P.marginal(element))
                per[f"{loss}/{method}"] = {
                    "theta": float(P.mean[i]),
                    "sigma": float(P.std[i]),
                    "theta_after": float(Q.mean[i]),
                    "sigma_after": float(Q.std[i]),
                    "kl_shift": part["shift"],
                    "kl_shape": part["shape"],
                    "removal_kl": removal_kl(P, element),
                    **rule(element, float(P.mean[i]), part["shift"]),
                }
        main = per[f"{MAIN[0]}/{MAIN[1]}"]
        tests[element] = {
            **main,
            "by_method": per,
            "unanimous": len({v["keep"] for v in per.values()}) == 1,
        }
    return tests


def eliminate(design: Design, plan: Plan, key: int) -> list[dict]:
    """Backward elimination: remove the weakest failing element, refit, test again."""
    names = list(design.names)
    steps = []
    while names:
        sub = design.columns(names)
        base = sub.distributions()
        tests = decision_tests(sub, plan, key * 10 + len(steps), base)
        failing = [e for e in names if not tests[e]["keep"]]
        removed = min(failing, key=lambda e: tests[e]["kl_shift"]) if failing else None
        steps.append(
            {
                "model": list(names),
                "r2": base[MAIN[0], "r2"],
                "base": {f"{lo}/{me}": base[lo, me].to_dict() for lo in LOSSES for me in METHODS},
                "tests": tests,
                "removed": removed,
            }
        )
        if removed is None:
            break
        names.remove(removed)
    return steps


def final_keep(steps: list[dict]) -> dict:
    kept = steps[-1]["model"] if steps[-1]["removed"] is None else []
    return {e: e in kept for e in NAMES}


# -- grid-level variant -------------------------------------------------------------------


def leaf_variant(
    data: FieldData,
    settings: WeakSettings,
    design: Design,
    plan: Plan,
    key: int,
    k: float = 10.0,
    qs=(0.05, 0.5),
    seeds: int = 3,
) -> tuple[list, dict]:
    """Move grid values of one leaf beyond three sigma of the neighbour-conditional law.

    Only the column of the element fed by the leaf is recomputed; the target
    and the other leaves keep the measured field.
    """
    form = weak_form(data, settings)
    model = fit_field_conditional(data.velocity)
    base = design.distributions()[MAIN]
    u = data.velocity
    rows = []
    for li, (leaf, element) in enumerate(LEAVES.items()):
        for qi, q in enumerate(qs):
            for s in range(seeds):
                moved = three_sigma_field(u, model, k, q, rng_for(plan, key, 50, li, qi, s))
                leaves = {"n01": u, "n03": u, "n13": u, "d": u, leaf: moved}
                if element == "A":
                    column = form.advection(leaves["n01"], leaves["n03"])
                elif element == "V":
                    column = form.viscous(leaves["n13"], data.nu)
                else:
                    column = form.damping(leaves["d"])
                X = design.X.copy()
                X[:, design.names.index(element)] = column
                after = distributions(X, design.y, design.names, design.clusters)[MAIN]
                i = design.names.index(element)
                part = kl_parts(after.marginal(element), base.marginal(element))
                rows.append(
                    {
                        "leaf": leaf,
                        "element": element,
                        "k": k,
                        "q": q,
                        "seed": s,
                        "rho": float(1 - after.mean[i] / base.mean[i]),
                        "shift": part["shift"],
                        "shape": part["shape"],
                    }
                )
    return rows, {
        "sigma_conditional": model.sigma.tolist(),
        "sigma_marginal": model.marginal_std.tolist(),
    }


# -- one data set -------------------------------------------------------------------------


def analyse(design: Design, plan: Plan, key: int, truth: dict | None = None) -> dict:
    """Everything reported for one data set, as plain JSON-ready data plus the refits."""
    base = design.distributions()
    refits = sweep(design, plan, key)
    rows = sweep_table(base, refits)
    replicates = bootstrap(design.X, design.y, design.clusters, plan.n_boot, seed=plan.seed + key)
    steps = eliminate(design, plan, key)
    curves = {
        e: {
            "q": curve(rows, e, "q"),
            "k": curve(rows, e, "k"),
            "q_shape": curve(rows, e, "q", "shape"),
            "k_shape": curve(rows, e, "k", "shape"),
            "q_placebo": curve(rows, e, "q", kind="placebo"),
            "q_huber": curve(rows, e, "q", loss="huber"),
            "q_fisher": curve(rows, e, "q", method="fisher"),
        }
        for e in design.names
    }
    critical = {
        e: {name: critical_fraction(curves[e][name]) for name in ("q", "q_huber", "q_fisher")}
        for e in design.names
    }
    S = {
        f"{lo}/{me}": {e: removal_kl(base[lo, me], e) for e in design.names}
        for lo in LOSSES
        for me in METHODS
    }
    return {
        "dataset": design.name,
        "meta": design.meta,
        "truth": truth,
        "plan": asdict(plan),
        "base": {f"{lo}/{me}": base[lo, me].to_dict() for lo in LOSSES for me in METHODS},
        "r2": {lo: base[lo, "r2"] for lo in LOSSES},
        "removal_kl": S,
        "bootstrap_std": replicates.std(axis=0, ddof=1).tolist(),
        "curves": curves,
        "critical_q": critical,
        "theory": {e: [attenuation(plan.q_sweep_k, q) for q in plan.qs] for e in design.names},
        "rows": rows,
        "steps": steps,
        "keep": final_keep(steps),
        "_replicates": replicates,
    }


def gaussian(payload: dict) -> Gaussian:
    return Gaussian.from_dict(payload)
