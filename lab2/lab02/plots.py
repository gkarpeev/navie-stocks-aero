"""Figures of the notebook and the slides; every function takes stored results and returns a Figure.

Colour rules hold in every figure: an element keeps its colour (A blue, V orange,
D aqua, the unidentifiable pressure branch grey); the distribution before an
intervention is grey and the one after it wears the colour of the element whose
input was moved; information estimates, error functions and kinds of
intervention differ by line style, never by colour; red and green mean only
"removed" and "added".

Sizes: a full-width figure is WIDTH inches wide. On a 4:3 slide (text width
4.7 in) it is scaled by about 0.7, which keeps its 8-10 pt text legible; the
graph figures share one scale (inches per layout unit), so a node has the
same size in every figure.
"""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from matplotlib.ticker import LogLocator, MaxNLocator

from . import graph
from .divergence import THRESHOLD
from .experiment import NAMES, critical_fraction
from .intervention import attenuation

WIDTH, NARROW = 6.6, 4.8
GRAPH_UNIT = 0.5  # inches per unit of graph.TREE_POSITIONS
FLOOR = 1e-6  # log axes: a measured zero shift is drawn at the floor

SURFACE = "#ffffff"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
ELEMENT = {"A": "#2a78d6", "V": "#eb6834", "D": "#1baf7a", "P": "#898781"}
BEFORE, BEFORE_WASH = "#52514e", "#d9d8d2"
REMOVED, ADDED = "#d03b3b", "#0ca30c"
KEPT_CELL = "#184f95"
KIND_FILL = {"field": "#d8eaf9", "parameter": "#ffe0b3", "operation": "#d9edd9"}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "lab02_blue",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
DIVERGING = LinearSegmentedColormap.from_list("lab02_div", ["#2a78d6", "#f0efec", "#e34948"])
DOTTED, DASHED = (0, (1, 1.5)), (0, (4, 2))
SYMBOL = {"A": r"\theta_A", "V": r"\theta_V", "D": r"\theta_D"}
ELEMENT_NAME = {"A": "адвекция A", "V": "вязкость V", "D": "контроль D (−αu)"}
METHOD_LABEL = {
    "godambe": "Фишер–Годамбе (кластеры окон)",
    "fisher": "Фишер, строки независимы",
    "bootstrap": "кластерный бутстрэп (проверка)",
}


def apply_style() -> None:
    # fontTools warns on every PDF that the DejaVu "created" stamp predates 1904; the stamp is
    # only font metadata, and the warning would bury real messages in the logs.
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "lines.linewidth": 1.8,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 8,
            "font.size": 9,
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "pdf.fonttype": 42,
        }
    )


def _normal(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def _fmt(value: float, std: float | None = None) -> str:
    """Value with as many digits as its uncertainty supports."""
    if std is None or std <= 0:
        return f"{value:.3g}"
    digits = max(0, 1 - int(np.floor(np.log10(std))))
    return f"{value:.{digits}f} ± {std:.{digits}f}"


def _sci(x: float) -> str:
    """Mathtext number: plain for 0.01..999, otherwise a·10^e."""
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    if -2 <= exp <= 2:
        return f"{x:.3g}"
    return rf"{x / 10**exp:.1f}\cdot10^{{{exp}}}"


def _bottom_legend(fig, handles, ncol, y=0.0):
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=ncol,
        bbox_to_anchor=(0.5, y),
        handlelength=1.8,
        columnspacing=1.2,
    )


# -- graph -------------------------------------------------------------------------------


VERTICES, EDGES = ("вершина", "вершины", "вершин"), ("ребро", "ребра", "рёбер")


def _count(n: int, forms: tuple[str, str, str]) -> str:
    """Russian plural: 1 ребро, 2 ребра, 5 рёбер."""
    if n % 10 == 1 and n % 100 != 11:
        form = forms[0]
    elif 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        form = forms[1]
    else:
        form = forms[2]
    return f"{n} {form}"


def _node_label(dag, node) -> str:
    return f"{int(node[1:])}\n{dag.graph.nodes[node]['label']}"


def _element_of(node: str) -> str | None:
    for name, nodes in graph.ELEMENT_NODES.items():
        if node in nodes:
            return name
    return None


def _graph_axes(nodes, extra_height=0.0):
    """Axes whose scale is GRAPH_UNIT inches per layout unit for the given nodes."""
    pos = graph.TREE_POSITIONS
    xs, ys = zip(*(pos[n] for n in nodes))
    x0, x1, y0, y1 = min(xs) - 0.55, max(xs) + 0.55, min(ys) - 0.45, max(ys) + 0.45
    height = (y1 - y0) * GRAPH_UNIT
    fig = plt.figure(figsize=((x1 - x0) * GRAPH_UNIT, height + extra_height))
    ax = fig.add_axes((0, 0, 1, height / (height + extra_height)))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _box(ax, x, y, fill, edge, lw, ls="-"):
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.42, y - 0.29),
            0.84,
            0.58,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            fc=fill,
            ec=edge,
            lw=lw,
            ls=ls,
            zorder=3,
        )
    )


def _arrow(ax, s, t, color, lw, ls="-", bend=0.0):
    pos = graph.TREE_POSITIONS
    ax.annotate(
        "",
        xy=pos[t],
        xytext=pos[s],
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            ls=ls,
            shrinkA=11,
            shrinkB=11,
            mutation_scale=9,
            connectionstyle=f"arc3,rad={bend}",
        ),
    )


def plot_tree_elements(dag=None):
    """The LR1 tree with the candidate control: nodes framed by the element they belong to."""
    dag = graph.candidate_tree() if dag is None else dag
    g = nx.DiGraph(dag.graph)
    pos = graph.TREE_POSITIONS
    fig, ax = _graph_axes(list(g))
    for s, t in g.edges():
        e = _element_of(s)
        _arrow(ax, s, t, ELEMENT[e] if e else MUTED, 1.3, "--" if e == "D" else "-")
    for node in g:
        e = _element_of(node)
        x, y = pos[node]
        _box(
            ax,
            x,
            y,
            KIND_FILL[dag.graph.nodes[node]["kind"]],
            ELEMENT[e] if e else MUTED,
            2.0 if e else 0.9,
            "--" if e == "D" else "-",
        )
        ax.text(
            x,
            y,
            _node_label(dag, node),
            ha="center",
            va="center",
            fontsize=7.5,
            color=INK,
            zorder=4,
            linespacing=1.05,
        )
    elements = [
        Line2D([], [], color=ELEMENT["A"], lw=2, label=r"A: $\theta_A$, ребро 5→11"),
        Line2D([], [], color=ELEMENT["P"], lw=2, label=r"P: $\theta_P = 1/\rho$, вершина 8"),
        Line2D(
            [], [], color=ELEMENT["V"], lw=2, label=r"V: $\theta_V = \nu/\nu_{ЛР1}$, вершина 12"
        ),
        Line2D(
            [], [], color=ELEMENT["D"], lw=2, ls="--", label=r"D: $\theta_D = \alpha$ — кандидат"
        ),
    ]
    kinds = [
        Patch(fc=KIND_FILL[k], ec=MUTED, lw=0.8, label=lab)
        for k, lab in (("field", "поле"), ("parameter", "параметр"), ("operation", "операция"))
    ]
    first = ax.legend(
        handles=elements, loc="upper left", fontsize=7.5, handlelength=1.6, borderaxespad=0.1
    )
    ax.add_artist(first)
    ax.legend(handles=kinds, loc="lower right", fontsize=7.5, handlelength=1.2, borderaxespad=0.1)
    return fig


def plot_decision_tree(keep: dict, show_after: bool = False):
    """Candidate tree with what the decisions remove (red) and add (green), or the tree after them."""
    before = graph.candidate_tree()
    after = graph.decided_tree(keep)
    diff = graph.graph_diff(before, after)
    removed_nodes, removed_edges, added_edges = (
        set(diff.removed_nodes),
        set(diff.removed_edges),
        set(diff.added_edges),
    )
    dag = after if show_after else before
    pos = graph.TREE_POSITIONS
    fig, ax = _graph_axes(list(dag.graph), extra_height=0.3)
    for s, t in sorted(graph.edge_pairs(dag) | added_edges):
        # An added edge bypasses removed nodes; the arc keeps it off the node it bypasses.
        if (s, t) in added_edges:
            _arrow(ax, s, t, ADDED, 2.0, bend=-0.3)
        elif (s, t) in removed_edges:
            _arrow(ax, s, t, REMOVED, 2.0)
        else:
            _arrow(ax, s, t, MUTED, 1.1)
    for node in dag.graph:
        x, y = pos[node]
        gone = node in removed_nodes
        _box(
            ax,
            x,
            y,
            KIND_FILL[dag.graph.nodes[node]["kind"]],
            REMOVED if gone else MUTED,
            2.2 if gone else 0.9,
        )
        ax.text(
            x,
            y,
            _node_label(dag, node),
            ha="center",
            va="center",
            fontsize=7.5,
            linespacing=1.05,
            color=REMOVED if gone else INK,
            zorder=4,
        )
    n_nodes, n_edges = diff.after if show_after else diff.before
    ax.set_title(
        f"{'после' if show_after else 'до'}: {_count(n_nodes, VERTICES)}, {_count(n_edges, EDGES)}",
        fontsize=9,
        pad=2,
    )
    if not show_after:
        handles = [
            Line2D(
                [],
                [],
                color=REMOVED,
                lw=2,
                label=f"удаляется: {_count(len(removed_nodes), VERTICES)}, {_count(len(removed_edges), EDGES)}",
            ),
            Line2D(
                [], [], color=ADDED, lw=2, label=f"добавляется: {_count(len(added_edges), EDGES)}"
            ),
            Line2D([], [], color=MUTED, lw=1.1, label="остаётся"),
        ]
        ax.legend(
            handles=handles, loc="upper left", fontsize=7.5, handlelength=1.6, borderaxespad=0.1
        )
    return fig


def plot_adjacency_diff(keep: dict):
    """Adjacency matrix of the candidate tree, Z[i, j] = 1 if j is an input of i, with the decisions."""
    before = graph.candidate_tree()
    after = graph.decided_tree(keep)
    removed_nodes = set(graph.graph_diff(before, after).removed_nodes)
    order = sorted(graph.TREE_POSITIONS, key=lambda n: int(n[1:]))
    z_before, z_after = graph.adjacency(before, order), graph.adjacency(after, order)
    canvas = np.ones((len(order), len(order), 3))
    canvas[(z_before == 1) & (z_after == 1)] = _rgb(KEPT_CELL)
    canvas[(z_before == 1) & (z_after == 0)] = _rgb(REMOVED)
    canvas[(z_before == 0) & (z_after == 1)] = _rgb(ADDED)
    fig, ax = plt.subplots(figsize=(4.4, 4.5))
    ax.grid(False)
    ax.imshow(canvas, interpolation="nearest")
    ticks = [f"{int(n[1:])} {before.graph.nodes[n]['label']}" for n in order]
    ax.set_xticks(range(len(order)), ticks, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(order)), ticks, fontsize=6.5)
    for labels in (ax.get_xticklabels(), ax.get_yticklabels()):
        for tick, node in zip(labels, order):
            if node in removed_nodes:
                tick.set_color(REMOVED)
    ax.set_xticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order), 1), minor=True)
    ax.grid(which="minor", color=GRID, lw=0.4)
    ax.tick_params(which="minor", length=0)
    ax.set_xlabel("j — вход")
    ax.set_ylabel("i — операция")
    fig.tight_layout()
    handles = [
        Patch(fc=REMOVED, label="удаляется"),
        Patch(fc=ADDED, label="добавляется"),
        Patch(fc=KEPT_CELL, label="остаётся"),
    ]
    _bottom_legend(fig, handles, 3)
    return fig


def plot_identifiability(ident: dict):
    """Correlation of the Fisher information of the 15 edge weights of the LR1 tree."""
    raw = np.array(
        [[np.nan if v is None else v for v in row] for row in ident["edge_fisher_correlation"]]
    )
    edges = [e.replace("n0", "").replace("n", "") for e in ident["edges"]]
    incidence = np.abs(np.array(ident["incidence"]))
    branches = ident["branches"]
    # An edge belongs to a branch when its derivative is that branch alone; n11→n16 carries A + P.
    branch = [branches[int(np.argmax(r))] if (r > 0.5).sum() == 1 else "AP" for r in incidence]
    rank = {"A": 0, "AP": 1, "P": 2, "V": 3}
    order = sorted(range(len(edges)), key=lambda i: (rank[branch[i]], i))
    corr = raw[np.ix_(order, order)]
    labels = [edges[i] for i in order]
    fig, ax = plt.subplots(figsize=(4.6, 3.9))
    ax.grid(False)
    cmap = DIVERGING.copy()
    cmap.set_bad("#f0efec")
    im = ax.imshow(np.ma.masked_invalid(corr), cmap=cmap, norm=TwoSlopeNorm(0.0, -1.0, 1.0))
    zero_rows = [k for k, i in enumerate(order) if np.isnan(raw[i, i])]
    if zero_rows:
        mid = np.mean(zero_rows)
        ax.text(mid, mid, "нет\nинформации", ha="center", va="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    colors = {"A": ELEMENT["A"], "P": ELEMENT["P"], "V": ELEMENT["V"], "AP": INK_2}
    for ticks in (ax.get_xticklabels(), ax.get_yticklabels()):
        for tick, i in zip(ticks, order):
            tick.set_color(colors[branch[i]])
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("корреляция информации Фишера", color=INK_2, fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_visible(False)
    ax.set_title(f"15 весов рёбер: ранг информации {ident['rank']}", fontsize=9)
    fig.tight_layout()
    return fig


# -- data --------------------------------------------------------------------------------


def plot_snapshots(fields: dict):
    """One velocity slice per data set: {title: (u_x slice, extent, (x label, y label, unit))}."""
    fig, axes = plt.subplots(1, len(fields), figsize=(WIDTH, 2.25))
    for ax, (title, (slice_, extent, label)) in zip(np.atleast_1d(axes), fields.items()):
        ax.grid(False)
        im = ax.imshow(slice_.T, origin="lower", extent=extent, cmap=SEQUENTIAL, aspect="equal")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel(label[0], fontsize=8)
        ax.set_ylabel(label[1], fontsize=8, labelpad=1)
        ax.tick_params(labelsize=7)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
        cb.set_label(label[2], color=INK_2, fontsize=7.5)
        cb.ax.tick_params(labelsize=7)
        cb.outline.set_visible(False)
    fig.tight_layout(w_pad=0.6)
    return fig


# -- the intervention --------------------------------------------------------------------


def plot_intervention(example: dict):
    """Standardized deviation of the input from its conditional mean, before against after.

    ``example`` holds the rows of one test-function component: mu, x, x_after, moved, sigma, k, element.
    """
    mu, x, x_after, moved = (np.asarray(example[key]) for key in ("mu", "x", "x_after", "moved"))
    sigma, k, element = example["sigma"], example["k"], example["element"]
    color = ELEMENT[element]
    z, z_after = (x - mu) / sigma, (x_after - mu) / sigma
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(WIDTH, 2.75))
    lim = k + 2
    band = dict(color=GRID, alpha=0.7, lw=0, zorder=0)
    ax0.axhspan(-3, 3, **band)
    ax0.scatter(
        z[~moved],
        z_after[~moved],
        s=3,
        color=BEFORE,
        lw=0,
        rasterized=True,
        label="окна без изменений",
    )
    ax0.scatter(
        z[moved],
        z_after[moved],
        s=7,
        color=color,
        lw=0,
        rasterized=True,
        label=f"вынесено: z' = ±{k:g}",
    )
    ax0.set_xlim(-6, 6)
    ax0.set_ylim(-lim, lim)
    ax0.set_yticks([-k, -3, 0, 3, k])
    ax0.set_xlabel(r"$z$ до: $(x-\mu)/\sigma$")
    ax0.set_ylabel(r"$z'$ после")
    ax0.legend(
        loc="upper left", fontsize=7.5, markerscale=2.2, handletextpad=0.2, borderaxespad=0.1
    )
    bins = np.linspace(-lim, lim, 97)
    ax1.axvspan(-3, 3, **band)
    ax1.hist(z, bins=bins, color=BEFORE_WASH, label="до: все окна")
    ax1.hist(z_after[moved], bins=bins, color=color, label="после: вынесенные")
    ax1.set_yscale("log")
    ax1.set_xticks([-k, -3, 0, 3, k])
    ax1.set_xlabel(r"$z$")
    ax1.set_ylabel("число строк")
    handles, _ = ax1.get_legend_handles_labels()
    handles.append(Patch(fc=GRID, label="±3σ условного закона"))
    ax1.legend(handles=handles, loc="upper left", fontsize=7.5, borderaxespad=0.1)
    fig.tight_layout(w_pad=1.2)
    return fig


# -- distributions -----------------------------------------------------------------------


def _axis_for(gaussians, pad=4.5):
    lo = min(m - pad * s for m, s in gaussians)
    hi = max(m + pad * s for m, s in gaussians)
    return np.linspace(lo, hi, 800)


def plot_methods(base: dict, bootstrap: np.ndarray, title: str, mark_values=(0.0, 1.0)):
    """Marginal densities of every parameter: Godambe, Fisher, cluster bootstrap."""
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.55))
    G, F = base["l2/godambe"], base["l2/fisher"]
    for i, (ax, e) in enumerate(zip(axes, NAMES)):
        gm, gs = G["mean"][i], np.sqrt(G["cov"][i][i])
        fs = np.sqrt(F["cov"][i][i])
        x = _axis_for([(gm, gs)])
        heights, *_ = ax.hist(
            bootstrap[:, i],
            bins=36,
            density=True,
            color=BEFORE_WASH,
            label=METHOD_LABEL["bootstrap"],
        )
        ax.plot(x, _normal(x, gm, gs), color=ELEMENT[e], label=METHOD_LABEL["godambe"])
        ax.plot(
            x, _normal(x, gm, fs), color=ELEMENT[e], lw=1.2, ls="--", label=METHOD_LABEL["fisher"]
        )
        ax.set_ylim(0, 1.2 * max(heights.max(), _normal(gm, gm, min(gs, fs))))
        # 0 is "no element", 1 the LR1 value; labels on the outer sides keep close lines legible.
        for j, v in enumerate(sorted(mark_values)):
            if x[0] <= v <= x[-1]:
                ax.axvline(v, color=INK_2, lw=0.7)
                ax.text(
                    v,
                    0.98,
                    f" {v:g} ",
                    transform=ax.get_xaxis_transform(),
                    fontsize=7.5,
                    color=INK_2,
                    va="top",
                    ha="right" if j == 0 else "left",
                )
        ax.set_title(
            f"${SYMBOL[e]}$ = {_fmt(gm, gs)}\n" + rf"$\sigma_Ф/\sigma_Г$ = {fs / gs:.2f}",
            fontsize=8.5,
        )
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.get_major_formatter().set_useOffset(False)
    handles, _ = axes[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=9.5, y=1.0)
    fig.tight_layout(w_pad=0.8)
    _bottom_legend(fig, handles, 3)
    return fig


def plot_losses(base: dict, title: str):
    """Marginal densities by error function (Godambe information for both)."""
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4))
    L, H = base["l2/godambe"], base["huber/godambe"]
    for i, (ax, e) in enumerate(zip(axes, NAMES)):
        lm, ls_ = L["mean"][i], np.sqrt(L["cov"][i][i])
        hm, hs = H["mean"][i], np.sqrt(H["cov"][i][i])
        x = _axis_for([(lm, ls_), (hm, hs)], pad=4.0)
        ax.plot(x, _normal(x, lm, ls_), color=ELEMENT[e], label="L2: сумма квадратов")
        ax.plot(
            x, _normal(x, hm, hs), color=ELEMENT[e], lw=1.5, ls=DOTTED, label="Хьюбер, c = 1.345"
        )
        ax.set_ylim(0, 1.5 * _normal(0, 0, min(ls_, hs)))  # headroom for the two lines of values
        if x[0] <= 0 <= x[-1]:
            ax.axvline(0, color=INK_2, lw=0.7)
        ax.set_title(f"${SYMBOL[e]}$", fontsize=9)
        ax.set_yticks([])
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.get_major_formatter().set_useOffset(False)
        ax.text(
            0.03,
            0.97,
            f"L2: {_fmt(lm, ls_)}\nХьюбер: {_fmt(hm, hs)}",
            transform=ax.transAxes,
            fontsize=7.5,
            color=INK_2,
            va="top",
            bbox=dict(fc=SURFACE, ec="none", pad=0.5),
        )
    handles, _ = axes[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=9.5, y=1.0)
    fig.tight_layout(w_pad=0.8)
    _bottom_legend(fig, handles, 2)
    return fig


def _density_panel(ax, P, Q, color):
    """P (grey) and Q (colour) on one axis; distributions narrower than a line become lines."""
    (pm, ps), (qm, qs) = P, Q
    lo, hi = min(pm - 4 * ps, qm - 4 * qs, 0.0), max(pm + 4 * ps, qm + 4 * qs, 0.0)
    span = hi - lo
    lo, hi = lo - 0.1 * span, hi + 0.1 * span
    ax.set_xlim(lo, hi)
    ax.set_yticks([])
    ax.set_ylim(0, 1.65)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.get_major_formatter().set_useOffset(False)
    if max(ps, qs) >= (hi - lo) / 80:
        x = np.linspace(lo, hi, 1500)
        p, q = _normal(x, pm, ps), _normal(x, qm, qs)
        top = max(p.max(), q.max())
        ax.fill_between(x, p / top, color=BEFORE_WASH, lw=0)
        ax.plot(x, p / top, color=BEFORE, lw=1.4)
        ax.fill_between(x, q / top, color=color, alpha=0.12, lw=0)
        ax.plot(x, q / top, color=color)
    else:
        # Both widths are below the line width at this scale: each distribution is a line.
        for centre, col in ((pm, BEFORE), (qm, color)):
            ax.plot([centre, centre], [0, 1.0], color=col, lw=2.2, solid_capstyle="butt")
        ax.text(
            0.5,
            0.45,
            "ширины меньше\nтолщины линии",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7,
            color=MUTED,
        )
    ax.text(
        0.03,
        0.97,
        f"P: {_fmt(pm, ps)}\nQ: {_fmt(qm, qs)}",
        transform=ax.transAxes,
        fontsize=7.5,
        color=INK_2,
        va="top",
        zorder=5,
        bbox=dict(fc=SURFACE, ec="none", pad=0.5),
    )


def plot_pq(test_rows: dict, title: str, truth: dict | None = None, k: float = 10.0):
    """P (before) against Q (after the full-strength intervention) for every element."""
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.7))
    for ax, e in zip(axes, NAMES):
        t = test_rows[e]
        _density_panel(
            ax, (t["theta"], t["sigma"]), (t["theta_after"], t["sigma_after"]), ELEMENT[e]
        )
        ax.axvline(0, color=INK_2, lw=0.7, ls=":")
        if truth is not None:
            ax.axvline(truth[e], ymax=0.62, color=INK, lw=0.9)
        verdict = "оставить" if t["keep"] else "удалить"
        if t["significant"] and not t["admissible"]:
            verdict += " (знак)"
        ax.set_title(
            f"${SYMBOL[e]}$ → {verdict}\n" + rf"$KL_{{сдв}} = {_sci(t['kl_shift'])}$", fontsize=8.5
        )
    handles = [
        Patch(fc=BEFORE_WASH, ec=BEFORE, label="P — до интервенции"),
        Patch(fc=SURFACE, ec=INK_2, label=f"Q — вход за {k:g}σ во всех окнах"),
        Line2D([], [], color=INK_2, lw=0.7, ls=":", label="θ = 0: элемента нет"),
    ]
    if truth is not None:
        handles.append(Line2D([], [], color=INK, lw=0.9, label="истинное θ"))
    fig.suptitle(title, fontsize=9.5, y=1.0)
    fig.tight_layout(w_pad=0.8)
    _bottom_legend(fig, handles, 2)
    return fig


# -- KL against the hyperparameters --------------------------------------------------------


def _star(ax, x, y, color):
    ax.plot([x], [y], marker="*", ms=11, color=color, mec=INK, mew=0.7, zorder=6, ls="none")


def _bounded(report: dict, key: str = "q") -> list[str]:
    """Elements whose shift is above the threshold already at the smallest q."""
    return [e for e, c in report["critical_q"].items() if c[key]["upper_bound"]]


def plot_kl_sweeps(report: dict, title: str, elements=NAMES):
    """Shift part of KL(after || before) against q (k = 10) and k (q = 5 %), with the theory."""
    fig, (axq, axk) = plt.subplots(1, 2, figsize=(WIDTH, 2.9), sharey=True)
    plan = report["plan"]
    S = report["removal_kl"]["l2/godambe"]
    lows, highs = [], []
    for e in elements:
        c = report["curves"][e]
        color = ELEMENT[e]
        for ax, cur, axis in ((axq, c["q"], "q"), (axk, c["k"], "k")):
            x = np.array(cur["x"])
            med, low, high = (
                np.maximum(np.array(cur[key]), FLOOR) for key in ("median", "low", "high")
            )
            ax.fill_between(x, low, high, color=color, alpha=0.12, lw=0)
            ax.plot(x, med, color=color, marker="o", ms=3, mec=SURFACE, mew=0.6)
            lows.append(med.min())
            highs.append(high.max())
            # Theory KL_shift = rho(k, q)^2 S_e holds for an element that is really there;
            # for an absent one the shift is noise and the curve would mean nothing.
            if report["keep"][e]:
                if axis == "q":
                    rho = np.array([attenuation(plan["q_sweep_k"], q) for q in x])
                else:
                    rho = np.array([attenuation(k, plan["k_sweep_q"]) for k in x])
                ax.plot(x, np.maximum(rho**2 * S[e], FLOOR), color=INK, lw=0.8, ls=DOTTED, zorder=5)
        crit = report["critical_q"][e]["q"]
        if crit["value"] is not None and not crit["upper_bound"]:
            _star(axq, crit["value"], THRESHOLD, color)
    top, bottom = max(highs) * 5, max(min(lows) / 3, 1e-5)
    for ax in (axq, axk):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom, top)
        ax.axhspan(THRESHOLD, top, color=ADDED, alpha=0.04, lw=0)
        ax.axhline(THRESHOLD, color=INK, lw=0.9)
    axq.set_ylabel(r"$KL_{сдв}$(после ‖ до), нат")
    bounded = [e for e in _bounded(report) if e in elements]
    if bounded:
        windows = int(np.ceil(plan["qs"][0] * report["meta"]["windows"]))
        axq.text(
            0.97,
            0.03,
            f"{', '.join(bounded)}: выше порога уже\nпри {windows} из {report['meta']['windows']} окон",
            transform=axq.transAxes,
            va="bottom",
            ha="right",
            fontsize=7,
            color=INK_2,
        )
    axq.set_xlabel(f"q — доля окон (k = {plan['q_sweep_k']:g})")
    axk.set_xlabel(f"k — вынос, σ (q = {plan['k_sweep_q']:g})")
    axq.xaxis.set_major_locator(LogLocator(numticks=4))
    axk.set_xticks([3, 5, 10, 20, 30], ["3", "5", "10", "20", "30"])
    axk.minorticks_off()
    handles = [
        Line2D([], [], color=ELEMENT[e], marker="o", ms=3, label=ELEMENT_NAME[e]) for e in elements
    ]
    handles += [
        Line2D([], [], color=INK, lw=0.9, label="порог 1.92: выше — оставить"),
        Line2D([], [], color=INK, lw=0.8, ls=DOTTED, label=r"теория $\rho(k,q)^2 S_e$"),
        Line2D(
            [], [], color=MUTED, marker="*", ms=9, ls="none", mec=INK, label="q*: медиана на пороге"
        ),
    ]
    fig.suptitle(title, fontsize=9.5, y=1.0)
    fig.tight_layout(w_pad=0.6)
    _bottom_legend(fig, handles, 3)
    return fig


def plot_kl_parts(report: dict, title: str, elements=NAMES):
    """Shift and shape parts of KL against q: the shape grows for real and null elements alike."""
    fig, ax = plt.subplots(figsize=(WIDTH, 2.8))
    for e in elements:
        c = report["curves"][e]
        x = np.array(c["q"]["x"])
        ax.plot(
            x,
            np.maximum(c["q"]["median"], FLOOR),
            color=ELEMENT[e],
            label=f"{ELEMENT_NAME[e]}: сдвиг",
        )
        ax.plot(
            x,
            np.maximum(c["q_shape"]["median"], FLOOR),
            color=ELEMENT[e],
            lw=1.3,
            ls="--",
            label=f"{ELEMENT_NAME[e]}: форма",
        )
    ax.axhline(THRESHOLD, color=INK, lw=0.9, label=f"порог {THRESHOLD:.2f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"q — доля окон с выносом (k = {report['plan']['q_sweep_k']:g})")
    ax.set_ylabel("часть KL(после ‖ до), нат")
    ax.set_title(title, fontsize=9.5)
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return fig


def plot_q_sensitivity(report: dict, title: str, element: str = "A"):
    """KL shift of one element against q: L2 and Huber for the 3σ move, and the conditional redraw.

    The redraw (placebo) replaces the input by a draw from its conditional law; it removes
    the information linearly, rho = q, while a move to k sigma has leverage k^2.
    """
    fig, ax = plt.subplots(figsize=(NARROW + 0.6, 2.7))
    color = ELEMENT[element]
    plan = report["plan"]
    S = report["removal_kl"]["l2/godambe"][element]
    # Label offsets: L2 crosses first and Huber next to it, so their labels go to opposite sides.
    curves = (
        ("q", "-", f"вынос за {plan['q_sweep_k']:g}σ, L2", (-7, 5), "right"),
        ("q_huber", DOTTED, f"вынос за {plan['q_sweep_k']:g}σ, Хьюбер", (7, -12), "left"),
        ("q_placebo", DASHED, "перерисовка из условного закона, L2", (-7, 5), "right"),
    )
    for key, ls, label, offset, ha in curves:
        c = report["curves"][element][key]
        ax.plot(c["x"], np.maximum(c["median"], FLOOR), color=color, ls=ls, label=label)
        crit = critical_fraction(c)
        if crit["value"] is not None:
            _star(ax, crit["value"], THRESHOLD, color)
            sign = r"\leq" if crit["upper_bound"] else "="
            ax.annotate(
                rf"$q^* {sign} {_sci(crit['value'])}$",
                (crit["value"], THRESHOLD),
                textcoords="offset points",
                xytext=offset,
                ha=ha,
                fontsize=7.5,
                color=INK_2,
            )
    x = np.geomspace(plan["qs"][0], 1.0, 60)
    ax.plot(
        x,
        [attenuation(plan["q_sweep_k"], q) ** 2 * S for q in x],
        color=INK,
        lw=0.8,
        ls=DOTTED,
        label=r"теория: $\rho(k,q)^2 S$ и $q^2 S$",
    )
    ax.plot(x, x**2 * S, color=INK, lw=0.8, ls=DOTTED)
    ax.axhline(THRESHOLD, color=INK, lw=0.9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, S * 5)
    ax.set_xlabel("q — доля окон")
    ax.set_ylabel(rf"$KL_{{сдв}}$ для ${SYMBOL[element]}$, нат")
    ax.set_title(title, fontsize=9.5)
    fig.tight_layout()
    handles, _ = ax.get_legend_handles_labels()
    _bottom_legend(fig, handles, 2)
    return fig


# -- checks ------------------------------------------------------------------------------


def plot_monte_carlo(summary: dict, count: int):
    """Width over spread on independent DNS realizations and the share of 'keep' decisions."""
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(WIDTH, 2.7))
    methods = ("fisher", "godambe", "bootstrap")
    grays = {"fisher": "#c3c2b7", "godambe": "#52514e", "bootstrap": "#898781"}
    width = 0.26
    xs = np.arange(len(NAMES))
    ratio = summary["l2"]["ratio"]
    ax0.axhline(1.0, color=INK, lw=0.9, zorder=1)
    for j, m in enumerate(methods):
        vals = np.array(ratio[m])
        bars = ax0.bar(xs + (j - 1) * width, vals, width * 0.92, color=grays[m], zorder=2)
        for b, v in zip(bars, vals):
            ax0.text(
                b.get_x() + b.get_width() / 2,
                v + 0.03,
                f"{v:.2f}",
                ha="center",
                fontsize=6.5,
                color=INK_2,
                bbox=dict(fc=SURFACE, ec="none", pad=0.4),
                zorder=3,
            )
    ax0.set_xticks(xs, [f"${SYMBOL[e]}$" for e in NAMES])
    ax0.set_ylabel("σ метода / разброс оценок")
    ax0.set_ylim(0, 1.2)
    ax0.set_title("Ширина (1 — верно)", fontsize=9)
    share = summary["keep_share"]
    for j, key in enumerate(("l2/fisher", "l2/godambe")):
        vals = np.array([share[key][e] for e in NAMES])
        m = key.split("/")[1]
        bars = ax1.bar(xs + (j - 0.5) * width * 1.1, vals, width, color=grays[m], zorder=2)
        for b, v in zip(bars, vals):
            label, x = f"{round(v * count)}/{count}", b.get_x() + b.get_width() / 2
            # A tall bar carries its label inside, vertically: side by side the labels would collide.
            if v >= 0.5:
                ax1.text(
                    x,
                    v - 0.04,
                    label,
                    rotation=90,
                    ha="center",
                    va="top",
                    fontsize=6.5,
                    zorder=3,
                    color=SURFACE if m == "godambe" else INK,
                )
            else:
                ax1.text(x, v + 0.03, label, ha="center", fontsize=6.5, color=INK_2, zorder=3)
    ax1.set_xticks(xs, [ELEMENT_NAME[e].split(" (")[0] for e in NAMES], fontsize=8)
    ax1.set_ylabel("доля «оставить»")
    ax1.set_ylim(0, 1.2)
    ax1.set_title(f"Решения на {count} реализациях DNS", fontsize=9)
    fig.tight_layout(w_pad=1.0)
    _bottom_legend(fig, [Patch(fc=grays[m], label=METHOD_LABEL[m]) for m in methods], 3)
    return fig


LEAF_LABEL = {"n01": "в. 1: u под ∇", "n03": "в. 3: перенос. u", "n13": "в. 13: u под Δ"}


def plot_leaf_vs_window(reports: dict, k: float = 10.0):
    """Lost share rho of a parameter: moving window inputs (main) versus grid values of a leaf (variant).

    Only elements that are kept are shown: for an absent parameter rho = 1 - theta'/theta
    divides noise by noise.
    """
    fig, axes = plt.subplots(1, len(reports), figsize=(WIDTH, 2.8), sharey=True)
    for ax, (title, rep) in zip(np.atleast_1d(axes), reports.items()):
        rows = rep["leaf_variant"]["rows"]
        qs = sorted({r["q"] for r in rows})
        grid_q = np.geomspace(qs[0] / 1.6, qs[-1] * 1.6, 60)
        ax.plot(grid_q, [attenuation(k, q) for q in grid_q], color=INK, lw=0.8, ls=DOTTED)
        S = rep["removal_kl"]["l2/godambe"]
        for element in ("A", "V"):
            if not rep["keep"][element]:
                continue
            win = rep["curves"][element]["q"]
            rho = [np.sqrt(win["median"][win["x"].index(q)] / S[element]) for q in qs]
            # rho does not depend on the element, so A and V coincide: concentric markers keep both visible.
            ax.plot(
                qs,
                rho,
                color=ELEMENT[element],
                lw=0,
                marker="o",
                ms=7 if element == "A" else 4,
                mec=SURFACE,
                mew=0.8,
                zorder=4,
            )
        for leaf, element, ls in (("n01", "A", "-"), ("n03", "A", "--"), ("n13", "V", "-")):
            if not rep["keep"][element]:
                continue
            rho = [
                np.median([abs(r["rho"]) for r in rows if r["leaf"] == leaf and r["q"] == q])
                for q in qs
            ]
            ax.plot(
                qs,
                rho,
                color=ELEMENT[element],
                lw=1.4,
                ls=ls,
                marker="o",
                ms=4,
                mfc=SURFACE,
                zorder=3,
            )
            ax.annotate(
                LEAF_LABEL[leaf],
                (qs[-1], rho[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=7,
                color=INK_2,
            )
        dropped = [e for e in ("A", "V") if not rep["keep"][e]]
        if dropped:
            ax.text(
                0.03,
                0.03,
                f"{', '.join(dropped)} не обнаружена:\nρ не определено",
                transform=ax.transAxes,
                fontsize=7,
                color=MUTED,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(qs[0] / 1.8, qs[-1] * 14)
        ax.set_xticks(qs, [f"{q:g}" for q in qs])
        ax.minorticks_off()
        ax.set_ylim(5e-5, 2)
        ax.set_xlabel(f"q (k = {k:g})")
        ax.set_title(title, fontsize=9)
    np.atleast_1d(axes)[0].set_ylabel(r"$\rho = 1-\theta'/\theta$")
    handles = [
        Line2D([], [], color=MUTED, lw=0, marker="o", ms=6, label="окна: вход ветви"),
        Line2D([], [], color=INK, lw=0.8, ls=DOTTED, label=r"теория $\rho(k,q)$"),
        Line2D(
            [],
            [],
            color=MUTED,
            lw=1.4,
            marker="o",
            ms=4,
            mfc=SURFACE,
            label="узлы сетки: значения листа",
        ),
    ]
    fig.tight_layout(w_pad=0.8)
    _bottom_legend(fig, handles, 3)
    return fig


def plot_cross(reports: dict, q: float = 0.05):
    """|shift| of every parameter, in its sigmas, when the input of one element is moved."""
    fig, axes = plt.subplots(1, len(reports), figsize=(WIDTH, 2.6))
    for ax, (title, rep) in zip(np.atleast_1d(axes), reports.items()):
        ax.grid(False)
        m = np.zeros((3, 3))
        for i, e in enumerate(NAMES):
            sel = [
                r
                for r in rep["rows"]
                if r["element"] == e
                and r["sweep"] == "q"
                and r["kind"] == "three_sigma"
                and r["q"] == q
                and r["loss"] == "l2"
                and r["method"] == "godambe"
            ]
            m[i] = np.median(np.abs(np.array([r["z"] for r in sel])), axis=0)
        top = max(m.max(), 10)
        ax.imshow(np.maximum(m, 1e-2), cmap=SEQUENTIAL, norm=LogNorm(1e-2, top))
        for i in range(3):
            for j in range(3):
                v = m[i, j]
                dark = np.log10(max(v, 1e-2)) > 0.6 * np.log10(top)
                text = f"{v:.2g}" if v < 100 else f"{float(f'{v:.2g}'):.0f}"
                ax.text(
                    j, i, text, ha="center", va="center", fontsize=8, color=SURFACE if dark else INK
                )
        ax.set_xticks(range(3), [f"${SYMBOL[e]}$" for e in NAMES])
        ax.set_yticks(range(3), [f"вход {e}" for e in NAMES])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("|сдвиг| в σ до интервенции", fontsize=8)
    fig.tight_layout(w_pad=1.5)
    return fig


def plot_scales(scales: list[dict]):
    """theta_A with 2 sigma and the decisions for several sizes of the test windows."""
    fig, ax = plt.subplots(figsize=(NARROW, 2.3))
    labels = [f"mx={s['settings']['mx']}, mt={s['settings']['mt']}" for s in scales]
    y = np.arange(len(scales))[::-1]
    for yi, s in zip(y, scales):
        ax.errorbar(
            s["theta"]["A"],
            yi,
            xerr=2 * s["sigma"]["A"],
            color=ELEMENT["A"],
            marker="o",
            ms=4,
            capsize=2.5,
            lw=1.4,
        )
        kept = ", ".join(e for e in NAMES if s["keep"][e])
        ax.text(
            1.02,
            yi,
            f"оставлено: {kept}",
            transform=ax.get_yaxis_transform(),
            va="center",
            fontsize=7.5,
            color=INK_2,
        )
    ax.axvline(1.0, color=INK_2, lw=0.7)
    ax.text(
        1.0,
        1.01,
        r" ЛР1: $\theta_A = 1$",
        transform=ax.get_xaxis_transform(),
        fontsize=7.5,
        color=INK_2,
    )
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.set_xlabel(r"$\theta_A \pm 2\sigma$")
    ax.set_title("Размер пробных окон", fontsize=9)
    fig.tight_layout()
    return fig


def _rgb(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i : i + 2], 16) / 255 for i in (0, 2, 4)])
