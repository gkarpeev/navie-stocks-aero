"""Structural parameters on the LR1 expression tree and their identifiability.

The analysed graph is the 16-node tree of the momentum right-hand side
F = -(grad u) u - grad(p)/rho + nu*Laplacian(u) built by ``navier_stokes_tree``
of the LR1 code. A structural parameter multiplies the value carried by an
edge; all edges equal to one reproduces the LR1 model exactly. The analysis
adds one candidate branch that LR1 does not have, the control -alpha*u
(nodes n17..n20): the method must remove it from data without friction and
keep it from data with friction.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from .legacy import OperatorDAG, navier_stokes_tree

# Roots of the three subtrees summed into the output: advection, pressure, viscosity.
BRANCHES = {"A": "n05", "P": "n10", "V": "n15"}
BRANCH_LABELS = {
    "A": "адвекция −(∇u)u",
    "P": "давление −∇p/ρ",
    "V": "вязкость νΔu",
    "D": "контроль −αu",
}
# Admissible sign of each structural parameter: advection transports, viscosity
# and friction dissipate. A significant parameter of the wrong sign is removed.
ADMISSIBLE_SIGN = {"A": 1.0, "V": 1.0, "D": 1.0}
# Nodes and edges whose removal a zero parameter means (control branch included).
ELEMENT_NODES = {
    "A": ("n01", "n02", "n03", "n04", "n05"),
    "P": ("n06", "n07", "n08", "n09", "n10"),
    "V": ("n12", "n13", "n14", "n15"),
    "D": ("n17", "n18", "n19"),
}


def edge_list(dag: OperatorDAG) -> list[tuple[str, str, int]]:
    """Edges (source, target, argument port) in a deterministic order."""
    return sorted(dag.graph.edges(keys=True))


def subtree_nodes(dag: OperatorDAG, root: str) -> set[str]:
    """Nodes whose value flows into ``root`` (inputs point to operations)."""
    return nx.ancestors(dag.graph, root) | {root}


def evaluate_weighted(dag: OperatorDAG, values: dict, operators: dict, weights=None) -> dict:
    """``OperatorDAG.evaluate`` with a multiplicative weight on every edge.

    Mirrors the upstream evaluation order and argument ports; a missing weight
    is 1, so ``weights=None`` reproduces ``dag.evaluate`` exactly (tested).
    """
    weights = {} if weights is None else weights
    unknown = set(weights) - set(edge_list(dag))
    if unknown:
        raise KeyError(f"Weights for edges absent from the graph: {sorted(unknown)}")
    results = dict(values)
    for node in nx.topological_sort(dag.graph):
        op = dag.graph.nodes[node]["op"]
        if op is None:
            if node not in results:
                raise KeyError(f"Missing input {node}")
            continue
        edges = sorted(dag.graph.in_edges(node, keys=True), key=lambda e: e[2])
        args = [weights.get(edge, 1.0) * results[edge[0]] for edge in edges]
        results[node] = operators[op](*args)
    return results


def periodic_operators(h: float) -> tuple[dict, callable]:
    """Fourth-order periodic finite differences for fields shaped (3, nx, ny, nz).

    Returns the operator table for ``evaluate_weighted`` and a curl. The
    convention grad(u)[i, j] = d_j u_i follows the LR1 tree (mv gives (u.grad)u).
    """

    def d1(f, axis):
        return (
            -np.roll(f, -2, axis)
            + 8 * np.roll(f, -1, axis)
            - 8 * np.roll(f, 1, axis)
            + np.roll(f, 2, axis)
        ) / (12 * h)

    def d2(f, axis):
        return (
            -np.roll(f, -2, axis)
            + 16 * np.roll(f, -1, axis)
            - 30 * f
            + 16 * np.roll(f, 1, axis)
            - np.roll(f, 2, axis)
        ) / (12 * h**2)

    def gradient(f):
        if f.ndim == 3:
            return np.stack([d1(f, axis) for axis in range(3)])
        return np.stack([np.stack([d1(f[i], axis) for axis in range(3)]) for i in range(3)])

    def laplacian(f):
        return np.stack([sum(d2(f[i], axis) for axis in range(3)) for i in range(3)])

    def curl(f):
        return np.stack(
            [
                d1(f[2], 1) - d1(f[1], 2),
                d1(f[0], 2) - d1(f[2], 0),
                d1(f[1], 0) - d1(f[0], 1),
            ]
        )

    operators = {
        "gradient": gradient,
        "laplacian": laplacian,
        "matvec": lambda m, v: np.einsum("ij...,j...->i...", m, v),
        "negate": lambda a: -a,
        "add": lambda a, b: a + b,
        "divide_scalar": lambda v, s: v / s,
        "scale": lambda s, v: s * v,
    }
    return operators, curl


def leaf_values(u: np.ndarray, p: np.ndarray, nu: float, rho: float) -> dict:
    """Inputs of the LR1 tree; every occurrence of u is a separate leaf."""
    return {"n01": u, "n03": u, "n13": u, "n06": p, "n08": rho, "n12": nu}


def edge_derivatives(dag, values, operators, step: float = 1e-3) -> dict:
    """dF/d(theta_e) at theta = 1 for every edge, by central differences.

    The tree is multilinear in all edge weights except the divisor rho, for
    which the O(step^2) error is far below the identifiability tolerance.
    """
    output = dag.graph.graph["output"]
    result = {}
    for edge in edge_list(dag):
        plus = evaluate_weighted(dag, values, operators, {edge: 1 + step})[output]
        minus = evaluate_weighted(dag, values, operators, {edge: 1 - step})[output]
        result[edge] = (plus - minus) / (2 * step)
    return result


@dataclass(frozen=True)
class Incidence:
    """Edge-to-branch map: dF/d(theta_e) = sum_b M[e, b] * T_b."""

    edges: tuple
    branches: tuple
    matrix: np.ndarray
    residual: float


def branch_incidence(dag, values, operators) -> Incidence:
    """Express every edge derivative through the branch outputs T_A, T_P, T_V.

    The fit is done in the space of the right-hand side, before any
    observation operator, so that zero pressure information is not confused
    with a missing branch. ``residual`` is the worst relative misfit.
    """
    full = evaluate_weighted(dag, values, operators)
    basis = np.stack([full[node].ravel() for node in BRANCHES.values()], axis=1)
    derivatives = edge_derivatives(dag, values, operators)
    rows, worst = [], 0.0
    for edge in edge_list(dag):
        column = derivatives[edge].ravel()
        coef, *_ = np.linalg.lstsq(basis, column, rcond=None)
        misfit = np.linalg.norm(basis @ coef - column) / max(np.linalg.norm(column), 1e-300)
        worst = max(worst, float(misfit))
        rows.append(coef)
    return Incidence(tuple(edge_list(dag)), tuple(BRANCHES), np.array(rows), worst)


def edge_fisher(incidence: Incidence, branch_fisher: np.ndarray) -> np.ndarray:
    """Fisher matrix of all edge weights from the Fisher matrix of (A, P, V)."""
    m = incidence.matrix
    return m @ branch_fisher @ m.T


def observed_branch_fisher(precision_av: np.ndarray) -> np.ndarray:
    """Embed the (A, V) information into (A, P, V); velocity data give P none.

    ``precision_av`` is the inverse marginal covariance of (theta_A, theta_V),
    i.e. their information after the nuisance control parameter is profiled out.
    """
    order = list(BRANCHES)
    out = np.zeros((3, 3))
    idx = [order.index("A"), order.index("V")]
    out[np.ix_(idx, idx)] = precision_av
    return out


def incidence_from_synthetic(
    velocity: np.ndarray, pressure: np.ndarray, nu: float, h: float
) -> Incidence:
    """Edge-to-branch map evaluated on one periodic frame with its true pressure."""
    tree = navier_stokes_tree()
    operators, _ = periodic_operators(h)
    incidence = branch_incidence(tree, leaf_values(velocity, pressure, nu, 1.0), operators)
    if incidence.residual > 1e-6:
        raise ValueError(f"Edge derivatives are not combinations of branches: {incidence.residual}")
    return incidence


def element_table() -> list[dict]:
    """Structural elements, their nodes and edges, and the parameter attached to each."""
    tree = candidate_tree()
    meaning = {
        "A": "θ_A, вес ребра n05→n11 (в ЛР1 равен 1)",
        "P": "θ_P = 1/ρ, вес ребра n08→n09",
        "V": "θ_V = ν/ν_ЛР1, значение вершины n12",
        "D": "θ_D = α, значение вершины n17 (в ЛР1 нет)",
    }
    rows = []
    for name, nodes in ELEMENT_NODES.items():
        edges = [(s, t) for s, t, _ in edge_list(tree) if s in nodes]
        rows.append(
            {
                "element": name,
                "label": BRANCH_LABELS[name],
                "nodes": list(nodes),
                "edges": [f"{s}→{t}" for s, t in edges],
                "parameter": meaning[name],
            }
        )
    return rows


def _advection_and_pressure(d: OperatorDAG, kinematic_pressure: bool) -> None:
    d.leaf("n01", "u")
    d.operation("n02", "grad", "gradient", ["n01"], "matrix3x3")
    d.leaf("n03", "u")
    d.operation("n04", "mv", "matvec", ["n02", "n03"])
    d.operation("n05", "−", "negate", ["n04"])
    if kinematic_pressure:
        d.leaf("n06", "P=p/ρ", shape="scalar")
        d.operation("n07", "grad", "gradient", ["n06"])
        d.operation("n10", "−", "negate", ["n07"])
    else:
        d.leaf("n06", "p", shape="scalar")
        d.operation("n07", "grad", "gradient", ["n06"])
        d.leaf("n08", "ρ", kind="parameter", shape="scalar")
        d.operation("n09", "÷s", "divide_scalar", ["n07", "n08"])
        d.operation("n10", "−", "negate", ["n09"])
    d.operation("n11", "+", "add", ["n05", "n10"])


def _control(d: OperatorDAG, below: str) -> str:
    d.leaf("n17", "−α", kind="parameter", shape="scalar")
    d.leaf("n18", "u")
    d.operation("n19", "×s", "scale", ["n17", "n18"])
    d.operation("n20", "+", "add", [below, "n19"])
    return "n20"


def simplified_tree(viscous: bool, kinematic_pressure: bool, control: bool = False) -> OperatorDAG:
    """Tree after the structural decisions, keeping the upstream node IDs.

    ``kinematic_pressure`` removes rho (n08) and the division n09: the pressure
    leaf becomes P = p/rho and the new edge n07→n10 appears. ``viscous=False``
    removes n12..n15 and the sum n16. ``control=True`` adds the branch -alpha*u
    with its own sum n20 above the rest.
    """
    d = OperatorDAG()
    _advection_and_pressure(d, kinematic_pressure)
    output = "n11"
    if viscous:
        d.leaf("n12", "ν", kind="parameter", shape="scalar")
        d.leaf("n13", "u")
        d.operation("n14", "Δ", "laplacian", ["n13"])
        d.operation("n15", "×s", "scale", ["n12", "n14"])
        d.operation("n16", "+", "add", ["n11", "n15"])
        output = "n16"
    if control:
        output = _control(d, output)
    d.graph.graph.update(output=output, representation="expression_tree")
    return d


def candidate_tree() -> OperatorDAG:
    """The LR1 tree with the candidate control branch: the graph the decisions start from."""
    d = simplified_tree(viscous=True, kinematic_pressure=False, control=True)
    d.graph.graph.update(equation="lr1_momentum_rhs_with_control")
    return d


def decided_tree(keep: dict) -> OperatorDAG:
    """Tree after the decisions ``keep[e]`` for e in A, V, D; rho is merged into P = p/rho.

    The advection branch is not removable here: without it the remaining terms
    are linear and the tree degenerates; the decisions keep it on every data set.
    """
    if not keep["A"]:
        raise ValueError("A tree without advection is not built: decisions keep A")
    return simplified_tree(viscous=keep["V"], kinematic_pressure=True, control=keep["D"])


def edge_pairs(dag: OperatorDAG) -> set[tuple[str, str]]:
    return {(s, t) for s, t, _ in dag.graph.edges(keys=True)}


@dataclass(frozen=True)
class GraphDiff:
    """What the decisions remove from and add to the candidate tree."""

    removed_nodes: tuple
    removed_edges: tuple
    added_edges: tuple
    before: tuple  # (nodes, edges) counts
    after: tuple


def graph_diff(before: OperatorDAG, after: OperatorDAG) -> GraphDiff:
    b_nodes, a_nodes = set(before.graph), set(after.graph)
    if a_nodes - b_nodes:
        raise ValueError(f"Decisions cannot create nodes: {sorted(a_nodes - b_nodes)}")
    b_edges, a_edges = edge_pairs(before), edge_pairs(after)
    return GraphDiff(
        tuple(sorted(b_nodes - a_nodes)),
        tuple(sorted(b_edges - a_edges)),
        tuple(sorted(a_edges - b_edges)),
        (len(b_nodes), len(b_edges)),
        (len(a_nodes), len(a_edges)),
    )


def adjacency(dag: OperatorDAG, order) -> np.ndarray:
    """Z[i, j] = 1 if node j is an input of node i (the LR1 convention), on a fixed order."""
    index = {n: i for i, n in enumerate(order)}
    z = np.zeros((len(order), len(order)), dtype=int)
    for s, t in edge_pairs(dag):
        z[index[t], index[s]] = 1
    return z


# Layout of the LR1 presentation tree (lab01.graphs.plot_tree), extended by the control
# branch on the right and its sum n20 above the LR1 output n16.
TREE_POSITIONS = {
    "n20": (1.6, 1.0),
    "n16": (0, 0),
    "n11": (-2.2, -1),
    "n15": (3.6, -1),
    "n05": (-4.1, -2),
    "n10": (-0.35, -2),
    "n12": (2.5, -2),
    "n14": (4.7, -2),
    "n04": (-4.1, -3),
    "n09": (-0.35, -3),
    "n13": (4.7, -3),
    "n02": (-5, -4),
    "n03": (-3.2, -4),
    "n07": (-1.3, -4),
    "n08": (0.6, -4),
    "n01": (-5, -5),
    "n06": (-1.3, -5),
    "n19": (6.3, -1),
    "n17": (5.8, -2),
    "n18": (6.9, -2),
}
