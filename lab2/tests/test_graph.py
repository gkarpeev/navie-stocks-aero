import subprocess

import numpy as np
import pytest

from lab02 import graph, legacy
from tests.helpers import periodic_field


def test_upstream_clone_is_the_pinned_commit():
    head = subprocess.run(
        ["git", "-C", str(legacy.UPSTREAM_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert head == legacy.UPSTREAM_COMMIT
    dirty = subprocess.run(
        ["git", "-C", str(legacy.UPSTREAM_ROOT), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert dirty == "", "the LR1 code must stay unmodified"


def _inputs(n=16):
    u = periodic_field(n, seed=1)
    p = periodic_field(n, seed=2)[0]
    return u, p, 2 * np.pi / n


def test_unit_weights_reproduce_upstream_evaluation():
    u, p, h = _inputs()
    tree = legacy.navier_stokes_tree()
    ops, _ = graph.periodic_operators(h)
    values = graph.leaf_values(u, p, nu=0.3, rho=1.7)
    upstream = tree.evaluate(values, ops)
    ours = graph.evaluate_weighted(tree, values, ops)
    for node in tree.graph:
        np.testing.assert_array_equal(np.asarray(ours[node]), np.asarray(upstream[node]))


def test_unknown_edge_weight_is_rejected():
    u, p, h = _inputs()
    ops, _ = graph.periodic_operators(h)
    with pytest.raises(KeyError):
        graph.evaluate_weighted(
            legacy.navier_stokes_tree(),
            graph.leaf_values(u, p, 1, 1),
            ops,
            {("n01", "n16", 0): 2.0},
        )


def test_edge_derivatives_are_branch_combinations():
    u, p, h = _inputs()
    incidence = graph.incidence_from_synthetic(u, p, nu=0.3, h=h)
    rows = dict(zip(incidence.edges, np.round(incidence.matrix, 6)))
    advection = [
        ("n01", "n02", 0),
        ("n02", "n04", 0),
        ("n03", "n04", 1),
        ("n04", "n05", 0),
        ("n05", "n11", 0),
    ]
    pressure = [("n06", "n07", 0), ("n07", "n09", 0), ("n09", "n10", 0), ("n10", "n11", 1)]
    viscous = [("n12", "n15", 0), ("n13", "n14", 0), ("n14", "n15", 1), ("n15", "n16", 1)]
    for edge in advection:
        np.testing.assert_allclose(rows[edge], [1, 0, 0], atol=1e-6)
    for edge in pressure:
        np.testing.assert_allclose(rows[edge], [0, 1, 0], atol=1e-6)
    for edge in viscous:
        np.testing.assert_allclose(rows[edge], [0, 0, 1], atol=1e-6)
    np.testing.assert_allclose(rows["n08", "n09", 1], [0, -1, 0], atol=1e-6)  # d/d(rho) of 1/rho
    np.testing.assert_allclose(rows["n11", "n16", 0], [1, 1, 0], atol=1e-6)


def test_curl_annihilates_the_pressure_branch():
    u, p, h = _inputs()
    tree = legacy.navier_stokes_tree()
    ops, curl = graph.periodic_operators(h)
    values = graph.evaluate_weighted(tree, graph.leaf_values(u, p, 0.3, 1.0), ops)
    assert np.abs(curl(values["n10"])).max() < 1e-12 * np.abs(values["n10"]).max()
    assert np.abs(curl(values["n05"])).max() > 1e-3


def test_edge_fisher_has_rank_two():
    u, p, h = _inputs()
    incidence = graph.incidence_from_synthetic(u, p, nu=0.3, h=h)
    precision = np.array([[4.0, 1.0], [1.0, 2.0]])
    eigen = np.linalg.eigvalsh(
        graph.edge_fisher(incidence, graph.observed_branch_fisher(precision))
    )
    assert np.sum(eigen > 1e-9 * eigen.max()) == 2


def test_simplified_trees_match_the_original():
    u, p, h = _inputs()
    rho, nu = 1.7, 0.3
    ops, _ = graph.periodic_operators(h)
    full = legacy.navier_stokes_tree().evaluate(graph.leaf_values(u, p, nu, rho), ops)
    kinematic = graph.simplified_tree(viscous=True, kinematic_pressure=True)
    values = {"n01": u, "n03": u, "n13": u, "n06": p / rho, "n12": nu}
    np.testing.assert_allclose(
        kinematic.evaluate(values, ops)["n16"], full["n16"], rtol=1e-12, atol=1e-12
    )
    euler = graph.simplified_tree(viscous=False, kinematic_pressure=True)
    np.testing.assert_allclose(
        euler.evaluate(values, ops)["n11"], full["n16"] - full["n15"], rtol=1e-12, atol=1e-12
    )
    assert len(euler.graph) == 9 and euler.graph.number_of_edges() == 8


def test_element_table_covers_all_tree_edges():
    edges = {e for row in graph.element_table() for e in row["edges"]}
    tree_edges = {f"{s}→{t}" for s, t, _ in graph.edge_list(legacy.navier_stokes_tree())}
    assert tree_edges - edges == {"n11→n16"}  # the sum edge carries A + P, see the incidence


def test_adjacency_follows_the_lr1_convention():
    tree = legacy.navier_stokes_tree()
    order, binary, _ = tree.matrices()  # LR1 code: Z[target, source]
    np.testing.assert_array_equal(graph.adjacency(tree, order), binary)


def test_candidate_tree_adds_the_control_branch():
    u, p, h = _inputs()
    alpha, nu, rho = 0.25, 0.3, 1.7
    ops, _ = graph.periodic_operators(h)
    full = legacy.navier_stokes_tree().evaluate(graph.leaf_values(u, p, nu, rho), ops)["n16"]
    values = {**graph.leaf_values(u, p, nu, rho), "n17": -alpha, "n18": u}
    out = graph.candidate_tree().evaluate(values, ops)["n20"]
    np.testing.assert_allclose(out, full - alpha * u, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "keep,after,added",
    [
        ({"A": True, "V": False, "D": False}, (9, 8), {("n07", "n10")}),
        ({"A": True, "V": True, "D": False}, (14, 13), {("n07", "n10")}),
        ({"A": True, "V": True, "D": True}, (18, 17), {("n07", "n10")}),
        ({"A": True, "V": False, "D": True}, (13, 12), {("n07", "n10"), ("n11", "n20")}),
    ],
)
def test_decisions_remove_and_add_the_expected_nodes_and_edges(keep, after, added):
    diff = graph.graph_diff(graph.candidate_tree(), graph.decided_tree(keep))
    assert diff.before == (20, 19)
    assert diff.after == after
    assert set(diff.added_edges) == added
    assert {"n08", "n09"} <= set(diff.removed_nodes)  # rho is merged into P = p/rho
    assert ("n12" in diff.removed_nodes) != keep["V"]
    assert ("n17" in diff.removed_nodes) != keep["D"]
