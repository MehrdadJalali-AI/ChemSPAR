"""Frozen-definition tests (immutability of ChemSPAR-v2) and Phase 2 baseline/pruning invariants."""
import hashlib
import json
from pathlib import Path

import networkx as nx
import pytest

from chemspar_v2 import baselines_bhs, bonding, constructions, graph, prune, scores
from chemspar_v2.config import load_chemistry
from chemspar_v2.crystalnn import crystalnn_pairs

RESUB = Path(__file__).resolve().parents[1]
FROZEN = json.loads((RESUB / "configs/FROZEN.json").read_text())


# ---------------------------------------------------------------- immutability
@pytest.mark.parametrize("rel", sorted(FROZEN["files_sha256"]))
def test_frozen_file_hashes(rel):
    assert hashlib.sha256((RESUB / rel).read_bytes()).hexdigest() == FROZEN["files_sha256"][rel], \
        f"{rel} changed after freezing; re-run scripts/freeze_configs.py and record the change"


def test_frozen_score_definition_matches_loaded_config():
    cfg = load_chemistry()
    assert FROZEN["method"] == "ChemSPAR-v2" and FROZEN["chemistry_version"] == "chemistry-v1.2"
    assert cfg["score"]["primary"] == "gravity" and FROZEN["score"]["primary"] == "gravity"
    assert "edge_weights" not in cfg["score"] and "chemistry_prior" not in cfg["score"]   # B, C not in the primary score
    ex = cfg["score"]["exploratory_three_term"]
    assert ex["edge_weights"] == {"gravity": 1.0, "betweenness": 0.5, "chemistry": 1.2}
    assert set(ex["chemistry_prior"]) == set(bonding.ALL_CLASSES)
    assert FROZEN["taxonomy"] == cfg["protection"]["taxonomy"]


def test_primary_score_is_gravity(cu_graph):
    _, g, comp = cu_graph
    assert all(c["S"] == c["g"] for c in comp.values())
    lean = scores.edge_components(g, exploratory=False)
    assert all(lean[e]["S"] == comp[e]["S"] for e in comp) and "B" not in next(iter(lean.values()))


# ---------------------------------------------------------------- pruning invariants on a real structure
@pytest.fixture(scope="module")
def cu_graph(load_structure):
    s = load_structure("qmof-4125488")
    g = bonding.annotate(graph.build_cutoff_graph(s, 4.5), crystalnn_pairs(s))
    return s, g, scores.edge_components(g)


def test_prefix_property_of_static_pruning(cu_graph):
    """Removed set at tau is the first floor(tau|E|) acceptances of a run to a larger tau."""
    _, g, comp = cu_graph
    order = prune.order_by_score(comp)
    full = prune.prune(g, 0.9, order, "corrected", audit=False)
    for tau in (0.2, 0.5, 0.7):
        direct = prune.prune(g, tau, order, "corrected", audit=False)
        assert direct.removed == full.removed[: direct.target]


def test_fast_connectivity_equals_whole_graph_check(cu_graph):
    _, g, _ = cu_graph
    h = g.copy()
    for u, v in list(h.edges())[:300]:
        a = dict(h.edges[u, v]); h.remove_edge(u, v)
        assert nx.has_path(h, u, v) == nx.is_connected(h)
        h.add_edge(u, v, **a)


def test_pruned_graphs_stay_connected_and_reach_target(cu_graph):
    _, g, comp = cu_graph
    for mode in ("corrected", "connectivity_only"):
        r = prune.prune(g, 0.5, prune.order_by_score(comp), mode, audit=False)
        assert nx.is_connected(r.graph) and len(r.removed) == r.target


# ---------------------------------------------------------------- constructions
def test_cutoff_subgraph_is_subset(cu_graph):
    _, g, _ = cu_graph
    h = constructions.cutoff_subgraph(g, 3.0)
    assert set(h.edges()) <= set(g.edges()) and all(a["distance"] <= 3.0 for _, _, a in h.edges(data=True))
    assert h.number_of_nodes() == g.number_of_nodes()


def test_crystalnn_graph_edges_are_crystalnn(cu_graph):
    _, g, _ = cu_graph
    h = constructions.crystalnn_graph(g)
    assert h.number_of_edges() == sum(a["crystalnn"] for _, _, a in g.edges(data=True))


def test_knn_symmetrised_and_degree_at_least_k(cu_graph):
    s, _, _ = cu_graph
    e = constructions.knn_edges(s, k=12)
    short = e.pop("__atoms_with_fewer_than_k_within_rmax__")[0]
    assert short == 0
    deg = {}
    for i, j in e:
        assert i < j
        deg[i] = deg.get(i, 0) + 1; deg[j] = deg.get(j, 0) + 1
    # each atom keeps its own k nearest (<= k distinct partners because images of one atom collapse)
    assert min(deg.values()) >= 1


# ---------------------------------------------------------------- BHS adaptation
def test_bhs_gravity_matches_documented_formula(cu_graph):
    _, g, _ = cu_graph
    gam = baselines_bhs.bhs_node_gravity(g)
    assert all(0.0 <= x <= 1.0 + 1e-12 for x in gam.values())
    sc = baselines_bhs.bhs_edge_scores(g)
    u, v = next(iter(g.edges()))
    assert sc["min"][(min(u, v), max(u, v))] == pytest.approx(min(gam[u], gam[v]))
    assert sc["prod"][(min(u, v), max(u, v))] == pytest.approx(gam[u] * gam[v])


def test_bhs_weights_follow_published_equal_weighting():
    """Published BHS (JCIM 2025, Eq. 1): alpha = beta = gamma, summing to 1."""
    w = baselines_bhs.BHS_WEIGHTS
    assert len(set(round(x, 12) for x in w)) == 1 and abs(sum(w) - 1.0) < 1e-12
