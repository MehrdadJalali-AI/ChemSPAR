"""Periodic edge-instance representation: mapping tests on representative multi-image structures.

Representative structures (selected by stated criteria from the fixed cohort, PHASE_3 notes):
  qmof-aa014ee  highest share of atom pairs with >1 periodic image within 4.5 A (80%), 34 atoms
  qmof-8643335  most self-image contacts among the 40 most multi-image structures (88)
  qmof-4846b1b  most multi-image metal-incident pairs (110)
  qmof-0338cb2  median-size structure (96 atoms, 1.8% multi-image pairs)
"""
import networkx as nx
import numpy as np
import pytest
import torch

from chemspar_v2 import periodic as P
from chemspar_v2 import views as V
from chemspar_v2.elements import role
from chemspar_v2.graph import build_cutoff_graph
from chemspar_v2.train import instances_to_data, node_features, tensor_instances, verify_view_tensor

REP = ["qmof-aa014ee", "qmof-8643335", "qmof-4846b1b", "qmof-0338cb2"]
TENSOR_VIEWS = ["Original-4.5", "Cutoff-3.5", "Cutoff-3.0", "CrystalNN", "kNN-12", "ChemSPAR-v2@0.2", "Random@0.2",
                "Random-Chem@0.2", "Distance@0.2", "Distance-Chem@0.2", "BHS-edge-min@0.2", "ChemSPAR-v2-massfree@0.2",
                "ChemSPAR-v2@0.5", "Random-Chem@0.5", "Distance-Chem@0.5", "BHS-edge-min@0.5",
                # final matrix: ungated partners (configs/final_v1.yaml)
                "ChemSPAR-v2-noGate@0.2", "BHS-edge-min-noGate@0.2", "ChemSPAR-v2-noGate@0.5", "Random@0.5",
                "Distance@0.5", "BHS-edge-min-noGate@0.5"]


@pytest.fixture(scope="module")
def rep(load_structure):
    return {q: (load_structure(q), V.compute_structure(load_structure(q), q)) for q in REP}


def test_canonical_key_roundtrip():
    for i, j, im in [(0, 3, (1, 0, -1)), (3, 0, (1, 0, -1)), (2, 2, (0, -1, 0)), (2, 2, (0, 1, 0))]:
        k = P.canonical(i, j, im)
        assert P.canonical(k[1], k[0], tuple(-x for x in k[2])) == k
        if i == j:
            assert k[2] > (0, 0, 0)


@pytest.mark.parametrize("qid", REP)
def test_instance_graph_matches_neighbour_list(rep, qid):
    s, res = rep[qid]
    c, p, im, d = s.get_neighbor_list(4.5)
    directed = sum(1 for a, b, x in zip(c, p, im) if not (a == b and not any(x)))
    assert directed == 2 * len(res["instances"])                  # every directed neighbour appears exactly once per direction
    for k in res["instances"]:
        assert res["dist"][k] == pytest.approx(P.instance_distance(s, k), abs=1e-6)


@pytest.mark.parametrize("qid", REP)
def test_minimum_image_pairs_are_contained(rep, qid):
    s, res = rep[qid]
    g = build_cutoff_graph(s, 4.5)
    pair_min = {}
    for (i, j, _), dd in res["dist"].items():
        if (i, j, _) in res["attrs"] and i != j:
            pair_min[(i, j)] = min(pair_min.get((i, j), 9e9), dd)
    assert set(pair_min) == {(min(u, v), max(u, v)) for u, v in g.edges()}
    for u, v, a in g.edges(data=True):
        assert pair_min[(min(u, v), max(u, v))] == pytest.approx(a["distance"], abs=1e-6)


def test_representatives_really_have_multiple_images(rep):
    s, res = rep["qmof-aa014ee"]
    pairs = {}
    for i, j, _ in res["instances"]:
        pairs[(i, j)] = pairs.get((i, j), 0) + 1
    assert sum(v > 1 for v in pairs.values()) / len(pairs) > 0.5
    assert sum(1 for i, j, _ in rep["qmof-8643335"][1]["instances"] if i == j) >= 50


@pytest.mark.parametrize("qid", REP)
def test_crystalnn_instances_exist_in_graph(rep, qid):
    s, res = rep[qid]
    assert res["cnn"] <= set(res["instances"])


@pytest.mark.parametrize("qid", REP)
def test_pruned_views_never_remove_protected_and_stay_connected(rep, qid):
    s, res = rep[qid]
    for view in ["ChemSPAR-v2@0.2", "ChemSPAR-v2@0.5", "Random-Chem@0.5", "Distance-Chem@0.5", "BHS-edge-min@0.5"]:
        kept = set(V.view_instances(res, view))
        removed = set(res["instances"]) - kept
        assert len(removed) == int(float(view.split("@")[1]) * len(res["instances"]))
        assert not any(res["attrs"][k]["protected"] for k in removed)
        h = nx.MultiGraph(); h.add_nodes_from(range(len(s))); h.add_edges_from((i, j) for i, j, _ in kept)
        assert nx.is_connected(h)


@pytest.mark.parametrize("qid", REP)
def test_prefix_property_on_instances(rep, qid):
    s, res = rep[qid]
    g, inst = res["graph"], res["instances"]
    order = P.order_by(res["val"]["gravity"], inst)
    full = P.prune_instances(g, 0.9, order, "corrected", audit=False)["removed"]
    for tau in (0.2, 0.5):
        direct = P.prune_instances(g, tau, order, "corrected", audit=False)["removed"]
        assert direct == full[:len(direct)]


@pytest.mark.parametrize("qid", REP)
def test_tensor_contains_exactly_the_view(rep, qid):
    s, res = rep[qid]
    z, r = node_features([x.specie.Z for x in s], [role(x.specie.symbol) for x in s])
    for view in TENSOR_VIEWS:
        kept = V.view_instances(res, view)
        d = instances_to_data(z, r, kept, [res["dist"][k] for k in kept], 1.0, qid)
        assert verify_view_tensor(d, kept), view
        # removed instances never reappear
        if "@" in view:
            removed = set(res["instances"]) - set(kept)
            assert not (set(tensor_instances(d)) & removed)
        # distances in the tensor equal the true periodic distances of each instance
        for (a, b, im), dist in zip(zip(d.edge_index[0].tolist(), d.edge_index[1].tolist(), d.edge_image.tolist()), d.edge_dist.tolist()):
            assert dist == pytest.approx(P.instance_distance(s, P.canonical(a, b, tuple(im))), abs=1e-4)


def test_partially_pruned_pairs_exist_and_are_respected(rep):
    """With instance-level pruning a pair can keep one image and lose another; the tensor must reflect that."""
    found = 0
    for qid in REP:
        s, res = rep[qid]
        kept = set(V.view_instances(res, "ChemSPAR-v2@0.5"))
        byp = {}
        for i, j, im in res["instances"]:
            byp.setdefault((i, j), []).append((i, j, im) in kept)
        found += sum(1 for v in byp.values() if any(v) and not all(v))
    assert found > 0


def test_self_image_instances_become_two_directed_self_loops(rep):
    s, res = rep["qmof-8643335"]
    k = next(k for k in res["instances"] if k[0] == k[1])
    z, r = node_features([x.specie.Z for x in s], [role(x.specie.symbol) for x in s])
    d = instances_to_data(z, r, [k], [res["dist"][k]], 0.0, "x")
    assert d.edge_index.tolist() == [[k[0], k[0]], [k[0], k[0]]]
    assert sorted(map(tuple, d.edge_image.tolist())) == sorted([k[2], tuple(-x for x in k[2])])


@pytest.mark.parametrize("qid", REP)
def test_tensor_store_masks_reproduce_views(rep, qid):
    """The compact store (base instances + per-view mask) must yield exactly view_instances for every trained view."""
    from chemspar_v2.tensor_store import view_data
    s, res = rep[qid]
    z, r = node_features([x.specie.Z for x in s], [role(x.specie.symbol) for x in s])
    inst = res["instances"]
    entry = dict(qmof_id=qid, z=z, roles=r, y=0.0, extras={},
                 inst=torch.tensor([[i, j, *im] for i, j, im in inst], dtype=torch.int32),
                 dist=torch.tensor([res["dist"][k] for k in inst], dtype=torch.float32))
    for c in ("CrystalNN", "kNN-12"):
        ex = [k for k in res["constructions"][c] if k not in set(inst)]
        if ex:
            entry["extras"][c] = (torch.tensor([[i, j, *im] for i, j, im in ex], dtype=torch.int32),
                                  torch.tensor([res["dist"][k] for k in ex], dtype=torch.float32))
    for view in TENSOR_VIEWS:
        kept = set(V.view_instances(res, view))
        mask = torch.tensor([k in kept for k in inst])
        d = view_data(entry, mask, view)
        assert verify_view_tensor(d, list(kept)), view


UNGATED = {"ChemSPAR-v2": "ChemSPAR-v2-noGate", "Distance-Chem": "Distance", "Random-Chem": "Random",
           "BHS-edge-min": "BHS-edge-min-noGate"}


def test_final_matrix_pairs_are_matched_in_code():
    """Each gated order has an ungated partner with the SAME ordering key, gate = connectivity only, sweep to 0.9."""
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/final_v1.yaml").read_text())
    assert cfg["views"]["ungated_partner"] == UNGATED
    assert len(set(cfg["display_names"].values())) == len(cfg["display_names"]) == 8
    for gated, ungated in UNGATED.items():
        og, mg, eg, tg = V.POLICIES[gated]
        ou, mu, eu, tu = V.POLICIES[ungated]
        assert og == ou and mg == "corrected" and mu == "connectivity_only" and eg == eu == () and tg == tu == 0.9
    assert len(cfg["views"]["trained"]) == cfg["views"]["n_trained_views"] == 8 and cfg["backbones"]["cgcnn"]["seeds"] == [1, 2, 3, 4, 5]
    for g in cfg["views"]["orders"]:                             # every gated order at tau 0.5 is trained
        assert f"{g}@0.5" in cfg["views"]["trained"]


@pytest.mark.parametrize("qid", REP)
def test_ungated_partners_same_size_connected_and_prefix(rep, qid):
    s, res = rep[qid]
    g, inst = res["graph"], res["instances"]
    for gated, ungated in UNGATED.items():
        for tau in (0.2, 0.5):
            kept = set(V.view_instances(res, f"{ungated}@{tau}"))
            assert len(set(inst) - kept) == len(set(inst) - set(V.view_instances(res, f"{gated}@{tau}")))  # matched sparsity
            h = nx.MultiGraph(); h.add_nodes_from(range(len(s))); h.add_edges_from((i, j) for i, j, _ in kept)
            assert nx.is_connected(h)
    # ungated prefix property (extension of Random/Distance from 0.2 to 0.9 leaves the 0.2 view unchanged)
    order = P.order_random(inst, 42, qid)
    full = P.prune_instances(g, 0.9, order, "connectivity_only", audit=False)["removed"]
    assert P.prune_instances(g, 0.2, order, "connectivity_only", audit=False)["removed"] == full[:int(0.2 * len(inst))]


def test_ungated_bhs_removes_protected_somewhere(rep):
    """The ungated partner is really ungated: at tau 0.5 it removes protected instances in at least one structure."""
    assert any(any(res["attrs"][k]["protected"] for k in set(res["instances"]) - set(V.view_instances(res, "BHS-edge-min-noGate@0.5")))
               for _, res in rep.values())
