"""Unit tests for the corrected ChemSPAR pipeline (src/chemspar_v2)."""
import networkx as nx
import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from chemspar_v2 import bonding, elements, graph, prune, scores
from chemspar_v2.config import load_chemistry
from chemspar_v2.crystalnn import crystalnn_pairs

# Real QMOF structures used as fixtures (all in the legacy 1,000-MOF debugging cohort):
U_MOF = "qmof-63f404f"     # U centre; U absent from the legacy metal set
CU_MOF = "qmof-4125488"    # Cu centre; recognised by both definitions
MIXED = "qmof-2235ac6"


# ---------------------------------------------------------------- elements
@pytest.mark.parametrize("el", ["Zn", "Cu", "Zr", "Fe", "Co", "Mg", "Al", "Li"])
def test_recognised_metals(el):
    assert elements.is_metal(el) and el in elements.LEGACY_METALS


@pytest.mark.parametrize("el", ["Cd", "Ag", "Hg", "U", "Gd", "Pb", "Sn", "Bi", "Mo", "Y", "La", "Ce"])
def test_previously_omitted_metals_now_recognised(el):
    assert elements.is_metal(el), el
    assert el not in elements.LEGACY_METALS
    assert elements.role(el) == "metal"


@pytest.mark.parametrize("el", ["B", "Si", "Ge", "As", "Sb", "Te"])
def test_metalloids_forced_nonmetal(el):
    assert not elements.is_metal(el)
    assert el in load_chemistry()["metals"]["forced_nonmetals"]


@pytest.mark.parametrize("el", ["H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Se"])
def test_nonmetals(el):
    assert not elements.is_metal(el)


def test_pymatgen_version_matches_config():
    from importlib.metadata import version
    assert version("pymatgen") == load_chemistry()["pymatgen_version_validated"]


# ---------------------------------------------------------------- graph
@pytest.mark.parametrize("qid", [U_MOF, CU_MOF, MIXED])
def test_graph_matches_legacy_edges_order_and_distances(load_structure, qid):
    pytest.importorskip("utils.legacy", reason="legacy pre-ChemSPAR code is not part of this repository")
    from utils.legacy import load_run_all
    s = load_structure(qid)
    g_legacy = load_run_all().build_graph_distance(s, 4.5)
    g_v2 = graph.build_cutoff_graph(s, 4.5)
    assert list(g_legacy.edges()) == list(g_v2.edges())
    d_l = np.array([d["distance"] for _, _, d in g_legacy.edges(data=True)])
    d_v = np.array([d["distance"] for _, _, d in g_v2.edges(data=True)])
    assert np.allclose(d_l, d_v, atol=1e-6)


def test_periodic_image_recorded():
    s = Structure(Lattice.cubic(5.0), ["C", "C"], [[0.1, 0.5, 0.5], [0.9, 0.5, 0.5]])
    g = graph.build_cutoff_graph(s, 4.5)
    a = g.edges[0, 1]
    assert a["distance"] == pytest.approx(1.0, abs=1e-6)
    assert a["image"] == (-1, 0, 0)            # atom 1 is realised through the -a image
    assert a["n_images_within_cutoff"] >= 2    # the in-cell copy at 4.0 A is also within the cutoff


# ---------------------------------------------------------------- edge taxonomy (chemistry-v1.1)
ACC = frozenset(load_chemistry()["protection"]["hydrogen_bond_acceptors"])


@pytest.mark.parametrize("ei,ej,mi,mj,rb,cn,expected", [
    ("C", "C", False, False, True, True, "covalent_radius"),        # radius wins, CrystalNN also present
    ("C", "S", False, False, True, False, "covalent_radius"),       # C-S now protected (legacy: 'vdw')
    ("C", "Cl", False, False, True, False, "covalent_radius"),      # C-halogen now protected
    ("H", "H", False, False, False, False, "other_contact"),        # geminal-type H...H no longer protected
    ("Cd", "O", True, False, True, False, "metal_ligand"),           # radius route
    ("Cd", "O", True, False, False, True, "metal_ligand"),           # CrystalNN route
    ("Cd", "C", True, False, False, False, "metal_contact"),         # long metal contact: removable
    ("U", "U", True, True, True, False, "metal_contact"),            # D6: radius route NOT used for M-M
    ("Cu", "Cu", True, True, False, True, "metal_metal_crystalnn"),  # D6: CrystalNN M-M protected
    ("H", "O", False, False, False, True, "hydrogen_bond_candidate"),
    ("Br", "H", False, False, False, True, "hydrogen_bond_candidate"),
    ("H", "C", False, False, False, True, "crystalnn_nonmetal"),    # C is not an acceptor
    ("C", "C", False, False, False, True, "crystalnn_nonmetal"),
    ("C", "O", False, False, False, False, "other_contact"),
])
def test_taxonomy_classes_and_precedence(ei, ej, mi, mj, rb, cn, expected):
    assert bonding.classify_edge(ei, ej, mi, mj, rb, cn, ACC) == expected
    assert (expected in bonding.PROTECTED_CLASSES) == (expected in ("covalent_radius", "metal_ligand", "metal_metal_crystalnn"))


def _toy():
    """15 A cubic cell; atoms placed so that only the intended short contacts exist."""
    species = ["Cd", "O", "C", "H", "H", "C", "S", "U", "U", "H", "O"]
    cart = [[5.0, 5.0, 5.0], [7.3, 5.0, 5.0], [5.0, 8.9, 5.0],       # Cd-O 2.30 A, Cd...C 3.90 A
            [10.0, 10.0, 10.0], [11.8, 10.0, 10.0],                  # H...H 1.80 A
            [1.0, 12.0, 1.0], [2.79, 12.0, 1.0],                     # C-S 1.79 A
            [10.0, 2.0, 12.0], [14.2, 2.0, 12.0],                    # U...U 4.20 A
            [2.0, 2.0, 10.0], [3.7, 2.0, 10.0]]                      # H...O 1.70 A
    s = Structure(Lattice.cubic(15.0), species, cart, coords_are_cartesian=True)
    return graph.build_cutoff_graph(s, 4.5)


def _mock_cnn(*pairs):
    d = {tuple(sorted(p)): {"images": {(0, 0, 0)}} for p in pairs}
    d["__failed_sites__"] = {"count": 0}
    return d


def test_covalent_radius_class_protected():
    g = bonding.annotate(_toy(), cnn_pairs=_mock_cnn())
    assert g.edges[5, 6]["bond_class"] == "covalent_radius" and g.edges[5, 6]["protected"]
    assert g.edges[0, 1]["bond_class"] == "metal_ligand" and g.edges[0, 1]["protection_route"] == "radius"


def test_metal_metal_only_protected_via_crystalnn():
    g = bonding.annotate(_toy(), cnn_pairs=_mock_cnn())
    uu = g.edges[7, 8]
    assert uu["radius_bonded"] and uu["bond_class"] == "metal_contact" and not uu["protected"]
    g = bonding.annotate(_toy(), cnn_pairs=_mock_cnn((7, 8)))
    assert g.edges[7, 8]["bond_class"] == "metal_metal_crystalnn" and g.edges[7, 8]["protected"]
    assert g.edges[7, 8]["protection_route"] == "both"


def test_hydrogen_bond_candidate_removable_by_default_protected_in_hbprotect():
    g = bonding.annotate(_toy(), cnn_pairs=_mock_cnn((9, 10)))
    hb = g.edges[9, 10]
    assert hb["bond_class"] == "hydrogen_bond_candidate" and not hb["protected"] and hb["protection_route"] == "crystalnn"
    extra = tuple(load_chemistry()["protection"]["variants"]["ChemSPAR-HBProtect"]["extra_protected_classes"])
    r = prune.prune(g, 0.99, [(9, 10)], "corrected", extra_protected_classes=extra)
    assert r.audit[0]["reason"] == "protected:hydrogen_bond_candidate:crystalnn" and r.removed == []
    assert r.audit[0]["protected_under_policy"] is True


def test_removable_contacts():
    g = bonding.annotate(_toy(), cnn_pairs=_mock_cnn())
    assert g.edges[0, 2]["bond_class"] == "metal_contact" and not g.edges[0, 2]["protected"]
    assert g.edges[3, 4]["bond_class"] == "other_contact" and not g.edges[3, 4]["protected"]
    for e in [(0, 2), (3, 4)]:
        a = g.edges[e]
        assert prune._checks(g, *e, "corrected")[0]["protected"] == "pass"


def test_classes_mutually_exclusive_on_real_structures(load_structure):
    for qid in (U_MOF, CU_MOF):
        s = load_structure(qid)
        g = bonding.annotate(graph.build_cutoff_graph(s, 4.5), crystalnn_pairs(s))
        for _, _, a in g.edges(data=True):
            assert a["bond_class"] in bonding.ALL_CLASSES
            assert a["protected"] == (a["bond_class"] in bonding.PROTECTED_CLASSES)


def test_tolerance_sensitivity_changes_radius_class():
    s = Structure(Lattice.cubic(15.0), ["C", "C"], [[5, 5, 5], [6.8, 5, 5]], coords_are_cartesian=True)   # 1.80 A
    lo = bonding.annotate(graph.build_cutoff_graph(s, 4.5), _mock_cnn(), tol=0.25)
    hi = bonding.annotate(graph.build_cutoff_graph(s, 4.5), _mock_cnn(), tol=0.40)
    assert lo.edges[0, 1]["bond_class"] == "other_contact"      # 1.80 > 2*0.73 + 0.25 = 1.71
    assert hi.edges[0, 1]["bond_class"] == "covalent_radius"    # 1.80 <= 1.86


def test_crystalnn_unavailable_is_recorded():
    g = bonding.annotate(_toy(), cnn_pairs=None)
    assert all(not a["crystalnn_available"] for _, _, a in g.edges(data=True))


# ---------------------------------------------------------------- gate
def _chain():
    """Path graph O-C-C-C with a chord C1-C3 so that some removals are feasible."""
    s = Structure(Lattice.cubic(30.0), ["O", "C", "C", "C"],
                  [[5, 5, 5], [6.2, 5, 5], [8.0, 5, 5], [7.0, 6.9, 5]], coords_are_cartesian=True)
    g = bonding.annotate(graph.build_cutoff_graph(s, 4.5), cnn_pairs={})
    return g


def test_gate_rejects_protected_and_records_route():
    g = _chain()
    e = (0, 1)
    assert g.edges[e]["protected"]
    r = prune.prune(g, 0.99, [e], "corrected", policy="t")
    assert r.removed == [] and r.audit[0]["reason"] == "protected:%s:%s" % (g.edges[e]["bond_class"], g.edges[e]["protection_route"])
    assert r.audit[0]["check_protected"] == "fail" and r.audit[0]["check_connectivity"] == "not_evaluated"


def test_gate_connectivity_and_degree_rules():
    g = nx.Graph()
    for n, el in enumerate(["C", "C", "C", "H"]):
        g.add_node(n, element=el)
    for u, v in [(0, 1), (1, 2), (2, 0), (2, 3)]:
        g.add_edge(u, v, distance=1.5, image=(0, 0, 0), n_images_within_cutoff=1, bond_class="other_contact",
                   radius_bonded=False, crystalnn=False, crystalnn_available=True, protection_route="none",
                   protected=False, C=0.2)
    r = prune.prune(g, 0.99, [(2, 3), (0, 1), (1, 2)], "connectivity_only")
    reasons = [a["reason"] for a in r.audit]
    assert reasons[0] == "connectivity"      # removing the only H edge would isolate H -> disconnects
    assert reasons[1] == "accepted"          # ring edge
    assert reasons[2] in {"degree", "connectivity"}   # C1 now has degree 1 -> non-H degree rule
    assert nx.is_connected(r.graph)


# ---------------------------------------------------------------- audit rows on a real structure
@pytest.fixture(scope="module")
def u_pruned(load_structure):
    s = load_structure(U_MOF)
    g = bonding.annotate(graph.build_cutoff_graph(s, 4.5), crystalnn_pairs(s))
    comp = scores.edge_components(g)
    r = prune.prune(g, 0.2, prune.order_by_score(comp), "corrected", components=comp, qmof_id=U_MOF, policy="ChemSPAR-v2")
    return g, comp, r


def test_audit_rows_recorded_before_deletion(u_pruned):
    g, comp, r = u_pruned
    acc = [a for a in r.audit if a["decision"] == "accepted"]
    assert len(acc) == len(r.removed) == r.target
    for a in acc:
        e = g.edges[a["atom_i"], a["atom_j"]]          # original graph
        assert a["distance"] == pytest.approx(e["distance"]) and a["distance"] > 0
        assert a["bond_class"] == e["bond_class"] and a["protection_route"] == e["protection_route"]
        assert a["S"] == pytest.approx(comp[(a["atom_i"], a["atom_j"])]["S"])
        assert not r.graph.has_edge(a["atom_i"], a["atom_j"])
    assert all(a["protection_route"] == "none" for a in acc)


def test_audit_rejections_have_reasons_and_graph_consistent(u_pruned):
    g, _, r = u_pruned
    rej = [a for a in r.audit if a["decision"] == "rejected"]
    assert all(a["reason"] != "accepted" for a in rej)
    assert r.graph.number_of_edges() == g.number_of_edges() - len(r.removed)
    assert nx.is_connected(r.graph)


def test_u_first_shell_protected(load_structure, u_pruned):
    g, _, r = u_pruned
    u = next(n for n, d in g.nodes(data=True) if d["element"] == "U")
    shell = [(min(u, v), max(u, v)) for v in g.neighbors(u) if g.edges[u, v]["protected"]]
    assert len(shell) >= 6
    assert all(r.graph.has_edge(*e) for e in shell)


# ---------------------------------------------------------------- scores identical to legacy for identical inputs
def test_score_components_match_legacy_on_recognised_metal(load_structure):
    pytest.importorskip("utils.legacy", reason="legacy pre-ChemSPAR code is not part of this repository")
    from utils.legacy import load_run_all
    import sys
    from sparsification.abh_static import compute_static_edge_scores
    s = load_structure(CU_MOF)
    g_legacy = load_run_all().build_graph_distance(s, 4.5)
    legacy = compute_static_edge_scores(g_legacy)
    g = bonding.annotate(graph.build_cutoff_graph(s, 4.5), cnn_pairs=None)
    v2 = scores.edge_components(g)
    for e, c in legacy.items():
        assert v2[e]["g"] == pytest.approx(c["gravity_score"], rel=1e-9)
        assert v2[e]["B"] == pytest.approx(c["bridge_score"], rel=1e-9)
