"""Edge annotation with the mutually exclusive edge taxonomy of configs/chemistry_v1.yaml (chemistry-v1.1).

For every edge we record:
  radius_bonded      d_min_image <= r_i + r_j + tol (Cordero radii via pymatgen)
  crystalnn          CrystalNN pair (either site lists the other, any image)
  protection_route   which lens(es) support the edge: none | crystalnn | radius | both (recorded for ALL edges)
  bond_class         exactly one class, first match in this order:
      covalent_radius          non-metal pair, radius_bonded                           -> protected
      metal_ligand             metal-non-metal pair, crystalnn OR radius_bonded        -> protected
      metal_metal_crystalnn    metal-metal pair, crystalnn (radius route not used)      -> protected
      hydrogen_bond_candidate  non-metal pair, crystalnn, not radius_bonded, H+acceptor -> removable
      crystalnn_nonmetal       other non-metal crystalnn pair, not radius_bonded        -> removable
      metal_contact            any other metal-involving pair                           -> removable
      other_contact            everything else                                          -> removable
  protected          True for the first three classes
  C                  former additive chemistry prior by class (exploratory 3-term ablation only; not in ChemSPAR-v2)
"""
from __future__ import annotations

import networkx as nx
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius

from .config import load_chemistry

PROTECTED_CLASSES = ("covalent_radius", "metal_ligand", "metal_metal_crystalnn")
ALL_CLASSES = PROTECTED_CLASSES + ("hydrogen_bond_candidate", "crystalnn_nonmetal", "metal_contact", "other_contact")


def radius_bonded(el_i: str, el_j: str, d: float, tol: float) -> bool:
    r = CovalentRadius.radius
    return d <= r[el_i] + r[el_j] + tol


def classify_edge(el_i: str, el_j: str, metal_i: bool, metal_j: bool, rb: bool, cn: bool,
                  acceptors: frozenset) -> str:
    if metal_i and metal_j:
        return "metal_metal_crystalnn" if cn else "metal_contact"
    if metal_i or metal_j:
        return "metal_ligand" if (cn or rb) else "metal_contact"
    if rb:
        return "covalent_radius"
    if cn:
        pair = {el_i, el_j}
        if "H" in pair and (pair - {"H"}) & acceptors:
            return "hydrogen_bond_candidate"
        return "crystalnn_nonmetal"
    return "other_contact"


def annotate(g: nx.Graph, cnn_pairs: dict | None, tol: float | None = None) -> nx.Graph:
    """Annotate every edge in place. `cnn_pairs=None` means CrystalNN was not computed
    (recorded as crystalnn_available=False; CrystalNN-dependent classes then cannot occur)."""
    cfg = load_chemistry()
    tol = cfg["bonding"]["radius_tolerance_angstrom"] if tol is None else tol
    prior = cfg["score"]["exploratory_three_term"]["chemistry_prior"]   # exploratory only (S3 ablation)
    acceptors = frozenset(cfg["protection"]["hydrogen_bond_acceptors"])
    for u, v, a in g.edges(data=True):
        eu, ev = g.nodes[u]["element"], g.nodes[v]["element"]
        mu, mv = g.nodes[u]["is_metal"], g.nodes[v]["is_metal"]
        rb = radius_bonded(eu, ev, a["distance"], tol)
        key = (u, v) if u < v else (v, u)
        cn = bool(cnn_pairs is not None and key in cnn_pairs)
        cn_min_image = None
        if cn:
            im = a["image"] if u < v else tuple(-x for x in a["image"])
            cn_min_image = im in cnn_pairs[key]["images"]
        cls = classify_edge(eu, ev, mu, mv, rb, cn, acceptors)
        route = "both" if (cn and rb) else "crystalnn" if cn else "radius" if rb else "none"
        pair_type = "metal-metal" if (mu and mv) else "metal-nonmetal" if (mu or mv) else "nonmetal-nonmetal"
        a.update(radius_bonded=rb, crystalnn=cn, crystalnn_available=cnn_pairs is not None,
                 crystalnn_image_is_min_image=cn_min_image, protection_route=route,
                 protected=cls in PROTECTED_CLASSES, pair_type=pair_type, bond_class=cls,
                 C=float(prior[cls]), radius_tolerance=tol)
    return g
