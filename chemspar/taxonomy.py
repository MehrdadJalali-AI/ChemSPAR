"""Element roles, the bond taxonomy and its two protection routes (radius and CrystalNN).

Every contact instance gets exactly one class, first match in this order (``chemistry_v1.yaml``):

==========================  ==================  ==========================================  =========
class                       pair                rule                                        protected
==========================  ==================  ==========================================  =========
covalent_radius             non-metal pair      radius-bonded                               yes
metal_ligand                metal-non-metal     CrystalNN or radius-bonded                  yes
metal_metal_crystalnn       metal-metal         CrystalNN (radius route not used)           yes
hydrogen_bond_candidate     non-metal pair      CrystalNN, not radius-bonded, H + acceptor  no
crystalnn_nonmetal          non-metal pair      CrystalNN, not radius-bonded (other)        no
metal_contact               metal-involving     any other                                   no
other_contact               non-metal pair      everything else                             no
==========================  ==================  ==========================================  =========

Radius route: ``d <= r_i + r_j + tolerance`` (Cordero covalent radii via pymatgen; tolerance 0.40 Å).
CrystalNN route: either site lists the other (that periodic image) as a CrystalNN neighbour.

The functions are copied unchanged from the paper code (``bonding.py``, ``crystalnn.py``, ``elements.py`` and
``periodic.annotate_instances`` at tag ``paper-v1``).
"""
from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Optional, Set

import networkx as nx
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.core import Element, Structure

from ._config import load_chemistry
from ._keys import Key, canonical

PROTECTED_CLASSES = ("covalent_radius", "metal_ligand", "metal_metal_crystalnn")
REMOVABLE_CLASSES = ("hydrogen_bond_candidate", "crystalnn_nonmetal", "metal_contact", "other_contact")
ALL_CLASSES = PROTECTED_CLASSES + REMOVABLE_CLASSES
ROUTES = ("none", "crystalnn", "radius", "both")
HALIDES = frozenset({"F", "Cl", "Br", "I"})
ROLES = ("metal", "oxygen", "nitrogen", "carbon", "hydrogen", "halide", "other")


# ------------------------------------------------------------------ elements
@lru_cache(maxsize=256)
def is_metal(symbol: str) -> bool:
    """pymatgen's ``Element.is_metal`` with the metalloids B, Si, Ge, As, Sb, Te forced to non-metal."""
    cfg = load_chemistry()["metals"]
    if symbol in cfg.get("forced_nonmetals", []):
        return False
    if symbol in cfg.get("forced_metals", []):
        return True
    return bool(Element(symbol).is_metal)


def role(symbol: str) -> str:
    """Element role used by node masses and node features (one of :data:`ROLES`)."""
    if is_metal(symbol):
        return "metal"
    return {"O": "oxygen", "N": "nitrogen", "C": "carbon", "H": "hydrogen"}.get(
        symbol, "halide" if symbol in HALIDES else "other")


# ------------------------------------------------------------------ routes and classes
def radius_bonded(el_i: str, el_j: str, d: float, tol: float) -> bool:
    """Radius route: ``d <= r_i + r_j + tol`` with Cordero covalent radii."""
    r = CovalentRadius.radius
    return d <= r[el_i] + r[el_j] + tol


def classify_edge(el_i: str, el_j: str, metal_i: bool, metal_j: bool, rb: bool, cn: bool,
                  acceptors: frozenset) -> str:
    """Bond class of one contact (first match wins; see the module docstring)."""
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


def crystalnn_pairs(structure: Structure, include_self: bool = False) -> dict:
    """CrystalNN neighbour pairs: ``{(i, j) with i<j: {"images": set of images of j relative to i}}``.

    A pair is included if CrystalNN lists j (any image) as a neighbour of i, or i as a neighbour of j. Sites for
    which CrystalNN raises are skipped and counted in ``pairs["__failed_sites__"]["count"]``.
    """
    cnn = CrystalNN()
    pairs: dict = {}
    failed = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(structure)):
            try:
                infos = cnn.get_nn_info(structure, i)
            except Exception:
                failed += 1
                continue
            for info in infos:
                j = int(info["site_index"])
                img = tuple(int(round(x)) for x in info["image"])
                if j == i and (not include_self or img == (0, 0, 0)):
                    continue
                key, im = ((i, j), img) if i < j else ((j, i), tuple(-x for x in img))
                pairs.setdefault(key, {"images": set()})["images"].add(im)
    pairs["__failed_sites__"] = {"count": failed}
    return pairs


def crystalnn_instances(cnn_pairs: dict) -> Set[Key]:
    """Map ``crystalnn_pairs(..., include_self=True)`` output to canonical instance keys."""
    out = set()
    for key, v in cnn_pairs.items():
        if key == "__failed_sites__":
            continue
        i, j = key
        for im in v["images"]:
            out.add(canonical(i, j, im))
    return out


def annotate_instances(g: nx.MultiGraph, cnn_inst: Optional[Set[Key]], tol: Optional[float] = None) -> nx.MultiGraph:
    """Annotate every instance in place with its routes, bond class and protection flag.

    ``cnn_inst=None`` means CrystalNN was not computed; CrystalNN-dependent classes then cannot occur.
    """
    cfg = load_chemistry()
    tol = cfg["bonding"]["radius_tolerance_angstrom"] if tol is None else tol
    acc = frozenset(cfg["protection"]["hydrogen_bond_acceptors"])
    for u, v, k, a in g.edges(keys=True, data=True):
        eu, ev = g.nodes[u]["element"], g.nodes[v]["element"]
        rb = radius_bonded(eu, ev, a["distance"], tol)
        cn = bool(cnn_inst is not None and a["instance"] in cnn_inst)
        cls = classify_edge(eu, ev, g.nodes[u]["is_metal"], g.nodes[v]["is_metal"], rb, cn, acc)
        a.update(radius_bonded=rb, crystalnn=cn, crystalnn_available=cnn_inst is not None, bond_class=cls,
                 protected=cls in PROTECTED_CLASSES,
                 protection_route="both" if (cn and rb) else "crystalnn" if cn else "radius" if rb else "none",
                 radius_tolerance=tol)
    return g


def default_tolerance() -> float:
    """Radius-route tolerance used in the paper (0.40 Å)."""
    return float(load_chemistry()["bonding"]["radius_tolerance_angstrom"])


__all__ = ["PROTECTED_CLASSES", "REMOVABLE_CLASSES", "ALL_CLASSES", "ROUTES", "ROLES", "is_metal", "role",
           "radius_bonded", "classify_edge", "crystalnn_pairs", "crystalnn_instances", "annotate_instances",
           "default_tolerance"]
