"""Periodic contact graph: one edge per contact instance ``(i, j, image)`` within the cutoff.

An instance is one periodic contact between atom ``i`` in the home cell and atom ``j`` in lattice image ``img``,
with ``|x_j + img - x_i| <= cutoff``. Every rule (taxonomy, orderings, gate, audit) works per instance, and a graph
view is exactly a set of retained instances. :func:`build_instance_graph` is copied unchanged from the paper code.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Dict, List, Optional, Union

import networkx as nx
from pymatgen.core import Structure

from ._keys import Img, Key, canonical
from .taxonomy import annotate_instances, crystalnn_instances, crystalnn_pairs, is_metal, role

DEFAULT_CUTOFF = 4.5


def build_instance_graph(structure: Structure, cutoff: float = DEFAULT_CUTOFF) -> nx.MultiGraph:
    """Instance multigraph on the atoms of the cell, edges keyed by image, in canonical instance order."""
    g = nx.MultiGraph()
    for n, site in enumerate(structure):
        el = site.specie.symbol
        g.add_node(n, element=el, Z=float(site.specie.Z), role=role(el), is_metal=is_metal(el))
    c, p, imgs, d = structure.get_neighbor_list(cutoff)
    inst: Dict[Key, float] = {}
    for a, b, im, dd in zip(c, p, imgs, d):
        k = canonical(int(a), int(b), tuple(int(round(x)) for x in im))
        if k[0] == k[1] and k[2] == (0, 0, 0):
            continue
        inst.setdefault(k, float(dd))
    for k in sorted(inst):
        g.add_edge(k[0], k[1], key=k[2], distance=inst[k], instance=k)
    g.graph["instances"] = sorted(inst)
    return g


@dataclass
class ContactGraph:
    """An annotated periodic contact graph of one structure (input to :func:`chemspar.prune`)."""

    graph: nx.MultiGraph
    structure: Structure
    structure_id: str = ""
    cutoff: float = DEFAULT_CUTOFF
    tolerance: Optional[float] = None
    crystalnn_failed_sites: int = 0
    _attrs: Dict[Key, dict] = field(default_factory=dict, repr=False)
    _cache: Dict[str, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._attrs = {a["instance"]: a for _, _, a in self.graph.edges(data=True)}

    # -------------------------------------------------------------- accessors
    @property
    def instances(self) -> List[Key]:
        """All contact instances in canonical (sorted) order."""
        return list(self.graph.graph["instances"])

    def __len__(self) -> int:
        return len(self.graph.graph["instances"])

    def attrs(self, key: Key) -> dict:
        """Annotation of one instance (distance, routes, bond class, protection flag)."""
        return self._attrs[key]

    @property
    def protected(self) -> List[Key]:
        """Protected instances (the first three taxonomy classes)."""
        return [k for k in self.instances if self._attrs[k]["protected"]]

    def bond_class_counts(self) -> Dict[str, int]:
        return dict(Counter(self._attrs[k]["bond_class"] for k in self.instances))

    def distance(self, key: Key) -> float:
        return float(self._attrs[key]["distance"])

    @property
    def elements(self) -> List[str]:
        return [self.graph.nodes[n]["element"] for n in self.graph.nodes]

    # -------------------------------------------------------------- export
    def to_networkx(self) -> nx.MultiGraph:
        """A copy of the annotated instance multigraph."""
        return self.graph.copy()

    def to_pyg(self, y: Optional[float] = None):
        """Full (unpruned) graph as a PyTorch Geometric ``Data`` object (requires the ``[pyg]`` extra)."""
        from .export import to_pyg
        return to_pyg(self, self.instances, y=y)


def build_graph(structure: Union[Structure, str, PathLike[str]], cutoff: float = DEFAULT_CUTOFF, *,
                tolerance: Optional[float] = None, crystalnn: bool = True,
                structure_id: Optional[str] = None) -> ContactGraph:
    """Build and annotate the periodic contact graph of a structure.

    Parameters
    ----------
    structure
        A pymatgen ``Structure`` or a path to any file pymatgen can read (e.g. a CIF).
    cutoff
        Contact cutoff in Å (4.5 in the paper).
    tolerance
        Radius-route tolerance in Å; ``None`` uses the paper value (0.40).
    crystalnn
        Run the CrystalNN route. With ``False`` the CrystalNN-dependent classes cannot occur (not the paper setting).
    structure_id
        Identifier written to the audit log and used to seed the random ordering. Defaults to the file stem when a
        path is given (QMOF CIFs are named ``<qmof_id>.cif``, which reproduces the paper's random orders).
    """
    if isinstance(structure, Structure):
        s, sid = structure, structure_id or ""
    else:
        path = Path(structure)
        s, sid = Structure.from_file(str(path)), structure_id if structure_id is not None else path.stem
    g = build_instance_graph(s, cutoff)
    failed = 0
    cnn = None
    if crystalnn:
        pairs = crystalnn_pairs(s, include_self=True)
        failed = pairs["__failed_sites__"]["count"]
        cnn = crystalnn_instances(pairs)
    annotate_instances(g, cnn, tol=tolerance)
    return ContactGraph(graph=g, structure=s, structure_id=sid, cutoff=cutoff, tolerance=tolerance,
                        crystalnn_failed_sites=failed)


__all__ = ["ContactGraph", "build_graph", "build_instance_graph", "canonical", "Key", "Img", "DEFAULT_CUTOFF"]
