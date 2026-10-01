"""Alternative graph constructions used as baselines (not pruning policies).

cutoff_subgraph   : minimum-image cutoff graph at r < 4.5 A (edge subset of the 4.5 A graph)
crystalnn_graph   : edges = CrystalNN pairs (either site lists the other; minimum-image pair key)
knn_graph         : periodic k-nearest-neighbour graph. For each atom, its k nearest periodic neighbours
                    (self-images excluded) are taken; the union over atoms is symmetrised and collapsed to one
                    edge per atom pair (minimum distance kept). Edges can exceed 4.5 A.
All constructions keep every atom; they are NOT guaranteed connected (connectivity is reported, not enforced).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from pymatgen.core import Structure


def cutoff_subgraph(g45: nx.Graph, r: float) -> nx.Graph:
    h = nx.Graph(); h.add_nodes_from(g45.nodes(data=True))
    h.add_edges_from((u, v, a) for u, v, a in g45.edges(data=True) if a["distance"] <= r)
    return h


def crystalnn_graph(g45: nx.Graph) -> nx.Graph:
    h = nx.Graph(); h.add_nodes_from(g45.nodes(data=True))
    h.add_edges_from((u, v, a) for u, v, a in g45.edges(data=True) if a["crystalnn"])
    return h


def knn_edges(structure: Structure, k: int = 12, r_max: float = 8.0) -> dict[tuple[int, int], tuple[float, tuple]]:
    """Return {(i, j): (distance, image_of_j)} for the symmetrised periodic k-NN graph."""
    centers, points, images, dists = structure.get_neighbor_list(r_max)
    out: dict[tuple[int, int], tuple[float, tuple]] = {}
    short = 0
    for i in range(len(structure)):
        m = (centers == i) & (points != i)
        idx = np.where(m)[0]
        if len(idx) < k:
            short += 1
        sel = idx[np.argsort(dists[idx], kind="stable")[:k]]
        for t in sel:
            j, d, im = int(points[t]), float(dists[t]), tuple(int(round(x)) for x in images[t])
            key, im_key = ((i, j), im) if i < j else ((j, i), tuple(-x for x in im))
            if key not in out or d < out[key][0]:
                out[key] = (d, im_key)
    out["__atoms_with_fewer_than_k_within_rmax__"] = (float(short), ())
    return out
