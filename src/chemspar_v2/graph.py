"""Minimum-image cutoff graph with periodic-image bookkeeping.

Edge set and edge order are designed to equal the legacy `build_graph_distance` (one undirected edge per
atom pair i<j whose minimum-image distance is <= cutoff, added in (i, j) lexicographic order); this is
verified by tests and by the legacy-vs-v2 regression in Phase 1. Unlike the legacy graph, every edge also
records the lattice image of j that realises the minimum distance and how many images of j lie within the
cutoff of i (multiplicity), so that periodic information is not silently discarded.
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from pymatgen.core import Structure

from .elements import is_metal, role


def build_cutoff_graph(structure: Structure, cutoff: float = 4.5) -> nx.Graph:
    g = nx.Graph()
    for i, site in enumerate(structure):
        el = site.specie.symbol
        g.add_node(i, element=el, Z=float(site.specie.Z), role=role(el), is_metal=is_metal(el),
                   frac=np.array(site.frac_coords), cart=np.array(site.coords))
    centers, points, images, dists = structure.get_neighbor_list(cutoff)
    best: dict[tuple[int, int], tuple[float, tuple[int, int, int]]] = {}
    mult: dict[tuple[int, int], int] = {}
    for c, p, img, d in zip(centers, points, images, dists):
        c, p = int(c), int(p)
        if c >= p:          # self-images (c == p) excluded as in the legacy graph; (p, c) duplicates skipped
            continue
        key = (c, p)
        mult[key] = mult.get(key, 0) + 1
        im = tuple(int(round(x)) for x in img)
        if key not in best or d < best[key][0] - 1e-12:
            best[key] = (float(d), im)
    for (i, j) in sorted(best):
        d, im = best[(i, j)]
        g.add_edge(i, j, distance=d, image=im, n_images_within_cutoff=mult[(i, j)])
    return g
