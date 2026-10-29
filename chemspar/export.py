"""Export of contact graphs and graph views to PyTorch Geometric, networkx and JSON.

PyG conversion follows the paper's training input (``train.instances_to_data`` at ``paper-v1``): each retained
instance becomes two directed edges, ``edge_dist`` holds the distances and ``edge_image`` the lattice image of the
target atom. Node features are the atomic number and a one-hot element role. No neighbour list is rebuilt.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from ._keys import Key, canonical
from .taxonomy import ROLES

if TYPE_CHECKING:
    from .gate import GraphView
    from .graph import ContactGraph

JSON_FORMAT = "chemspar-view"
JSON_FORMAT_VERSION = 1


def _torch():
    try:
        import torch
        from torch_geometric.data import Data
    except ImportError as e:  # pragma: no cover - depends on the optional extra
        raise ImportError("PyG export needs the optional extra: pip install 'chemspar[pyg]'") from e
    return torch, Data


def node_features(elements_Z: Sequence[int], roles: Sequence[str]):
    """Atomic numbers (long) and one-hot roles (float) in the order of :data:`chemspar.taxonomy.ROLES`."""
    torch, _ = _torch()
    z = torch.tensor(list(elements_Z), dtype=torch.long)
    r = torch.zeros(len(roles), len(ROLES))
    for k, ro in enumerate(roles):
        r[k, ROLES.index(ro)] = 1.0
    return z, r


def instances_to_data(z, roles, instances: Sequence[Key], distances: Sequence[float], y: Optional[float] = None,
                      structure_id: str = ""):
    """PyG ``Data`` with paired directed copies of each instance (paper representation)."""
    torch, Data = _torch()
    src: List[int] = []
    dst: List[int] = []
    dist: List[float] = []
    img: List[List[int]] = []
    for (i, j, im), d in zip(instances, distances):
        src += [i, j]; dst += [j, i]; dist += [d, d]
        img += [list(im), [-x for x in im]]
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    kw: Dict[str, Any] = dict(z=z, roles=roles, edge_index=ei, edge_dist=torch.tensor(dist, dtype=torch.float32),
                              edge_image=torch.tensor(img, dtype=torch.int16).view(-1, 3), structure_id=structure_id,
                              num_nodes=len(z))
    if y is not None:
        kw["y"] = torch.tensor([y], dtype=torch.float32)
    return Data(**kw)


def data_instances(d) -> List[Key]:
    """Recover the sorted canonical instances encoded in a PyG ``Data`` object (inverse of the export)."""
    keys = sorted(canonical(int(a), int(b), tuple(int(x) for x in im))
                  for a, b, im in zip(d.edge_index[0].tolist(), d.edge_index[1].tolist(), d.edge_image.tolist()))
    if len(keys) % 2 or any(keys[2 * t] != keys[2 * t + 1] for t in range(len(keys) // 2)):
        raise ValueError("tensor edges are not paired directed copies of canonical instances")
    return keys[::2]


def to_pyg(graph: ContactGraph, instances: Sequence[Key], y: Optional[float] = None):
    nodes = graph.graph.nodes
    z, r = node_features([int(nodes[n]["Z"]) for n in nodes], [nodes[n]["role"] for n in nodes])
    return instances_to_data(z, r, list(instances), [graph.distance(k) for k in instances], y=y,
                             structure_id=graph.structure_id)


def view_to_dict(view: GraphView) -> dict:
    base = view.base
    return {
        "format": JSON_FORMAT,
        "format_version": JSON_FORMAT_VERSION,
        "structure_id": base.structure_id,
        "cutoff": base.cutoff,
        "radius_tolerance": base.tolerance,
        "policy": view.policy,
        "ordering": view.ordering,
        "gate": view.gate,
        "extra_protected": list(view.extra_protected),
        "seed": view.seed,
        "tau": view.tau,
        "target": view.target,
        "n_instances": len(base),
        "atoms": [{"index": int(n), "element": d["element"], "role": d["role"]}
                  for n, d in base.graph.nodes(data=True)],
        "retained": [_inst(base, k) for k in view.instances],
        "removed": [[k[0], k[1], list(k[2])] for k in view.removed],
    }


def _inst(base: ContactGraph, k: Key) -> dict:
    a = base.attrs(k)
    return {"i": k[0], "j": k[1], "image": list(k[2]), "distance": a["distance"], "bond_class": a["bond_class"],
            "protected": bool(a["protected"]), "route": a["protection_route"]}


def view_to_json(view: GraphView, path=None, indent: Optional[int] = None) -> str:
    s = json.dumps(view_to_dict(view), indent=indent)
    if path is not None:
        Path(path).write_text(s, encoding="utf-8")
    return s


__all__ = ["node_features", "instances_to_data", "data_instances", "to_pyg", "view_to_dict", "view_to_json"]
