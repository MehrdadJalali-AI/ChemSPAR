"""Periodic edge-instance representation (pruning unit = periodic edge instance).

An edge instance is one periodic contact between atom i (home cell) and atom j in lattice image `img`, with
d = |x_j + img - x_i| <= cutoff. Canonical key (one per undirected instance):
    (i, j, img)  with i < j;   (j, i, -img) is the same instance
    (i, i, img)  self-image contact, with img lexicographically > (0, 0, 0)
The instance graph is a networkx.MultiGraph on the atoms of the cell whose edges are keyed by `img`.
Every rule (radius bonding, CrystalNN, taxonomy, gravity, gate, audit) is evaluated per instance, and a graph
view is exactly a set of retained instances. Both backbones consume exactly that set (see train.instances_to_data).
"""
from __future__ import annotations

import hashlib

import networkx as nx
import numpy as np
from pymatgen.core import Structure

from .bonding import PROTECTED_CLASSES, classify_edge, radius_bonded
from .config import load_chemistry
from .elements import is_metal, role

Img = tuple[int, int, int]
Key = tuple[int, int, Img]


def canonical(i: int, j: int, img: Img) -> Key:
    img = tuple(int(x) for x in img)
    neg = tuple(-x for x in img)
    if i < j:
        return (i, j, img)
    if i > j:
        return (j, i, neg)
    return (i, i, img if img > (0, 0, 0) else neg)


def build_instance_graph(structure: Structure, cutoff: float = 4.5) -> nx.MultiGraph:
    g = nx.MultiGraph()
    for n, site in enumerate(structure):
        el = site.specie.symbol
        g.add_node(n, element=el, Z=float(site.specie.Z), role=role(el), is_metal=is_metal(el))
    c, p, imgs, d = structure.get_neighbor_list(cutoff)
    inst: dict[Key, float] = {}
    for a, b, im, dd in zip(c, p, imgs, d):
        k = canonical(int(a), int(b), tuple(int(round(x)) for x in im))
        if k[0] == k[1] and k[2] == (0, 0, 0):
            continue
        inst.setdefault(k, float(dd))
    for k in sorted(inst):
        g.add_edge(k[0], k[1], key=k[2], distance=inst[k], instance=k)
    g.graph["instances"] = sorted(inst)
    return g


def crystalnn_instances(cnn_pairs: dict) -> set[Key]:
    """Map crystalnn.crystalnn_pairs(..., include_self=True) output to canonical instance keys."""
    out = set()
    for key, v in cnn_pairs.items():
        if key == "__failed_sites__":
            continue
        i, j = key
        for im in v["images"]:
            out.add(canonical(i, j, im))
    return out


def annotate_instances(g: nx.MultiGraph, cnn_inst: set[Key] | None, tol: float | None = None) -> nx.MultiGraph:
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


# ---------------------------------------------------------------- gravity on the instance multigraph
def instance_gravity(g: nx.MultiGraph, mode: str = "full") -> dict[Key, dict[str, float]]:
    cfg = load_chemistry()["score"]
    w, rw, lam = cfg["node_mass_weights"], cfg["role_weights"], cfg["hybrid_lambda"]
    simple = nx.Graph(g)                                  # shortest-path centralities ignore parallel instances
    deg = {n: g.degree(n) for n in g}                     # multigraph degree (self-image counts 2)
    n_nodes = max(g.number_of_nodes() - 1, 1)
    btw = nx.betweenness_centrality(simple, normalized=True)
    cls = nx.closeness_centrality(simple)
    nsum = {n: 0.0 for n in g}
    for u, v, a in g.edges(data=True):
        nsum[u] += a["distance"]; nsum[v] += a["distance"]
    max_n = max(nsum.values()) or 1.0
    max_z = max(d["Z"] for _, d in g.nodes(data=True))
    m = {}
    for n, d in g.nodes(data=True):
        topo = w["degree"] * deg[n] / n_nodes + w["betweenness"] * btw[n] + w["closeness"] * cls[n] + w["neighbour_distance"] * nsum[n] / max_n
        chem = w["z"] * d["Z"] / max_z + w["role"] * rw.get(d["role"], rw["other"])
        m[n] = {"full": topo + chem, "chemistry": chem, "topology": topo}[mode]
    out = {}
    for u, v, k, a in g.edges(keys=True, data=True):
        du, dv = deg[u], deg[v]
        dh = max(lam * (1 + abs(du - dv) / max(du, dv, 1)) + (1 - lam) * a["distance"], 1e-6)
        out[a["instance"]] = {"g": m[u] * m[v] / dh ** 2, "d_hyb": dh, "S": m[u] * m[v] / dh ** 2}
    return out


# ---------------------------------------------------------------- gate + pruning on instances
def _check(g: nx.MultiGraph, key: Key, mode: str, extra: tuple) -> tuple[dict, str | None]:
    u, v, img = key
    a = g.edges[u, v, img]
    out = {"protected": "not_evaluated", "degree": "not_evaluated", "connectivity": "not_evaluated"}
    if mode == "corrected":
        if a["protected"] or a["bond_class"] in extra:
            out["protected"] = "fail"
            return out, f"protected:{a['bond_class']}:{a['protection_route']}"
        out["protected"] = "pass"
    for n in {u, v}:
        loss = 2 if u == v else 1
        if g.nodes[n]["element"] != "H" and g.degree(n) - loss <= 0:
            out["degree"] = "fail"
            return out, "degree"
    out["degree"] = "pass"
    if u == v:
        out["connectivity"] = "pass"                      # a self-image contact never disconnects the cell graph
        return out, None
    attrs = dict(a)
    g.remove_edge(u, v, key=img)
    ok = nx.has_path(g, u, v)
    g.add_edge(u, v, key=img, **attrs)
    out["connectivity"] = "pass" if ok else "fail"
    return out, None if ok else "connectivity"


def prune_instances(g0: nx.MultiGraph, tau: float, order: list[Key], mode: str, *, comp: dict | None = None,
                    qmof_id: str = "", policy: str = "", audit: bool = True, extra: tuple = ()) -> dict:
    g = g0.copy()
    target = int(tau * g.number_of_edges())
    removed, rows, rej = [], [], {}
    for rank, key in enumerate(order):
        if len(removed) >= target:
            break
        u, v, img = key
        if not g.has_edge(u, v, key=img):
            continue
        outcome, reason = _check(g, key, mode, tuple(extra))
        if audit:
            a = g.edges[u, v, img]
            c = (comp or {}).get(key, {})
            rows.append(dict(qmof_id=qmof_id, policy=policy, gate_mode=mode, tau=tau, target=target, rank=rank,
                             atom_i=u, atom_j=v, image_j="%d,%d,%d" % img, element_i=g.nodes[u]["element"],
                             element_j=g.nodes[v]["element"], distance=a["distance"], bond_class=a["bond_class"],
                             radius_bonded=a["radius_bonded"], radius_tolerance=a["radius_tolerance"], crystalnn=a["crystalnn"],
                             protection_route=a["protection_route"], protected_under_policy=bool(a["protected"] or a["bond_class"] in extra),
                             check_protected=outcome["protected"], check_degree=outcome["degree"],
                             check_connectivity=outcome["connectivity"], g=c.get("g"), S=c.get("S"),
                             decision="rejected" if reason else "accepted", reason=reason or "accepted",
                             removed_before_attempt=len(removed)))
        if reason:
            rej[reason] = rej.get(reason, 0) + 1
            continue
        g.remove_edge(u, v, key=img)
        removed.append(key)
    return {"graph": g, "removed": removed, "target": target, "audit": rows, "rejections": rej}


def order_by(values: dict[Key, float], instances: list[Key], descending: bool = False) -> list[Key]:
    """Stable sort of the canonical instance list (ties keep canonical order)."""
    return sorted(instances, key=lambda k: (-values[k] if descending else values[k]))


def order_random(instances: list[Key], seed: int, qmof_id: str) -> list[Key]:
    h = int(hashlib.sha256(f"{seed}:{qmof_id}".encode()).hexdigest()[:16], 16)
    idx = np.random.default_rng(h).permutation(len(instances))
    return [instances[t] for t in idx]


def knn_instances(structure: Structure, k: int = 12, r_max: float = 8.0) -> list[Key]:
    """Each atom's k nearest periodic neighbours (own images included, zero-distance self excluded), canonicalised, union."""
    c, p, imgs, d = structure.get_neighbor_list(r_max)
    keep = set()
    for a in range(len(structure)):
        m = np.where(c == a)[0]
        m = m[np.argsort(d[m], kind="stable")[:k]]
        for t in m:
            keep.add(canonical(a, int(p[t]), tuple(int(round(x)) for x in imgs[t])))
    return sorted(keep)


def instance_distance(structure: Structure, key: Key) -> float:
    i, j, img = key
    return float(np.linalg.norm(structure.lattice.get_cartesian_coords(structure[j].frac_coords + np.array(img) - structure[i].frac_coords)))
