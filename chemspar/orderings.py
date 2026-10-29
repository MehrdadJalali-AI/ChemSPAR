"""Edge-ordering policies: the order in which the gate attempts to remove contact instances.

The gate accepts any ordering. Built-in orderings (all from the paper):

* ``gravity``  – ascending gravity score ``S = m_i m_j / d_hyb^2`` (weakest first).
* ``distance`` – longest contact first.
* ``random``   – a seeded permutation; the seed is ``sha256("<seed>:<structure_id>")``.
* ``bhs``      – ascending ``min(gamma_i, gamma_j)``, an edge-level adaptation of the Black Hole Strategy node score.

Ablation orderings from the paper's supplementary analyses are also registered (``bhs_prod``, ``massfree``,
``gravity_chem``, ``gravity_topo``). Score functions and sort rules are copied unchanged from the paper code
(``periodic.instance_gravity``, ``periodic.order_by``, ``periodic.order_random``, ``views._bhs_node_gravity``).

A user ordering can be passed to :func:`chemspar.prune` as

* a registered name (``"distance"``),
* an :class:`Ordering` or any callable ``f(graph) -> order``, where ``order`` is one of the forms below,
* a sequence of instance keys in removal order (instances not listed are never attempted),
* a rank array of length ``len(graph)``: ``rank[k]`` is the position of ``graph.instances[k]`` in the removal
  order (a permutation of ``0..n-1``).

Use :func:`from_scores` to turn per-instance scores into an ordering with the paper's stable tie-breaking
(ties keep canonical instance order).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Sequence, Union

import networkx as nx
import numpy as np

from ._config import bhs_weights, load_chemistry
from ._keys import Key

if TYPE_CHECKING:
    from .graph import ContactGraph

OrderFunc = Callable[..., Sequence[Key]]


# ------------------------------------------------------------------ scores (copied from the paper code)
def instance_gravity(g: nx.MultiGraph, mode: str = "full") -> Dict[Key, Dict[str, float]]:
    """Gravity components per instance: ``g``, ``d_hyb`` and the score ``S`` (``mode``: full, chemistry, topology)."""
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


def bhs_node_gravity(g: nx.MultiGraph) -> Dict[int, float]:
    """BHS node score: equal-weight sum of min-max scaled degree, betweenness and 1/d-weighted degree."""
    wd, wb, ww = bhs_weights()
    n1 = max(g.number_of_nodes() - 1, 1)
    simple = nx.Graph(g)
    raw = {"deg": {n: g.degree(n) / n1 for n in g}, "btw": nx.betweenness_centrality(simple, normalized=True),
           "w": {n: 0.0 for n in g}}
    for u, v, a in g.edges(data=True):
        raw["w"][u] += 1.0 / a["distance"]; raw["w"][v] += 1.0 / a["distance"]
    mm = {}
    for k, d in raw.items():
        lo, hi = min(d.values()), max(d.values())
        mm[k] = {n: 0.0 if hi == lo else (x - lo) / (hi - lo) for n, x in d.items()}
    return {n: wd * mm["deg"][n] + wb * mm["btw"][n] + ww * mm["w"][n] for n in g}


def order_by(values: Mapping[Key, float], instances: Sequence[Key], descending: bool = False) -> List[Key]:
    """Stable sort of the canonical instance list (ties keep canonical order)."""
    return sorted(instances, key=lambda k: (-values[k] if descending else values[k]))


def order_random(instances: Sequence[Key], seed: int, structure_id: str) -> List[Key]:
    """Seeded permutation; the generator seed is ``sha256(f"{seed}:{structure_id}")`` (independent of cohort order)."""
    h = int(hashlib.sha256(f"{seed}:{structure_id}".encode()).hexdigest()[:16], 16)
    idx = np.random.default_rng(h).permutation(len(instances))
    return [instances[t] for t in idx]


# ------------------------------------------------------------------ ordering objects and registry
@dataclass(frozen=True)
class Ordering:
    """A named ordering policy. ``func(graph, seed)`` returns instance keys in removal order."""

    name: str
    func: Callable[[ContactGraph, int], Sequence[Key]]
    description: str = ""

    def __call__(self, graph: ContactGraph, seed: int = 42) -> List[Key]:
        return list(self.func(graph, seed))


def _gravity(mode: str) -> Callable[[ContactGraph, int], List[Key]]:
    def f(cg: ContactGraph, seed: int) -> List[Key]:
        comp = instance_gravity(cg.graph, mode=mode)
        return order_by({k: comp[k]["S"] for k in cg.instances}, cg.instances)
    return f


def _massfree(cg: ContactGraph, seed: int) -> List[Key]:
    comp = instance_gravity(cg.graph)
    return order_by({k: 1.0 / comp[k]["d_hyb"] ** 2 for k in cg.instances}, cg.instances)


def _distance(cg: ContactGraph, seed: int) -> List[Key]:
    return order_by({k: cg.distance(k) for k in cg.instances}, cg.instances, descending=True)


def _random(cg: ContactGraph, seed: int) -> List[Key]:
    return order_random(cg.instances, seed, cg.structure_id)


def _bhs(kind: str) -> Callable[[ContactGraph, int], List[Key]]:
    def f(cg: ContactGraph, seed: int) -> List[Key]:
        gam = bhs_node_gravity(cg.graph)
        if kind == "min":
            val = {k: min(gam[k[0]], gam[k[1]]) for k in cg.instances}
        else:
            val = {k: gam[k[0]] * gam[k[1]] for k in cg.instances}
        return order_by(val, cg.instances)
    return f


gravity = Ordering("gravity", _gravity("full"), "ascending gravity score (weakest contact first)")
distance = Ordering("distance", _distance, "longest contact first")
random = Ordering("random", _random, "seeded random permutation (seed and structure_id)")
bhs = Ordering("bhs", _bhs("min"), "ascending min(gamma_i, gamma_j), BHS-derived")
bhs_prod = Ordering("bhs_prod", _bhs("prod"), "ascending gamma_i * gamma_j (ablation)")
massfree = Ordering("massfree", _massfree, "ascending 1/d_hyb^2 (ablation)")
gravity_chem = Ordering("gravity_chem", _gravity("chemistry"), "gravity with chemistry-only node mass (ablation)")
gravity_topo = Ordering("gravity_topo", _gravity("topology"), "gravity with topology-only node mass (ablation)")

_REGISTRY: Dict[str, Ordering] = {o.name: o for o in
                                  (gravity, distance, random, bhs, bhs_prod, massfree, gravity_chem, gravity_topo)}


def register(name: str, func: Callable[[ContactGraph, int], Sequence[Key]], description: str = "",
             overwrite: bool = False) -> Ordering:
    """Register a custom ordering under ``name``; ``func(graph, seed)`` returns keys in removal order."""
    if name in _REGISTRY and not overwrite:
        raise ValueError(f"ordering {name!r} is already registered")
    o = Ordering(name, func, description)
    _REGISTRY[name] = o
    return o


def get(name: str) -> Ordering:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown ordering {name!r}; available: {', '.join(sorted(_REGISTRY))}") from None


def available() -> List[str]:
    return sorted(_REGISTRY)


def from_scores(scores: Union[Mapping[Key, float], Sequence[float]], descending: bool = False) -> Callable:
    """Ordering from per-instance scores (ascending = removed first), ties in canonical order.

    ``scores`` is a mapping ``key -> score`` or a sequence aligned with ``graph.instances``.
    """
    def f(cg: ContactGraph) -> List[Key]:
        inst = cg.instances
        if isinstance(scores, Mapping):
            val = dict(scores)
        else:
            arr = list(scores)
            if len(arr) != len(inst):
                raise ValueError(f"expected {len(inst)} scores, got {len(arr)}")
            val = dict(zip(inst, arr))
        return order_by(val, inst, descending=descending)
    return f


def resolve(ordering: Union[str, Ordering, Callable, Sequence], graph: ContactGraph, seed: int = 42) -> List[Key]:
    """Turn any accepted ordering form into a validated list of instance keys in removal order."""
    order: Any
    if isinstance(ordering, str):
        order = get(ordering)(graph, seed)
    elif isinstance(ordering, Ordering):
        order = ordering(graph, seed)
    elif callable(ordering):
        order = ordering(graph)
    else:
        order = ordering
    return _validate(order, graph)


def _validate(order, graph: ContactGraph) -> List[Key]:
    inst = graph.instances
    arr = list(order.tolist() if isinstance(order, np.ndarray) else order)
    # rank array: integers, one per instance
    if arr and all(isinstance(x, (int, np.integer)) and not isinstance(x, bool) for x in arr):
        if len(arr) != len(inst):
            raise ValueError(f"a rank array needs one rank per instance ({len(inst)}), got {len(arr)}")
        ranks = [int(x) for x in arr]
        if sorted(ranks) != list(range(len(inst))):
            raise ValueError("a rank array must be a permutation of 0..n-1")
        out: List[Any] = [None] * len(inst)
        for k, r in zip(inst, ranks):
            out[r] = k
        return out
    valid = set(inst)
    keys: List[Key] = []
    seen = set()
    for x in arr:
        i, j, img = x
        k = (int(i), int(j), tuple(int(t) for t in img))
        if k not in valid:
            raise ValueError(f"{k} is not a canonical contact instance of this graph")
        if k in seen:
            raise ValueError(f"instance {k} appears more than once in the ordering")
        seen.add(k)
        keys.append(k)
    return keys


__all__ = ["Ordering", "gravity", "distance", "random", "bhs", "bhs_prod", "massfree", "gravity_chem",
           "gravity_topo", "register", "get", "available", "from_scores", "resolve", "instance_gravity",
           "bhs_node_gravity", "order_by", "order_random"]
