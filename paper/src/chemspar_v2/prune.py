"""Constraint gate, static pruning loop and audit record for chemspar_v2.

Gate modes
  'corrected'          : protected taxonomy class (+ extra_protected_classes) -> non-H degree -> connectivity
  'connectivity_only'  : non-H degree -> connectivity (the 'without gate' / connectivity-only baselines)
Every attempted removal produces one audit row, built BEFORE any deletion, containing endpoints, elements,
distance, periodic image, protection route, the outcome of each gate check ('pass', 'fail' or 'not_evaluated'),
all score components, the final score, the decision and the reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import networkx as nx
import numpy as np

GATE_MODES = ("corrected", "connectivity_only")


@dataclass
class PruneResult:
    graph: nx.Graph
    removed: list[tuple[int, int]]
    target: int
    audit: list[dict] = field(default_factory=list)
    rejections: dict[str, int] = field(default_factory=dict)


def is_protected(a: dict, extra_protected_classes: tuple = ()) -> bool:
    return bool(a["protected"] or a["bond_class"] in extra_protected_classes)


def _checks(g: nx.Graph, u: int, v: int, mode: str, extra: tuple = ()) -> tuple[dict[str, str], str | None]:
    out = {"protected": "not_evaluated", "degree": "not_evaluated", "connectivity": "not_evaluated"}
    a = g.edges[u, v]
    if mode == "corrected":
        if is_protected(a, extra):
            out["protected"] = "fail"
            return out, f"protected:{a['bond_class']}:{a['protection_route']}"
        out["protected"] = "pass"
    for n in (u, v):
        if g.nodes[n]["element"] != "H" and g.degree(n) <= 1:
            out["degree"] = "fail"
            return out, "degree"
    out["degree"] = "pass"
    # Removing (u, v) increases the number of connected components iff u and v are no longer joined.
    # For a connected graph this is identical to "graph still connected"; the early-exit path search is
    # much cheaper than a whole-graph traversal (equivalence covered by unit tests).
    attrs = dict(a)
    g.remove_edge(u, v)
    connected = nx.has_path(g, u, v)
    g.add_edge(u, v, **attrs)
    out["connectivity"] = "pass" if connected else "fail"
    return out, None if connected else "connectivity"


def prune(g0: nx.Graph, tau: float, order: Iterable[tuple[int, int]], mode: str, *,
          components: dict | None = None, qmof_id: str = "", policy: str = "", audit: bool = True,
          extra_protected_classes: tuple = ()) -> PruneResult:
    """Visit edges in `order` (lowest retention priority first) and remove up to floor(tau*|E|) of them.
    `extra_protected_classes` adds taxonomy classes to the protected set (e.g. ChemSPAR-HBProtect)."""
    if mode not in GATE_MODES:
        raise ValueError(mode)
    g = g0.copy()
    target = int(tau * g.number_of_edges())
    res = PruneResult(graph=g, removed=[], target=target, rejections={})
    for rank, (u, v) in enumerate(order):
        if len(res.removed) >= target:
            break
        if not g.has_edge(u, v):
            continue
        outcome, reason = _checks(g, u, v, mode, tuple(extra_protected_classes))
        if audit:
            a = g.edges[u, v]
            comp = (components or {}).get((min(u, v), max(u, v)), {})
            res.audit.append({
                "qmof_id": qmof_id, "policy": policy, "gate_mode": mode, "tau": tau, "target": target,
                "rank": rank, "atom_i": min(u, v), "atom_j": max(u, v),
                "element_i": g.nodes[min(u, v)]["element"], "element_j": g.nodes[max(u, v)]["element"],
                "distance": a["distance"], "image_j": "%d,%d,%d" % a["image"],
                "n_images_within_cutoff": a["n_images_within_cutoff"],
                "bond_class": a["bond_class"], "radius_bonded": a["radius_bonded"], "crystalnn": a["crystalnn"],
                "crystalnn_available": a["crystalnn_available"], "protection_route": a["protection_route"],
                "protected_under_policy": is_protected(a, tuple(extra_protected_classes)),
                "radius_tolerance": a.get("radius_tolerance"),
                "check_protected": outcome["protected"], "check_degree": outcome["degree"],
                "check_connectivity": outcome["connectivity"],
                "g": comp.get("g"), "B": comp.get("B"), "C": comp.get("C", a.get("C")), "S": comp.get("S"),
                "decision": "rejected" if reason else "accepted", "reason": reason or "accepted",
                "removed_before_attempt": len(res.removed),
            })
        if reason:
            res.rejections[reason] = res.rejections.get(reason, 0) + 1
            continue
        g.remove_edge(u, v)
        res.removed.append((min(u, v), max(u, v)))
    return res


# ---- orderings ------------------------------------------------------------------------------------
def order_by_score(components: dict, key: str = "S") -> list[tuple[int, int]]:
    """Ascending score; ties keep the dict insertion order (graph edge order), as in the published code."""
    return sorted(components, key=lambda e: components[e][key])


def order_longest_first(g: nx.Graph) -> list[tuple[int, int]]:
    edges = [(min(u, v), max(u, v)) for u, v in g.edges()]
    return sorted(edges, key=lambda e: -g.edges[e]["distance"])


def order_random(g: nx.Graph, seed: int, qmof_id: str) -> list[tuple[int, int]]:
    """Per-structure generator: default_rng(sha256('<seed>:<qmof_id>')) - independent of cohort order."""
    import hashlib
    h = int(hashlib.sha256(f"{seed}:{qmof_id}".encode()).hexdigest()[:16], 16)
    edges = [(min(u, v), max(u, v)) for u, v in g.edges()]
    np.random.default_rng(h).shuffle(edges)
    return edges
