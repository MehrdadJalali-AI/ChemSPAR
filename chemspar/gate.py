"""The protection gate and the static pruning loop.

Instances are visited once, in ordering rank, until ``floor(tau * |E|)`` instances have been removed. Each
attempt passes through three checks, in this order:

1. **protected** – reject if the instance's bond class is protected (gated runs only);
2. **degree** – reject if a non-hydrogen endpoint would lose its last contact;
3. **connectivity** – reject if removing the instance would disconnect its two endpoints in the cell graph
   (lattice image vectors are not taken into account; a self-image contact never disconnects the cell graph).

The ungated partner of a policy (``gate=False``, internal mode ``connectivity_only``) skips check 1 only.
``_check`` and ``prune_instances`` are copied unchanged from the paper code (``periodic.py`` at ``paper-v1``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import networkx as nx

from ._keys import Key
from .audit import AuditLog
from .graph import ContactGraph
from .orderings import Ordering, instance_gravity, resolve

GATE_MODES = ("corrected", "connectivity_only")


def _check(g: nx.MultiGraph, key: Key, mode: str, extra: tuple) -> Tuple[dict, Optional[str]]:
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


def prune_instances(g0: nx.MultiGraph, tau: float, order: List[Key], mode: str, *, comp: Optional[dict] = None,
                    qmof_id: str = "", policy: str = "", audit: bool = True, extra: tuple = ()) -> dict:
    """Low-level pruning loop (paper implementation). Returns graph, removed keys, target, audit rows, rejections."""
    g = g0.copy()
    target = int(tau * g.number_of_edges())
    removed: List[Key] = []; rows: List[dict] = []; rej: Dict[str, int] = {}
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


# ------------------------------------------------------------------ public API
@dataclass
class GraphView:
    """The retained contact instances of one pruning run."""

    base: ContactGraph
    graph: nx.MultiGraph
    removed: List[Key]
    tau: float
    target: int
    ordering: str
    gate: bool
    extra_protected: Tuple[str, ...] = ()
    policy: str = ""
    seed: int = 42
    _retained: List[Key] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        gone = set(self.removed)
        self._retained = [k for k in self.base.instances if k not in gone]

    @property
    def instances(self) -> List[Key]:
        """Retained instances in canonical order."""
        return list(self._retained)

    def __len__(self) -> int:
        return len(self._retained)

    @property
    def removed_fraction(self) -> float:
        n = len(self.base)
        return len(self.removed) / n if n else 0.0

    def protected_retention(self) -> float:
        """Fraction of the base graph's protected instances that are retained (1.0 if none are protected)."""
        prot = self.base.protected
        if not prot:
            return 1.0
        kept = set(self._retained)
        return sum(k in kept for k in prot) / len(prot)

    def is_connected(self) -> bool:
        """Connectivity of the cell graph (image vectors ignored, as in the gate)."""
        return self.graph.number_of_nodes() <= 1 or nx.is_connected(self.graph)

    def to_networkx(self) -> nx.MultiGraph:
        return self.graph.copy()

    def to_pyg(self, y: Optional[float] = None):
        """PyTorch Geometric ``Data`` built directly from the retained instances (requires ``[pyg]``)."""
        from .export import to_pyg
        return to_pyg(self.base, self._retained, y=y)

    def to_dict(self) -> dict:
        from .export import view_to_dict
        return view_to_dict(self)

    def to_json(self, path=None, indent: Optional[int] = None) -> str:
        """JSON string of the view; also written to ``path`` when given."""
        from .export import view_to_json
        return view_to_json(self, path=path, indent=indent)


OrderingLike = Union[str, Ordering, Callable, Sequence]


def _gravity_components(graph: ContactGraph) -> dict:
    if "gravity" not in graph._cache:
        graph._cache["gravity"] = instance_gravity(graph.graph)
    return graph._cache["gravity"]  # type: ignore[return-value]


def prune(graph: ContactGraph, ordering: OrderingLike = "gravity", tau: float = 0.5, gate: bool = True, *,
          seed: int = 42, extra_protected: Sequence[str] = (), policy: Optional[str] = None,
          audit: bool = True) -> Tuple[GraphView, AuditLog]:
    """Remove up to ``floor(tau * |E|)`` contact instances in ``ordering`` rank, through the gate.

    Parameters
    ----------
    graph
        Output of :func:`chemspar.build_graph`.
    ordering
        A registered name (``"gravity"``, ``"distance"``, ``"random"``, ``"bhs"``, ...), an
        :class:`~chemspar.orderings.Ordering`, a callable ``f(graph)``, a sequence of instance keys in removal
        order, or a rank array (see :mod:`chemspar.orderings`).
    tau
        Removal budget as a fraction of all instances, ``0 <= tau <= 1``.
    gate
        ``True``: full gate (protected, degree, connectivity). ``False``: the ungated partner (degree and
        connectivity only).
    seed
        Seed of the random ordering (42 in the paper).
    extra_protected
        Additional bond classes to protect (e.g. ``("hydrogen_bond_candidate",)``).
    policy
        Label written to the audit log (defaults to the ordering name).
    audit
        Record the audit log. The ``g`` and ``S`` audit fields always hold the gravity score, as in the paper.

    Returns
    -------
    (GraphView, AuditLog)
    """
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be between 0 and 1")
    name = ordering if isinstance(ordering, str) else ordering.name if isinstance(ordering, Ordering) else "custom"
    order = resolve(ordering, graph, seed=seed)
    mode = "corrected" if gate else "connectivity_only"
    comp = _gravity_components(graph) if audit else None
    label = policy if policy is not None else name
    r = prune_instances(graph.graph, tau, order, mode, comp=comp, qmof_id=graph.structure_id, policy=label,
                        audit=audit, extra=tuple(extra_protected))
    view = GraphView(base=graph, graph=r["graph"], removed=r["removed"], tau=tau, target=r["target"], ordering=name,
                     gate=gate, extra_protected=tuple(extra_protected), policy=label, seed=seed)
    return view, AuditLog(r["audit"])


# ------------------------------------------------------------------ named policies of the paper
@dataclass(frozen=True)
class Policy:
    name: str
    internal_key: str
    ordering: str
    gate: bool
    extra_protected: Tuple[str, ...] = ()
    paper_main: bool = True


POLICIES: Dict[str, Policy] = {p.name: p for p in (
    Policy("Gravity-Chem", "ChemSPAR-v2", "gravity", True),
    Policy("Gravity", "ChemSPAR-v2-noGate", "gravity", False),
    Policy("Distance-Chem", "Distance-Chem", "distance", True),
    Policy("Distance", "Distance", "distance", False),
    Policy("Random-Chem", "Random-Chem", "random", True),
    Policy("Random", "Random", "random", False),
    Policy("BHS-Chem", "BHS-edge-min", "bhs", True),
    Policy("BHS", "BHS-edge-min-noGate", "bhs", False),
    # supplementary ablations
    Policy("Gravity-HBProtect", "ChemSPAR-v2-HBProtect", "gravity", True, ("hydrogen_bond_candidate",), False),
    Policy("BHS-prod-Chem", "BHS-edge-prod", "bhs_prod", True, (), False),
    Policy("Massfree-Chem", "ChemSPAR-v2-massfree", "massfree", True, (), False),
    Policy("Gravity-chemmass-Chem", "ChemSPAR-v2-chemmass", "gravity_chem", True, (), False),
    Policy("Gravity-topomass-Chem", "ChemSPAR-v2-topomass", "gravity_topo", True, (), False),
)}
"""Named policies; ``internal_key`` is the name used in the paper's code, configs and Zenodo files."""

PAPER_POLICIES = tuple(n for n, p in POLICIES.items() if p.paper_main)


def get_policy(name: str) -> Policy:
    """Look up a policy by paper name or internal key."""
    if name in POLICIES:
        return POLICIES[name]
    for p in POLICIES.values():
        if p.internal_key == name:
            return p
    raise KeyError(f"unknown policy {name!r}; available: {', '.join(POLICIES)}")


def prune_policy(graph: ContactGraph, policy: str, tau: float, *, seed: int = 42,
                 audit: bool = True) -> Tuple[GraphView, AuditLog]:
    """Run one of the paper's named policies (e.g. ``"Distance-Chem"`` or its internal key)."""
    p = get_policy(policy)
    return prune(graph, p.ordering, tau, gate=p.gate, seed=seed, extra_protected=p.extra_protected,
                 policy=p.internal_key, audit=audit)


__all__ = ["GATE_MODES", "GraphView", "prune", "prune_instances", "Policy", "POLICIES", "PAPER_POLICIES",
           "get_policy", "prune_policy"]
