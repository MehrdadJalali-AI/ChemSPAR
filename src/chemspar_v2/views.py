"""Single source of truth for graph views on periodic edge instances (used by Phase 2, tensors and tests).

`compute_structure(structure, qmof_id)` builds the 4.5 A instance multigraph, CrystalNN instances, taxonomy,
gravity scores, all pruning policies (removal ranks) and all construction instance sets.
`view_instances(res, view)` returns the exact retained instance list of a named view, e.g. "ChemSPAR-v2@0.2",
"Random-Chem@0.5", "Original-4.5", "Cutoff-3.0", "CrystalNN", "kNN-12".
Pruning unit = periodic edge instance (i, j, image); removed instances never reappear anywhere downstream.
"""
from __future__ import annotations

import networkx as nx
from pymatgen.core import Structure

from . import periodic as P
from .baselines_bhs import BHS_WEIGHTS
from .config import load_chemistry
from .crystalnn import crystalnn_pairs

SWEEP_TAUS = (0.2, 0.5, 0.6, 0.7, 0.8, 0.9)
# policy: (ordering key, gate mode, extra protected classes, tau_max)
POLICIES = {
    "ChemSPAR-v2": ("gravity", "corrected", (), 0.9),
    "ChemSPAR-v2-HBProtect": ("gravity", "corrected", ("hydrogen_bond_candidate",), 0.9),
    "ChemSPAR-v2-noGate": ("gravity", "connectivity_only", (), 0.9),
    "Random-Chem": ("random", "corrected", (), 0.9),
    "Distance-Chem": ("longest", "corrected", (), 0.9),
    "BHS-edge-min": ("bhs_min", "corrected", (), 0.9),
    "BHS-edge-prod": ("bhs_prod", "corrected", (), 0.9),
    # ungated partners (connectivity check only, no chemistry gate); extended to 0.9 for the gated/ungated matrix
    "Random": ("random", "connectivity_only", (), 0.9),
    "Distance": ("longest", "connectivity_only", (), 0.9),
    "BHS-edge-min-noGate": ("bhs_min", "connectivity_only", (), 0.9),
    "ChemSPAR-v2-massfree": ("massfree", "corrected", (), 0.5),
    "ChemSPAR-v2-chemmass": ("gravity_chem", "corrected", (), 0.5),
    "ChemSPAR-v2-topomass": ("gravity_topo", "corrected", (), 0.5),
}
CONSTRUCTIONS = ("Original-4.5", "Cutoff-3.5", "Cutoff-3.0", "CrystalNN", "kNN-12")


def _bhs_node_gravity(g: nx.MultiGraph) -> dict:
    wd, wb, ww = BHS_WEIGHTS
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


def compute_structure(structure: Structure, qmof_id: str, audit: bool = False, random_seed: int = 42) -> dict:
    g = P.build_instance_graph(structure, 4.5)
    inst = g.graph["instances"]
    cnn = P.crystalnn_instances(crystalnn_pairs(structure, include_self=True))
    P.annotate_instances(g, cnn)
    alt = {t: P.annotate_instances(g.copy(), cnn, tol=t) for t in (0.25, 0.55)}
    comp = P.instance_gravity(g)
    chem = P.instance_gravity(g, mode="chemistry")
    topo = P.instance_gravity(g, mode="topology")
    gam = _bhs_node_gravity(g)
    attrs = {a["instance"]: a for _, _, a in g.edges(data=True)}
    val = {
        "gravity": {k: comp[k]["S"] for k in inst},
        "gravity_chem": {k: chem[k]["S"] for k in inst},
        "gravity_topo": {k: topo[k]["S"] for k in inst},
        "massfree": {k: 1.0 / comp[k]["d_hyb"] ** 2 for k in inst},
        "bhs_min": {k: min(gam[k[0]], gam[k[1]]) for k in inst},
        "bhs_prod": {k: gam[k[0]] * gam[k[1]] for k in inst},
        "distance": {k: attrs[k]["distance"] for k in inst},
    }
    orders = {key: P.order_by(v, inst) for key, v in val.items() if key != "distance"}
    orders["longest"] = P.order_by(val["distance"], inst, descending=True)
    orders["random"] = P.order_random(inst, random_seed, qmof_id)
    ranks, rej, audits = {}, {}, []
    for name, (okey, mode, extra, tmax) in POLICIES.items():
        r = P.prune_instances(g, tmax, orders[okey], mode, comp=comp, qmof_id=qmof_id, policy=name, audit=True, extra=extra)
        ranks[name] = {k: t for t, k in enumerate(r["removed"])}
        rej[name] = {}
        for tau in [t for t in SWEEP_TAUS if t <= tmax + 1e-9]:
            tgt = int(tau * len(inst))
            rows = [a for a in r["audit"] if a["removed_before_attempt"] < tgt]
            counts = {}
            for a in rows:
                if a["decision"] == "rejected":
                    counts[a["reason"]] = counts.get(a["reason"], 0) + 1
            rej[name][tau] = dict(target=tgt, removed=min(len(r["removed"]), tgt), **counts)
        if audit and tmax >= 0.9:
            audits += r["audit"]
    knn = set(P.knn_instances(structure, 12))
    cons = {"Original-4.5": list(inst),
            "Cutoff-3.5": [k for k in inst if attrs[k]["distance"] <= 3.5],
            "Cutoff-3.0": [k for k in inst if attrs[k]["distance"] <= 3.0],
            "CrystalNN": sorted(cnn),
            "kNN-12": sorted(knn)}
    dist = {k: attrs[k]["distance"] for k in inst}
    for k in cons["CrystalNN"] + cons["kNN-12"]:
        if k not in dist:
            dist[k] = P.instance_distance(structure, k)
    return dict(qmof_id=qmof_id, graph=g, instances=inst, attrs=attrs, alt=alt, comp=comp, val=val, ranks=ranks,
                rejections=rej, constructions=cons, dist=dist, cnn=cnn, knn=knn, audit=audits)


def view_instances(res: dict, view: str) -> list:
    if view in CONSTRUCTIONS:
        return list(res["constructions"][view])
    name, tau = view.rsplit("@", 1)
    tau = float(tau)
    k = int(tau * len(res["instances"]))
    rk = res["ranks"][name]
    return [e for e in res["instances"] if not (e in rk and rk[e] < k)]
