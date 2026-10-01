"""ChemSPAR-v2 edge score: gravity only (configs/chemistry_v1.yaml, chemistry-v1.2).

Primary score  S_ij = g_ij = m_i m_j / max(d_hyb, 1e-6)^2   (ascending S = removal priority)
Exploratory    S3_ij = 1.0 g_ij + 0.5 B_ij + 1.2 C_ij       (former three-term score; structural SI ablation only)
d_hyb = lam * (1 + |deg i - deg j| / max(deg i, deg j, 1)) + (1 - lam) * d_ij,
m_i   = 0.45 D + 0.55 Q + 0.20 L + 0.40 N + 0.60 Z~ + 0.90 rho   (NetworkX normalised centralities).
g_ij uses the published formula; only the node roles (v2 metal definition) differ from the legacy code.
"""
from __future__ import annotations

import networkx as nx

from .config import load_chemistry


def node_mass(g: nx.Graph, mode: str = "full") -> dict[int, float]:
    cfg = load_chemistry()["score"]
    w, rw = cfg["node_mass_weights"], cfg["role_weights"]
    deg = nx.degree_centrality(g)
    btw = nx.betweenness_centrality(g, normalized=True)
    cls = nx.closeness_centrality(g)
    max_z = max((d["Z"] for _, d in g.nodes(data=True)), default=1.0)
    nsum = {n: sum(g.edges[n, nb]["distance"] for nb in g.neighbors(n)) for n in g.nodes}
    max_n = max(nsum.values(), default=1.0) or 1.0
    out = {}
    for n, d in g.nodes(data=True):
        topo = w["degree"] * deg[n] + w["betweenness"] * btw[n] + w["closeness"] * cls[n] + w["neighbour_distance"] * nsum[n] / max_n
        chem = w["z"] * d["Z"] / max_z + w["role"] * rw.get(d["role"], rw["other"])
        out[n] = {"full": topo + chem, "chemistry": chem, "topology": topo}[mode]
    return out


def edge_components(g: nx.Graph, mode: str = "full", exploratory: bool = True) -> dict[tuple[int, int], dict[str, float]]:
    """Per-edge components: g, d_hyb and the primary score S (= g). With exploratory=True also B, C and the former
    three-term score S3 (needs edge betweenness, which ChemSPAR-v2 itself does not use; the timing benchmark uses
    exploratory=False)."""
    cfg = load_chemistry()["score"]
    assert cfg["primary"] == "gravity"
    lam, ew = cfg["hybrid_lambda"], cfg["exploratory_three_term"]["edge_weights"]
    m = node_mass(g, mode)
    eb = nx.edge_betweenness_centrality(g) if exploratory else {}
    out = {}
    for u, v, a in g.edges(data=True):
        du, dv = g.degree(u), g.degree(v)
        dh = max(lam * (1 + abs(du - dv) / max(du, dv, 1)) + (1 - lam) * a["distance"], 1e-6)
        grav = m[u] * m[v] / dh ** 2
        b = eb.get((u, v), eb.get((v, u), 0.0))
        c = a["C"]
        key = (u, v) if u < v else (v, u)
        out[key] = {"g": grav, "d_hyb": dh, "S": grav}
        if exploratory:
            out[key].update(B=b, C=c, S3=ew["gravity"] * grav + ew["betweenness"] * b + ew["chemistry"] * c)
    return out
