"""BHS-derived edge-ranking baseline (OUR adaptation; the Black Hole Strategy itself ranks MOF nodes, not edges).

Inherited from the published BHS node score (Jalali et al., JCIM 2025, Eq. 1; equal weights in all main
experiments; weights read from configs/baselines_v1.yaml `bhs_node_score`):
    gamma_i = 1/3 * minmax(degree_centrality_i) + 1/3 * minmax(betweenness_i) + 1/3 * minmax(weighted_degree_i)
    with min-max scaling within a community and edge weights that express similarity (larger = stronger).
Adaptation choices (ours, fixed before any result):
    * community       = the whole atomistic graph of one structure (no community detection);
    * edge weight     = 1 / d_ij (shorter contact = stronger), the analogue of a similarity weight;
    * edge score      = min(gamma_i, gamma_j)  ('BHS-edge-min')  or  gamma_i * gamma_j  ('BHS-edge-prod');
    * removal order   = ascending edge score (BHS retains high-gravity nodes, so low-gravity edges go first);
    * gate            = the same corrected gate as ChemSPAR-v2.
Not inherited: BHS community stratification, test-node selection and representative subset sampling.
"""
from __future__ import annotations

import networkx as nx
import numpy as np

def _load_weights():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/baselines_v1.yaml").read_text())["bhs_node_score"]["weights"]
    return (cfg["degree"], cfg["betweenness"], cfg["neighbourhood_influence"])


BHS_WEIGHTS = _load_weights()


def _minmax(v: dict) -> dict:
    a = np.array(list(v.values()), dtype=float)
    lo, hi = a.min(), a.max()
    return {k: (0.0 if hi == lo else (x - lo) / (hi - lo)) for k, x in v.items()}


def bhs_node_gravity(g: nx.Graph) -> dict[int, float]:
    wd, wb, ww = BHS_WEIGHTS
    deg = _minmax(nx.degree_centrality(g))
    btw = _minmax(nx.betweenness_centrality(g, normalized=True))
    wsum = _minmax({n: sum(1.0 / g.edges[n, nb]["distance"] for nb in g.neighbors(n)) for n in g.nodes})
    return {n: wd * deg[n] + wb * btw[n] + ww * wsum[n] for n in g.nodes}


def bhs_edge_scores(g: nx.Graph) -> dict[str, dict[tuple[int, int], float]]:
    gam = bhs_node_gravity(g)
    out = {"min": {}, "prod": {}}
    for u, v in g.edges():
        e = (min(u, v), max(u, v))
        out["min"][e] = min(gam[u], gam[v])
        out["prod"][e] = gam[u] * gam[v]
    return out


def order_ascending(scores: dict[tuple[int, int], float]) -> list[tuple[int, int]]:
    return sorted(scores, key=lambda e: scores[e])   # stable: ties keep graph edge order
