"""Structural comparison of the chemical protection gate and a size-matched geometric gate.

Pre-specified before any of these counts were computed (Task C of the strengthening prompt):

For each structure, P is the number of chemically protected instances (stored `protected`
flag, chemistry-v1.2). The geometric gate protects the P shortest instances. Ties keep
the canonical instance order, which is the stable sort used by periodic.order_by.
The non-hydrogen degree check and the connectivity check are the checks in
periodic._check; this script repeats that function for mode "corrected" and an empty
extra-class tuple so the run does not import pymatgen. The chemical-gate removed set
is compared with the stored tau = 0.5 rank prefix before the geometric result is used.

Random order uses periodic.order_random with seed 42. BHS-derived order uses the stored
bhs_min scores, ascending, with canonical ties.

Stopping rule, fixed before the run: if both pooled removed-set Jaccards
(Random-Geo vs Random-Chem, and BHS-Geo vs BHS-Chem) are at least 0.98, the two gates
are practically the same and training cannot separate the explanations.

No training. Writes only results/geogate/.
"""
from __future__ import annotations

import glob
import hashlib
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
VIEWS = R / "results" / "phase2i" / "views"
OUT = R / "results" / "geogate"
TAU = 0.5


def order_by(values: dict, instances: list) -> list:
    return sorted(instances, key=lambda k: values[k])


def order_random(instances: list, seed: int, qmof_id: str) -> list:
    h = int(hashlib.sha256(f"{seed}:{qmof_id}".encode()).hexdigest()[:16], 16)
    idx = np.random.default_rng(h).permutation(len(instances))
    return [instances[t] for t in idx]


def build(E: pd.DataFrame, n_atoms: int, protected: np.ndarray) -> tuple[nx.MultiGraph, list]:
    els = {}
    for i, j, a, b in zip(E.i, E.j, E.el_i, E.el_j):
        els[int(i)] = a
        els[int(j)] = b
    if len(els) != n_atoms:
        raise AssertionError((len(els), n_atoms))
    g = nx.MultiGraph()
    for n in range(n_atoms):
        g.add_node(n, element=els[n])
    inst = []
    for row_i, (i, j, im, d, prot) in enumerate(zip(E.i, E.j, E.image, E.distance, protected)):
        key = (int(i), int(j), tuple(int(x) for x in str(im).split(",")))
        g.add_edge(key[0], key[1], key=key[2], distance=float(d), instance=key, protected=bool(prot))
        inst.append(key)
    if inst != sorted(inst):
        raise AssertionError("instance rows are not in canonical order")
    g.graph["instances"] = inst
    return g, inst


def _check(g: nx.MultiGraph, key: tuple) -> str | None:
    """periodic._check for mode 'corrected' and extra=(). Returns the rejection reason, or None."""
    u, v, img = key
    a = g.edges[u, v, img]
    if a["protected"]:
        return "protected"
    for n in {u, v}:
        loss = 2 if u == v else 1
        if g.nodes[n]["element"] != "H" and g.degree(n) - loss <= 0:
            return "degree"
    if u == v:
        return None
    attrs = dict(a)
    g.remove_edge(u, v, key=img)
    ok = nx.has_path(g, u, v)
    g.add_edge(u, v, key=img, **attrs)
    return None if ok else "connectivity"


def prune(g0: nx.MultiGraph, order: list) -> list:
    g = g0.copy()
    target = int(TAU * g.number_of_edges())
    removed = []
    for key in order:
        if len(removed) >= target:
            break
        u, v, img = key
        if not g.has_edge(u, v, key=img):
            continue
        if _check(g, key):
            continue
        g.remove_edge(u, v, key=img)
        removed.append(key)
    return removed


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def _stored_removed(inst, rank, k: int) -> list:
    rank = np.asarray(rank)
    sel = np.flatnonzero((rank >= 0) & (rank < k))
    order = sel[np.argsort(rank[sel], kind="mergesort")]
    return [inst[i] for i in order]


def by_structure():
    cohort = pd.read_csv(R / "data" / "cohort_5000_v1.csv")
    natoms = dict(zip(cohort.qmof_id, cohort.n_atoms))
    cols = ["qmof_id", "i", "j", "image", "distance", "el_i", "el_j", "protected", "bhs_min",
            "rank__Random_Chem", "rank__BHS_edge_min"]
    for f in sorted(glob.glob(str(VIEWS / "inst_*.parquet"))):
        E = pd.read_parquet(f, columns=cols)
        for q, e in E.groupby("qmof_id", sort=False):
            yield q, e.reset_index(drop=True), int(natoms[q])


def one(args):
    q, e, n = args
    warnings.filterwarnings("ignore")
    g, inst = build(e, n, e.protected.values)
    N = len(inst)
    k = int(TAU * N)
    chem = {key for key, prot in zip(inst, e.protected.values) if bool(prot)}
    Pcount = len(chem)
    shortest = sorted(inst, key=lambda key: g.edges[key[0], key[1], key[2]]["distance"])
    geo = set(shortest[:Pcount])
    g_geo, _ = build(e, n, np.array([key in geo for key in inst]))
    od_random = order_random(inst, 42, q)
    od_bhs = order_by({key: float(v) for key, v in zip(inst, e.bhs_min.values)}, inst)
    out = {
        "qmof_id": q,
        "n_instances": N,
        "P": Pcount,
        "jaccard_protected": _jaccard(chem, geo),
        "protected_intersection": len(chem & geo),
    }
    orders = {"Random-Chem": ("rank__Random_Chem", od_random), "BHS-edge-min": ("rank__BHS_edge_min", od_bhs)}
    for pol, (col, order) in orders.items():
        removed = prune(g, order)
        stored = _stored_removed(inst, e[col].values, k)
        out[f"chem_rank_ok__{pol}"] = removed == stored
        geo_removed = prune(g_geo, order)
        A, B = set(removed), set(geo_removed)
        out[f"jaccard_removed__{pol}"] = _jaccard(A, B)
        out[f"removed_chem__{pol}"] = len(A)
        out[f"removed_geo__{pol}"] = len(B)
        out[f"removed_intersection__{pol}"] = len(A & B)
        out[f"only_chem__{pol}"] = len(A - B)
        out[f"only_geo__{pol}"] = len(B - A)
    return out


def pooled_from_parts(df: pd.DataFrame, pol: str) -> float:
    inter = int(df[f"removed_intersection__{pol}"].sum())
    union = inter + int(df[f"only_chem__{pol}"].sum()) + int(df[f"only_geo__{pol}"].sum())
    return 1.0 if union == 0 else inter / union


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    work = list(by_structure())
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    rows = []
    with Pool(n_workers) as pool:
        for i, row in enumerate(pool.imap(one, work, chunksize=4), start=1):
            rows.append(row)
            if i % 250 == 0:
                print(f"{i} structures", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "geogate_per_structure.csv", index=False)
    prot_inter = int(df.protected_intersection.sum())
    p_sum = int(df.P.sum())
    prot_union = 2 * p_sum - prot_inter
    summary = {
        "definition": (
            "P = number of chemically protected instances; geometric gate protects the P shortest "
            "instances, ties kept in canonical instance order (stable sort by distance); "
            "non-hydrogen degree and connectivity checks unchanged; tau = 0.5; "
            "random seed 42; BHS order from stored bhs_min."
        ),
        "n_structures": int(len(df)),
        "chemical_rank_mismatches": {
            "Random-Chem": int((~df["chem_rank_ok__Random-Chem"]).sum()),
            "BHS-edge-min": int((~df["chem_rank_ok__BHS-edge-min"]).sum()),
        },
        "protected_set": {
            "pooled_jaccard": (prot_inter / prot_union) if prot_union else 1.0,
            "pooled_intersection": prot_inter,
            "pooled_P": p_sum,
            "per_structure_mean": float(df.jaccard_protected.mean()),
            "per_structure_median": float(df.jaccard_protected.median()),
            "per_structure_p5": float(df.jaccard_protected.quantile(0.05)),
            "per_structure_min": float(df.jaccard_protected.min()),
        },
        "removed_set": {},
        "stopping_rule": "stop without training if both pooled removed-set Jaccards are >= 0.98",
    }
    for pol, label in (("Random-Chem", "Random-Geo_vs_Random-Chem"), ("BHS-edge-min", "BHS-Geo_vs_BHS-Chem")):
        summary["removed_set"][label] = {
            "pooled_jaccard": pooled_from_parts(df, pol),
            "per_structure_mean": float(df[f"jaccard_removed__{pol}"].mean()),
            "per_structure_median": float(df[f"jaccard_removed__{pol}"].median()),
            "per_structure_p5": float(df[f"jaccard_removed__{pol}"].quantile(0.05)),
            "per_structure_min": float(df[f"jaccard_removed__{pol}"].min()),
            "n_identical": int((df[f"jaccard_removed__{pol}"] == 1).sum()),
        }
    vals = [summary["removed_set"][k]["pooled_jaccard"] for k in summary["removed_set"]]
    summary["stopping_rule_met"] = bool(all(v >= 0.98 for v in vals) and all(
        v == 0 for v in summary["chemical_rank_mismatches"].values()))
    (OUT / "geogate_structural_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
