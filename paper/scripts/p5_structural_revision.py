"""Revision round 1 (reviewer comments R1.3, R1.5, R1.7): structural analyses only, no training.

The instance graph of every cohort structure is rebuilt from the stored instance views (results/phase2i/views:
contact instances, distances, elements, CrystalNN flags), so CrystalNN is not rerun. The rebuilt taxonomy, gravity
scores, BHS-derived scores and the stored removal ranks of all 8 policies up to tau = 0.9 are checked for exact
equality before anything else is used ("verify").

  python p5_structural_revision.py verify 200     # exactness check on the first N cohort structures
  python p5_structural_revision.py sweep          # all 5,000 structures: pruning to tau = 0.95 for 8 policies,
                                                  # complete audit logs, sweep endpoints at tau 0.5/0.8/0.9/0.95
  python p5_structural_revision.py weights        # +-20% perturbation of each gravity weight, gated, tau = 0.5
Outputs: results/revision/{verify.json, sweep_per_structure.parquet, sweep_summary.csv, weights_summary.csv},
         results/revision/audit_full/audit_*.parquet (complete per-attempt logs; deposited on Zenodo)
"""
from __future__ import annotations

import glob
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from chemspar_v2 import periodic as P  # noqa: E402
from chemspar_v2 import views as V  # noqa: E402
from chemspar_v2.config import load_chemistry  # noqa: E402
from chemspar_v2.elements import is_metal, role  # noqa: E402

VIEWS = R / "results/phase2i/views"
O = R / "results/revision"
POLS = ["ChemSPAR-v2", "ChemSPAR-v2-noGate", "Distance-Chem", "Distance", "Random-Chem", "Random", "BHS-edge-min", "BHS-edge-min-noGate"]
TAUS = (0.5, 0.8, 0.9, 0.95)
TMAX = 0.95
WEIGHT_KEYS = ["degree", "betweenness", "closeness", "neighbour_distance", "z", "role"]


def rebuild(E: pd.DataFrame, n_atoms: int) -> nx.MultiGraph:
    """Instance multigraph identical to periodic.build_instance_graph + annotate_instances, from stored rows."""
    from pymatgen.core import Element
    els = {}
    for i, j, a, b in zip(E.i, E.j, E.el_i, E.el_j):
        els[i] = a; els[j] = b
    assert len(els) == n_atoms, (len(els), n_atoms)
    g = nx.MultiGraph()
    for n in range(n_atoms):
        el = els[n]
        g.add_node(n, element=el, Z=float(Element(el).Z), role=role(el), is_metal=is_metal(el))
    inst = []
    for i, j, im, d in zip(E.i, E.j, E.image, E.distance):
        k = (int(i), int(j), tuple(int(x) for x in im.split(",")))
        g.add_edge(k[0], k[1], key=k[2], distance=float(d), instance=k)
        inst.append(k)
    assert inst == sorted(inst)
    g.graph["instances"] = inst
    cnn = {k for k, c in zip(inst, E.crystalnn) if c}
    P.annotate_instances(g, cnn)
    return g


def orders(g, qid, comp):
    inst = g.graph["instances"]
    gam = V._bhs_node_gravity(g)
    dist = {a["instance"]: a["distance"] for _, _, a in g.edges(data=True)}
    return {"gravity": P.order_by({k: comp[k]["S"] for k in inst}, inst),
            "longest": P.order_by(dist, inst, descending=True),
            "random": P.order_random(inst, 42, qid),
            "bhs_min": P.order_by({k: min(gam[k[0]], gam[k[1]]) for k in inst}, inst)}, gam


def by_structure():
    cohort = pd.read_csv(R / "data/cohort_5000_v1.csv")
    natoms = dict(zip(cohort.qmof_id, cohort.n_atoms))
    for f in sorted(glob.glob(str(VIEWS / "inst_*.parquet"))):
        E = pd.read_parquet(f)
        for q, e in E.groupby("qmof_id", sort=False):
            yield q, e.reset_index(drop=True), natoms[q]


def verify_one(args):
    q, e, n = args
    warnings.filterwarnings("ignore")
    g = rebuild(e, n)
    inst = g.graph["instances"]
    comp = P.instance_gravity(g)
    out = dict(qmof_id=q, class_ok=bool((e.bond_class.values == np.array([g.edges[k[0], k[1], k[2]]["bond_class"] for k in inst])).all()),
               g_ok=bool(np.allclose([comp[k]["g"] for k in inst], e.g.values, rtol=0, atol=0)))
    od, gam = orders(g, q, comp)
    out["bhs_ok"] = bool(np.allclose([min(gam[k[0]], gam[k[1]]) for k in inst], e.bhs_min.values, rtol=0, atol=0))
    for pol in POLS:
        okey, mode, extra, _ = V.POLICIES[pol]
        rem = P.prune_instances(g, 0.9, od[okey], mode, audit=False, extra=extra)["removed"]
        rk = {k: t for t, k in enumerate(rem)}
        out[f"rank_ok::{pol}"] = bool((np.array([rk.get(k, -1) for k in inst]) == e["rank__" + pol.replace("-", "_")].values).all())
    return out


def shell_complete(e, removed_mask, r=3.0):
    m = e.distance.values <= r
    a = np.r_[e.i.values[m], e.j.values[m]]; lost = np.r_[removed_mask[m], removed_mask[m]]
    s = pd.Series(lost).groupby(a).max()
    return int((~s.astype(bool)).sum()), int(len(s))


def sweep_one(args):
    q, e, n = args
    warnings.filterwarnings("ignore")
    g = rebuild(e, n)
    inst = g.graph["instances"]; N = len(inst)
    comp = P.instance_gravity(g)
    od, _ = orders(g, q, comp)
    idx = {k: t for t, k in enumerate(inst)}
    prot = e.protected.values
    rows, audit = [], []
    for pol in POLS:
        okey, mode, extra, _ = V.POLICIES[pol]
        r = P.prune_instances(g, TMAX, od[okey], mode, comp=comp, qmof_id=q, policy=pol, audit=True, extra=extra)
        rem = r["removed"]
        stored = e["rank__" + pol.replace("-", "_")].values
        n09 = int((stored >= 0).sum())
        rank_ok = all(idx[x] == int(np.where(stored == t)[0][0]) for t, x in enumerate(rem[:n09])) if n09 else True
        # each instance is visited at most once per pass (R2.5)
        ranks = [a["rank"] for a in r["audit"]]
        assert len(ranks) == len(set(ranks))
        for tau in TAUS:
            tgt = int(tau * N); k = min(tgt, len(rem))
            mask = np.zeros(N, bool); mask[[idx[x] for x in rem[:k]]] = True
            att = [a for a in r["audit"] if a["removed_before_attempt"] < tgt]
            c, na = shell_complete(e, mask)
            rows.append(dict(qmof_id=q, policy=pol, tau=tau, rank_ok_to_0p9=rank_ok, n_instances=N, target=tgt, removed=k, short=tgt - k,
                             attempts=len(att), rejected=sum(a["decision"] == "rejected" for a in att),
                             protected_total=int(prot.sum()), protected_removed=int((mask & prot).sum()),
                             shell_complete=c, shell_atoms=na))
        for a in r["audit"]:
            a.pop("g", None); a.pop("S", None)
        audit += r["audit"]
    return rows, audit


def weights_one(args):
    q, e, n, variants = args
    warnings.filterwarnings("ignore")
    g = rebuild(e, n)
    inst = g.graph["instances"]; N = len(inst); tgt = int(0.5 * N)
    base = set(inst[t] for t in np.where((e.rank__ChemSPAR_v2.values >= 0) & (e.rank__ChemSPAR_v2.values < tgt))[0])
    out = []
    cent = centralities(g)
    for name, cfg in variants:
        S = gravity_with(g, cfg, cent)
        rem = set(P.prune_instances(g, 0.5, P.order_by(S, inst), "corrected", audit=False)["removed"])
        out.append(dict(qmof_id=q, variant=name, jaccard=len(rem & base) / len(rem | base), protected_removed=sum(g.edges[k[0], k[1], k[2]]["protected"] for k in rem)))
    return out


def centralities(g):
    simple = nx.Graph(g)
    return nx.betweenness_centrality(simple, normalized=True), nx.closeness_centrality(simple)


def gravity_with(g, cfg, cent):
    """periodic.instance_gravity with explicit weights (same formula; used only for the perturbation check)."""
    w, rw, lam = cfg["node_mass_weights"], cfg["role_weights"], cfg["hybrid_lambda"]
    deg = {n: g.degree(n) for n in g}; n1 = max(g.number_of_nodes() - 1, 1)
    btw, cls = cent
    nsum = {n: 0.0 for n in g}
    for u, v, a in g.edges(data=True):
        nsum[u] += a["distance"]; nsum[v] += a["distance"]
    mx = max(nsum.values()) or 1.0; mz = max(d["Z"] for _, d in g.nodes(data=True))
    m = {n: w["degree"] * deg[n] / n1 + w["betweenness"] * btw[n] + w["closeness"] * cls[n] + w["neighbour_distance"] * nsum[n] / mx
         + w["z"] * d["Z"] / mz + w["role"] * rw.get(d["role"], rw["other"]) for n, d in g.nodes(data=True)}
    out = {}
    for u, v, k, a in g.edges(keys=True, data=True):
        du, dv = deg[u], deg[v]
        dh = max(lam * (1 + abs(du - dv) / max(du, dv, 1)) + (1 - lam) * a["distance"], 1e-6)
        out[a["instance"]] = m[u] * m[v] / dh ** 2
    return out


def main():
    mode = sys.argv[1]
    O.mkdir(parents=True, exist_ok=True)
    if mode == "verify":
        nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        work = [x for _, x in zip(range(nmax), by_structure())]
        with Pool(9) as pool:
            res = pool.map(verify_one, work, chunksize=2)
        D = pd.DataFrame(res)
        summ = {c: int(D[c].sum()) for c in D.columns if c != "qmof_id"}
        summ["n"] = len(D)
        (O / "verify.json").write_text(json.dumps(summ, indent=2)); print(json.dumps(summ, indent=1))
    elif mode == "sweep":
        A = O / "audit_full"; A.mkdir(exist_ok=True)
        rows, buf, part = [], [], 0
        it = by_structure()
        with Pool(9) as pool:
            for k, (rr, aa) in enumerate(pool.imap(sweep_one, it, chunksize=4)):
                rows += rr; buf += aa
                if (k + 1) % 250 == 0:
                    pd.DataFrame(buf).to_parquet(A / f"audit_{part:03d}.parquet", index=False); buf = []; part += 1
                    print(f"{k + 1} structures", flush=True)
        if buf:
            pd.DataFrame(buf).to_parquet(A / f"audit_{part:03d}.parquet", index=False)
        D = pd.DataFrame(rows); D.to_parquet(O / "sweep_per_structure.parquet", index=False)
        S = D.groupby(["policy", "tau"]).agg(structures=("qmof_id", "nunique"), target=("target", "sum"), removed=("removed", "sum"),
                                             structures_short=("short", lambda s: int((s > 0).sum())), attempts=("attempts", "sum"),
                                             rejected=("rejected", "sum"), protected_total=("protected_total", "sum"),
                                             protected_removed=("protected_removed", "sum"), shell_complete=("shell_complete", "sum"),
                                             shell_atoms=("shell_atoms", "sum")).reset_index()
        S["achieved_fraction_pct"] = 100 * S.removed / D.groupby(["policy", "tau"]).n_instances.sum().values
        S["protected_retention_pct"] = 100 * (1 - S.protected_removed / S.protected_total)
        S["shell_complete_pct"] = 100 * S.shell_complete / S.shell_atoms
        S.to_csv(O / "sweep_summary.csv", index=False); print(S.round(3).to_string(index=False))
        print("rank prefix identical to stored views (structure x policy):", int(D.drop_duplicates(["qmof_id", "policy"]).rank_ok_to_0p9.sum()), "of", D.drop_duplicates(["qmof_id", "policy"]).shape[0])
    elif mode == "weights":
        base = load_chemistry()["score"]
        variants = []
        for key in WEIGHT_KEYS:
            for f in (0.8, 1.2):
                c = json.loads(json.dumps(base)); c["node_mass_weights"][key] *= f; variants.append((f"{key}x{f}", c))
        for f in (0.8, 1.2):
            c = json.loads(json.dumps(base)); c["hybrid_lambda"] = base["hybrid_lambda"] * f; variants.append((f"hybrid_lambda x{f}", c))
        c = json.loads(json.dumps(base)); variants.append(("unperturbed", c))
        work = [(q, e, n, variants) for q, e, n in by_structure()]
        with Pool(9) as pool:
            res = [r for part in pool.imap(weights_one, work, chunksize=4) for r in part]
        D = pd.DataFrame(res)
        S = D.groupby("variant").agg(mean_jaccard=("jaccard", "mean"), min_jaccard=("jaccard", "min"),
                                     p05_jaccard=("jaccard", lambda s: s.quantile(0.05)), protected_removed=("protected_removed", "sum")).reset_index()
        S.to_csv(O / "weights_summary.csv", index=False); print(S.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
