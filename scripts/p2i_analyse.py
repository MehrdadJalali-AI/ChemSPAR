"""Phase 2 analysis on periodic edge INSTANCES (ChemSPAR-v2 = gravity + gate; pruning unit = instance).

Pre-specified HBProtect rule (unchanged from the pair analysis, written before these results): lowest tau in
{0.5..0.9} with >= 25% of HB-candidate-containing structures differing, stably at the next tau, >= 99% achieved.
Outputs: results/phase2i/*.csv
"""
from __future__ import annotations

import glob
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

RESUB = Path(__file__).resolve().parents[1]
V = RESUB / "results/phase2i/views"
O = RESUB / "results/phase2i"
TAUS = (0.2, 0.5, 0.6, 0.7, 0.8, 0.9)
SWEEP = ["ChemSPAR-v2", "ChemSPAR-v2-HBProtect", "ChemSPAR-v2-noGate", "Random-Chem", "Distance-Chem", "BHS-edge-min", "BHS-edge-prod"]
HALF = ["ChemSPAR-v2-massfree", "ChemSPAR-v2-chemmass", "ChemSPAR-v2-topomass"]
TAU02 = ["Random", "Distance"]
PROTECTED = {"covalent_radius", "metal_ligand", "metal_metal_crystalnn"}
col = lambda p: "rank__" + p.replace("-", "_")


def load():
    E = pd.concat((pd.read_parquet(f) for f in sorted(glob.glob(str(V / "inst_*.parquet")))), ignore_index=True)
    S = pd.concat((pd.read_parquet(f) for f in sorted(glob.glob(str(V / "stats_*.parquet")))), ignore_index=True)
    T = pd.concat((pd.read_parquet(f) for f in sorted(glob.glob(str(V / "timing_*.parquet")))), ignore_index=True)
    X = [pd.read_parquet(f) for f in sorted(glob.glob(str(V / "extra_*.parquet")))]
    E["n_inst"] = E.qmof_id.map(E.groupby("qmof_id").size())
    E["shell"] = pd.cut(E.distance, [0, 2.5, 3.5, 4.5001], labels=["<2.5", "2.5-3.5", "3.5-4.5"], right=False).astype(str)
    E["pair"] = ["-".join(sorted([("M" if mi else a), ("M" if mj else b)])) for a, b, mi, mj in zip(E.el_i, E.el_j, E.metal_i, E.metal_j)]
    return E, S, T, (pd.concat(X) if X else pd.DataFrame())


def removed(E, p, tau):
    r = E[col(p)]
    return (r >= 0) & (r < (E.n_inst * tau).astype(int))


def jac(E, a, b):
    s = pd.DataFrame({"q": E.qmof_id, "i": a & b, "u": a | b}).groupby("q")[["i", "u"]].sum()
    return s.i / s.u.replace(0, np.nan)


def main():
    E, S, T, X = load()
    n_struct = E.qmof_id.nunique()
    # ---- sparsity validation
    rej = [c for c in S.columns if c.startswith("rej::")]
    S[rej] = S[rej].fillna(0)
    val = S.groupby(["view", "tau"]).agg(target=("target", "sum"), removed=("removed", "sum"),
                                        short_of_target=("removed", lambda s: int((s < S.loc[s.index, "target"]).sum())),
                                        **{c.replace("rej::", "rej_"): (c, "sum") for c in rej}).reset_index()
    val["achieved_pct"] = 100 * val.removed / val.target
    val.to_csv(O / "p2i_sparsity_validation.csv", index=False)
    # ---- constructions (instances; connectivity of the cell graph)
    cons = []
    n_atoms = pd.read_csv(RESUB / "data/cohort_5000_v1.csv").set_index("qmof_id").n_atoms
    for name, m in [("Original-4.5", pd.Series(True, index=E.index)), ("Cutoff-3.5", E.in_cut35), ("Cutoff-3.0", E.in_cut30),
                    ("CrystalNN", E.in_crystalnn), ("kNN-12", E.in_knn12)]:
        sel = E[m]
        grp = dict(tuple(sel.groupby("qmof_id")[["i", "j"]]))
        comp, iso = [], []
        for q, n in n_atoms.items():
            h = nx.Graph(); h.add_nodes_from(range(n))
            if q in grp:
                h.add_edges_from(zip(grp[q].i, grp[q].j))
            comp.append(nx.number_connected_components(h)); iso.append(sum(1 for x in h if h.degree(x) == 0))
        extra = 0 if X.empty else int((X.construction == name).sum())
        cons.append(dict(view=name, mean_instances=(m.sum() + extra) / n_struct, pct_of_original=100 * m.sum() / len(E),
                         instances_beyond_4p5=extra, connected_frac=float(np.mean(np.array(comp) == 1)),
                         mean_components=float(np.mean(comp)), structures_with_isolated=int(np.sum(np.array(iso) > 0)),
                         **{f"retained_{c}_pct": 100 * (m & (E.bond_class == c)).sum() / max((E.bond_class == c).sum(), 1)
                            for c in ["covalent_radius", "metal_ligand", "metal_metal_crystalnn", "hydrogen_bond_candidate", "metal_contact", "other_contact"]}))
    pd.DataFrame(cons).to_csv(O / "p2i_constructions.csv", index=False)
    # ---- taxonomy and multi-image bookkeeping
    pairs = E.groupby(["qmof_id", "i", "j"]).size()
    tax = dict(instances=len(E), self_image=int(E.self_image.sum()), atom_pairs=len(pairs),
               pairs_with_multiple_instances=int((pairs > 1).sum()), **{f"class_{k}": int(v) for k, v in E.bond_class.value_counts().items()})
    pd.Series(tax).to_csv(O / "p2i_taxonomy.csv")
    # ---- chemical validity and composition
    chem, comp_rows = [], []
    ml = E.bond_class == "metal_ligand"; hb = E.bond_class == "hydrogen_bond_candidate"
    for p in SWEEP + HALF + TAU02:
        for tau in (TAUS if p in SWEEP else (0.2, 0.5) if p in HALF else (0.2,)):
            r = removed(E, p, tau)
            chem.append(dict(policy=p, tau=tau, removed=int(r.sum()),
                             protected_removed_tol025=int((r & E.class_tol025.isin(PROTECTED)).sum()),
                             protected_removed_tol040=int((r & E.protected).sum()),
                             protected_removed_tol055=int((r & E.class_tol055.isin(PROTECTED)).sum()),
                             crystalnn_retained_pct=100 * (1 - (r & E.crystalnn).sum() / E.crystalnn.sum()),
                             metal_ligand_retained_pct=100 * (1 - (r & ml).sum() / ml.sum()),
                             structures_losing_metal_ligand=int((r & ml).groupby(E.qmof_id).any().sum()),
                             hb_candidates_retained_pct=100 * (1 - (r & hb).sum() / max(hb.sum(), 1)),
                             metal_contact_removed_pct=100 * (r & (E.bond_class == "metal_contact")).sum() / (E.bond_class == "metal_contact").sum(),
                             self_image_removed=int((r & E.self_image).sum())))
            R = E[r]
            d = dict(policy=p, tau=tau, mean_distance=R.distance.mean())
            for k, v in R.bond_class.value_counts(normalize=True).items():
                d[f"class_{k}"] = v
            for k, v in R.shell.value_counts(normalize=True).items():
                d[f"shell_{k}"] = v
            for k, v in R.pair.value_counts(normalize=True).head(6).items():
                d[f"pair_{k}"] = v
            comp_rows.append(d)
    pd.DataFrame(chem).to_csv(O / "p2i_chemical_validity.csv", index=False)
    pd.DataFrame(comp_rows).fillna(0).to_csv(O / "p2i_removed_composition.csv", index=False)
    # ---- Jaccard
    jr = []
    for tau, pols in [(0.2, SWEEP + HALF + TAU02), (0.5, SWEEP + HALF), (0.7, SWEEP)]:
        R = {p: removed(E, p, tau) for p in pols}
        for a, b in combinations(pols, 2):
            j = jac(E, R[a], R[b])
            jr.append(dict(tau=tau, A=a, B=b, mean=j.mean(), sd=j.std(ddof=1), median=j.median(), q25=j.quantile(.25), q75=j.quantile(.75),
                           structures_identical=int((~(R[a] != R[b]).groupby(E.qmof_id).any()).sum())))
    pd.DataFrame(jr).to_csv(O / "p2i_removed_set_jaccard.csv", index=False)
    # ---- HBProtect
    has_hb = E[hb].qmof_id.unique()
    rows = []
    vv = val.set_index(["view", "tau"])
    for tau in TAUS:
        a, b = removed(E, "ChemSPAR-v2", tau), removed(E, "ChemSPAR-v2-HBProtect", tau)
        diff = (a != b).groupby(E.qmof_id).any()
        rows.append(dict(tau=tau, structures_differing=int(diff.sum()),
                         frac_differing_among_hb_structures=float(diff.reindex(has_hb).mean()) if len(has_hb) else np.nan,
                         mean_jaccard_removed=float(jac(E, a, b).mean()), min_jaccard_removed=float(jac(E, a, b).min()),
                         hb_candidates_total=int(hb.sum()), hb_removed_ChemSPAR=int((a & hb).sum()), hb_removed_HBProtect=int((b & hb).sum()),
                         protected_removed_ChemSPAR=int((a & E.protected).sum()), protected_removed_HBProtect=int((b & E.protected).sum()),
                         achieved_pct_ChemSPAR=float(vv.loc[("ChemSPAR-v2", tau), "achieved_pct"]),
                         achieved_pct_HBProtect=float(vv.loc[("ChemSPAR-v2-HBProtect", tau), "achieved_pct"])))
    H = pd.DataFrame(rows); H.to_csv(O / "p2i_hbprotect_vs_chemspar.csv", index=False)
    fd = []
    for q, e in E[E.qmof_id.isin(has_hb)].groupby("qmof_id"):
        a = e[e[col("ChemSPAR-v2")] >= 0].sort_values(col("ChemSPAR-v2"))[["i", "j", "image"]].values.tolist()
        b = e[e[col("ChemSPAR-v2-HBProtect")] >= 0].sort_values(col("ChemSPAR-v2-HBProtect"))[["i", "j", "image"]].values.tolist()
        k = next((t for t, (x, y) in enumerate(zip(a, b)) if x != y), None)
        if k is None and len(a) != len(b):
            k = min(len(a), len(b))
        fd.append(dict(qmof_id=q, first_divergence_tau=np.nan if k is None else (k + 1) / len(e)))
    FD = pd.DataFrame(fd); FD.to_csv(O / "p2i_hbprotect_first_divergence.csv", index=False)
    grid = [0.5, 0.6, 0.7, 0.8, 0.9]; Hs = H.set_index("tau"); chosen = None
    for k, t in enumerate(grid[:-1]):
        if (Hs.loc[t, "frac_differing_among_hb_structures"] >= 0.25 and Hs.loc[grid[k + 1], "frac_differing_among_hb_structures"] >= 0.25
                and Hs.loc[t, "achieved_pct_ChemSPAR"] >= 99 and Hs.loc[t, "achieved_pct_HBProtect"] >= 99):
            chosen = t; break
    pd.Series(dict(chosen_tau=chosen)).to_csv(O / "p2i_hbprotect_tau_decision.csv")
    # ---- instance vs pair projection: pairs partially pruned by instance-level policies
    pp = []
    for p, tau in [("ChemSPAR-v2", 0.2), ("ChemSPAR-v2", 0.5), ("Random-Chem", 0.2), ("Distance-Chem", 0.2), ("BHS-edge-min", 0.2)]:
        r = removed(E, p, tau)
        g = pd.DataFrame({"q": E.qmof_id, "i": E.i, "j": E.j, "r": r}).groupby(["q", "i", "j"]).r.agg(["sum", "size"])
        multi = g[g["size"] > 1]
        pp.append(dict(policy=p, tau=tau, multi_instance_pairs=len(multi), partially_pruned_pairs=int(((multi["sum"] > 0) & (multi["sum"] < multi["size"])).sum()),
                       fully_pruned_multi_pairs=int((multi["sum"] == multi["size"]).sum())))
    pd.DataFrame(pp).to_csv(O / "p2i_partial_pair_pruning.csv", index=False)
    T.describe().T.to_csv(O / "p2i_build_timing_summary.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(pd.Series(tax).to_string()); print(pd.DataFrame(cons).round(2).T.to_string())
    print(val[["view", "tau", "achieved_pct", "short_of_target"]].round(2).to_string(index=False))
    print(pd.DataFrame(chem).round(3).to_string(index=False))
    print(H.round(4).to_string(index=False)); print("first divergence:", FD.first_divergence_tau.describe().round(3).to_dict(), "| chosen:", chosen)
    print(pd.DataFrame(pp).to_string(index=False))
    J = pd.DataFrame(jr); print(J[J.A == "ChemSPAR-v2"][["tau", "B", "mean", "structures_identical"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
