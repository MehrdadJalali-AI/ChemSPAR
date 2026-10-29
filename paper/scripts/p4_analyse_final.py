"""Phase 4/5 pre-registered analysis of the final matrix (configs/final_v1.yaml `statistics`, `trade_off`).
Written BEFORE any final fit exists. Reads results/final/fits/*.json and results/final/structural/structural_tradeoff.csv.

  python p4_analyse_final.py [cgcnn|graphsage] [--seeds 1-10]

Families (Holm within family, separately for t and Wilcoxon):
  F1 gate effect:          X-gated - X-ungated, X in 4 orders, tau in {0.2, 0.5}             (8)
  F2 ordering under gate:  all pairwise differences among the 4 gated orders, per tau        (12)
  F3 vs full graph:        gated view - Original-4.5, with the D14 equivalence CI rule        (8)
Descriptive (no multiplicity claim): ungated views and constructions vs Original-4.5.
D15 trigger (CGCNN only): any F3 95% CI neither inside [-0.02, 0.02] nor disjoint from it.
Pareto (per tau, 8 pruned views): maximise protected retention (tol 0.40) and shell completeness (3.0 A); minimise
CGCNN MAE and cost (training s/epoch from the cost benchmark if present, else omitted and stated).
Outputs: results/final/analysis/<backbone>_*.csv
"""
from __future__ import annotations

import glob
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

R = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((R / "configs/final_v1.yaml").read_text())
O = R / "results/final/analysis"
DELTA = 0.02
INV = {k: d for d, k in CFG["display_names"].items()}


def disp(view: str) -> str:
    if "@" not in view:
        return view
    p, t = view.split("@")
    return f"{INV[p]}@{t}"


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj


def compare(M: pd.DataFrame, a: str, b: str) -> dict:
    d = (M[a] - M[b]).dropna()
    n = len(d); mu = d.mean(); sd = d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
    t_p = stats.ttest_rel(M.loc[d.index, a], M.loc[d.index, b]).pvalue if n > 1 else np.nan
    w_p = stats.wilcoxon(d, alternative="two-sided", method="exact").pvalue if n > 1 and (d != 0).any() else 1.0
    return dict(A=disp(a), B=disp(b), n_pairs=n, mean_A=M.loc[d.index, a].mean(), sd_A=M.loc[d.index, a].std(ddof=1),
                mean_B=M.loc[d.index, b].mean(), sd_B=M.loc[d.index, b].std(ddof=1), mean_diff=mu, sd_diff=sd,
                ci_low=mu - half, ci_high=mu + half, d_z=mu / sd if sd > 0 else np.nan, p_t=t_p, p_wilcoxon=w_p,
                n_A_better=int((d < 0).sum()))


def family(M, pairs, name):
    ident = CFG["views"].get("identical_views", {})
    same = [(a, b) for a, b in pairs if ident.get(a) == b or ident.get(b) == a]
    pairs = [x for x in pairs if x not in same]
    rows = [dict(family=name, **compare(M, a, b)) for a, b in pairs if a in M and b in M]
    if not rows:
        return pd.DataFrame()
    F = pd.DataFrame(rows)
    F["p_t_holm"], F["p_wilcoxon_holm"] = holm(F.p_t), holm(F.p_wilcoxon)
    F["verdict"] = np.where((F.p_t_holm < 0.05) & ((F.ci_low > 0) | (F.ci_high < 0)),
                            np.where(F.mean_diff < 0, "A lower MAE", "A higher MAE"), "not distinguishable")
    if same:
        F = pd.concat([F, pd.DataFrame([dict(family=name, A=disp(a), B=disp(b), mean_diff=0.0,
                                             verdict="identical views; difference 0 by construction (excluded from Holm)") for a, b in same])],
                      ignore_index=True)
    return F


def main():
    bb = sys.argv[1] if len(sys.argv) > 1 else "cgcnn"
    seeds = CFG["backbones"][bb]["seeds"]
    if "--seeds" in sys.argv:
        lo, hi = map(int, sys.argv[sys.argv.index("--seeds") + 1].split("-")); seeds = list(range(lo, hi + 1))
    O.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(Path(f).read_text()) for f in glob.glob(str(R / f"results/final/fits/{bb}__*.json"))]
    D = pd.DataFrame(recs)
    D = D[D.seed.isin(seeds)]
    M = D.pivot(index="seed", columns="view", values="MAE")
    ident = CFG["views"].get("identical_views", {})
    for a, b in ident.items():                                   # identical views: metrics equal by construction
        if b in M:
            M[a] = M[b]
    v = CFG["views"]; taus = v["taus"]; orders = v["orders"]; ung = v["ungated_partner"]
    F1 = family(M, [(f"{g}@{t}", f"{ung[g]}@{t}") for t in taus for g in orders], "F1_gate_effect")
    F2 = family(M, [(f"{a}@{t}", f"{b}@{t}") for t in taus for a, b in combinations(orders, 2)], "F2_ordering_under_gate")
    F3 = family(M, [(f"{g}@{t}", "Original-4.5") for t in taus for g in orders], "F3_vs_full_graph")
    if len(F3):
        F3["equivalent_D14"] = (F3.ci_low > -DELTA) & (F3.ci_high < DELTA)
        F3["outside_margin"] = (F3.ci_low > DELTA) | (F3.ci_high < -DELTA)
        F3["unresolved"] = ~F3.equivalent_D14 & ~F3.outside_margin
    desc = pd.DataFrame([dict(family="descriptive_vs_full", **compare(M, x, "Original-4.5")) for x in M.columns if x != "Original-4.5"
                         and not any(x == f"{g}@{t}" for g in orders for t in taus)])
    tag = f"{bb}_seeds{min(seeds)}-{max(seeds)}"
    for name, T in (("F1", F1), ("F2", F2), ("F3", F3), ("descriptive", desc)):
        T.to_csv(O / f"{tag}_{name}.csv", index=False)
    summ = D.groupby("view")[["MAE", "RMSE", "R2", "Spearman", "Kendall", "best_epoch"]].agg(["mean", "std", "count"])
    summ.index = [disp(i) for i in summ.index]; summ.to_csv(O / f"{tag}_summary.csv")
    out = dict(backbone=bb, seeds=seeds, n_fits=len(D), complete_views=int((M.notna().sum() == len(seeds)).sum()))
    if bb == "cgcnn" and len(F3):
        out["D15_extension_triggered"] = bool(F3.unresolved.any())
        out["D15_unresolved"] = F3[F3.unresolved].A.tolist()
    # Pareto per tau (structural axes + CGCNN MAE; cost added when the benchmark exists)
    st = R / "results/final/structural/structural_tradeoff.csv"
    if bb == "cgcnn" and st.exists():
        S = pd.read_csv(st, index_col=0)
        cost_f = R / "results/final/cost/cost_summary.csv"
        C = pd.read_csv(cost_f, index_col=0) if cost_f.exists() else None
        rows = []
        for t in taus:
            vs = [x for x in (f"{p}@{t}" for g in orders for p in (g, ung[g])) if x in M]
            X = pd.DataFrame({"prot": S.loc[vs, "protected_retention_tol040_pct"], "shell": S.loc[vs, "shell_complete_r3.0_pct"],
                              "mae": M[vs].mean()})
            if C is not None:
                X["cost"] = C.loc[vs, "cgcnn_train_seconds_per_epoch"]
            sign = {"prot": 1, "shell": 1, "mae": -1, "cost": -1}
            for a in X.index:
                dom = any(all(sign[c] * X.loc[b, c] >= sign[c] * X.loc[a, c] for c in X.columns) and
                          any(sign[c] * X.loc[b, c] > sign[c] * X.loc[a, c] for c in X.columns) for b in X.index if b != a)
                rows.append(dict(tau=t, view=disp(a), **X.loc[a].to_dict(), pareto_nondominated=not dom))
        P = pd.DataFrame(rows); P.to_csv(O / f"{tag}_pareto.csv", index=False)
        out["pareto_axes"] = list(X.columns)
    (O / f"{tag}_decision.json").write_text(json.dumps(out, indent=2))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    for T in (F1, F2, F3):
        if len(T):
            print(T[["family", "A", "B", "n_pairs", "mean_diff", "ci_low", "ci_high", "p_t_holm", "p_wilcoxon_holm", "verdict"]].round(4).to_string(index=False))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
