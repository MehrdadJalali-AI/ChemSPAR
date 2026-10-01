"""Results tables, macros and Figure 2 from the final analysis (no hand-typed numbers).
Inputs: results/final/fits/*.json, results/final/analysis/cgcnn_seeds1-5_{F1,F2,F3,descriptive,pareto}.csv,
        results/final/cost/cost_summary.csv, results/final/cost/prep_rep*.csv, results/final/structural/structural_tradeoff.csv
Outputs: manuscript/tables/{tab_pred,tab_tests,tab_cost,si_perseed,si_tests_full,si_cost_stages,results_macros}.tex,
         figures/fig2_results.{pdf,png} (copied to manuscript/figures)
"""
import glob
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

R = Path(__file__).resolve().parents[1]
T = R / "manuscript/tables"
A = R / "results/final/analysis"
CFG = yaml.safe_load((R / "configs/final_v1.yaml").read_text())
INV = {k: d for d, k in CFG["display_names"].items()}
ORDER = ["Original-4.5", "Cutoff-3.5", "Distance-Chem@0.5", "ChemSPAR-v2@0.5", "Random-Chem@0.5", "Random@0.5", "BHS-edge-min@0.5", "BHS-edge-min-noGate@0.5"]


def disp(v):
    return v if "@" not in v else INV[v.split("@")[0]]


def w(name, rows):
    (T / name).write_text("\n".join(rows) + "\n\\bottomrule\n")


def bname(b):
    return b.split('@')[0].replace('Original-4.5', 'full graph')


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main():
    D = pd.DataFrame([json.loads(Path(f).read_text()) for f in glob.glob(str(R / "results/final/fits/*.json"))])
    assert len(D) == 40
    g = D.groupby("view")
    rows = []
    for v in ORDER:
        x = g.get_group(v)
        gate = "" if "@" not in v else ("yes" if v.split("@")[0] in CFG["views"]["orders"] else "no")
        rows.append(f"{disp(v)} & {gate} & {x.MAE.mean():.3f} $\\pm$ {x.MAE.std():.3f} & {x.RMSE.mean():.3f} & {x.R2.mean():.3f} & {x.Spearman.mean():.3f} \\\\")
    w("tab_pred.tex", rows)
    F = pd.concat([pd.read_csv(A / f"cgcnn_seeds1-5_{k}.csv") for k in ("F1", "F2", "F3")])
    fam = {"F1_gate_effect": "Gate effect", "F2_ordering_under_gate": "Ordering (gated)", "F3_vs_full_graph": "vs.\\ full graph"}
    rows = []
    for r in F.itertuples():
        ver = {"A lower MAE": "lower", "A higher MAE": "higher", "not distinguishable": "n.d."}[r.verdict]
        eq = ""
        if r.family == "F3_vs_full_graph":
            eq = " (equiv.)" if r.equivalent_D14 else ""
        rows.append(f"{fam[r.family]} & {r.A.split('@')[0]} -- {bname(r.B)} & {r.mean_diff:+.4f} & [{r.ci_low:+.4f}, {r.ci_high:+.4f}] & {r.d_z:.2f} & {pfmt(r.p_t_holm)} & {r.n_A_better}/5 & {ver}{eq} \\\\")
    w("tab_tests.tex", rows)
    rows = [f"{fam[r.family]} & {r.A.split('@')[0]} -- {r.B.split('@')[0]} & {pfmt(r.p_t)} & {pfmt(r.p_t_holm)} & {r.p_wilcoxon:.4f} & {r.p_wilcoxon_holm:.4f} \\\\" for r in F.itertuples()]
    w("si_tests_full.tex", rows)
    rows = []
    for v in ORDER:
        x = g.get_group(v).sort_values("seed")
        rows.append(f"{disp(v)}" + "".join(f" & {m:.4f}" for m in x.MAE) + f" & {x.best_epoch.mean():.0f} \\\\")
    w("si_perseed.tex", rows)
    C = pd.read_csv(R / "results/final/cost/cost_summary.csv", index_col=0)
    rows = [f"{disp(v)} & {C.loc[v].prep_seconds_per_structure_mean:.3f} & {C.loc[v].prep_hours_5000_extrapolated:.2f} & {C.loc[v].cgcnn_train_seconds_per_epoch:.1f} $\\pm$ {C.loc[v].cgcnn_train_seconds_per_epoch_sd:.1f} & "
            f"{C.loc[v].cgcnn_full_budget_hours_computed:.2f} & {C.loc[v].inference_seconds_test:.2f} & {C.loc[v].peak_rss_mb / 1000:.1f} \\\\" for v in ORDER]
    w("tab_cost.tex", rows)
    P = pd.concat([pd.read_csv(f).assign(rep=k) for k, f in enumerate(sorted(glob.glob(str(R / "results/final/cost/prep_rep*.csv"))))])
    st = [c for c in P.columns if c not in ("qmof_id", "n_instances", "rep") and not c.startswith("tensorise_")]
    per = P.groupby("rep")[st].sum() / P.qmof_id.nunique()
    nice = {"graph_4p5": "contact instances, 4.5 Å", "graph_3p5": "contact instances, 3.5 Å", "crystalnn": "CrystalNN neighbours", "taxonomy": "taxonomy",
            "score_gravity": "gravity score", "score_distance": "distance order", "score_random": "random order", "score_bhs": "BHS-derived score"}
    rows = [f"{nice.get(c, c.replace('prune_', 'pruning, ').replace('ChemSPAR-v2', 'Gravity-Chem').replace('BHS-edge-min-noGate', 'BHS').replace('BHS-edge-min', 'BHS-Chem'))} & {1000 * per[c].mean():.2f} & {1000 * per[c].std():.2f} \\\\" for c in st]
    w("si_cost_stages.tex", rows)
    # ---- macros for the text
    f3 = pd.read_csv(A / "cgcnn_seeds1-5_F3.csv").set_index("A"); f1 = pd.read_csv(A / "cgcnn_seeds1-5_F1.csv").set_index("A")
    f2 = pd.read_csv(A / "cgcnn_seeds1-5_F2.csv")
    m = {}
    mae = g.MAE.mean(); sd = g.MAE.std()
    for v, k in [("Original-4.5", "Full"), ("Cutoff-3.5", "CutA"), ("Distance-Chem@0.5", "DC"), ("ChemSPAR-v2@0.5", "GC"), ("Random-Chem@0.5", "RC"),
                 ("Random@0.5", "RU"), ("BHS-edge-min@0.5", "BC"), ("BHS-edge-min-noGate@0.5", "BU")]:
        m[f"Mae{k}"] = f"{mae[v]:.3f}"; m[f"Sd{k}"] = f"{sd[v]:.3f}"
    def ci(r):
        return f"{r.mean_diff:+.4f}", f"{r.ci_low:+.4f}", f"{r.ci_high:+.4f}", pfmt(r.p_t_holm)
    for key, r in [("GateR", f1.loc["Random-Chem@0.5"]), ("GateB", f1.loc["BHS-Chem@0.5"]), ("FullDC", f3.loc["Distance-Chem@0.5"]),
                   ("FullGC", f3.loc["Gravity-Chem@0.5"]), ("FullRC", f3.loc["Random-Chem@0.5"]), ("FullBC", f3.loc["BHS-Chem@0.5"])]:
        a, b, c, p = ci(r)
        m[f"D{key}"], m[f"L{key}"], m[f"H{key}"], m[f"P{key}"] = a, b, c, p
    r = f2[(f2.A == "Gravity-Chem@0.5") & (f2.B == "Distance-Chem@0.5")].iloc[0]
    m["DGCvsDC"], m["LGCvsDC"], m["HGCvsDC"], m["PGCvsDC"] = ci(r)
    for k in ("GateR", "GateB", "FullDC", "GCvsDC"):
        m[f"Abs{k}"] = f"{abs(float(m[f'D{k}'])):.4f}"
    m["PFtwoMax"] = pfmt(f2.p_t_holm.max())
    m["TrainFull"] = f"{C.loc['Original-4.5'].cgcnn_train_seconds_per_epoch:.1f}"
    m["TrainHalfMin"] = f"{C.loc[ORDER[2:]].cgcnn_train_seconds_per_epoch.min():.1f}"; m["TrainHalfMax"] = f"{C.loc[ORDER[2:]].cgcnn_train_seconds_per_epoch.max():.1f}"
    m["TrainSdMax"] = f"{C.loc[ORDER[2:]].cgcnn_train_seconds_per_epoch_sd.max():.1f}"
    m["PrepGatedMin"] = f"{C.loc[[v for v in ORDER[2:] if v.split('@')[0] in CFG['views']['orders']]].prep_seconds_per_structure_mean.min():.2f}"
    m["PrepGatedMax"] = f"{C.loc[[v for v in ORDER[2:] if v.split('@')[0] in CFG['views']['orders']]].prep_seconds_per_structure_mean.max():.2f}"
    m["CnnShareMin"] = f"{C.loc[[v for v in ORDER[2:] if C.loc[v].crystalnn_share_pct > 0]].crystalnn_share_pct.min():.0f}"
    m["CnnShareMax"] = f"{C.loc[[v for v in ORDER[2:] if C.loc[v].crystalnn_share_pct > 0]].crystalnn_share_pct.max():.0f}"
    m["RssFull"] = f"{C.loc['Original-4.5'].peak_rss_mb / 1000:.1f}"; m["RssDC"] = f"{C.loc['Distance-Chem@0.5'].peak_rss_mb / 1000:.1f}"
    m["HoursFull"] = f"{C.loc['Original-4.5'].cgcnn_full_budget_hours_computed:.2f}"; m["HoursDC"] = f"{C.loc['Distance-Chem@0.5'].cgcnn_full_budget_hours_computed:.2f}"
    m["HoursSaved"] = f"{C.loc['Original-4.5'].cgcnn_full_budget_hours_computed - C.loc['Distance-Chem@0.5'].cgcnn_full_budget_hours_computed:.2f}"
    m["PrepHoursDC"] = f"{C.loc['Distance-Chem@0.5'].prep_hours_5000_extrapolated:.2f}"
    m["CostDiffRCDC"] = f"{C.loc['Distance-Chem@0.5'].cgcnn_train_seconds_per_epoch - C.loc['Random-Chem@0.5'].cgcnn_train_seconds_per_epoch:.2f}"
    m["DescCut"], m["LDescCut"], m["HDescCut"] = [f"{x:+.4f}" for x in pd.read_csv(A / "cgcnn_seeds1-5_descriptive.csv").set_index("A").loc["Cutoff-3.5", ["mean_diff", "ci_low", "ci_high"]]]
    (T / "results_macros.tex").write_text("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))
    fig2(D, F, C)


def fig2(D, F, C):
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    S = pd.read_csv(R / "results/final/structural/structural_tradeoff.csv", index_col=0)
    col = {"Original-4.5": "#555555", "Cutoff-3.5": "#999999", "Distance-Chem@0.5": "#2a9d8f", "ChemSPAR-v2@0.5": "#3f72af",
           "Random-Chem@0.5": "#e3b23c", "Random@0.5": "#e3b23c", "BHS-edge-min@0.5": "#d1495b", "BHS-edge-min-noGate@0.5": "#d1495b"}
    fig = plt.figure(figsize=(7.09, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.62, wspace=0.32, width_ratios=[1.15, 1.0])
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, :]), fig.add_subplot(gs[0, 1])]
    # a: paired per-seed MAE
    a = ax[0]
    M = D.pivot(index="seed", columns="view", values="MAE")[ORDER]
    for s, row in M.iterrows():
        a.plot(range(len(ORDER)), row.values, color="#cccccc", lw=0.5, zorder=1)
    for k, v in enumerate(ORDER):
        gated = "@" in v and v.split("@")[0] in CFG["views"]["orders"]
        fc = col[v] if (gated or "@" not in v) else "white"
        a.scatter([k] * len(M), M[v], s=12, fc=fc, ec=col[v], lw=0.8, zorder=2)
        a.plot([k - 0.28, k + 0.28], [M[v].mean()] * 2, color="k", lw=1.0, zorder=3)
    a.set_xticks(range(len(ORDER)), [disp(v) for v in ORDER], rotation=55, ha="right")
    a.set_ylabel("Test MAE (eV)"); a.set_title("Bandgap error, 5 paired seeds", fontsize=7)
    a.spines[["top", "right"]].set_visible(False)
    a.text(-0.16, 1.06, "a", transform=a.transAxes, fontsize=9, fontweight="bold")
    # b: forest plot
    b = ax[1]
    fam = ["F1_gate_effect", "F3_vs_full_graph", "F2_ordering_under_gate"]
    lab = {"F1_gate_effect": "gated − ungated", "F3_vs_full_graph": "gated − full graph", "F2_ordering_under_gate": "gated − gated"}
    rows = []
    for f in fam:
        for r in F[F.family == f].itertuples():
            rows.append((f"{r.A.split('@')[0]} − {r.B.split('@')[0]}".replace("Original-4.5", "Full"), r.mean_diff, r.ci_low, r.ci_high, r.verdict, f))
    y = np.arange(len(rows))[::-1]
    b.axvspan(-0.02, 0.02, color="#eef3f8", zorder=0)
    b.axvline(0, color="#777777", lw=0.6)
    for yy, (n, m_, lo, hi, ver, f) in zip(y, rows):
        c = "#2a9d8f" if ver == "A lower MAE" else ("#d1495b" if ver == "A higher MAE" else "#888888")
        b.plot([lo, hi], [yy, yy], color=c, lw=1.2); b.scatter([m_], [yy], s=12, color=c, zorder=3)
    b.set_yticks(y, [r[0] for r in rows], fontsize=6)
    for k, f in enumerate(fam):
        ys = [yy for yy, r in zip(y, rows) if r[5] == f]
        b.text(1.01, np.mean(ys), lab[f], transform=b.get_yaxis_transform(), fontsize=6.3, va="center", ha="left", color="#555555", style="italic")
        if k < len(fam) - 1:
            b.axhline(min(ys) - 0.5, color="#bbbbbb", lw=0.5)
    b.set_ylim(-0.7, len(rows) - 0.3)
    b.text(0.995, 0.02, "green: A has lower MAE · red: A higher · grey: not distinguishable (Holm-adjusted)\nshaded: ±0.02 eV equivalence margin", transform=b.transAxes, fontsize=5.8, color="#555555", ha="right", va="bottom")
    b.set_xlabel("Paired ΔMAE, A − B (eV), mean and 95% CI"); b.set_title("Pre-registered comparisons (Holm)", fontsize=7)
    b.spines[["top", "right"]].set_visible(False)
    b.text(-0.30, 1.04, "c", transform=b.transAxes, fontsize=9, fontweight="bold")
    # c: trade-off: shell completeness vs MAE
    c = ax[2]
    for v in ORDER[2:]:
        gated = v.split("@")[0] in CFG["views"]["orders"]
        c.scatter(S.loc[v, "shell_complete_r3.0_pct"], D[D.view == v].MAE.mean(), s=10 + 0.4 * S.loc[v, "protected_retention_tol040_pct"],
                  fc=col[v] if gated else "white", ec=col[v], lw=0.9, zorder=3)
        xr = S.loc[v, "shell_complete_r3.0_pct"]
        c.annotate(disp(v), (xr, D[D.view == v].MAE.mean()), xytext=(-5, 5) if xr > 80 else (5, 2), ha="right" if xr > 80 else "left", textcoords="offset points", fontsize=5.8)
    c.axhline(D[D.view == "Original-4.5"].MAE.mean(), color="#555555", lw=0.6, ls="--")
    c.text(100, D[D.view == "Original-4.5"].MAE.mean() + 0.002, "full graph", ha="right", fontsize=5.8, color="#555555")
    c.set_xlabel("Atoms with complete 3.0 Å shell (%)"); c.set_ylabel("Mean test MAE (eV)")
    c.set_title("Trade-off at τ = 0.5", fontsize=7); c.set_xlim(-8, 112)
    c.spines[["top", "right"]].set_visible(False)
    c.text(-0.2, 1.06, "b", transform=c.transAxes, fontsize=9, fontweight="bold")
    c.text(0.98, 0.70, "filled: gated · open: ungated\nmarker size: share of\nprotected bonds kept", transform=c.transAxes, fontsize=5.8, color="#555555", ha="right", va="top")
    out = R / "figures/fig2_results"
    fig.savefig(f"{out}.pdf", bbox_inches="tight"); fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight")
    shutil.copy(f"{out}.pdf", R / "manuscript/figures/fig2_results.pdf")


if __name__ == "__main__":
    main()
