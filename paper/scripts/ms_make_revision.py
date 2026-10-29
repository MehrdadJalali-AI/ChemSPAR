"""Revision round 1: tables and macros generated only from saved outputs (no hand-typed numbers).
Inputs: results/revision/{sweep_summary.csv, weights_summary.csv, voronoi_construction.csv, posthoc_distance_vs_cutoff35.json,
        audit_full/*.parquet}, results/final/check/check_results.json, results/final/cost/cost_summary.csv,
        results/final/audit/rejections_tau0.5.csv
Outputs: manuscript/tables/{revision_macros.tex, tab_sweep.tex, si_sweep_full.tex, si_weights.tex, tab_audit_rejections.tex}
         and the Voronoi row appended to tab_constructions.tex (after ms_make_tables.py has written it)
"""
import glob
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

R = Path(__file__).resolve().parents[1]
T = R / "manuscript/tables"
V = R / "results/revision"
NAME = {"ChemSPAR-v2": ("Gravity", "yes"), "ChemSPAR-v2-noGate": ("Gravity", "no"), "Distance-Chem": ("Distance", "yes"), "Distance": ("Distance", "no"),
        "Random-Chem": ("Random", "yes"), "Random": ("Random", "no"), "BHS-edge-min": ("BHS-derived", "yes"), "BHS-edge-min-noGate": ("BHS-derived", "no")}
ORDER = list(NAME)


def w(name, rows):
    (T / name).write_text("\n".join(rows) + "\n\\bottomrule\n")


def main():
    m = {}
    # ---- route counts (R1.2, R2.4)
    rt = json.loads((R / "results/final/check/check_results.json").read_text())["route"]["by_class_route"]
    ml = {k.split("|")[1]: v for k, v in rt.items() if k.startswith("metal_ligand|")}
    mm = {k.split("|")[1]: v for k, v in rt.items() if k.startswith("metal_metal_crystalnn|")}
    tot_prot = sum(rt.values())
    m.update(NMLtotal=f"{sum(ml.values()):,}", NMLboth=f"{ml['both']:,}", NMLcnnOnly=f"{ml['crystalnn']:,}", NMLradOnly=f"{ml['radius']:,}",
             NMMcnnOnly=f"{mm.get('crystalnn', 0):,}", PctCnnOnly=f"{100 * (ml['crystalnn'] + mm.get('crystalnn', 0)) / tot_prot:.2f}")
    # ---- post hoc (R2.2)
    ph = json.loads((V / "posthoc_distance_vs_cutoff35.json").read_text())
    m.update(PHdiff=f"{ph['mean_diff']:+.4f}", PHlo=f"{ph['ci_low']:+.4f}", PHhi=f"{ph['ci_high']:+.4f}", PHnbetter=str(ph["n_distance_lower"]),
             PHp=f"{ph['p_t']:.2f}")
    # ---- memory range (leftover)
    C = pd.read_csv(R / "results/final/cost/cost_summary.csv", index_col=0)
    half = C.loc[[v for v in C.index if "@0.5" in v]].peak_rss_mb / 1000
    m.update(RssHalfMin=f"{half.min():.1f}", RssHalfMax=f"{half.max():.1f}")
    # ---- sweep (R1.3)
    S = pd.read_csv(V / "sweep_summary.csv")
    rows = []
    for pol in ORDER:
        x = S[(S.policy == pol) & (S.tau.isin([0.8, 0.9, 0.95]))].set_index("tau")
        n, g = NAME[pol]
        rows.append(f"{n} & {g} & " + " & ".join(f"{x.loc[t, 'achieved_fraction_pct']:.1f} & {int(x.loc[t, 'structures_short']):,} & {x.loc[t, 'protected_retention_pct']:.1f} & {x.loc[t, 'shell_complete_pct']:.1f}"
                                                  for t in (0.8, 0.9, 0.95)) + " \\\\")
    w("tab_sweep.tex", rows)
    rows = []
    for pol in ORDER:
        for t in (0.5, 0.8, 0.9, 0.95):
            r = S[(S.policy == pol) & (S.tau == t)].iloc[0]; n, g = NAME[pol]
            rows.append(f"{n} & {g} & {t} & {int(r.target):,} & {int(r.removed):,} & {int(r.structures_short):,} & {int(r.attempts):,} & {int(r.rejected):,} & {r.protected_retention_pct:.2f} & {r.shell_complete_pct:.2f} \\\\")
    w("si_sweep_full.tex", rows)
    g = S[S.policy.isin(["ChemSPAR-v2", "Distance-Chem", "Random-Chem", "BHS-edge-min"])].set_index(["policy", "tau"])
    u = S.set_index(["policy", "tau"])
    assert g.xs(0.95, level="tau").achieved_fraction_pct.nunique() == 1
    m.update(SwShortEight=f"{int(g.loc[('Distance-Chem', 0.8), 'structures_short']):,}", SwShortNine=f"{int(g.loc[('Distance-Chem', 0.9), 'structures_short']):,}",
             SwShortNineFive=f"{int(g.loc[('Distance-Chem', 0.95), 'structures_short']):,}", SwCeil=f"{g.loc[('Distance-Chem', 0.95), 'achieved_fraction_pct']:.1f}",
             SwUngCeil=f"{u.loc[('Distance', 0.95), 'achieved_fraction_pct']:.1f}",
             SwRandEight=f"{u.loc[('Random', 0.8), 'protected_retention_pct']:.1f}", SwBhsEight=f"{u.loc[('BHS-edge-min-noGate', 0.8), 'protected_retention_pct']:.1f}",
             SwDistEight=f"{u.loc[('Distance', 0.8), 'protected_retention_pct']:.1f}", SwGravEight=f"{u.loc[('ChemSPAR-v2-noGate', 0.8), 'protected_retention_pct']:.1f}",
             SwDistNine=f"{u.loc[('Distance', 0.9), 'protected_retention_pct']:.1f}", SwGravNine=f"{u.loc[('ChemSPAR-v2-noGate', 0.9), 'protected_retention_pct']:.1f}",
             SwDistNineFive=f"{u.loc[('Distance', 0.95), 'protected_retention_pct']:.1f}", SwGravNineFive=f"{u.loc[('ChemSPAR-v2-noGate', 0.95), 'protected_retention_pct']:.1f}",
             SwShellDCEight=f"{g.loc[('Distance-Chem', 0.8), 'shell_complete_pct']:.1f}", SwShellDCNine=f"{g.loc[('Distance-Chem', 0.9), 'shell_complete_pct']:.1f}")
    # ---- full audit logs (R1.5)
    files = sorted(glob.glob(str(V / "audit_full/*.parquet")))
    nrows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    nbytes = sum(Path(f).stat().st_size for f in files)
    m.update(AuditFullRows=f"{nrows:,}", AuditFullMB=f"{nbytes / 1e6:.0f}", AuditFullBytesPerRow=f"{nbytes / nrows:.1f}",
             AuditFullPerStructPolicyKB=f"{nbytes / 5000 / 8 / 1e3:.1f}")
    # ---- Table 9 with total attempts (R2.5)
    A = pd.read_csv(R / "results/final/audit/rejections_tau0.5.csv", index_col=0)
    prot = [c for c in A.columns if c.startswith("rej::protected")]
    disp = {"ChemSPAR-v2": "Gravity-Chem", "Distance-Chem": "Distance-Chem", "Random-Chem": "Random-Chem", "BHS-edge-min": "BHS-Chem", "Random": "Random", "BHS-edge-min-noGate": "BHS"}
    rows = []
    for k in disp:
        r = A.loc[k]
        att = int(r.removed + r[[c for c in A.columns if c.startswith("rej::")]].sum())
        sw = int(S[(S.policy == k) & (S.tau == 0.5)].attempts.iloc[0])
        assert att == sw, (k, att, sw)            # stored counts == recomputed full logs
        rows.append(f"{disp[k]} & {int(r[prot].sum()):,} & {int(r['rej::degree']):,} & {int(r['rej::connectivity']):,} & {int(r.removed):,} & {att:,} \\\\")
    w("tab_audit_rejections.tex", rows)
    # ---- weights (R1.7)
    W = V / "weights_summary.csv"
    if W.exists():
        Wd = pd.read_csv(W)
        pert = Wd[Wd.variant != "unperturbed"]
        unp = Wd[Wd.variant == "unperturbed"].iloc[0]
        assert abs(unp.mean_jaccard - 1) < 1e-12 and unp.min_jaccard == 1
        import yaml
        sc = yaml.safe_load((R / "configs/chemistry_v1.yaml").read_text())["score"]
        nice = {"degree": "degree ($w_1$)", "betweenness": "betweenness ($w_2$)", "closeness": "closeness ($w_3$)",
                "neighbour_distance": "summed contact distance ($w_4$)", "z": "atomic number ($w_5$)", "role": "element role ($w_6$)"}
        rows = []
        order = ["degree", "betweenness", "closeness", "neighbour_distance", "z", "role", "hybrid_lambda"]
        pert = pert.assign(_k=[order.index(v.replace("hybrid_lambda ", "hybrid_lambda").split("x")[0]) for v in pert.variant]).sort_values(["_k", "variant"])
        for r in pert.itertuples():
            key, f = r.variant.replace("hybrid_lambda ", "hybrid_lambda").split("x")
            f = float(f)
            if key == "hybrid_lambda":
                lam = sc["hybrid_lambda"]
                name, base, new = "$d^{\\mathrm{hyb}}$ split $(\\lambda, 1-\\lambda)$", f"({lam:.2f}, {1 - lam:.2f})", f"({lam * f:.2f}, {1 - lam * f:.2f})"
            else:
                b = sc["node_mass_weights"][key]; name, base, new = nice[key], f"{b:.2f}", f"{b * f:.2f}"
            rows.append(f"{name} & {base} & {new} & {r.mean_jaccard:.4f} & {r.p05_jaccard:.4f} & {r.min_jaccard:.4f} & {int(r.protected_removed)} \\\\")
        w("si_weights.tex", rows)
        mass = pert[~pert.variant.str.startswith("hybrid_lambda")]; split = pert[pert.variant.str.startswith("hybrid_lambda")]
        m.update(WtMassMin=f"{mass.mean_jaccard.min():.4f}", WtMassMax=f"{mass.mean_jaccard.max():.4f}",
                 WtSplitMin=f"{split.mean_jaccard.min():.4f}", WtSplitMax=f"{split.mean_jaccard.max():.4f}",
                 WtProtRemoved=str(int(pert.protected_removed.max())))
    # ---- periodic dimensionality (final corrections, item 3)
    Dd = V / "periodic_dimensionality_summary.csv"
    if Dd.exists():
        Dm = pd.read_csv(Dd).set_index("view")
        full = pd.read_csv(V / "periodic_dimensionality.csv")["Original-4.5|dim_atom0"]
        assert (full == 3).all()
        for v, k in [("Distance-Chem@0.5", "DC"), ("ChemSPAR-v2@0.5", "GC"), ("Random-Chem@0.5", "RC"), ("BHS-edge-min@0.5", "BC"),
                     ("Distance@0.5", "DU"), ("ChemSPAR-v2-noGate@0.5", "GU"), ("Random@0.5", "RU"), ("BHS-edge-min-noGate@0.5", "BU"), ("Cutoff-3.5", "Cut")]:
            m[f"Dim{k}"] = f"{int(Dm.loc[v, 'lowered_dim_atom0']):,}"
            assert Dm.loc[v, "lowered_dim_atom0"] == Dm.loc[v, "changed_dim_atom0"] == Dm.loc[v, "changed_dim_max"] and Dm.loc[v, "multi_component"] == 0
    # ---- Voronoi (R1.1)
    Vo = V / "voronoi_construction.csv"
    if Vo.exists():
        v = pd.read_csv(Vo).iloc[0]
        line = f"Voronoi & {v.mean_instances:,.1f} & {v.pct_of_full:.1f} & {v.connected_pct:.1f} & {v.covalent_radius_kept_pct:.2f} & {v.metal_ligand_kept_pct:.2f} & {v.metal_metal_kept_pct:.1f} \\\\"
        tc = (T / "tab_constructions.tex").read_text()
        if "Voronoi" not in tc:
            (T / "tab_constructions.tex").write_text(tc.replace("\\bottomrule", line + "\n\\bottomrule"))
        m["VoronoiSentence"] = (f"The parameter-free Voronoi construction keeps {v.pct_of_full:.1f}\\% as many instances as the full graph "
                                f"({int(v.instances_beyond_4p5):,} of them beyond 4.5\\,\\AA{{}}), {v.covalent_radius_kept_pct:.2f}\\% of radius-bonded non-metal and "
                                f"{v.metal_ligand_kept_pct:.2f}\\% of metal--ligand contacts, and loses protected contacts in {int(v.structures_losing_protected):,} structures"
                                + (f" ({int(v.failed)} structures could not be tessellated)" if int(v.failed) else "") + ".")
    else:
        m["VoronoiSentence"] = ""
    (T / "revision_macros.tex").write_text("% generated by scripts/ms_make_revision.py\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))
    print(json.dumps(m, indent=1))


if __name__ == "__main__":
    main()
