"""Supporting-information tables generated only from result files. Output: manuscript/tables/si_*.tex"""
from pathlib import Path

import pandas as pd
import yaml

R = Path(__file__).resolve().parents[1]
T = R / "manuscript/tables"
CFG = yaml.safe_load((R / "configs/final_v1.yaml").read_text())


def w(name, rows):
    (T / name).write_text("\n".join(rows) + "\n\\bottomrule\n")


def main():
    S = pd.read_csv(R / "results/final/structural/structural_tradeoff.csv", index_col=0)
    rows = []
    for v, r in S.iterrows():
        g = "" if pd.isna(r.gated) else ("gated" if r.gated in (True, "True") else "ungated")
        rows.append(f"{r.display_name} & {g} & {r.retained_instances_pct:.1f} & {r.protected_retention_tol025_pct:.3f} & {r.protected_retention_tol040_pct:.3f} & "
                    f"{r.protected_retention_tol055_pct:.3f} & {r['shell_complete_r2.5_pct']:.1f} & {r['shell_complete_r3.0_pct']:.1f} & "
                    f"{r['shell_complete_r3.5_pct']:.1f} & {r.connected_structures_pct:.1f} \\\\")
    w("si_structural_all.tex", rows)
    rows = []
    for v, r in S.iterrows():
        rows.append(f"{r.display_name} & " + " & ".join(f"{r[f'loss_frac_{k}']:.2f}" for k in ("H", "C", "N", "O", "metal", "other")) +
                    (f" & {r.removed_H_involving_pct:.1f}" if not pd.isna(r.removed_H_involving_pct) else " & --") + " \\\\")
    w("si_role_loss.tex", rows)
    H = pd.read_csv(R / "results/phase2i/p2i_hbprotect_vs_chemspar.csv")
    w("si_hbprotect.tex", [f"{r.tau} & {r.structures_differing:,} & {100 * r.frac_differing_among_hb_structures:.1f} & {r.mean_jaccard_removed:.4f} & "
                           f"{r.hb_removed_ChemSPAR:,} & {r.hb_removed_HBProtect:,} \\\\" for r in H.itertuples()])
    C = pd.read_csv(R / "results/phase2i/p2i_chemical_validity.csv")
    names = {"ChemSPAR-v2-massfree": "Gravity-Chem, mass-free ($1/d_\\mathrm{hyb}^2$)", "ChemSPAR-v2-chemmass": "Gravity-Chem, chemical mass only",
             "ChemSPAR-v2-topomass": "Gravity-Chem, topological mass only", "ChemSPAR-v2-HBProtect": "Gravity-Chem + HB protection",
             "BHS-edge-prod": "BHS-Chem, product score"}
    C = C[C.policy.isin(names) & C.tau.isin([0.2, 0.5])]
    w("si_variants.tex", [f"{names[r.policy]} & {r.tau} & {r.protected_removed_tol055:,} & {r.crystalnn_retained_pct:.2f} & {r.hb_candidates_retained_pct:.1f} & "
                          f"{r.metal_contact_removed_pct:.1f} \\\\" for r in C.itertuples()])
    B = pd.read_csv(R / "results/final/budget_extension/budget_extension_summary.csv")
    inv = {k: d for d, k in CFG["display_names"].items()}
    lab = lambda v: v if "@" not in v else f"{inv[v.split('@')[0]]}@{v.split('@')[1]}"
    w("si_budget.tex", [f"{lab(r.view)} & {r.best_val_mae:.4f} & {r.best_epoch} & {r.first_epoch_within_1pct} \\\\" for r in B.itertuples()])
    print("ok")


if __name__ == "__main__":
    main()
