"""Audit-log summary for the manuscript (no training). Sources:
  results/phase2i/views/audit_*.parquet  full per-attempt rows, first 200 cohort structures, all policies, tau up to 0.9
  results/phase2i/views/stats_*.parquet  per structure x policy x tau rejection counts, all 5,000 structures
Outputs (results/final/audit/): audit_storage.csv, rejections_tau0.5.csv, audit_example_rows.csv
"""
import glob
from pathlib import Path
import pandas as pd

R = Path(__file__).resolve().parents[1]
V = R / "results/phase2i/views"
O = R / "results/final/audit"
POLS = ["ChemSPAR-v2", "Distance-Chem", "Random-Chem", "Random", "BHS-edge-min", "BHS-edge-min-noGate"]


def main():
    O.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(V / "audit_*.parquet")))
    A = pd.concat(pd.read_parquet(f) for f in files)
    nbytes = sum(Path(f).stat().st_size for f in files)
    st = []
    for p in POLS:
        a = A[A.policy == p]
        st.append(dict(policy=p, structures=a.qmof_id.nunique(), attempt_rows_tau0p9=len(a), columns=A.shape[1],
                       rows_per_structure=len(a) / a.qmof_id.nunique()))
    S = pd.DataFrame(st); S["parquet_bytes_per_row_all_policies"] = nbytes / len(A)
    S.to_csv(O / "audit_storage.csv", index=False)
    T = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(V / "stats_*.parquet"))))
    T = T[(T.tau == 0.5) & T.view.isin(POLS)]
    rej = [c for c in T.columns if c.startswith("rej::")]
    G = T.groupby("view")[["target", "removed"] + rej].sum()
    G["rejected_total"] = G[rej].sum(1)
    G["structures"] = T.groupby("view").qmof_id.nunique()
    G["structures_target_not_reached"] = T.assign(short=T.removed < T.target).groupby("view").short.sum()
    G.to_csv(O / "rejections_tau0.5.csv")
    ex = A[(A.policy == "ChemSPAR-v2") & (A.qmof_id == A.qmof_id.iloc[0])]
    ex = pd.concat([ex[ex.decision == "rejected"].head(3), ex[ex.decision == "accepted"].head(2)])
    ex.to_csv(O / "audit_example_rows.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(S.round(1).to_string(index=False)); print(G.T.to_string())


if __name__ == "__main__":
    main()
