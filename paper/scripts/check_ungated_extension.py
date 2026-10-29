"""After adding ungated partners (Random/Distance extended to tau 0.9; new BHS-edge-min-noGate): verify that every
pre-existing column is identical to the previous build, and that the Random@0.2 / Distance@0.2 removed sets are
unchanged (prefix property). Output: results/phase2i/ungated_extension_check.csv"""
import glob
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]
A = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(R / "results/phase2i/views_prev_before_ungated_extension/inst_*.parquet")))).reset_index(drop=True)
B = pd.concat(pd.read_parquet(f) for f in sorted(glob.glob(str(R / "results/phase2i/views/inst_*.parquet")))).reset_index(drop=True)
key = ["qmof_id", "i", "j", "image"]
assert len(A) == len(B) and A[key].equals(B[key])
rows = [dict(check="new columns", value=",".join(sorted(set(B.columns) - set(A.columns)))),
        dict(check="dropped columns", value=",".join(sorted(set(A.columns) - set(B.columns))))]
for c in A.columns:
    if c in key:
        continue
    rows.append(dict(check=f"identical::{c}", value=bool(A[c].equals(B[c]))))
n = A.groupby("qmof_id").size()
for pol in ("Random", "Distance"):
    k = A.qmof_id.map((n * 0.2).astype(int))
    a = (A[f"rank__{pol}"] >= 0) & (A[f"rank__{pol}"] < k); b = (B[f"rank__{pol}"] >= 0) & (B[f"rank__{pol}"] < k)
    rows.append(dict(check=f"{pol}@0.2 removed set identical (all 5,000 structures)", value=bool((a == b).all())))
    rows.append(dict(check=f"{pol}: old ranks equal new ranks where old rank >= 0", value=bool((A[f'rank__{pol}'][A[f'rank__{pol}'] >= 0] == B[f'rank__{pol}'][A[f'rank__{pol}'] >= 0]).all())))
d = pd.DataFrame(rows); d.to_csv(R / "results/phase2i/ungated_extension_check.csv", index=False)
print(d[~d.check.str.startswith("identical::") | (d.value == False)].to_string(index=False))
print("changed pre-existing columns:", list(d[d.check.str.startswith("identical::") & (d.value == False)].check))
