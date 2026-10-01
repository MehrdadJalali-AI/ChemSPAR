"""Phase 4 structural axes of the pre-specified trade-off (configs/final_v1.yaml `trade_off`), for all 21 trained views
on all 5,000 structures. No training; deterministic; reads the stored instance views (results/phase2i/views).

Per view (pooled over structures unless stated):
  protected_retention_tol{025,040,055}_pct   retained / all protected instances (chemistry-v1.2; tol 0.40 = default)
  structures_losing_protected_tol040          structures with >= 1 removed protected instance (default tolerance)
  hb_candidate_retention_pct                  hydrogen_bond_candidate instances retained
  shell_complete_r{2.5,3.0,3.5}_pct           % of atoms (with >= 1 instance within r) whose instances with d <= r are all retained
  loss_frac_<role>                            mean over atoms of that role of the fraction of incident instances removed
  removed_H_involving_pct                     % of removed 4.5 A instances with at least one H end
  connected_structures_pct                    % of structures whose view graph (incl. construction extras) is connected
Output: results/final/structural/structural_tradeoff.csv
"""
from __future__ import annotations

import glob
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from chemspar_v2.bonding import PROTECTED_CLASSES  # noqa: E402

V = R / "results/phase2i/views"
O = R / "results/final/structural"
CFG = yaml.safe_load((R / "configs/final_v1.yaml").read_text())
CONS = {"Original-4.5": None, "Cutoff-3.5": "in_cut35", "Cutoff-3.0": "in_cut30", "CrystalNN": "in_crystalnn", "kNN-12": "in_knn12"}
PRUNED = {f"{p}@{t}": (p, t) for t in CFG["views"]["taus"]
          for g in CFG["views"]["orders"] for p in (g, CFG["views"]["ungated_partner"][g])}
VIEWS = list(CONS) + list(PRUNED)
SHELLS = (2.5, 3.0, 3.5)


def role(el, metal):
    return "metal" if metal else {"H": "H", "C": "C", "O": "O", "N": "N"}.get(el, "other")


def retained_mask(E: pd.DataFrame, view: str) -> np.ndarray:
    if view in CONS:
        return np.ones(len(E), bool) if CONS[view] is None else E[CONS[view]].values.astype(bool)
    pol, tau = PRUNED[view]
    rk = E["rank__" + pol.replace("-", "_")].values
    k = E.qmof_id.map((E.groupby("qmof_id").size() * tau).astype(int)).values
    return ~((rk >= 0) & (rk < k))


def one_file(f: str) -> list[dict]:
    E = pd.read_parquet(f)
    xf = f.replace("inst_", "extra_")
    X = pd.read_parquet(xf) if Path(xf).exists() else pd.DataFrame(columns=["qmof_id", "construction", "i", "j"])
    prot = {t: E[c].isin(PROTECTED_CLASSES).values for t, c in (("025", "class_tol025"), ("040", "bond_class"), ("055", "class_tol055"))}
    hb = (E.bond_class == "hydrogen_bond_candidate").values
    hinv = ((E.el_i == "H") | (E.el_j == "H")).values
    ends = pd.DataFrame({"q": np.r_[E.qmof_id.values, E.qmof_id.values], "atom": np.r_[E.i.values, E.j.values],
                         "role": [role(a, m) for a, m in zip(np.r_[E.el_i.values, E.el_j.values], np.r_[E.metal_i.values, E.metal_j.values])],
                         "d": np.r_[E.distance.values, E.distance.values]})
    natoms = ends.groupby("q").atom.max() + 1
    out = []
    for v in VIEWS:
        keep = retained_mask(E, v)
        rem = ~keep
        row = dict(view=v, file=Path(f).name, n_structures=E.qmof_id.nunique(), n_instances=len(E), n_retained=int(keep.sum()),
                   n_removed=int(rem.sum()), n_removed_H=int((rem & hinv).sum()), n_hb=int(hb.sum()), n_hb_kept=int((keep & hb).sum()))
        for t, p in prot.items():
            row[f"n_prot_{t}"], row[f"n_prot_kept_{t}"] = int(p.sum()), int((keep & p).sum())
        row["structures_losing_protected_tol040"] = int(pd.Series(rem & prot["040"]).groupby(E.qmof_id.values).any().sum())
        ends["lost"] = np.r_[rem, rem]
        per_atom = ends.groupby(["q", "atom", "role"]).lost.mean().reset_index()
        for r_, grp in per_atom.groupby("role"):
            row[f"sum_loss_{r_}"], row[f"n_atoms_{r_}"] = float(grp.lost.sum()), len(grp)
        for r in SHELLS:
            sh = ends[ends.d <= r].groupby(["q", "atom"]).lost.max()
            row[f"n_atoms_shell_{r}"], row[f"n_complete_shell_{r}"] = len(sh), int((sh == 0).sum())
        # connectivity (constructions may add instances beyond 4.5 A as extras)
        conn = 0
        xs = X[X.construction == v] if v in ("CrystalNN", "kNN-12") else X.iloc[:0]
        xg = dict(tuple(xs.groupby("qmof_id"))) if len(xs) else {}
        for q, idx in E.groupby("qmof_id", sort=False).indices.items():
            k = idx[keep[idx]]
            i, j = E.i.values[k], E.j.values[k]
            if q in xg:
                i, j = np.r_[i, xg[q].i.values], np.r_[j, xg[q].j.values]
            n = int(natoms[q])
            conn += connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)), directed=False)[0] == 1
        row["n_connected"] = int(conn)
        out.append(row)
    return out


def main():
    O.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(V / "inst_*.parquet")))
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as pool:
        rows = [r for part in pool.map(one_file, files, chunksize=1) for r in part]
    A = pd.DataFrame(rows).groupby("view", sort=False).sum(numeric_only=True)
    S = pd.DataFrame(index=A.index)
    S["retained_instances_pct"] = 100 * A.n_retained / A.n_instances
    for t in ("025", "040", "055"):
        S[f"protected_retention_tol{t}_pct"] = 100 * A[f"n_prot_kept_{t}"] / A[f"n_prot_{t}"]
    S["structures_losing_protected_tol040"] = A.structures_losing_protected_tol040
    S["hb_candidate_retention_pct"] = 100 * A.n_hb_kept / A.n_hb
    for r in SHELLS:
        S[f"shell_complete_r{r}_pct"] = 100 * A[f"n_complete_shell_{r}"] / A[f"n_atoms_shell_{r}"]
    for r_ in ("H", "C", "N", "O", "metal", "other"):
        S[f"loss_frac_{r_}"] = A[f"sum_loss_{r_}"] / A[f"n_atoms_{r_}"]
    S["removed_H_involving_pct"] = np.where(A.n_removed > 0, 100 * A.n_removed_H / A.n_removed.clip(lower=1), np.nan)
    S["connected_structures_pct"] = 100 * A.n_connected / A.n_structures
    inv = {k: d for d, k in CFG["display_names"].items()}
    S.insert(0, "display_name", [v if v in CONS else f"{inv[PRUNED[v][0]]}@{PRUNED[v][1]}" for v in S.index])
    S.insert(1, "gated", [None if v in CONS else PRUNED[v][0] in CFG["views"]["orders"] for v in S.index])
    S.to_csv(O / "structural_tradeoff.csv")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print(S.round(2).to_string())


if __name__ == "__main__":
    main()
