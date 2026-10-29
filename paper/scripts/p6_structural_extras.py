"""Final corrections (structural only, no training).

  python p6_structural_extras.py voronoi   # per-instance Voronoi membership -> same endpoints as Supplementary Tables S1-S2
  python p6_structural_extras.py dim       # periodic dimensionality of the full graph and of each tau = 0.5 view

Voronoi: pymatgen VoronoiNN(cutoff=13, allow_pathological=True) as in p5_voronoi.py; the membership of every stored 4.5 A
instance is written to results/revision/voronoi_masks/ and the endpoints are computed with the same code as
p4_structural_tradeoff.py (retention at three tolerances, shell completeness, role loss; connectivity including Voronoi
contacts beyond 4.5 A is taken from voronoi_per_structure.csv).

Dimensionality: the gate's connectivity check tests the finite quotient graph (unit-cell atoms; image vectors ignored).
Here the periodic net is analysed with its image vectors: a spanning forest assigns each atom a lattice offset, every
non-tree edge (i, j, n) contributes the cycle vector p_i + n - p_j, and the dimensionality of the component is the rank
of these vectors (0-3). Reported per view: structures whose dimensionality (of the component containing atom 0, and the
maximum over components) differs from the full 4.5 A graph.
Outputs: results/revision/{structural_voronoi.csv, periodic_dimensionality.csv, periodic_dimensionality_summary.csv}
"""
import glob
import json
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src")); sys.path.insert(0, str(R / "scripts"))
from chemspar_v2.paths import CIF_DIR  # noqa: E402
from chemspar_v2.periodic import canonical  # noqa: E402

V = R / "results/phase2i/views"
O = R / "results/revision"
DIM_VIEWS = {"Original-4.5": None, "Cutoff-3.5": "in_cut35"}
POLS = ["ChemSPAR-v2", "ChemSPAR-v2-noGate", "Distance-Chem", "Distance", "Random-Chem", "Random", "BHS-edge-min", "BHS-edge-min-noGate"]


# ------------------------------------------------------------------ Voronoi membership
def vor_one(args):
    q, e = args
    warnings.filterwarnings("ignore")
    from pymatgen.analysis.local_env import VoronoiNN
    from pymatgen.core import Structure
    nn = VoronoiNN(cutoff=13.0, allow_pathological=True).get_all_nn_info(Structure.from_file(CIF_DIR / f"{q}.cif"))
    vor = set()
    for i, lst in enumerate(nn):
        for d in lst:
            im = tuple(int(round(x)) for x in d["image"]); j = int(d["site_index"])
            if not (i == j and im == (0, 0, 0)):
                vor.add(canonical(i, j, im))
    return q, np.array([(int(a), int(b), tuple(int(x) for x in im.split(","))) in vor for a, b, im in zip(e.i, e.j, e.image)])


def voronoi():
    M = O / "voronoi_masks"; M.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(V / "inst_*.parquet")))
    with Pool(9) as pool:
        for f in files:
            out = M / Path(f).name.replace("inst_", "mask_")
            if out.exists():
                continue
            E = pd.read_parquet(f, columns=["qmof_id", "i", "j", "image"])
            res = dict(pool.map(vor_one, list(E.groupby("qmof_id", sort=False)), chunksize=2))
            mask = np.concatenate([res[q] for q in E.qmof_id.drop_duplicates()])
            assert len(mask) == len(E)
            pd.DataFrame({"in_voronoi": mask}).to_parquet(out, index=False)
            print(out.name, flush=True)
    # endpoints with the code of p4_structural_tradeoff (Voronoi treated as a construction column)
    import p4_structural_tradeoff as ST
    ST.CONS = {"Voronoi": "in_voronoi"}; ST.VIEWS = ["Voronoi"]; ST.PRUNED = {}
    orig_read = pd.read_parquet

    def read_with_mask(path, *a, **k):
        df = orig_read(path, *a, **k)
        if Path(path).name.startswith("inst_") and "columns" not in k:
            df["in_voronoi"] = orig_read(M / Path(path).name.replace("inst_", "mask_")).in_voronoi.values
        return df
    ST.pd.read_parquet = read_with_mask
    rows = [r for f in files for r in ST.one_file(f)]
    A = pd.DataFrame(rows).groupby("view").sum(numeric_only=True).iloc[0]
    S = {"retained_instances_pct_of_4p5": 100 * A.n_retained / A.n_instances}
    for t in ("025", "040", "055"):
        S[f"protected_retention_tol{t}_pct"] = 100 * A[f"n_prot_kept_{t}"] / A[f"n_prot_{t}"]
    S["structures_losing_protected_tol040"] = A.structures_losing_protected_tol040
    S["hb_candidate_retention_pct"] = 100 * A.n_hb_kept / A.n_hb
    for r in (2.5, 3.0, 3.5):
        S[f"shell_complete_r{r}_pct"] = 100 * A[f"n_complete_shell_{r}"] / A[f"n_atoms_shell_{r}"]
    for r_ in ("H", "C", "N", "O", "metal", "other"):
        S[f"loss_frac_{r_}"] = A[f"sum_loss_{r_}"] / A[f"n_atoms_{r_}"]
    S["removed_H_involving_pct"] = 100 * A.n_removed_H / A.n_removed
    vps = pd.read_csv(O / "voronoi_per_structure.csv")
    S["connected_structures_pct_incl_beyond_4p5"] = 100 * vps.connected.mean()
    S["pct_of_full_incl_beyond_4p5"] = pd.read_csv(O / "voronoi_construction.csv").pct_of_full.iloc[0]
    pd.DataFrame([S]).to_csv(O / "structural_voronoi.csv", index=False)
    print(pd.Series(S).round(4).to_string())


# ------------------------------------------------------------------ periodic dimensionality
def net_dims(n_atoms, i, j, img):
    """Dimensionality (rank of cycle vectors) of each connected component of a periodic quotient graph."""
    adj = [[] for _ in range(n_atoms)]
    for t, (a, b) in enumerate(zip(i, j)):
        adj[a].append((b, t, 1)); adj[b].append((a, t, -1))
    pos = [None] * n_atoms; comp = [-1] * n_atoms; tree = np.zeros(len(i), bool); c = 0
    for s in range(n_atoms):
        if comp[s] >= 0:
            continue
        pos[s] = np.zeros(3, int); comp[s] = c; stack = [s]
        while stack:
            u = stack.pop()
            for v, t, sgn in adj[u]:
                if comp[v] < 0:
                    comp[v] = c; pos[v] = pos[u] + sgn * img[t]; tree[t] = True; stack.append(v)
        c += 1
    vecs = {k: [] for k in range(c)}
    for t in np.where(~tree)[0]:
        a, b = i[t], j[t]
        vecs[comp[a]].append(pos[a] + img[t] - pos[b])
    dims = {k: (int(np.linalg.matrix_rank(np.array(v))) if v else 0) for k, v in vecs.items()}
    return dims[comp[0]], max(dims.values()), c


def dim_one(args):
    q, e, n = args
    i = e.i.values.astype(int); j = e.j.values.astype(int)
    img = np.array([[int(x) for x in s.split(",")] for s in e.image])
    N = len(e); k = int(0.5 * N)
    masks = {"Original-4.5": np.ones(N, bool), "Cutoff-3.5": e.in_cut35.values.astype(bool)}
    for p in POLS:
        rk = e["rank__" + p.replace("-", "_")].values
        masks[f"{p}@0.5"] = ~((rk >= 0) & (rk < k))
    out = {"qmof_id": q}
    for v, m in masks.items():
        d0, dmax, nc = net_dims(n, i[m], j[m], img[m])
        out[f"{v}|dim_atom0"] = d0; out[f"{v}|dim_max"] = dmax; out[f"{v}|components"] = nc
    return out


def dim():
    cohort = pd.read_csv(R / "data/cohort_5000_v1.csv"); natoms = dict(zip(cohort.qmof_id, cohort.n_atoms))
    cols = ["qmof_id", "i", "j", "image", "in_cut35"] + ["rank__" + p.replace("-", "_") for p in POLS]
    work = []
    for f in sorted(glob.glob(str(V / "inst_*.parquet"))):
        E = pd.read_parquet(f, columns=cols)
        work += [(q, e, natoms[q]) for q, e in E.groupby("qmof_id", sort=False)]
    with Pool(9) as pool:
        rows = pool.map(dim_one, work, chunksize=8)
    D = pd.DataFrame(rows); D.to_csv(O / "periodic_dimensionality.csv", index=False)
    views = ["Original-4.5", "Cutoff-3.5"] + [f"{p}@0.5" for p in POLS]
    base0, basem = D["Original-4.5|dim_atom0"], D["Original-4.5|dim_max"]
    summ = [dict(view=v, structures=len(D), changed_dim_atom0=int((D[f"{v}|dim_atom0"] != base0).sum()),
                 lowered_dim_atom0=int((D[f"{v}|dim_atom0"] < base0).sum()), changed_dim_max=int((D[f"{v}|dim_max"] != basem).sum()),
                 multi_component=int((D[f"{v}|components"] > 1).sum())) for v in views]
    S = pd.DataFrame(summ); S.to_csv(O / "periodic_dimensionality_summary.csv", index=False)
    print(base0.value_counts().sort_index().to_string()); print(S.to_string(index=False))


if __name__ == "__main__":
    {"voronoi": voronoi, "dim": dim}[sys.argv[1]]()
