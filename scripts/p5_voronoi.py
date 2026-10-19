"""Revision round 1 (R1.1): Voronoi-based neighbour construction, structural endpoints only (no training).
pymatgen VoronoiNN (default parameters except cutoff = 13 A and allow_pathological = True): every pair of atoms whose
Voronoi cells share a face becomes a contact instance (canonical key per periodic image), as in parameter-free Voronoi
edge selection. Endpoints match Table 2: mean instances (incl. contacts beyond 4.5 A), % of the 4.5 A graph,
% connected, and retention of radius-bonded non-metal, metal-ligand and CrystalNN metal-metal contacts.
Output: results/revision/voronoi_construction.csv, voronoi_per_structure.csv
"""
import glob
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from chemspar_v2.paths import CIF_DIR  # noqa: E402
from chemspar_v2.periodic import canonical  # noqa: E402

O = R / "results/revision"


def one(args):
    q, e = args
    warnings.filterwarnings("ignore")
    from pymatgen.analysis.local_env import VoronoiNN
    from pymatgen.core import Structure
    s = Structure.from_file(CIF_DIR / f"{q}.cif")
    try:
        nn = VoronoiNN(cutoff=13.0, allow_pathological=True).get_all_nn_info(s)
    except Exception as ex:  # recorded, not hidden
        return dict(qmof_id=q, failed=str(ex)[:120])
    vor = set()
    for i, lst in enumerate(nn):
        for d in lst:
            im = tuple(int(round(x)) for x in d["image"])
            j = int(d["site_index"])
            if i == j and im == (0, 0, 0):
                continue
            vor.add(canonical(i, j, im))
    base = {(int(a), int(b), tuple(int(x) for x in im.split(","))): (c, p) for a, b, im, c, p in zip(e.i, e.j, e.image, e.bond_class, e.protected)}
    kept = [k for k in base if k in vor]
    out = dict(qmof_id=q, failed="", n_base=len(base), n_vor=len(vor), n_vor_within_4p5=len(kept), n_vor_beyond_4p5=len(vor) - len(kept))
    for cls in ("covalent_radius", "metal_ligand", "metal_metal_crystalnn"):
        tot = [k for k, (c, _) in base.items() if c == cls]
        out[f"n_{cls}"] = len(tot); out[f"kept_{cls}"] = sum(k in vor for k in tot)
    n = len(s)
    i = [k[0] for k in vor]; j = [k[1] for k in vor]
    out["connected"] = bool(connected_components(coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)), directed=False)[0] == 1)
    return out


def main():
    O.mkdir(parents=True, exist_ok=True)
    cols = ["qmof_id", "i", "j", "image", "bond_class", "protected"]
    work = []
    for f in sorted(glob.glob(str(R / "results/phase2i/views/inst_*.parquet"))):
        E = pd.read_parquet(f, columns=cols)
        work += [(q, e) for q, e in E.groupby("qmof_id", sort=False)]
    with Pool(9) as pool:
        rows = pool.map(one, work, chunksize=4)
    D = pd.DataFrame(rows); D.to_csv(O / "voronoi_per_structure.csv", index=False)
    ok = D[D.failed == ""]
    S = dict(view="Voronoi", structures=len(ok), failed=int((D.failed != "").sum()),
             mean_instances=ok.n_vor.mean(), pct_of_full=100 * ok.n_vor.mean() / ok.n_base.mean(),
             instances_beyond_4p5=int(ok.n_vor_beyond_4p5.sum()), connected_pct=100 * ok.connected.mean(),
             covalent_radius_kept_pct=100 * ok.kept_covalent_radius.sum() / ok.n_covalent_radius.sum(),
             metal_ligand_kept_pct=100 * ok.kept_metal_ligand.sum() / ok.n_metal_ligand.sum(),
             metal_metal_kept_pct=100 * ok.kept_metal_metal_crystalnn.sum() / ok.n_metal_metal_crystalnn.sum(),
             structures_losing_protected=int(((ok.kept_covalent_radius < ok.n_covalent_radius) | (ok.kept_metal_ligand < ok.n_metal_ligand)
                                              | (ok.kept_metal_metal_crystalnn < ok.n_metal_metal_crystalnn)).sum()))
    pd.DataFrame([S]).to_csv(O / "voronoi_construction.csv", index=False)
    print(pd.DataFrame([S]).round(3).T.to_string())


if __name__ == "__main__":
    main()
