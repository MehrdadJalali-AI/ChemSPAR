"""Phase 2 on periodic edge INSTANCES (pruning unit = (i, j, image)); single source of truth = chemspar_v2.views.

Per structure writes one row per 4.5 A instance with taxonomy (tol 0.40; classes also at 0.25 / 0.55), gravity
components, and removal ranks of every policy (rank__<policy>, -1 = never removed; the removed set at tau is
{0 <= rank < floor(tau * n_instances)} by the prefix property). Construction membership flags: in_cut35, in_cut30,
in_crystalnn, in_knn12. Instances of the CrystalNN / kNN constructions beyond 4.5 A (if any) go to extra_*.parquet.
Also: per structure x policy x tau rejection counts (stats_*), full audit rows for the first 200 cohort structures.
Output: results/phase2i/views/*.parquet
"""
from __future__ import annotations

import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

RESUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESUB / "src"))
from chemspar_v2.paths import CIF_DIR, QMOF_DIR  # noqa: E402
CIF = CIF_DIR
OUT = RESUB / "results/phase2i/views"
N_AUDIT, CHUNK = 200, 50


def one(args):
    idx, qid = args
    warnings.filterwarnings("ignore")
    from pymatgen.core import Structure
    from chemspar_v2 import views as V
    t0 = time.perf_counter()
    s = Structure.from_file(CIF / f"{qid}.cif")
    res = V.compute_structure(s, qid, audit=idx < N_AUDIT)
    cons = {k: set(v) for k, v in res["constructions"].items()}
    rows = []
    for k in res["instances"]:
        a = res["attrs"][k]
        i, j, im = k
        row = dict(qmof_id=qid, i=i, j=j, image="%d,%d,%d" % im, self_image=i == j,
                   el_i=res["graph"].nodes[i]["element"], el_j=res["graph"].nodes[j]["element"],
                   metal_i=res["graph"].nodes[i]["is_metal"], metal_j=res["graph"].nodes[j]["is_metal"],
                   distance=a["distance"], bond_class=a["bond_class"], route=a["protection_route"], protected=a["protected"],
                   crystalnn=a["crystalnn"], class_tol025=res["alt"][0.25].edges[i, j, im]["bond_class"],
                   class_tol055=res["alt"][0.55].edges[i, j, im]["bond_class"],
                   g=res["comp"][k]["g"], d_hyb=res["comp"][k]["d_hyb"], bhs_min=res["val"]["bhs_min"][k],
                   in_cut35=k in cons["Cutoff-3.5"], in_cut30=k in cons["Cutoff-3.0"],
                   in_crystalnn=k in cons["CrystalNN"], in_knn12=k in cons["kNN-12"])
        for name, rk in res["ranks"].items():
            row["rank__" + name.replace("-", "_")] = rk.get(k, -1)
        rows.append(row)
    inst = set(res["instances"])
    extra = [dict(qmof_id=qid, i=k[0], j=k[1], image="%d,%d,%d" % k[2], distance=res["dist"][k], construction=c)
             for c in ("CrystalNN", "kNN-12") for k in res["constructions"][c] if k not in inst]
    stats = [dict(qmof_id=qid, view=p, tau=t, n_instances=len(inst), **{("rej::" + a if a not in ("target", "removed") else a): b for a, b in d.items()})
             for p, per in res["rejections"].items() for t, d in per.items()]
    return rows, extra, stats, res["audit"], dict(qmof_id=qid, n_atoms=len(s), n_instances=len(inst), seconds=time.perf_counter() - t0)


def main(limit: int | None = None):
    OUT.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(RESUB / "data/cohort_5000_v1.csv").qmof_id.tolist()
    if limit:
        ids = ids[:limit]
    work = list(enumerate(ids))
    t0 = time.time()
    with Pool(9) as pool:
        for c0 in range(0, len(work), CHUNK * 9):
            block = work[c0:c0 + CHUNK * 9]
            tag = f"{c0:05d}"
            if (OUT / f"inst_{tag}.parquet").exists() and not limit:
                continue
            res = pool.map(one, block, chunksize=1)
            pd.DataFrame([r for x in res for r in x[0]]).to_parquet(OUT / f"inst_{tag}.parquet", index=False)
            ex = [r for x in res for r in x[1]]
            if ex:
                pd.DataFrame(ex).to_parquet(OUT / f"extra_{tag}.parquet", index=False)
            pd.DataFrame([r for x in res for r in x[2]]).to_parquet(OUT / f"stats_{tag}.parquet", index=False)
            aud = [r for x in res for r in x[3]]
            if aud:
                pd.DataFrame(aud).to_parquet(OUT / f"audit_{tag}.parquet", index=False)
            pd.DataFrame([x[4] for x in res]).to_parquet(OUT / f"timing_{tag}.parquet", index=False)
            print(f"{c0 + len(block)}/{len(work)} structures, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
