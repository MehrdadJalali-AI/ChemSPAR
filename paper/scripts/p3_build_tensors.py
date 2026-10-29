"""Phase 3: build the compact tensor store from the STORED instance views (results/phase2i/views) and prove that
every tensor encodes exactly its named view.

Proof 1 (all structures x all trained views): tensor edges -> canonical instance keys == the view's retained
        instance set derived from the stored ranks/flags (chemspar_v2.train.verify_view_tensor).
Proof 2 (independent path, 100 structures = every 50th cohort structure): the view recomputed from the CIF with
        chemspar_v2.views.compute_structure/view_instances equals the stored-view tensor.
Outputs: results/phase3/tensors/base.pt, masks.pt, tensor_verification.csv, tensor_manifest.csv
"""
from __future__ import annotations

import glob
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RESUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESUB / "src"))
from chemspar_v2.paths import CIF_DIR, QMOF_DIR  # noqa: E402
CIF = CIF_DIR
VIEWS_DIR = RESUB / "results/phase2i/views"
OUT = RESUB / "results/phase3/tensors"
PRUNED = {"ChemSPAR-v2@0.2": ("ChemSPAR-v2", 0.2), "Random@0.2": ("Random", 0.2), "Random-Chem@0.2": ("Random-Chem", 0.2),
          "Distance@0.2": ("Distance", 0.2), "Distance-Chem@0.2": ("Distance-Chem", 0.2), "BHS-edge-min@0.2": ("BHS-edge-min", 0.2),
          "ChemSPAR-v2-massfree@0.2": ("ChemSPAR-v2-massfree", 0.2),
          "ChemSPAR-v2@0.5": ("ChemSPAR-v2", 0.5), "Random-Chem@0.5": ("Random-Chem", 0.5),
          "Distance-Chem@0.5": ("Distance-Chem", 0.5), "BHS-edge-min@0.5": ("BHS-edge-min", 0.5),
          # final matrix (configs/final_v1.yaml): ungated partners of every gated order at both taus
          "ChemSPAR-v2-noGate@0.2": ("ChemSPAR-v2-noGate", 0.2), "BHS-edge-min-noGate@0.2": ("BHS-edge-min-noGate", 0.2),
          "ChemSPAR-v2-noGate@0.5": ("ChemSPAR-v2-noGate", 0.5), "Random@0.5": ("Random", 0.5), "Distance@0.5": ("Distance", 0.5),
          "BHS-edge-min-noGate@0.5": ("BHS-edge-min-noGate", 0.5)}
CONS = {"Original-4.5": None, "Cutoff-3.5": "in_cut35", "Cutoff-3.0": "in_cut30", "CrystalNN": "in_crystalnn", "kNN-12": "in_knn12"}
VIEWS = list(CONS) + list(PRUNED)


def _img(s):
    return [int(x) for x in s.split(",")]


def entry_and_masks(args):
    qid, y, e, extra = args
    warnings.filterwarnings("ignore")
    from pymatgen.core import Structure
    from chemspar_v2.elements import role
    from chemspar_v2.train import node_features
    s = Structure.from_file(CIF / f"{qid}.cif")
    z, roles = node_features([x.specie.Z for x in s], [role(x.specie.symbol) for x in s])
    inst = torch.tensor([[i, j, *_img(im)] for i, j, im in zip(e.i, e.j, e.image)], dtype=torch.int32)
    entry = dict(qmof_id=qid, z=z, roles=roles, y=float(y), inst=inst, dist=torch.tensor(e.distance.values, dtype=torch.float32), extras={})
    for c in ("CrystalNN", "kNN-12"):
        x = extra[extra.construction == c] if extra is not None else None
        if x is not None and len(x):
            entry["extras"][c] = (torch.tensor([[i, j, *_img(im)] for i, j, im in zip(x.i, x.j, x.image)], dtype=torch.int32),
                                  torch.tensor(x.distance.values, dtype=torch.float32))
    n = len(e)
    masks = {}
    for v, colname in CONS.items():
        masks[v] = torch.ones(n, dtype=torch.bool) if colname is None else torch.tensor(e[colname].values)
    for v, (pol, tau) in PRUNED.items():
        rk = e["rank__" + pol.replace("-", "_")].values
        k = int(tau * n)
        masks[v] = torch.tensor(~((rk >= 0) & (rk < k)))
    return entry, masks


def verify(args):
    """Proof 1 for one structure: tensor keys == stored view keys, for every view."""
    entry, masks = args
    from chemspar_v2.tensor_store import view_data, view_keys
    from chemspar_v2.train import verify_view_tensor
    out = {}
    for v in VIEWS:
        d = view_data(entry, masks[v], v)
        out[v] = verify_view_tensor(d, view_keys(entry, masks[v], v))
    return entry["qmof_id"], out


def independent(args):
    """Proof 2: recompute the view from the CIF and compare with the stored-view tensor."""
    entry, masks = args
    warnings.filterwarnings("ignore")
    from pymatgen.core import Structure
    from chemspar_v2 import views as V
    from chemspar_v2.tensor_store import view_data
    from chemspar_v2.train import verify_view_tensor
    res = V.compute_structure(Structure.from_file(CIF / f"{entry['qmof_id']}.cif"), entry["qmof_id"])
    return entry["qmof_id"], {v: verify_view_tensor(view_data(entry, masks[v], v), V.view_instances(res, v)) for v in VIEWS}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(RESUB / "data/cohort_5000_v1.csv")
    cols = ["qmof_id", "i", "j", "image", "distance"] + [c for c in CONS.values() if c] + sorted({"rank__" + p.replace("-", "_") for p, _ in PRUNED.values()})
    E = pd.concat(pd.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(VIEWS_DIR / "inst_*.parquet"))))
    X = [pd.read_parquet(f) for f in sorted(glob.glob(str(VIEWS_DIR / "extra_*.parquet")))]
    X = pd.concat(X) if X else pd.DataFrame(columns=["qmof_id", "construction"])
    ge, gx = dict(tuple(E.groupby("qmof_id", sort=False))), dict(tuple(X.groupby("qmof_id"))) if len(X) else {}
    args = [(q, y, ge[q], gx.get(q)) for q, y in zip(cohort.qmof_id, cohort.bandgap)]
    with Pool(9) as pool:
        built = pool.map(entry_and_masks, args, chunksize=16)
        v1 = dict(pool.map(verify, built, chunksize=16))
        sample = built[::50]
        v2 = dict(pool.map(independent, sample, chunksize=1))
    torch.save([b[0] for b in built], OUT / "base.pt")
    torch.save({v: [b[1][v] for b in built] for v in VIEWS}, OUT / "masks.pt")
    rows = [dict(qmof_id=q, view=v, proof1_stored_view=v1[q][v], proof2_independent_recompute=v2.get(q, {}).get(v)) for q in cohort.qmof_id for v in VIEWS]
    ver = pd.DataFrame(rows); ver.to_csv(OUT / "tensor_verification.csv", index=False)
    man = []
    for v in VIEWS:
        m = [b[1][v] for b in built]
        extra = sum(len(b[0]["extras"].get(v, ([], []))[0]) for b in built)
        man.append(dict(view=v, structures=len(m), mean_instances=float(np.mean([int(x.sum()) for x in m])) + extra / len(m),
                        instances_beyond_4p5=extra, proof1_all_pass=bool(ver[ver.view == v].proof1_stored_view.all()),
                        proof2_checked=int(ver[ver.view == v].proof2_independent_recompute.notna().sum()),
                        proof2_all_pass=bool(ver[(ver.view == v) & ver.proof2_independent_recompute.notna()].proof2_independent_recompute.astype(bool).all())))
    pd.DataFrame(man).to_csv(OUT / "tensor_manifest.csv", index=False)
    print(pd.DataFrame(man).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
