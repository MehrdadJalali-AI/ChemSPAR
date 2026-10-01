"""Phase 3 smoke tests (one seed; NOT results; never used for comparisons).

1. For every view x backbone: load tensors (full cohort), check integrity, train `--epochs` epochs with seed 0,
   record per-epoch wall time, peak RSS, and hashes of the initial weights and of the first-epoch mini-batch order
   (pairing check: must be identical across views of the same backbone).
2. Determinism: Original-4.5 trained twice with the same seed must give identical test predictions.
3. Budget pilot (Original-4.5 only, one seed, not comparative): long run to read the validation learning curve.
Outputs: results/phase3/smoke/*.csv
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

RESUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESUB / "src"))
from chemspar_v2.train import BACKBONES, FitConfig, fit  # noqa: E402

T = RESUB / "results/phase3/tensors"
O = RESUB / "results/phase3/smoke"


_BASE, _MASKS = None, None


def load_view(name):
    """Views are materialised from the verified tensor store (base instances + per-view mask)."""
    global _BASE, _MASKS
    from chemspar_v2.tensor_store import view_data
    if _BASE is None:
        _BASE = torch.load(T / "base.pt", weights_only=False)
        _MASKS = torch.load(T / "masks.pt", weights_only=False)
    data = [view_data(e, m, name) for e, m in zip(_BASE, _MASKS[name])]
    cohort = pd.read_csv(RESUB / "data/cohort_5000_v1.csv")
    split = dict(zip(cohort.qmof_id, cohort.split))
    parts = {k: [d for d in data if split[d.qmof_id] == k] for k in ("train", "val", "test")}
    return data, parts


def integrity(name, data):
    bad = sum(1 for d in data if d.edge_index.numel() and (d.edge_index.min() < 0 or d.edge_index.max() >= d.num_nodes))
    return dict(view=name, structures=len(data), index_errors=bad,
                structures_with_isolated_atoms=sum(int((torch.bincount(d.edge_index[0], minlength=d.num_nodes) == 0).any()) for d in data),
                mean_directed_edges=float(np.mean([d.edge_index.shape[1] for d in data])),
                finite_distances=bool(all(torch.isfinite(d.edge_dist).all() for d in data)))


def init_hash(backbone, seed):
    torch.manual_seed(seed)
    m = BACKBONES[backbone]()
    h = hashlib.sha256()
    for k, v in m.state_dict().items():
        h.update(k.encode()); h.update(v.cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def order_hash(train, seed, bs=32):
    from torch_geometric.loader import DataLoader
    gen = torch.Generator().manual_seed(seed)
    ids = [q for b in DataLoader(train, batch_size=bs, shuffle=True, generator=gen) for q in b.qmof_id]
    return hashlib.sha256("|".join(ids).encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--pilot-epochs", type=int, default=0)
    args = ap.parse_args()
    O.mkdir(parents=True, exist_ok=True)
    man = pd.read_csv(T / "tensor_manifest.csv")
    views = args.views or man.view.tolist()
    rows, integ = [], []
    torch.set_num_threads(8)
    for v in views:
        data, parts = load_view(v)
        integ.append(integrity(v, data))
        for bb in ("graphsage", "cgcnn"):
            t0 = time.perf_counter()
            r = fit(parts["train"], parts["val"], parts["test"], FitConfig(backbone=bb, epochs=args.epochs), seed=0)
            rows.append(dict(view=v, backbone=bb, epochs=args.epochs, seconds_per_epoch=r.timing["train_seconds"] / args.epochs,
                             wall_seconds=time.perf_counter() - t0, peak_rss_mb=r.timing["peak_rss_mb"],
                             init_hash=init_hash(bb, 0), order_hash=order_hash(parts["train"], 0),
                             smoke_val_mae_last=r.history[-1]["val_mae"], finite=bool(np.isfinite(r.metrics["MAE"]))))
            print(rows[-1], flush=True)
    pd.DataFrame(integ).to_csv(O / "smoke_integrity.csv", index=False)
    pd.DataFrame(rows).to_csv(O / "smoke_runs.csv", index=False)
    # determinism: Original twice
    data, parts = load_view("Original-4.5")
    det = []
    for bb in ("graphsage", "cgcnn"):
        a = fit(parts["train"], parts["val"], parts["test"], FitConfig(backbone=bb, epochs=2), seed=7)
        b = fit(parts["train"], parts["val"], parts["test"], FitConfig(backbone=bb, epochs=2), seed=7)
        pa = np.array([x["y_pred"] for x in a.predictions]); pb = np.array([x["y_pred"] for x in b.predictions])
        det.append(dict(backbone=bb, bitwise_identical=bool((pa == pb).all()), max_abs_diff=float(np.abs(pa - pb).max())))
    pd.DataFrame(det).to_csv(O / "smoke_determinism.csv", index=False); print(det)
    if args.pilot_epochs:
        for bb in ("graphsage", "cgcnn"):
            r = fit(parts["train"], parts["val"], parts["test"], FitConfig(backbone=bb, epochs=args.pilot_epochs), seed=0)
            pd.DataFrame(r.history).assign(backbone=bb).to_csv(O / f"budget_pilot_{bb}.csv", index=False)
            print(bb, "best epoch", r.metrics["best_epoch"], "best val MAE", round(r.metrics["best_val_mae"], 4), "s/epoch", round(r.timing["train_seconds"] / args.pilot_epochs, 1))


if __name__ == "__main__":
    main()
