"""Phase 4 pre-run: CGCNN epoch-budget extension (validation MAE only; no test metric is read or written).

Why: the pilot showed the Phase 3 budget (E = 125, from one Original-4.5 run) is short: in 9 of 16 pilot CGCNN fits the
running-best validation MAE still improved by > 1% over epochs 100-125 (results/pilot/fits/*__history.csv).
Rule (fixed before this run; same 1% plateau criterion as scripts/p3_budget_pilot.py, applied to more than one view):
  * pilot seed 1001 (disjoint from the final seeds 1-20), 300 epochs, three views, one per sparsity regime and the same
    ordering policy, chosen before any longer run: Original-4.5, ChemSPAR-v2@0.2, ChemSPAR-v2@0.5;
  * per view: e* = first epoch at which the running-best validation MAE is within 1% of that view's best over 300 epochs;
  * E_final = smallest multiple of 25 >= max_view e*, capped at 250 (cap disclosed if binding);
  * a view is flagged 'horizon too short' if its best validation MAE falls in epochs 271-300.
Usage: python p4_budget_extension.py <view>   (one process per view; 3 torch threads each). Output: results/final/budget_extension/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from chemspar_v2.tensor_store import view_data  # noqa: E402
from chemspar_v2.train import FitConfig, fit  # noqa: E402

VIEWS = ("Original-4.5", "ChemSPAR-v2@0.2", "ChemSPAR-v2@0.5")
SEED, EPOCHS, THREADS, CAP = 1001, 300, 3, 250
O = R / "results/final/budget_extension"


def run(view: str):
    assert view in VIEWS
    O.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(THREADS)
    base = torch.load(R / "results/phase3/tensors/base.pt", weights_only=False)
    mask = torch.load(R / "results/phase3/tensors/masks.pt", weights_only=False)[view]
    split = dict(pd.read_csv(R / "data/cohort_5000_v1.csv")[["qmof_id", "split"]].values)
    data = [view_data(e, m, view) for e, m in zip(base, mask)]
    del base
    parts = {k: [d for d in data if split[d.qmof_id] == k] for k in ("train", "val")}
    r = fit(parts["train"], parts["val"], parts["val"][:1], FitConfig(backbone="cgcnn", epochs=EPOCHS), seed=SEED)
    pd.DataFrame(r.history).to_csv(O / f"{view}__history.csv", index=False)   # validation only; the 1-structure 'test' is a dummy


def decide():
    rows = []
    for v in VIEWS:
        h = pd.read_csv(O / f"{v}__history.csv")
        rb = h.val_mae.cummin()
        best = rb.iloc[-1]
        e_star = int(h.epoch[rb <= 1.01 * best].iloc[0])
        rows.append(dict(view=v, best_val_mae=best, best_epoch=int(h.epoch[h.val_mae.idxmin()]), first_epoch_within_1pct=e_star,
                         horizon_too_short=bool(h.epoch[h.val_mae.idxmin()] > 0.9 * EPOCHS)))
    d = pd.DataFrame(rows)
    e = int(-(-d.first_epoch_within_1pct.max() // 25) * 25)
    out = dict(rule_E=e, E_final=min(e, CAP), cap_binding=e > CAP, per_view=rows)
    d.to_csv(O / "budget_extension_summary.csv", index=False)
    (O / "budget_decision.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    decide() if sys.argv[1] == "decide" else run(sys.argv[1])
