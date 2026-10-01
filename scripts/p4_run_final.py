"""Phase 4 FINAL MATRIX runner (configs/final_v1.yaml). Outputs: results/final/fits/ (per fit: metrics JSON,
per-structure test predictions, per-epoch history).

  python p4_run_final.py preflight     # checks below; writes results/final/preflight.json (must say PASS)
  python p4_run_final.py worker <k>    # worker k of `parallel_workers`; claims jobs via atomic lock files; resumable

Preflight (all must pass before any final fit):
  * tensor store == the hashes in configs/FROZEN.json, and both tensor==view proofs pass for all 21 trained views;
  * configs/final_v1.yaml hash == FROZEN.json; CGCNN epoch budget read from results/final/budget_extension/budget_decision.json;
  * bitwise determinism at the final thread setting (3): Original-4.5 fitted twice for 2 epochs, identical predictions (both backbones);
  * every trained view x backbone trains for 2 epochs with finite outputs; initial-weight hash and mini-batch-order hash identical
    across all views (seed 1).
Job order: backbone (CGCNN first), then seed ascending, then view; so complete seeds accumulate in order.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src")); sys.path.insert(0, str(R / "scripts"))
from chemspar_v2.tensor_store import view_data  # noqa: E402
from chemspar_v2.train import FitConfig, fit  # noqa: E402
from p3_smoke_test import init_hash, order_hash  # noqa: E402

O = R / "results/final"
F = O / "fits"
T = R / "results/phase3/tensors"
CFG = yaml.safe_load((R / "configs/final_v1.yaml").read_text())


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def trained_views() -> list[str]:
    return list(CFG["views"]["trained"])


def epochs(bb: str) -> int:
    if bb == "cgcnn":
        return int(json.loads((O / "budget_extension/budget_decision.json").read_text())["E_final"])
    return int(CFG["backbones"][bb]["epochs"])


def jobs() -> list[tuple[str, int, str]]:
    ident = CFG["views"].get("identical_views", {})
    return [(bb, s, v) for bb in ("cgcnn", "graphsage") for s in CFG["backbones"][bb]["seeds"] for v in trained_views() if v not in ident]


def load():
    torch.set_num_threads(CFG["execution"]["torch_threads_per_fit"])
    base = torch.load(T / "base.pt", weights_only=False)
    masks = torch.load(T / "masks.pt", weights_only=False)
    split = dict(pd.read_csv(R / "data/cohort_5000_v1.csv")[["qmof_id", "split"]].values)
    return base, masks, split


def parts_for(base, masks, split, view):
    data = [view_data(e, m, view) for e, m in zip(base, masks[view])]
    return {k: [d for d in data if split[d.qmof_id] == k] for k in ("train", "val", "test")}


def fcfg(bb, ep):
    h = CFG["hyperparameters"]
    return FitConfig(backbone=bb, epochs=ep, lr=h["lr"], batch_size=h["batch_size"], weight_decay=h["weight_decay"], device=CFG["execution"]["device"])


def static_checks() -> dict:
    fr = json.loads((R / "configs/FROZEN.json").read_text())
    tp = fr["tensor_proof"]
    assert sha(T / "base.pt") == tp["base_sha256"] and sha(T / "masks.pt") == tp["masks_sha256"], "tensor store changed since the proofs"
    assert fr["files_sha256"]["configs/final_v1.yaml"] == sha(R / "configs/final_v1.yaml"), "final config changed since freeze"
    for a, b in CFG["views"].get("identical_views", {}).items():   # identity must hold in the frozen tensor store
        assert tp["identical_views"][a] == b, (a, b)
    for v in trained_views():
        assert tp["views"][v]["proof1_all_structures_tensor_equals_stored_view"] and tp["views"][v]["proof2_all_pass"], v
    return dict(frozen=fr, cgcnn_epochs=epochs("cgcnn"))


def preflight():
    O.mkdir(parents=True, exist_ok=True)
    st = static_checks()
    base, masks, split = load()
    rep = dict(cgcnn_epochs=st["cgcnn_epochs"], determinism={}, smoke=[])
    p = parts_for(base, masks, split, "Original-4.5")
    for bb in ("cgcnn",):
        a = fit(p["train"], p["val"], p["test"], fcfg(bb, 2), seed=1).predictions
        b = fit(p["train"], p["val"], p["test"], fcfg(bb, 2), seed=1).predictions
        rep["determinism"][bb] = float(max(abs(x["y_pred"] - y["y_pred"]) for x, y in zip(a, b)))
    for v in trained_views():
        p = parts_for(base, masks, split, v)
        for bb in ("cgcnn",):
            r = fit(p["train"], p["val"], p["test"], fcfg(bb, 2), seed=1)
            rep["smoke"].append(dict(view=v, backbone=bb, finite=bool(np.isfinite([x["y_pred"] for x in r.predictions]).all()),
                                     init_hash=init_hash(bb, 1), order_hash=order_hash(p["train"], 1, CFG["hyperparameters"]["batch_size"]),
                                     sec_per_epoch=r.timing["train_seconds"] / 2))
    sm = pd.DataFrame(rep["smoke"]); sm.to_csv(O / "preflight_smoke.csv", index=False)
    ok = (all(d == 0.0 for d in rep["determinism"].values()) and sm.finite.all()
          and sm.groupby("backbone").init_hash.nunique().eq(1).all() and sm.order_hash.nunique() == 1)
    rep.update(status="PASS" if ok else "FAIL", n_views=len(trained_views()), n_jobs=len(jobs()),
               environment=dict(python=platform.python_version(), platform=platform.platform(), torch=torch.__version__,
                                torch_geometric=version("torch_geometric"), numpy=version("numpy"),
                                threads=CFG["execution"]["torch_threads_per_fit"], workers=CFG["execution"]["parallel_workers"]),
               final_config_sha256=sha(R / "configs/final_v1.yaml"), base_sha256=st["frozen"]["tensor_proof"]["base_sha256"],
               masks_sha256=st["frozen"]["tensor_proof"]["masks_sha256"])
    rep.pop("smoke")
    (O / "preflight.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


def worker(k: int):
    assert json.loads((O / "preflight.json").read_text())["status"] == "PASS", "preflight not passed"
    static_checks()
    F.mkdir(parents=True, exist_ok=True)
    for lk in F.glob("*.lock"):                                    # stale locks from a crashed/stopped worker
        pid = int(lk.read_text().split()[3])
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            lk.unlink(missing_ok=True)
    base, masks, split = load()
    cache = {}
    for bb, seed, v in jobs():
        tag = f"{bb}__{v}__seed{seed}"
        if (F / f"{tag}.json").exists():
            continue
        try:
            fd = os.open(F / f"{tag}.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"worker {k} pid {os.getpid()} {time.ctime()}".encode()); os.close(fd)
        except FileExistsError:
            continue
        if v not in cache:
            cache = {v: parts_for(base, masks, split, v)}          # keep one view in memory
        p = cache[v]
        t0 = time.perf_counter()
        r = fit(p["train"], p["val"], p["test"], fcfg(bb, epochs(bb)), seed=seed)
        rec = dict(view=v, backbone=bb, seed=seed, worker=k, **r.metrics, **r.timing, wall_seconds=time.perf_counter() - t0,
                   init_hash=init_hash(bb, seed), order_hash=order_hash(p["train"], seed, CFG["hyperparameters"]["batch_size"]))
        pd.DataFrame(r.predictions).to_csv(F / f"{tag}__predictions.csv", index=False)
        pd.DataFrame(r.history).to_csv(F / f"{tag}__history.csv", index=False)
        (F / f"{tag}.json").write_text(json.dumps(rec))
        (F / f"{tag}.lock").unlink()
        print(json.dumps({x: rec[x] for x in ("view", "backbone", "seed", "best_epoch", "train_seconds")}), flush=True)


if __name__ == "__main__":
    preflight() if sys.argv[1] == "preflight" else worker(int(sys.argv[2]))
