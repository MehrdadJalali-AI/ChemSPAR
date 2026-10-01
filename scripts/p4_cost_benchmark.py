"""Phase 4 cost benchmark (configs/final_v1.yaml `cost_benchmark`). Run AFTER the final matrix, with nothing else running.

Preprocessing (stage-wise, per structure; single Python process; 3 repeats, each a fresh process):
  subsample = every 25th cohort structure (200 structures; fixed rule, cohort order)
  stages: graph_4p5, graph_3p5, crystalnn, taxonomy, score_gravity, score_distance, score_random, score_bhs,
          prune_<policy> (to tau = 0.5; gated and ungated), tensorise_<view>
  per-view preprocessing = sum of the stages that view needs; reported per structure and scaled to 5,000 structures
  (labelled as extrapolated). Ungated views do not need CrystalNN or the taxonomy.
Training/inference (per view; 3 repeats, each a fresh process, 8 torch threads, full cohort from the frozen tensor store):
  5 epochs, seconds/epoch = mean of epochs 2-5; test inference seconds; peak RSS. Full-budget training time =
  seconds/epoch x 250 (labelled as computed, not measured).
Deviation from final-v1.1 text, disclosed: preprocessing is timed on a 200-structure subsample, not all 5,000.
  python p4_cost_benchmark.py all            # orchestrates fresh processes
Outputs: results/final/cost/{prep_rep*.csv, train_*.json, cost_summary.csv, environment.json}
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from chemspar_v2.paths import CIF_DIR, QMOF_DIR  # noqa: E402
O = R / "results/final/cost"
CIF = CIF_DIR
TAU, E_FULL, REPEATS = 0.5, 250, 3
VIEWS = ["Original-4.5", "Cutoff-3.5", "ChemSPAR-v2@0.5", "Distance-Chem@0.5", "Random-Chem@0.5", "Random@0.5",
         "BHS-edge-min@0.5", "BHS-edge-min-noGate@0.5"]
NEEDS = {  # stages per view (tensorise added per view)
    "Original-4.5": ["graph_4p5"],
    "Cutoff-3.5": ["graph_3p5"],
    "ChemSPAR-v2@0.5": ["graph_4p5", "crystalnn", "taxonomy", "score_gravity", "prune_ChemSPAR-v2"],
    "Distance-Chem@0.5": ["graph_4p5", "crystalnn", "taxonomy", "score_distance", "prune_Distance-Chem"],
    "Random-Chem@0.5": ["graph_4p5", "crystalnn", "taxonomy", "score_random", "prune_Random-Chem"],
    "Random@0.5": ["graph_4p5", "score_random", "prune_Random"],
    "BHS-edge-min@0.5": ["graph_4p5", "crystalnn", "taxonomy", "score_bhs", "prune_BHS-edge-min"],
    "BHS-edge-min-noGate@0.5": ["graph_4p5", "score_bhs", "prune_BHS-edge-min-noGate"],
}


def prep(rep: int):
    warnings.filterwarnings("ignore")
    from pymatgen.core import Structure
    from chemspar_v2 import periodic as P
    from chemspar_v2 import views as V
    from chemspar_v2.crystalnn import crystalnn_pairs
    from chemspar_v2.elements import role
    from chemspar_v2.train import instances_to_data, node_features
    ids = pd.read_csv(R / "data/cohort_5000_v1.csv").qmof_id.tolist()[::25]
    if os.environ.get("COST_DRYRUN"):
        ids = ids[:2]
    rows = []
    for qid in ids:
        s = Structure.from_file(CIF / f"{qid}.cif")
        t, r = {}, {}

        def tm(name, fn):
            t0 = time.perf_counter(); out = fn(); t[name] = time.perf_counter() - t0; return out
        g = tm("graph_4p5", lambda: P.build_instance_graph(s, 4.5))
        g35 = tm("graph_3p5", lambda: P.build_instance_graph(s, 3.5))
        cnn = tm("crystalnn", lambda: P.crystalnn_instances(crystalnn_pairs(s, include_self=True)))
        tm("taxonomy", lambda: P.annotate_instances(g, cnn))
        inst = g.graph["instances"]
        comp = tm("score_gravity", lambda: P.instance_gravity(g))
        orders = {"gravity": tm("order_gravity", lambda: P.order_by({k: comp[k]["S"] for k in inst}, inst))}
        t["score_gravity"] += t.pop("order_gravity")
        dist = {a["instance"]: a["distance"] for _, _, a in g.edges(data=True)}
        orders["longest"] = tm("score_distance", lambda: P.order_by(dist, inst, descending=True))
        orders["random"] = tm("score_random", lambda: P.order_random(inst, 42, qid))

        def bhs():
            gam = V._bhs_node_gravity(g)
            return P.order_by({k: min(gam[k[0]], gam[k[1]]) for k in inst}, inst)
        orders["bhs_min"] = tm("score_bhs", bhs)
        kept = {}
        for pol in ["ChemSPAR-v2", "Distance-Chem", "Random-Chem", "Random", "BHS-edge-min", "BHS-edge-min-noGate"]:
            okey, mode, extra, _ = V.POLICIES[pol]
            res = tm(f"prune_{pol}", lambda: P.prune_instances(g, TAU, orders[okey], mode, audit=True, extra=extra))
            rm = set(res["removed"]); kept[f"{pol}@{TAU}"] = [k for k in inst if k not in rm]
        kept["Original-4.5"] = list(inst); kept["Cutoff-3.5"] = list(g35.graph["instances"])
        z, ro = node_features([x.specie.Z for x in s], [role(x.specie.symbol) for x in s])
        d35 = {a["instance"]: a["distance"] for _, _, a in g35.edges(data=True)}
        for v in VIEWS:
            dd = d35 if v == "Cutoff-3.5" else dist
            tm(f"tensorise_{v}", lambda: instances_to_data(z, ro, kept[v], [dd[k] for k in kept[v]], 0.0, qid))
        rows.append(dict(qmof_id=qid, n_instances=len(inst), **t))
    pd.DataFrame(rows).to_csv(O / (f"prep_rep{rep}.csv" if not os.environ.get("COST_DRYRUN") else "dryrun_prep.csv"), index=False)


def train(view: str, rep: int):
    import torch
    from chemspar_v2.tensor_store import view_data
    from chemspar_v2.train import FitConfig, fit
    torch.set_num_threads(8)
    t0 = time.perf_counter()
    base = torch.load(R / "results/phase3/tensors/base.pt", weights_only=False)
    mask = torch.load(R / "results/phase3/tensors/masks.pt", weights_only=False)[view]
    split = dict(pd.read_csv(R / "data/cohort_5000_v1.csv")[["qmof_id", "split"]].values)
    data = [view_data(e, m, view) for e, m in zip(base, mask)]
    t_load = time.perf_counter() - t0
    parts = {k: [d for d in data if split[d.qmof_id] == k] for k in ("train", "val", "test")}
    r = fit(parts["train"], parts["val"], parts["test"], FitConfig(backbone="cgcnn", epochs=5), seed=0)
    out = dict(view=view, repeat=rep, load_and_mask_seconds=t_load, train_seconds_5ep=r.timing["train_seconds"],
               seconds_per_epoch_mean5=r.timing["train_seconds"] / 5, inference_seconds=r.timing["inference_seconds"],
               peak_rss_mb=r.timing["peak_rss_mb"], n_edges_directed=int(sum(d.edge_index.shape[1] for d in data)))
    (O / f"train_{view}_rep{rep}.json").write_text(json.dumps(out))


def summarise():
    P_ = pd.concat([pd.read_csv(O / f"prep_rep{k}.csv").assign(repeat=k) for k in range(REPEATS)])
    per_rep = P_.groupby("repeat").sum(numeric_only=True)
    n = P_.qmof_id.nunique()
    T_ = pd.DataFrame([json.loads(f.read_text()) for f in sorted(O.glob("train_*.json"))])
    rows = []
    for v in VIEWS:
        st = NEEDS[v] + [f"tensorise_{v}"]
        tot = per_rep[st].sum(1)                                  # seconds for n structures, per repeat
        tv = T_[T_.view == v]
        rows.append(dict(view=v, prep_stages="+".join(st), prep_seconds_per_structure_mean=tot.mean() / n,
                         prep_seconds_per_structure_sd=tot.std(ddof=1) / n,
                         prep_hours_5000_extrapolated=tot.mean() / n * 5000 / 3600,
                         crystalnn_share_pct=100 * per_rep["crystalnn"].mean() / tot.mean() if "crystalnn" in st else 0.0,
                         cgcnn_train_seconds_per_epoch=tv.seconds_per_epoch_mean5.mean(), cgcnn_train_seconds_per_epoch_sd=tv.seconds_per_epoch_mean5.std(ddof=1),
                         cgcnn_full_budget_hours_computed=tv.seconds_per_epoch_mean5.mean() * E_FULL / 3600,
                         inference_seconds_test=tv.inference_seconds.mean(), peak_rss_mb=tv.peak_rss_mb.mean(),
                         directed_edges_total=int(tv.n_edges_directed.iloc[0])))
    S = pd.DataFrame(rows).set_index("view"); S.to_csv(O / "cost_summary.csv")
    pd.set_option("display.width", 250); print(S.round(3).to_string())


def main():
    O.mkdir(parents=True, exist_ok=True)
    import torch
    (O / "environment.json").write_text(json.dumps(dict(python=platform.python_version(), platform=platform.platform(),
                                                        machine=platform.machine(), torch=torch.__version__, torch_threads_train=8,
                                                        prep="single Python process", subsample="every 25th cohort structure (200)")))
    me = str(Path(__file__).resolve())
    for k in range(REPEATS):
        if not (O / f"prep_rep{k}.csv").exists():
            subprocess.run([sys.executable, me, "prep", str(k)], check=True)
        for v in VIEWS:
            if not (O / f"train_{v}_rep{k}.json").exists():
                subprocess.run([sys.executable, me, "train", v, str(k)], check=True)
    summarise()


if __name__ == "__main__":
    a = sys.argv[1]
    {"all": main, "summarise": summarise}.get(a, lambda: None)() if a in ("all", "summarise") else (
        prep(int(sys.argv[2])) if a == "prep" else train(sys.argv[2], int(sys.argv[3])))
