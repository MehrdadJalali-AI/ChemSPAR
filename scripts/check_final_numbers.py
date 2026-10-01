"""Final pre-submission check (no experiments): independently recompute manuscript tables from saved outputs and
compare with the generated table files. Writes results/final/check/check_results.json and prints a summary.
Recomputation paths are deliberately different from the generators (raw per-fit JSON, raw parquet, scipy directly)."""
import glob
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path(__file__).resolve().parents[1]
T = R / "manuscript/tables"
OUT = R / "results/final/check"
res = {}


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (m - r) * p[i])); adj[i] = run
    return adj


def nums(line):
    return [float(x.replace(",", "")) for x in re.findall(r"[-+]?\d[\d,]*\.?\d*", line.replace("$\\pm$", " "))]


# ---- Table 1: taxonomy
tx = pd.read_csv(R / "results/phase2i/p2i_taxonomy.csv", header=None, index_col=0, skiprows=1)[1]
cls = {k[6:]: int(v) for k, v in tx.items() if k.startswith("class_")}
prot = cls["covalent_radius"] + cls["metal_ligand"] + cls["metal_metal_crystalnn"]
res["T1"] = dict(class_sum=sum(cls.values()), instances=int(tx["instances"]), sum_ok=sum(cls.values()) == int(tx["instances"]),
                 protected=prot, protected_pct=round(100 * prot / int(tx["instances"]), 3))

# ---- Tables 3-5: consistency of "structures losing protected" (T3/T4) with "structures differing" (T5) for ungated views
S = pd.read_csv(R / "results/final/structural/structural_tradeoff.csv", index_col=0)
G = pd.read_csv(R / "results/final/structural/gated_vs_ungated_identity.csv")
pair = {"ChemSPAR_v2": ("ChemSPAR-v2", "ChemSPAR-v2-noGate"), "Distance_Chem": ("Distance-Chem", "Distance"),
        "Random_Chem": ("Random-Chem", "Random"), "BHS_edge_min": ("BHS-edge-min", "BHS-edge-min-noGate")}
rows = []
for r in G.itertuples():
    g, u = pair[r.gated]
    lu = int(S.loc[f"{u}@{r.tau}", "structures_losing_protected_tol040"]); lg = int(S.loc[f"{g}@{r.tau}", "structures_losing_protected_tol040"])
    rows.append(dict(order=g, tau=r.tau, structures_differing_T5=r.structures_differing, ungated_losing_protected_T3T4=lu, gated_losing=lg,
                     retention_ungated=round(S.loc[f"{u}@{r.tau}", "protected_retention_tol040_pct"], 4),
                     consistent=(lg == 0) and (lu <= r.structures_differing)))
res["T345"] = rows

# ---- Table 6 + 7 from raw per-fit JSON
D = pd.DataFrame([json.loads(Path(f).read_text()) for f in glob.glob(str(R / "results/final/fits/*.json"))])
M = D.pivot(index="seed", columns="view", values="MAE")
t6 = {v: (round(M[v].mean(), 3), round(M[v].std(ddof=1), 3)) for v in M}
gen6 = {}
inv = {"Original-4.5": "Original-4.5", "Cutoff-3.5": "Cutoff-3.5", "Distance-Chem": "Distance-Chem@0.5", "Gravity-Chem": "ChemSPAR-v2@0.5",
       "Random-Chem": "Random-Chem@0.5", "Random": "Random@0.5", "BHS-Chem": "BHS-edge-min@0.5", "BHS": "BHS-edge-min-noGate@0.5"}
for line in (T / "tab_pred.tex").read_text().splitlines():
    if "&" in line:
        name = line.split("&")[0].strip(); x = nums(line.split("&", 2)[2])
        gen6[inv[name]] = (x[0], x[1])
res["T6"] = [dict(view=v, recomputed=t6[v], table=gen6[v], ok=t6[v] == gen6[v]) for v in gen6]
fam = {"F1": [("Random-Chem@0.5", "Random@0.5"), ("BHS-edge-min@0.5", "BHS-edge-min-noGate@0.5")],
       "F2": list(combinations(["ChemSPAR-v2@0.5", "Distance-Chem@0.5", "Random-Chem@0.5", "BHS-edge-min@0.5"], 2)),
       "F3": [(g, "Original-4.5") for g in ["ChemSPAR-v2@0.5", "Distance-Chem@0.5", "Random-Chem@0.5", "BHS-edge-min@0.5"]]}
t7 = []
for f, prs in fam.items():
    out = []
    for a, b in prs:
        d = M[a] - M[b]; n = len(d); mu = d.mean(); sd = d.std(ddof=1); h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n)
        out.append(dict(family=f, A=a, B=b, diff=round(mu, 4), lo=round(mu - h, 4), hi=round(mu + h, 4), dz=round(mu / sd, 2),
                        p=stats.ttest_rel(M[a], M[b]).pvalue, n_better=int((d < 0).sum())))
    for o, ph in zip(out, holm([o["p"] for o in out])):
        o["p_holm"] = ph
    t7 += out
gen7 = [l for l in (T / "tab_tests.tex").read_text().splitlines() if "&" in l]
mism = []
for o, l in zip(t7, gen7):
    x = nums(l.split("&", 2)[2])
    ph = "<0.001" if o["p_holm"] < 0.001 else f"{o['p_holm']:.3f}"
    ok = abs(x[0] - o["diff"]) < 1e-9 and abs(x[1] - o["lo"]) < 1e-9 and abs(x[2] - o["hi"]) < 1e-9 and abs(x[3] - o["dz"]) < 1e-9 and ph in l
    if not ok:
        mism.append((o, l))
res["T7"] = dict(n=len(t7), mismatches=[str(m) for m in mism], rows=[{k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()} for o in t7])

# ---- Table 8 arithmetic
C = pd.read_csv(R / "results/final/cost/cost_summary.csv", index_col=0)
res["T8"] = dict(hours_check=[dict(view=v, s_per_struct=round(C.loc[v].prep_seconds_per_structure_mean, 4),
                                   hours=round(C.loc[v].prep_seconds_per_structure_mean * 5000 / 3600, 3),
                                   table_hours=round(C.loc[v].prep_hours_5000_extrapolated, 3)) for v in C.index],
                 saving=round(C.loc["Original-4.5"].cgcnn_full_budget_hours_computed - C.loc["Distance-Chem@0.5"].cgcnn_full_budget_hours_computed, 3),
                 full_h=round(C.loc["Original-4.5"].cgcnn_full_budget_hours_computed, 3), dc_h=round(C.loc["Distance-Chem@0.5"].cgcnn_full_budget_hours_computed, 3),
                 rss_gb={v: round(C.loc[v].peak_rss_mb / 1000, 2) for v in C.index})

# ---- Table 9: removed = sum floor(0.5 |E_i|) from raw per-structure instance counts
coh = pd.read_csv(R / "data/cohort_5000_v1.csv")
n_i = pd.concat(pd.read_parquet(f, columns=["qmof_id"]) for f in sorted(glob.glob(str(R / "results/phase2i/views/inst_*.parquet")))).qmof_id.value_counts()
A = pd.read_csv(R / "results/final/audit/rejections_tau0.5.csv", index_col=0)
res["T9"] = dict(sum_floor=int(np.floor(0.5 * n_i.values).sum()), n_structures=int(len(n_i)), table_removed={k: int(v) for k, v in A.removed.items()},
                 ok=all(int(v) == int(np.floor(0.5 * n_i.values).sum()) for v in A.removed))

# ---- CrystalNN-only protection route (saved taxonomy columns)
E = pd.concat(pd.read_parquet(f, columns=["bond_class", "route", "protected"]) for f in sorted(glob.glob(str(R / "results/phase2i/views/inst_*.parquet"))))
P = E[E.protected]
res["route"] = dict(protected=int(len(P)), by_route=P.route.value_counts().to_dict(),
                    by_class_route={f"{a}|{b}": int(n) for (a, b), n in P.groupby(["bond_class", "route"]).size().items()})
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "check_results.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps({k: (v if k not in ("T7",) else {"n": v["n"], "mismatches": v["mismatches"]}) for k, v in res.items()}, indent=1, default=str))
