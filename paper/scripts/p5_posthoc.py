"""Revision round 1 (R2.2, R2.3): post-hoc, exploratory paired comparison from the EXISTING final fits (no training).
Distance-Chem@0.5 vs Cutoff-3.5 (not part of the pre-specified families; no multiplicity correction; labelled post hoc).
Output: results/revision/posthoc_distance_vs_cutoff35.json"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

R = Path(__file__).resolve().parents[1]
D = pd.DataFrame([json.loads(Path(f).read_text()) for f in glob.glob(str(R / "results/final/fits/*.json"))])
M = D.pivot(index="seed", columns="view", values="MAE")
d = M["Distance-Chem@0.5"] - M["Cutoff-3.5"]
n = len(d); h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
S = pd.read_csv(R / "results/final/structural/structural_tradeoff.csv", index_col=0)
out = dict(comparison="Distance-Chem@0.5 minus Cutoff-3.5 (post hoc, exploratory)", n_pairs=n, mean_diff=d.mean(), ci_low=d.mean() - h,
           ci_high=d.mean() + h, d_z=d.mean() / d.std(ddof=1), p_t=stats.ttest_rel(M["Distance-Chem@0.5"], M["Cutoff-3.5"]).pvalue,
           p_wilcoxon=stats.wilcoxon(d, method="exact").pvalue, n_distance_lower=int((d < 0).sum()), per_seed=d.round(5).tolist(),
           retained_pct_distance_chem=S.loc["Distance-Chem@0.5", "retained_instances_pct"], retained_pct_cutoff35=S.loc["Cutoff-3.5", "retained_instances_pct"],
           structures_losing_protected_cutoff35=int(S.loc["Cutoff-3.5", "structures_losing_protected_tol040"]))
(R / "results/revision").mkdir(parents=True, exist_ok=True)
(R / "results/revision/posthoc_distance_vs_cutoff35.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
