"""Candidate screen for a second QMOF target. No training. Training-split correlations only.

Candidates named in the strengthening prompt:
  outputs.pbe.cbm, outputs.pbe.vbm, outputs.pbe.energy_total (per atom).

energy_total in the QMOF README is the DFT total energy in eV, not a per-atom quantity.
The per-atom candidate is that column divided by info.natoms from the same table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from scipy import stats

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from chemspar_v2.paths import QMOF_DIR  # noqa: E402

OUT = R / "results" / "target2"
COLS = [
    "qmof_id",
    "info.natoms",
    "outputs.pbe.bandgap",
    "outputs.pbe.cbm",
    "outputs.pbe.vbm",
    "outputs.pbe.energy_total",
]


def dist(s: pd.Series) -> dict:
    x = s.astype(float)
    mean = float(x.mean())
    return {
        "n": int(x.shape[0]),
        "n_missing": int(x.isna().sum()),
        "mean": mean,
        "std": float(x.std(ddof=1)),
        "min": float(x.min()),
        "p25": float(x.quantile(0.25)),
        "median": float(x.median()),
        "p75": float(x.quantile(0.75)),
        "max": float(x.max()),
        "mean_absolute_deviation_from_mean": float((x - mean).abs().mean()),
    }


def corr(y: pd.Series, bandgap: pd.Series) -> dict:
    pearson = stats.pearsonr(bandgap, y)
    spearman = stats.spearmanr(bandgap, y)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(R / "data" / "cohort_5000_v1.csv")
    q = pd.read_csv(QMOF_DIR / "qmof.csv", usecols=COLS, low_memory=False)
    m = cohort.merge(q, on="qmof_id", how="left", indicator=True)
    if (m["_merge"] != "both").any():
        raise SystemExit(f"cohort ids missing from qmof.csv: {(m['_merge'] != 'both').sum()}")
    m["energy_per_atom"] = m["outputs.pbe.energy_total"] / m["info.natoms"]
    bandgap_mismatch = int((m.bandgap - m["outputs.pbe.bandgap"]).abs().gt(1e-8).sum())
    natoms_mismatch = int((m.n_atoms != m["info.natoms"]).sum())
    train = m[m.split == "train"].copy()
    full = m
    report = {
        "cohort_n": int(len(m)),
        "train_n": int(len(train)),
        "bandgap_mismatch_vs_cohort_csv": bandgap_mismatch,
        "natoms_mismatch_vs_cohort_n_atoms": natoms_mismatch,
        "missing_on_cohort": {c: int(m[c].isna().sum()) for c in COLS if c != "qmof_id"},
        "candidates": {},
    }
    series = {
        "outputs.pbe.cbm": m["outputs.pbe.cbm"],
        "outputs.pbe.vbm": m["outputs.pbe.vbm"],
        "outputs.pbe.energy_total": m["outputs.pbe.energy_total"],
        "outputs.pbe.energy_total_per_atom": m["energy_per_atom"],
    }
    train_series = {
        "outputs.pbe.cbm": train["outputs.pbe.cbm"],
        "outputs.pbe.vbm": train["outputs.pbe.vbm"],
        "outputs.pbe.energy_total": train["outputs.pbe.energy_total"],
        "outputs.pbe.energy_total_per_atom": train["energy_per_atom"],
    }
    for name in series:
        report["candidates"][name] = {
            "cohort_distribution": dist(series[name]),
            "training_split_distribution": dist(train_series[name]),
            "correlation_with_bandgap_training_split": corr(train_series[name], train.bandgap),
        }
    # gap identity: bandgap vs cbm - vbm on the cohort
    gap = m["outputs.pbe.cbm"] - m["outputs.pbe.vbm"]
    report["cbm_minus_vbm_vs_bandgap"] = {
        "max_abs_diff_eV": float((gap - m["outputs.pbe.bandgap"]).abs().max()),
        "n_diff_gt_1e-6": int((gap - m["outputs.pbe.bandgap"]).abs().gt(1e-6).sum()),
    }
    (OUT / "candidate_screen.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
