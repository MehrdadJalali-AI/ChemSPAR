"""Graphical abstract (Elsevier: landscape, >= 1328 x 531 px, readable at small size) from real renders and results.
Left: Cd environment of qmof-0338cb2, full graph; middle: gate; right: same site after Distance-Chem at tau = 0.5;
far right: mean test MAE (final analysis) for the full graph and gated/ungated views."""
import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

import importlib.util
R = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("c", R / "scripts/fig1_compose.py"); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
plt.rcParams.update({"font.family": "Arial", "font.size": 10})
B = R / "figures/fig1_build"
D = pd.DataFrame([json.loads(Path(f).read_text()) for f in glob.glob(str(R / "results/final/fits/*.json"))])
mae = D.groupby("view").MAE.mean()

fig = plt.figure(figsize=(13.28, 5.31), dpi=100)
a = fig.add_axes([0.0, 0.08, 0.24, 0.80]); c.show(a, c.crop(B / "ax2_b_local.png"))
a.set_title("Periodic MOF graph (all contacts ≤ 4.5 Å)", fontsize=12)
g = fig.add_axes([0.245, 0.2, 0.15, 0.6]); g.set_axis_off(); g.set_xlim(0, 1); g.set_ylim(0, 1)
g.add_patch(FancyBboxPatch((0.05, 0.25), 0.9, 0.5, boxstyle="round,pad=0.02,rounding_size=0.06", fc="#e8f0fb", ec="#5a7fb5", lw=1.2))
g.text(0.5, 0.62, "ChemSPAR gate", ha="center", fontsize=13, fontweight="bold")
g.text(0.5, 0.42, "protected bonds\nnever removed;\nevery decision\nlogged", ha="center", fontsize=10, va="center", linespacing=1.15)
fig.add_artist(FancyArrowPatch((0.395, 0.5), (0.415, 0.5), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=22, lw=2, color="#333333"))
b = fig.add_axes([0.415, 0.08, 0.24, 0.80]); c.show(b, c.crop(B / "ax2_c_Distance.png"))
b.set_title("Half the contacts removed\n(distance ordering + gate; red = removed)", fontsize=11.5)
h = fig.add_axes([0.80, 0.2, 0.19, 0.62])
rows = [("Full graph", "Original-4.5", "#555555", True), ("Distance + gate", "Distance-Chem@0.5", "#2a9d8f", True),
        ("Random + gate", "Random-Chem@0.5", "#e3b23c", True), ("Random, no gate", "Random@0.5", "#e3b23c", False),
        ("BHS + gate", "BHS-edge-min@0.5", "#d1495b", True), ("BHS, no gate", "BHS-edge-min-noGate@0.5", "#d1495b", False)]
for k, (lab, v, col, filled) in enumerate(rows):
    h.barh(k, mae[v], color=col if filled else "white", edgecolor=col, lw=1.5, height=0.7)
    h.text(mae[v] + 0.003, k, f"{mae[v]:.3f}", va="center", fontsize=9)
h.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=10); h.invert_yaxis()
h.set_xlim(0.35, 0.51); h.set_xlabel("Bandgap test MAE (eV)", fontsize=10)
h.set_title("The gate improves accuracy", fontsize=12)
h.spines[["top", "right"]].set_visible(False)
fig.savefig(R / "figures/graphical_abstract.png", dpi=150, bbox_inches="tight"); fig.savefig(R / "figures/graphical_abstract.pdf", bbox_inches="tight")
