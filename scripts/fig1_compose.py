"""Compose Figure 1 (180 mm wide) from real renders of qmof-0338cb2 and real audit rows.
Inputs: figures/fig1_build/ax0_a_supercell.png (view along a), ax2_b_local.png, ax2_c_<policy>.png (view along c),
        figures/fig1_build/fig1_data_ax2.json (contact counts), audit rows recomputed from the CIF (Gravity-Chem, tau = 0.5).
Output: figures/fig1_main.{pdf,png}
"""
import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from chemspar_v2.paths import CIF_DIR, QMOF_DIR  # noqa: E402
B = R / "figures/fig1_build"
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6})
INK, PROT, KEEP, RM = "#222222", "#3b3f45", "#aab0b8", "#d1495b"
COL = {"Cd": "#6a4c93", "S": "#e3b23c", "N": "#3f72af", "C": "#4d5359", "H": "#f2f2f2"}
POLS = ["Gravity", "Distance", "Random", "BHS-derived"]


def crop(p, pad=20):
    im = Image.open(p).convert("RGB")
    a = np.asarray(im); m = (a < 245).any(2)
    ys, xs = np.where(m)
    return im.crop((max(xs.min() - pad, 0), max(ys.min() - pad, 0), min(xs.max() + pad, im.width), min(ys.max() + pad, im.height)))


def show(ax, im):
    ax.imshow(im); ax.set_axis_off()


def label(ax, s, x=-0.02, y=1.0):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")


def audit_rows():
    warnings.filterwarnings("ignore")
    from pymatgen.core import Structure
    from chemspar_v2 import periodic as P
    from chemspar_v2 import views as V
    from chemspar_v2.crystalnn import crystalnn_pairs
    qid = "qmof-0338cb2"
    s = Structure.from_file(CIF_DIR / f"{qid}.cif")
    g = P.build_instance_graph(s, 4.5); P.annotate_instances(g, P.crystalnn_instances(crystalnn_pairs(s, include_self=True)))
    inst = g.graph["instances"]; comp = P.instance_gravity(g)
    res = P.prune_instances(g, 0.5, P.order_random(inst, 42, qid), "corrected", comp=comp, qmof_id=qid, policy="Random-Chem")
    rows = res["audit"]
    rej = [r for r in rows if r["decision"] == "rejected"][:2]
    acc = [r for r in rows if r["decision"] == "accepted"]
    pick = sorted(rej + [acc[0], acc[1]], key=lambda r: r["rank"])
    return pick, sum(r["decision"] == "rejected" for r in rows), len(res["removed"])


def main():
    D = json.loads((B / "fig1_data_ax2.json").read_text())
    global FORM
    from pymatgen.core import Composition
    FORM = re.sub(r"([A-Za-z])(\d+)", r"\1$_{\2}$", Composition(D["formula"]).hill_formula.replace(" ", ""))
    C = D["local"]["contacts"]; N = D["local"]["nodes"]
    fig = plt.figure(figsize=(7.09, 5.3))
    # ---- row 1
    a = fig.add_axes([0.0, 0.49, 0.33, 0.50]); show(a, crop(B / "ax0_a_supercell.png")); label(a, "a")
    a.text(0.5, -0.01, f"{D['qmof_id']} ({FORM}); 2×2×1 cells, protected bonds", transform=a.transAxes, ha="center", va="top", fontsize=6.5)
    b = fig.add_axes([0.34, 0.49, 0.27, 0.50]); show(b, crop(B / "ax2_b_local.png")); label(b, "b")
    npr = sum(c["prot"] for c in C)
    b.text(0.5, -0.01, f"Cd site: {len(C)} contacts ≤ 4.5 Å ({npr} protected)", transform=b.transAxes, ha="center", va="top", fontsize=6.5)
    # ---- panel c: gate + audit (vector)
    c = fig.add_axes([0.63, 0.47, 0.37, 0.52]); c.set_xlim(0, 1); c.set_ylim(0, 1); c.set_axis_off(); label(c, "c", x=-0.03)
    c.text(0.02, 0.95, "Ordering policy", fontsize=7.5, fontweight="bold", va="top")
    for k, p in enumerate(POLS):
        c.add_patch(FancyBboxPatch((0.02 + k * 0.245, 0.80), 0.22, 0.075, boxstyle="round,pad=0.005,rounding_size=0.02", fc="#fff4e0", ec="#d9a441", lw=0.6))
        c.text(0.13 + k * 0.245, 0.8375, p, ha="center", va="center", fontsize=6.3)
    c.add_patch(FancyArrowPatch((0.5, 0.79), (0.5, 0.72), arrowstyle="-|>", mutation_scale=8, lw=0.8, color=INK))
    c.add_patch(FancyBboxPatch((0.02, 0.43), 0.96, 0.28, boxstyle="round,pad=0.005,rounding_size=0.03", fc="#e8f0fb", ec="#5a7fb5", lw=0.7))
    c.text(0.05, 0.685, "Protection gate (each candidate, in rank order)", fontsize=7.2, fontweight="bold", va="top")
    checks = ["protected class (bond)?", "non-H atom loses last contact?", "graph would disconnect?"]
    for k, t in enumerate(checks):
        y = 0.615 - k * 0.06
        c.add_patch(Circle((0.075, y), 0.022, fc="white", ec="#5a7fb5", lw=0.7)); c.text(0.075, y, str(k + 1), ha="center", va="center", fontsize=6)
        c.text(0.115, y, t, va="center", fontsize=6.5); c.text(0.95, y, "→ keep", va="center", ha="right", fontsize=6.5, color=PROT, fontweight="bold")
    c.text(0.115, 0.455, "otherwise", va="center", fontsize=6.5); c.text(0.95, 0.455, "→ remove", va="center", ha="right", fontsize=6.5, color=RM, fontweight="bold")
    rows, nrej, nrem = audit_rows()
    c.add_patch(FancyArrowPatch((0.5, 0.425), (0.5, 0.37), arrowstyle="-|>", mutation_scale=8, lw=0.8, color=INK))
    c.text(0.02, 0.345, f"Audit log excerpt (Random + gate, τ = 0.5: {nrem} removed, {nrej} blocked)", fontsize=6.6, fontweight="bold", va="top")
    hdr = ["rank", "pair", "d (Å)", "class", "decision"]
    xs = [0.03, 0.14, 0.29, 0.43, 0.80]
    for x, h in zip(xs, hdr):
        c.text(x, 0.27, h, fontsize=6, color="#555555", va="center")
    c.plot([0.02, 0.98], [0.245, 0.245], color="#999999", lw=0.5)
    for k, r in enumerate(rows):
        y = 0.205 - k * 0.055
        dec = "blocked (protected)" if r["decision"] == "rejected" else "removed"
        vals = [str(r["rank"]), f"{r['element_i']}–{r['element_j']}", f"{r['distance']:.2f}", r["bond_class"].replace("_", " "), dec]
        for x, v in zip(xs, vals):
            c.text(x, y, v, fontsize=6, va="center", color=(PROT if dec.startswith("blocked") else RM) if x == xs[-1] else INK)
    # ---- row 2: four orderings
    for k, p in enumerate(POLS):
        ax = fig.add_axes([0.005 + k * 0.25, 0.045, 0.24, 0.33]); show(ax, crop(B / f"ax2_c_{p}.png"))
        rm = [x for x in C if x[f"rm_{p}"]]
        nh = sum(1 for x in rm if "H" in (N[x["a"]]["el"], N[x["b"]]["el"]))
        ax.set_title(f"{p} + gate", fontsize=7.5, fontweight="bold", pad=1)
        ax.text(0.5, -0.02, f"{len(rm)} removed · {nh} involve H · 0 protected", transform=ax.transAxes, ha="center", va="top", fontsize=6.3)
        if k == 0:
            label(ax, "d", y=1.16)
    fig.text(0.5, 0.425, "The same Cd environment after each ordering policy under the gate, τ = 0.5 (half of all contacts removed)", ha="center", fontsize=7)
    # legend
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=PROT, lw=2.2, label="protected bond"), Line2D([], [], color=KEEP, lw=0.8, label="removable, kept"),
         Line2D([], [], color=RM, lw=1.2, label="removed")] + \
        [Line2D([], [], marker="o", ls="", ms=5, markerfacecolor=COL[e], markeredgecolor="k", markeredgewidth=0.4, label=e) for e in ("Cd", "S", "N", "C", "H")]
    fig.legend(handles=h, loc="lower center", ncol=8, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, -0.025), handlelength=1.6, columnspacing=1.2)
    fig.savefig(R / "figures/fig1_main.pdf", dpi=600, bbox_inches="tight"); fig.savefig(R / "figures/fig1_main.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
