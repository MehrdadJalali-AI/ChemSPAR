"""Figure (manuscript): schematic workflow of the ChemSPAR framework. A flowchart only; no crystal structure is drawn.
Every label is taken from the frozen definitions: configs/chemistry_v1.yaml (taxonomy, gate order, tolerance),
configs/baselines_v1.yaml (base graph, pruning unit), configs/final_v1.yaml (ordering policies, ungated meaning),
src/chemspar_v2/periodic.py (_check, prune_instances).
Output: figures/fig_workflow_schematic.{pdf,png}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

R = Path(__file__).resolve().parents[1]
CHEM = yaml.safe_load((R / "configs/chemistry_v1.yaml").read_text())
BASE = yaml.safe_load((R / "configs/baselines_v1.yaml").read_text())
O = R / "figures"

INK, PROT, REM, GATE, POLC, OUT = "#222222", "#1b1b1b", "#b22222", "#e8f0fb", "#fff4e0", "#eaf6ea"


def box(ax, x, y, w, h, title, body, fc, fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018", fc=fc, ec=INK, lw=1.0))
    ax.text(x + w / 2, y + h - 0.028, title, ha="center", va="top", fontsize=9.3, fontweight="bold", color=INK)
    ax.text(x + 0.012, y + h - 0.07, body, ha="left", va="top", fontsize=fs, color=INK, linespacing=1.35)


def arrow(ax, p, q, text=None, color=INK, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=1.2, color=color, ls=ls))
    if text:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.018, text, ha="center", va="bottom", fontsize=7.6, color=color)


def main():
    O.mkdir(parents=True, exist_ok=True)
    tol = CHEM["bonding"]["radius_tolerance_angstrom"]
    cut = BASE["base_graph"]["cutoff_angstrom"]
    tax = CHEM["protection"]["taxonomy"]
    prot = [t["class"] for t in tax if t["protected"]]
    remv = [t["class"] for t in tax if not t["protected"]]
    fig, ax = plt.subplots(figsize=(15, 5.4))
    ax.set_xlim(0, 1.5); ax.set_ylim(0, 0.62); ax.axis("off")
    T, B = 0.33, 0.02                       # bottom y of the top and bottom rows
    box(ax, 0.01, T, 0.17, 0.26, "1  Input", "QMOF CIF\n(DFT-relaxed,\nperiodic cell)", "#f2f2f2")
    box(ax, 0.21, T, 0.23, 0.26, "2  Periodic contacts",
        f"all atom pairs within {cut} Å,\none edge per lattice image;\n(i, j, image) = pruning unit;\nself-image contacts included", "#f2f2f2")
    box(ax, 0.47, T, 0.33, 0.26, "3  Taxonomy (first match wins)",
        "protected (never removable):\n" + "\n".join(f"  • {c}" for c in prot) +
        "\nremovable (audited):\n" + "\n".join(f"  • {c}" for c in remv[:2]) + "\n  • " + " · ".join(remv[2:]) +
        f"\nradius route: d ≤ r_i + r_j + {tol:.2f} Å", "#f2f2f2", fs=7.8)
    box(ax, 0.83, T, 0.29, 0.26, "4  Ordering policy (plug-in)",
        "ranks the candidate contacts:\n  • Gravity (m_i m_j / d_hyb²)\n  • Distance (longest first)\n  • Random (seeded)\n"
        "  • BHS-derived (min node score)", POLC)
    box(ax, 1.15, T, 0.34, 0.26, "6a  Sparse graph view",
        "exactly the retained instances;\ntensor built from the view\n(no neighbour list rebuilt);\nverified: tensor = view", OUT)
    box(ax, 0.83, B, 0.29, 0.27, "5  Protection gate",
        "per candidate, in rank order,\nuntil ⌊τ·|E|⌋ contacts are removed:\n  ① protected class?  → keep\n"
        "  ② non-H atom would lose its\n      last contact?  → keep\n  ③ graph disconnects?  → keep\n  otherwise → remove", GATE)
    box(ax, 1.15, B, 0.34, 0.27, "6b  Audit log (every attempt)",
        "structure; atoms i, j; image; elements;\ndistance; taxonomy class; protection\nroute (CrystalNN / radius / both);\n"
        "outcome of ①–③; decision; reason; rank", OUT)
    box(ax, 0.47, B + 0.07, 0.33, 0.20, "Matched ungated partner",
        "same ordering policy, same τ;\ncheck ① removed, checks ② and ③ kept.\nIsolates the effect of chemical protection.", "#fbeaea")
    my = T + 0.13
    arrow(ax, (0.18, my), (0.21, my)); arrow(ax, (0.44, my), (0.47, my)); arrow(ax, (0.80, my), (0.83, my))
    arrow(ax, (0.975, T), (0.975, B + 0.27)); arrow(ax, (1.12, B + 0.22), (1.15, T + 0.05), "retained")
    arrow(ax, (1.12, B + 0.10), (1.15, B + 0.10), "logged")
    arrow(ax, (0.80, B + 0.17), (0.83, B + 0.17), color=REM, ls="--")
    ax.text(0.75, 0.605, "ChemSPAR: chemistry-constrained, auditable sparsification of periodic MOF graphs (schematic; no structure drawn)",
            ha="center", va="bottom", fontsize=10.5, color=INK)
    fig.savefig(O / "fig_workflow_schematic.pdf", bbox_inches="tight"); fig.savefig(O / "fig_workflow_schematic.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
