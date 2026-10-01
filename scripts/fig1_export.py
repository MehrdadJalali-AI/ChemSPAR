"""Figure 1 data export (real data only): qmof-0338cb2 from its QMOF CIF and the stored instance views.
Outputs figures/fig1_build/fig1_data.json with
  supercell: atoms (element, xyz) of a 2x2x1 supercell, bonds = PROTECTED contact instances whose both ends lie in it,
             cell vectors (for the unit-cell outline)
  local:     the Cd environment of fig_case_structure (site + all periodic atom images within 4.5 A), with every stored
             contact instance among them: protected flag and removal at tau = 0.5 under the four gated orderings
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Structure

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from chemspar_v2.paths import CIF_DIR, QMOF_DIR  # noqa: E402
from chemspar_v2.bonding import PROTECTED_CLASSES  # noqa: E402
from chemspar_v2.periodic import canonical  # noqa: E402

QID, TAU, SC = "qmof-0338cb2", 0.5, (2, 2, 1)
CIF = CIF_DIR / f"{QID}.cif"
POL = {"Gravity": "rank__ChemSPAR_v2", "Distance": "rank__Distance_Chem", "Random": "rank__Random_Chem", "BHS-derived": "rank__BHS_edge_min"}


def main():
    s = Structure.from_file(CIF)
    E = next(e for e in (pd.read_parquet(f) for f in sorted(glob.glob(str(R / "results/phase2i/views/inst_*.parquet")))) if (e.qmof_id == QID).any())
    E = E[E.qmof_id == QID].reset_index(drop=True)
    E["img"] = [tuple(int(x) for x in im.split(",")) for im in E.image]
    E["prot"] = E.bond_class.isin(PROTECTED_CLASSES)
    k = int(TAU * len(E))
    for p, c in POL.items():
        E[f"rm_{p}"] = (E[c] >= 0) & (E[c] < k)
    key = {(r.i, r.j, r.img): t for t, r in enumerate(E.itertuples())}
    L = s.lattice
    # --- supercell with protected bonds
    cells = [(a, b, c) for a in range(SC[0]) for b in range(SC[1]) for c in range(SC[2])]
    idx = {}
    atoms = []
    for cell in cells:
        for i, site in enumerate(s):
            idx[(i, cell)] = len(atoms)
            atoms.append(dict(el=site.specie.symbol, xyz=L.get_cartesian_coords(site.frac_coords + np.array(cell)).round(4).tolist(), edge=False))
    bonds = []
    def atom(i, cell):                                   # add an atom image on demand (completes edge fragments)
        if (i, cell) not in idx:
            idx[(i, cell)] = len(atoms)
            atoms.append(dict(el=s[i].specie.symbol, xyz=L.get_cartesian_coords(s[i].frac_coords + np.array(cell)).round(4).tolist(), edge=True))
        return idx[(i, cell)]
    core = list(idx.items())
    for r in E[E.prot].itertuples():                   # protected bonds of every core atom, both directions
        for cell in cells:
            ci = (r.i, cell); cj = (r.j, tuple(np.array(cell) + np.array(r.img)))
            if ci in dict(core) or cj in dict(core):
                bonds.append([atom(*ci), atom(*cj)])
    bonds = sorted({tuple(sorted(b)) for b in bonds})
    # --- local Cd environment
    c0 = min(i for i, x in enumerate(s) if x.specie.symbol == "Cd")
    nodes = [(c0, (0, 0, 0))] + [(x.index, tuple(int(v) for v in np.round(x.image))) for x in s.get_neighbors(s[c0], 4.5)]
    pos = [L.get_cartesian_coords(s[n].frac_coords + np.array(g)) for n, g in nodes]
    contacts = []
    for a in range(len(nodes)):
        for b in range(a + 1, len(nodes)):
            (ia, ga), (ib, gb) = nodes[a], nodes[b]
            t = key.get(canonical(ia, ib, tuple(int(x) for x in np.array(gb) - np.array(ga))))
            if t is not None:
                r = E.loc[t]
                assert abs(np.linalg.norm(pos[b] - pos[a]) - r.distance) < 1e-3
                contacts.append(dict(a=a, b=b, d=round(float(r.distance), 3), prot=bool(r.prot), cls=r.bond_class,
                                     **{f"rm_{p}": bool(r[f"rm_{p}"]) for p in POL}))
    view_axis = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    v = L.matrix[view_axis] / np.linalg.norm(L.matrix[view_axis]); z = np.array([0, 0, 1.0])
    ax = np.cross(v, z); sn = np.linalg.norm(ax); cs = float(v @ z)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    Rm = np.eye(3) + K + K @ K * ((1 - cs) / sn ** 2) if sn > 1e-9 else np.eye(3)
    rot = lambda x: (Rm @ np.array(x)).round(4).tolist()
    for a in atoms:
        a["xyz"] = rot(a["xyz"])
    pos = [np.array(rot(p)) for p in pos]
    cellv = [rot(c) for c in (L.matrix * np.array(SC)[:, None])]
    out = dict(qmof_id=QID, formula=s.composition.reduced_formula, tau=TAU,
               supercell=dict(dims=SC, atoms=atoms, bonds=bonds, cell=cellv, view_axis=view_axis),
               local=dict(center=0, nodes=[dict(el=s[n].specie.symbol, xyz=p.round(4).tolist()) for (n, _), p in zip(nodes, pos)], contacts=contacts))
    (R / "figures/fig1_build/fig1_data.json").write_text(json.dumps(out))
    print(len(atoms), "supercell atoms,", len(bonds), "protected bonds;", len(nodes), "local atoms,", len(contacts), "local contacts")
    print({p: sum(c[f"rm_{p}"] for c in contacts) for p in POL})


if __name__ == "__main__":
    main()
