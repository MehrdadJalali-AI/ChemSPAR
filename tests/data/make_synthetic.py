"""Generate the small synthetic test crystals in tests/data/synthetic/.

These are NOT real materials. They are tiny MOF-like cells (a metal node bridged by formate-like O-C(H)-O linkers
along each axis, optionally with a guest water molecule) built so that every bond class of the taxonomy occurs,
and small enough for CrystalNN to run in seconds. They are used for invariant tests and for checking that the
package reproduces the paper code exactly. Run from the repository root:  python tests/data/make_synthetic.py
"""
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure

OUT = Path(__file__).parent / "synthetic"


def cubic_formate(metal: str, a: float, mo: float = 1.98, co: float = 1.25, ligand: str = "O",
                  water: bool = False, jitter: float = 0.0, seed: int = 0) -> Structure:
    species, cart = [metal], [[0.0, 0.0, 0.0]]
    for ax in range(3):
        perp = (ax + 1) % 3
        for x, el in ((mo, ligand), (a - mo, ligand)):
            p = [0.0, 0.0, 0.0]; p[ax] = x; species.append(el); cart.append(p)
        c = [0.0, 0.0, 0.0]; c[ax] = a / 2; species.append("C"); cart.append(c)
        h = [0.0, 0.0, 0.0]; h[ax] = a / 2; h[perp] = 1.08; species.append("H"); cart.append(h)
    if water:
        o = np.array([a / 2, a / 2, a / 2])
        species += ["O", "H", "H"]
        cart += [o.tolist(), (o + [0.76, 0.59, 0.0]).tolist(), (o + [-0.76, 0.59, 0.0]).tolist()]
    cart = np.array(cart)
    if jitter:
        cart = cart + np.random.default_rng(seed).normal(0.0, jitter, cart.shape)
    return Structure(Lattice.cubic(a), species, cart, coords_are_cartesian=True)


def main():
    OUT.mkdir(exist_ok=True)
    s = {
        "syn-zn-formate": cubic_formate("Zn", 6.40),
        "syn-cu-formate-water": cubic_formate("Cu", 6.60, water=True),
        "syn-mg-formate-jitter": cubic_formate("Mg", 6.50, jitter=0.05, seed=1),
        "syn-co-amide-water-jitter": cubic_formate("Co", 6.70, ligand="N", water=True, jitter=0.04, seed=2),
    }
    sc = cubic_formate("Zn", 6.40, jitter=0.03, seed=3)
    sc.make_supercell([2, 1, 1])
    sc.replace(0, "Cu")
    s["syn-zncu-formate-2x1x1"] = sc
    for name, st in s.items():
        st.to(filename=str(OUT / f"{name}.cif"))
        print(name, len(st), "atoms")


if __name__ == "__main__":
    main()
