"""Generate golden fixtures with the PAPER code (paper/src/chemspar_v2, identical to tag paper-v1).

Run in the paper's pinned environment (Python 3.9.6 and paper/requirements.txt):

    python tests/data/make_golden.py tests/data/synthetic tests/data/qmof

For every CIF it writes tests/data/golden/<structure_id>.json. The structure id is the file stem, which for QMOF
CIFs is the QMOF id used to seed the paper's random orderings. Bond-class counts, removal targets, rejection
counts and hashes of the removed instances and of the audit rows are stored for every policy at tau = 0.2 and 0.5.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import warnings
from collections import Counter
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper" / "src"))
sys.path.insert(0, str(ROOT / "tests"))
warnings.filterwarnings("ignore")

from chemspar_v2 import periodic as P  # noqa: E402
from chemspar_v2 import views as V  # noqa: E402
from golden_lib import POLICY_KEYS, TAUS, fingerprint  # noqa: E402
from pymatgen.core import Structure  # noqa: E402


def paper_run(structure, sid):
    """The paper's own per-structure computation, plus the audit rows of every policy."""
    res = V.compute_structure(structure, sid, audit=True)
    g, inst = res["graph"], res["instances"]
    orders = {key: P.order_by(v, inst) for key, v in res["val"].items() if key != "distance"}
    orders["longest"] = P.order_by(res["val"]["distance"], inst, descending=True)
    orders["random"] = P.order_random(inst, 42, sid)
    out = {}
    for name in POLICY_KEYS:
        okey, mode, extra, tmax = V.POLICIES[name]
        r = P.prune_instances(g, tmax, orders[okey], mode, comp=res["comp"], qmof_id=sid, policy=name,
                              audit=True, extra=extra)
        ranks = res["ranks"][name]
        assert [k for k, _ in sorted(ranks.items(), key=lambda t: t[1])] == r["removed"]
        out[name] = (r["removed"], r["audit"])
    return res, out


def main(dirs):
    outdir = ROOT / "tests" / "data" / "golden"
    outdir.mkdir(exist_ok=True)
    src = ROOT / "paper" / "src" / "chemspar_v2"
    src_hash = hashlib.sha256(b"".join(f.name.encode() + f.read_bytes() for f in sorted(src.glob("*.py")))).hexdigest()
    for d in dirs:
        for cif in sorted(Path(d).glob("*.cif")):
            sid = cif.stem
            res, runs = paper_run(Structure.from_file(str(cif)), sid)
            n = len(res["instances"])
            fx = {
                "structure_id": sid,
                "source": str(cif.relative_to(ROOT)),
                "generated_with": {"code": "paper/src/chemspar_v2 (paper-v1)", "paper_src_sha256": src_hash,
                                   "python": platform.python_version(),
                                   **{p: version(p) for p in ("pymatgen", "numpy", "networkx", "scipy")}},
                "n_atoms": res["graph"].number_of_nodes(),
                "n_instances": n,
                "bond_class_counts": dict(Counter(a["bond_class"] for a in res["attrs"].values())),
                "views": {f"{name}@{tau}": fingerprint(n, removed, rows, tau)
                          for name, (removed, rows) in runs.items() for tau in TAUS},
            }
            # cross-check against the paper's own view and rejection bookkeeping
            for name in POLICY_KEYS:
                for tau in TAUS:
                    f = fx["views"][f"{name}@{tau}"]
                    assert len(V.view_instances(res, f"{name}@{tau}")) == n - f["removed"]
                    rj = res["rejections"][name][tau]
                    assert rj["target"] == f["target"] and rj["removed"] == f["removed"]
                    assert {k: v for k, v in rj.items() if k not in ("target", "removed")} == f["rejections"]
            (outdir / f"{sid}.json").write_text(json.dumps(fx, indent=1, sort_keys=True) + "\n")
            print(f"{sid}: {fx['n_atoms']} atoms, {n} instances")


if __name__ == "__main__":
    main(sys.argv[1:] or [str(ROOT / "tests" / "data" / "synthetic"), str(ROOT / "tests" / "data" / "qmof")])
