"""Phase 2: build the fixed 5,000-MOF cohort and its single fixed split (frozen; used by every method).

Procedure (deterministic):
  1. Candidate order = numpy.random.default_rng(COHORT_SEED).permutation of all 20,372 qmof_ids in qmof.csv (sorted first).
  2. Walk the permutation; a structure is eligible iff (a) outputs.pbe.bandgap is present, (b) its CIF in
     relaxed_structures_full_v1 parses with pymatgen, (c) the 4.5 A minimum-image graph is connected.
  3. Keep the first 5,000 eligible structures in permutation order.
  4. Split: default_rng(SPLIT_SEED).permutation of the 5,000 -> first 3,500 train, next 750 val, last 750 test.
Outputs: data/cohort_5000_v1.csv (qmof_id, split, bandgap, n_atoms, n_edges_4p5, perm_rank),
         data/cohort_5000_v1_exclusions.csv (every rejected candidate examined, with reason).
"""
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

RESUB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RESUB / "src"))
from chemspar_v2.paths import CIF_DIR, QMOF_DIR  # noqa: E402
QMOF = QMOF_DIR
CIF = QMOF / "relaxed_structures_full_v1"
COHORT_SEED, SPLIT_SEED, N = 20260929, 20260930, 5000


def check(args):
    rank, qid, bg = args
    warnings.filterwarnings("ignore")
    import networkx as nx
    from pymatgen.core import Structure
    from chemspar_v2.graph import build_cutoff_graph
    if not np.isfinite(bg):
        return dict(perm_rank=rank, qmof_id=qid, eligible=False, reason="no_bandgap")
    try:
        s = Structure.from_file(CIF / f"{qid}.cif")
    except Exception as exc:
        return dict(perm_rank=rank, qmof_id=qid, eligible=False, reason=f"cif_parse:{type(exc).__name__}")
    g = build_cutoff_graph(s, 4.5)
    if g.number_of_edges() == 0 or not nx.is_connected(g):
        return dict(perm_rank=rank, qmof_id=qid, eligible=False, reason="disconnected_4p5", n_atoms=len(s))
    return dict(perm_rank=rank, qmof_id=qid, eligible=True, reason="", bandgap=float(bg), n_atoms=len(s),
                n_edges_4p5=g.number_of_edges())


def main():
    q = pd.read_csv(QMOF / "qmof.csv", low_memory=False)[["qmof_id", "outputs.pbe.bandgap"]].sort_values("qmof_id").reset_index(drop=True)
    perm = np.random.default_rng(COHORT_SEED).permutation(len(q))
    cand = [(r, q.qmof_id[i], q["outputs.pbe.bandgap"][i]) for r, i in enumerate(perm)]
    rows, start = [], 0
    with Pool(9) as pool:
        while sum(r["eligible"] for r in rows) < N and start < len(cand):
            batch = cand[start:start + 1000]; start += 1000
            rows += pool.map(check, batch, chunksize=4)
    df = pd.DataFrame(rows).sort_values("perm_rank")
    elig = df[df.eligible].head(N).copy()
    last_rank = int(elig.perm_rank.max())
    examined = df[df.perm_rank <= last_rank]
    examined[~examined.eligible].to_csv(RESUB / "data/cohort_5000_v1_exclusions.csv", index=False)
    order = np.random.default_rng(SPLIT_SEED).permutation(len(elig))
    split = np.empty(len(elig), dtype=object)
    split[order[:3500]], split[order[3500:4250]], split[order[4250:]] = "train", "val", "test"
    elig["split"] = split
    elig = elig[["qmof_id", "split", "bandgap", "n_atoms", "n_edges_4p5", "perm_rank"]]
    elig.to_csv(RESUB / "data/cohort_5000_v1.csv", index=False)
    print("examined", len(examined), "| excluded", int((~examined.eligible).sum()), examined[~examined.eligible].reason.value_counts().to_dict())
    print(elig.split.value_counts().to_dict(), "| atoms mean %.1f max %d | edges mean %.1f" % (elig.n_atoms.mean(), elig.n_atoms.max(), elig.n_edges_4p5.mean()))


if __name__ == "__main__":
    (RESUB / "data").mkdir(exist_ok=True)
    main()
