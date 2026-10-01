"""CrystalNN neighbour pairs mapped onto minimum-image atom pairs."""
from __future__ import annotations

import warnings

from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Structure


def crystalnn_pairs(structure: Structure, include_self: bool = False) -> dict[tuple[int, int], dict]:
    """Return {(i, j) with i<j: {"images": set of (image of j relative to i), "failed_sites": int}}.

    A pair is included if CrystalNN lists j (any image) as a neighbour of i, or i as a neighbour of j.
    Sites for which CrystalNN raises are counted in the returned '__failed_sites__' entry and skipped.
    include_self=True also records contacts of a site with its own periodic image (key (i, i)); used by the
    periodic edge-instance pipeline. The pair pipeline keeps the default (False) and is unchanged.
    """
    cnn = CrystalNN()
    pairs: dict[tuple[int, int], dict] = {}
    failed = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(structure)):
            try:
                infos = cnn.get_nn_info(structure, i)
            except Exception:
                failed += 1
                continue
            for info in infos:
                j = int(info["site_index"])
                img = tuple(int(round(x)) for x in info["image"])
                if j == i and (not include_self or img == (0, 0, 0)):
                    continue
                # store the image of the higher-index atom relative to the lower-index atom
                key, im = ((i, j), img) if i < j else ((j, i), tuple(-x for x in img))
                pairs.setdefault(key, {"images": set()})["images"].add(im)
    pairs["__failed_sites__"] = {"count": failed}
    return pairs
