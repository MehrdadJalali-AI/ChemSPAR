"""Compact tensor store: one base table of all 4.5 A periodic edge instances per structure + one boolean mask per
graph view (+ any construction instances beyond 4.5 A). Loading a view = masking; no neighbour list is rebuilt.

base entry (per structure): qmof_id, z, roles, y, inst (int32 [N, 5] = i, j, img_a, img_b, img_c; canonical order),
                            dist (float32 [N]), extras {construction: (int32 [M, 5], float32 [M])}
view mask: bool [N] (retained instances among the base instances), in cohort order.
"""
from __future__ import annotations

import torch

from .train import ViewData

CONSTRUCTIONS_WITH_EXTRAS = ("CrystalNN", "kNN-12")


def view_data(entry: dict, mask: torch.Tensor, view: str) -> ViewData:
    inst, dist = entry["inst"][mask], entry["dist"][mask]
    if view in CONSTRUCTIONS_WITH_EXTRAS and view in entry.get("extras", {}):
        xi, xd = entry["extras"][view]
        inst, dist = torch.cat([inst, xi]), torch.cat([dist, xd])
    i, j, img = inst[:, 0].long(), inst[:, 1].long(), inst[:, 2:5].to(torch.int16)
    ei = torch.stack([torch.cat([i, j]), torch.cat([j, i])]) if len(i) else torch.zeros(2, 0, dtype=torch.long)
    return ViewData(z=entry["z"], roles=entry["roles"], edge_index=ei, edge_dist=torch.cat([dist, dist]).float(),
                    edge_image=torch.cat([img, -img]) if len(i) else torch.zeros(0, 3, dtype=torch.int16),
                    y=torch.tensor([entry["y"]], dtype=torch.float32), qmof_id=entry["qmof_id"], num_nodes=len(entry["z"]))


def view_keys(entry: dict, mask: torch.Tensor, view: str) -> list:
    inst = entry["inst"][mask]
    keys = [(int(a), int(b), (int(c), int(d), int(e))) for a, b, c, d, e in inst.tolist()]
    if view in CONSTRUCTIONS_WITH_EXTRAS and view in entry.get("extras", {}):
        keys += [(int(a), int(b), (int(c), int(d), int(e))) for a, b, c, d, e in entry["extras"][view][0].tolist()]
    return keys
