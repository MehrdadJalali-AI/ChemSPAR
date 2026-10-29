"""Fingerprints shared by the golden-fixture generator (paper code) and the golden tests (package).

A fingerprint records, for one structure, policy and tau: the number of instances, the removal target, the
number of removed instances, the rejection counts by reason, a hash of the removed instance keys in removal
order, and hashes of the audit rows (discrete fields exactly; float fields rounded to 10 decimals).
"""
from __future__ import annotations

import hashlib
import json

TAUS = (0.2, 0.5)

# paper internal keys of all policies, in the order of the paper's views.POLICIES
POLICY_KEYS = ("ChemSPAR-v2", "ChemSPAR-v2-HBProtect", "ChemSPAR-v2-noGate", "Random-Chem", "Distance-Chem",
               "BHS-edge-min", "BHS-edge-prod", "Random", "Distance", "BHS-edge-min-noGate",
               "ChemSPAR-v2-massfree", "ChemSPAR-v2-chemmass", "ChemSPAR-v2-topomass")
MAIN_POLICY_KEYS = ("ChemSPAR-v2", "ChemSPAR-v2-noGate", "Distance-Chem", "Distance", "Random-Chem", "Random",
                    "BHS-edge-min", "BHS-edge-min-noGate")

DISCRETE = ("policy", "gate_mode", "rank", "atom_i", "atom_j", "image_j", "element_i", "element_j", "bond_class",
            "radius_bonded", "radius_tolerance", "crystalnn", "protection_route", "protected_under_policy",
            "check_protected", "check_degree", "check_connectivity", "decision", "reason", "removed_before_attempt")
FLOATS = ("distance", "g", "S")


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def key_list(keys) -> list:
    return [[int(k[0]), int(k[1]), [int(x) for x in k[2]]] for k in keys]


def fingerprint(n_instances: int, removed_in_order: list, rows: list, tau: float) -> dict:
    """``removed_in_order``: all removals of a run at tau >= ``tau``; ``rows``: its audit rows."""
    target = int(tau * n_instances)
    removed = removed_in_order[:target]
    cut = [r for r in rows if r["removed_before_attempt"] < target]
    rej: dict = {}
    for r in cut:
        if r["decision"] == "rejected":
            rej[r["reason"]] = rej.get(r["reason"], 0) + 1
    disc = [[r[f] for f in DISCRETE] for r in cut]
    flo = [[None if r[f] is None else round(float(r[f]), 10) for f in FLOATS] for r in cut]
    return {"target": target, "removed": len(removed), "attempts": len(cut), "rejections": rej,
            "removed_sha256": _h(key_list(removed)), "audit_discrete_sha256": _h(disc),
            "audit_float_sha256": _h(flo)}
