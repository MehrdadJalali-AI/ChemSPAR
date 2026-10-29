"""Audit log: one row per attempted removal, written before the instance is (or is not) removed.

The fields and their order are those of the paper's audit logs (``periodic.prune_instances`` at ``paper-v1``).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Dict, List, Union

AUDIT_FIELDS = (
    "qmof_id", "policy", "gate_mode", "tau", "target", "rank", "atom_i", "atom_j", "image_j", "element_i",
    "element_j", "distance", "bond_class", "radius_bonded", "radius_tolerance", "crystalnn", "protection_route",
    "protected_under_policy", "check_protected", "check_degree", "check_connectivity", "g", "S", "decision",
    "reason", "removed_before_attempt",
)
"""Column order of the audit log. ``qmof_id`` holds the structure identifier (paper field name kept)."""

CHECK_VALUES = ("pass", "fail", "not_evaluated")


@dataclass
class AuditRecord:
    """One removal attempt."""

    qmof_id: str
    policy: str
    gate_mode: str
    tau: float
    target: int
    rank: int
    atom_i: int
    atom_j: int
    image_j: str
    element_i: str
    element_j: str
    distance: float
    bond_class: str
    radius_bonded: bool
    radius_tolerance: float
    crystalnn: bool
    protection_route: str
    protected_under_policy: bool
    check_protected: str
    check_degree: str
    check_connectivity: str
    g: Any
    S: Any
    decision: str
    reason: str
    removed_before_attempt: int


@dataclass
class AuditLog:
    """All removal attempts of one pruning run, in attempt order."""

    rows: List[Dict[str, Any]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return (AuditRecord(**r) for r in self.rows)

    @property
    def attempts(self) -> int:
        return len(self.rows)

    @property
    def removed(self) -> int:
        return sum(r["decision"] == "accepted" for r in self.rows)

    @property
    def rejected(self) -> int:
        return sum(r["decision"] == "rejected" for r in self.rows)

    def rejections(self) -> Dict[str, int]:
        """Rejection counts by reason, e.g. ``{"protected:metal_ligand:both": 12, "degree": 3}``."""
        return dict(Counter(r["reason"] for r in self.rows if r["decision"] == "rejected"))

    def rejections_by_check(self) -> Dict[str, int]:
        """Rejection counts by gate check (``protected``, ``degree``, ``connectivity``)."""
        return dict(Counter(r["reason"].split(":", 1)[0] for r in self.rows if r["decision"] == "rejected"))

    def summary(self) -> Dict[str, Any]:
        """Counts of attempts, removals and rejections, plus removals by bond class."""
        first = self.rows[0] if self.rows else {}
        return {
            "structure_id": first.get("qmof_id"),
            "policy": first.get("policy"),
            "gate_mode": first.get("gate_mode"),
            "tau": first.get("tau"),
            "target": first.get("target"),
            "attempts": self.attempts,
            "removed": self.removed,
            "rejected": self.rejected,
            "rejected_by_check": self.rejections_by_check(),
            "rejected_by_reason": self.rejections(),
            "removed_by_bond_class": dict(Counter(r["bond_class"] for r in self.rows if r["decision"] == "accepted")),
        }

    def to_dataframe(self):
        import pandas as pd
        return pd.DataFrame(self.rows, columns=list(AUDIT_FIELDS))

    def to_parquet(self, path: Union[str, PathLike[str]]) -> None:
        self.to_dataframe().to_parquet(path, index=False)

    def to_csv(self, path: Union[str, PathLike[str]]) -> None:
        self.to_dataframe().to_csv(path, index=False)

    def write(self, path: Union[str, PathLike[str]]) -> None:
        """Write as Parquet or CSV depending on the file extension."""
        p = str(path)
        if p.endswith(".csv"):
            self.to_csv(p)
        elif p.endswith((".parquet", ".pq")):
            self.to_parquet(p)
        else:
            raise ValueError("audit log path must end in .parquet or .csv")


__all__ = ["AUDIT_FIELDS", "AuditRecord", "AuditLog"]
