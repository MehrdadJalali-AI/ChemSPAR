"""ChemSPAR: auditable, chemistry-constrained sparsification of periodic graphs of metal–organic frameworks.

>>> from chemspar import build_graph, prune, orderings
>>> g = build_graph("structure.cif", cutoff=4.5)
>>> view, log = prune(g, ordering=orderings.distance, tau=0.5, gate=True)
>>> log.summary()
"""
from __future__ import annotations

__version__ = "1.0.0"

from . import orderings, taxonomy  # noqa: E402
from .audit import AUDIT_FIELDS, AuditLog, AuditRecord  # noqa: E402
from .gate import PAPER_POLICIES, POLICIES, GraphView, Policy, get_policy, prune, prune_policy  # noqa: E402
from .graph import ContactGraph, build_graph  # noqa: E402
from .orderings import Ordering  # noqa: E402

__all__ = ["__version__", "build_graph", "prune", "prune_policy", "orderings", "taxonomy", "ContactGraph",
           "GraphView", "AuditLog", "AuditRecord", "AUDIT_FIELDS", "Ordering", "Policy", "POLICIES",
           "PAPER_POLICIES", "get_policy"]
