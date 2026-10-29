"""Command-line interface: ``chemspar prune``, ``chemspar policies`` and ``chemspar orderings``."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__, orderings
from .gate import POLICIES, get_policy, prune
from .graph import DEFAULT_CUTOFF, build_graph


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chemspar", description="Auditable, chemistry-constrained sparsification "
                                "of periodic contact graphs.")
    p.add_argument("--version", action="version", version=f"chemspar {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("prune", help="prune one structure and write the graph view and audit log")
    pr.add_argument("structure", help="CIF (or any file pymatgen can read)")
    g = pr.add_mutually_exclusive_group()
    g.add_argument("--ordering", default="gravity", help="ordering name (see `chemspar orderings`)")
    g.add_argument("--policy", help="named paper policy, e.g. Distance-Chem (see `chemspar policies`)")
    pr.add_argument("--tau", type=float, required=True, help="removal budget, fraction of instances")
    pr.add_argument("--no-gate", action="store_true", help="ungated partner: degree and connectivity checks only")
    pr.add_argument("--seed", type=int, default=42, help="seed of the random ordering (default 42)")
    pr.add_argument("--id", dest="structure_id", help="structure identifier (default: file stem)")
    pr.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF, help="contact cutoff in Å (default 4.5)")
    pr.add_argument("--out", help="write the graph view as JSON")
    pr.add_argument("--log", help="write the audit log (.parquet or .csv)")

    sub.add_parser("policies", help="list the named policies of the paper")
    sub.add_parser("orderings", help="list registered orderings")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "policies":
        for name, p in POLICIES.items():
            tag = "" if p.paper_main else "  (ablation)"
            print(f"{name:24s} ordering={p.ordering:13s} gate={'on' if p.gate else 'off'}  "
                  f"internal={p.internal_key}{tag}")
        return 0
    if args.command == "orderings":
        for name in orderings.available():
            print(f"{name:13s} {orderings.get(name).description}")
        return 0

    graph = build_graph(args.structure, cutoff=args.cutoff, structure_id=args.structure_id)
    if args.policy:
        pol = get_policy(args.policy)
        if args.no_gate:
            print("error: --no-gate cannot be combined with --policy", file=sys.stderr)
            return 2
        view, log = prune(graph, pol.ordering, args.tau, gate=pol.gate, seed=args.seed,
                          extra_protected=pol.extra_protected, policy=pol.internal_key)
    else:
        view, log = prune(graph, args.ordering, args.tau, gate=not args.no_gate, seed=args.seed)
    if args.out:
        view.to_json(args.out)
    if args.log:
        log.write(args.log)
    summary = log.summary()
    summary.update(instances=len(graph), retained=len(view), protected_retention=view.protected_retention())
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
