"""Shared paths and cached graphs for the test suite (kept out of conftest.py so the name cannot clash with
paper/tests/conftest.py)."""
from functools import cache
from pathlib import Path

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
DATA = TESTS / "data"

SYNTHETIC = sorted((DATA / "synthetic").glob("*.cif"))
QMOF = sorted((DATA / "qmof").glob("*.cif"))
ALL_CIFS = SYNTHETIC + QMOF


def cif_id(p: Path) -> str:
    return p.stem


@cache
def graph_for(path: str):
    from chemspar import build_graph
    return build_graph(path)
