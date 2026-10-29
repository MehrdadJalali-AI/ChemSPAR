import sys
import warnings
from functools import cache
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
DATA = TESTS / "data"
sys.path.insert(0, str(TESTS))
warnings.filterwarnings("ignore")

SYNTHETIC = sorted((DATA / "synthetic").glob("*.cif"))
QMOF = sorted((DATA / "qmof").glob("*.cif"))
ALL_CIFS = SYNTHETIC + QMOF


def cif_id(p: Path) -> str:
    return p.stem


@cache
def graph_for(path: str):
    from chemspar import build_graph
    return build_graph(path)


@pytest.fixture(params=SYNTHETIC, ids=cif_id)
def synthetic_graph(request):
    return graph_for(str(request.param))


@pytest.fixture
def small_graph():
    return graph_for(str(DATA / "synthetic" / "syn-cu-formate-water.cif"))
