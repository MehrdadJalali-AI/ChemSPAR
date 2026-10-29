import sys
import warnings
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from helpers import DATA, SYNTHETIC, cif_id, graph_for  # noqa: E402


@pytest.fixture(params=SYNTHETIC, ids=cif_id)
def synthetic_graph(request):
    return graph_for(str(request.param))


@pytest.fixture
def small_graph():
    return graph_for(str(DATA / "synthetic" / "syn-cu-formate-water.cif"))
