import sys
import warnings
from pathlib import Path

import pytest

RESUB = Path(__file__).resolve().parents[1]
REPO = RESUB.parent
sys.path.insert(0, str(RESUB / "src"))
from chemspar_v2.paths import CIF_DIR, QMOF_DIR  # noqa: E402
sys.path.insert(0, str(REPO / "src"))
CIF_DIR = CIF_DIR
warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def cif_dir():
    if not CIF_DIR.exists():
        pytest.skip("full QMOF CIF directory not available")
    return CIF_DIR


@pytest.fixture(scope="session")
def load_structure(cif_dir):
    from pymatgen.core import Structure

    def _load(qmof_id):
        return Structure.from_file(cif_dir / f"{qmof_id}.cif")
    return _load
