"""Location of the QMOF database (Rosen et al.). Set the environment variable QMOF_DIR to the folder that contains
qmof.csv, qmof.json and relaxed_structures_full_v1/ (one CIF per structure). The default matches the authors' layout."""
import os
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parents[4] / "QMOF/qmof_database/qmof_database"
QMOF_DIR = Path(os.environ.get("QMOF_DIR", _DEFAULT))
CIF_DIR = QMOF_DIR / "relaxed_structures_full_v1"
