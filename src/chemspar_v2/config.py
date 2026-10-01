from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

RESUB = Path(__file__).resolve().parents[2]
DEFAULT_CHEMISTRY = RESUB / "configs" / "chemistry_v1.yaml"


@lru_cache(maxsize=4)
def load_chemistry(path: str | None = None) -> dict:
    with open(path or DEFAULT_CHEMISTRY, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
