"""Element classification for chemspar_v2 (versioned by configs/chemistry_v1.yaml)."""
from __future__ import annotations

from functools import lru_cache

from pymatgen.core import Element

from .config import load_chemistry

# The published pipeline's hard-coded metal set (src/experiments/run_all.py:27), kept only for comparisons.
LEGACY_METALS = frozenset({
    "Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Sc", "Ti", "Zr", "V",
    "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Al", "Ga", "In",
})
HALIDES = frozenset({"F", "Cl", "Br", "I"})


@lru_cache(maxsize=256)
def is_metal(symbol: str) -> bool:
    cfg = load_chemistry()["metals"]
    if symbol in cfg.get("forced_nonmetals", []):
        return False
    if symbol in cfg.get("forced_metals", []):
        return True
    return bool(Element(symbol).is_metal)


def role(symbol: str) -> str:
    """Element role used by node masses and GNN features (same categories as the legacy code)."""
    if is_metal(symbol):
        return "metal"
    return {"O": "oxygen", "N": "nitrogen", "C": "carbon", "H": "hydrogen"}.get(
        symbol, "halide" if symbol in HALIDES else "other")


def classify(symbol: str) -> dict:
    el = Element(symbol)
    return {
        "element": symbol,
        "Z": el.Z,
        "pymatgen_is_metal": bool(el.is_metal),
        "pymatgen_is_metalloid": bool(el.is_metalloid),
        "v2_metal": is_metal(symbol),
        "legacy_metal": symbol in LEGACY_METALS,
        "v2_role": role(symbol),
    }
