"""Bundled, versioned configuration files.

The two YAML files in ``chemspar/data`` are byte-identical copies of ``paper/configs/chemistry_v1.yaml`` and
``paper/configs/baselines_v1.yaml`` used for the paper; their SHA-256 hashes are recorded in
``paper/configs/FROZEN.json`` and checked by the test suite.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from importlib.resources import files

import yaml

CHEMISTRY_FILE = "chemistry_v1.yaml"
BASELINES_FILE = "baselines_v1.yaml"

#: SHA-256 of the bundled files, identical to ``paper/configs/FROZEN.json``.
FROZEN_SHA256 = {
    CHEMISTRY_FILE: "c9751209a35659b84b5b15e6591447cf2982edc40e69a6dd7f364a2e31d6b917",
    BASELINES_FILE: "808fa26413a531580b42944c5fcc0bc50fe4824badf59edb2ac57de026e947f0",
}


def _read_bytes(name: str) -> bytes:
    return (files("chemspar") / "data" / name).read_bytes()


def config_sha256(name: str) -> str:
    """SHA-256 of a bundled configuration file."""
    return hashlib.sha256(_read_bytes(name)).hexdigest()


@lru_cache(maxsize=1)
def load_chemistry() -> dict:
    """The chemistry definition (taxonomy, tolerances, gravity weights) used in the paper."""
    return yaml.safe_load(_read_bytes(CHEMISTRY_FILE).decode("utf-8"))


@lru_cache(maxsize=1)
def bhs_weights() -> tuple[float, float, float]:
    """BHS node-score weights (degree, betweenness, neighbourhood influence) used in the paper."""
    cfg = yaml.safe_load(_read_bytes(BASELINES_FILE).decode("utf-8"))["bhs_node_score"]["weights"]
    return (cfg["degree"], cfg["betweenness"], cfg["neighbourhood_influence"])
