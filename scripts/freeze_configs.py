"""Freeze the method definition: write configs/FROZEN.json with SHA-256 hashes of the chemistry config,
the baselines config and the cohort file, plus a canonical-JSON hash of the parsed chemistry config.
Re-running this script is the ONLY sanctioned way to change the frozen definition (and must be recorded in the
revision tracker). Unit tests (tests/test_frozen.py) fail if any frozen file changes without re-freezing."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import yaml

RESUB = Path(__file__).resolve().parents[1]
import sys  # noqa: E402
sys.path.insert(0, str(RESUB / "src"))
from chemspar_v2.paths import QMOF_DIR  # noqa: E402
FILES = ["configs/chemistry_v1.yaml", "configs/baselines_v1.yaml", "configs/final_v1.yaml", "data/cohort_5000_v1.csv"]
OPTIONAL = ["results/final/budget_extension/budget_decision.json"]
QMOF = QMOF_DIR


def tensor_proof():
    """Summary of results/phase3/tensors/tensor_manifest.csv if the tensor store exists (else 'not yet built')."""
    import pandas as pd
    m = RESUB / "results/phase3/tensors/tensor_manifest.csv"
    if not m.exists():
        return "not yet built"
    df = pd.read_csv(m)
    import torch
    fin = yaml.safe_load((RESUB / "configs/final_v1.yaml").read_text())["views"].get("identical_views", {})
    masks = torch.load(RESUB / "results/phase3/tensors/masks.pt", weights_only=False)
    ident = {a: b for a, b in fin.items() if a in masks and b in masks and all(bool((x == y).all()) for x, y in zip(masks[a], masks[b]))}
    return {"identical_views": ident, "manifest_sha256": sha(m), "base_sha256": sha(RESUB / "results/phase3/tensors/base.pt"),
            "masks_sha256": sha(RESUB / "results/phase3/tensors/masks.pt"),
            "views": {r.view: {"proof1_all_structures_tensor_equals_stored_view": bool(r.proof1_all_pass),
                               "proof2_independent_recompute_checked": int(r.proof2_checked),
                               "proof2_all_pass": bool(r.proof2_all_pass)} for r in df.itertuples()}}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canonical(path: Path) -> str:
    return hashlib.sha256(json.dumps(yaml.safe_load(path.read_text()), sort_keys=True, default=str).encode()).hexdigest()


def main():
    chem = yaml.safe_load((RESUB / "configs/chemistry_v1.yaml").read_text())
    base = yaml.safe_load((RESUB / "configs/baselines_v1.yaml").read_text())
    import platform
    pkgs = ["pymatgen", "numpy", "scipy", "networkx", "pandas", "pyarrow", "torch", "torch_geometric", "scikit-learn", "ase", "PyYAML"]
    frozen = {
        "method": chem["method"],
        "method_framing": "ChemSPAR: auditable, chemistry-constrained sparsification framework; gravity (internal key ChemSPAR-v2), distance, random and BHS-derived orders are alternative ordering policies under the same protection gate, each with an ungated partner (configs/final_v1.yaml; author decision 2026-09-30).",
        "method_framing_superseded": "ChemSPAR-v2 combines gravity-based edge ordering with explicit, auditable chemistry constraints. (Phase 3; superseded 2026-09-30)",
        "optional_files_sha256": {f: sha(RESUB / f) for f in OPTIONAL if (RESUB / f).exists()},
        "chemistry_version": chem["version"].split()[0],
        "files_sha256": {f: sha(RESUB / f) for f in FILES},
        "chemistry_canonical_sha256": canonical(RESUB / "configs/chemistry_v1.yaml"),
        "score": {"primary": chem["score"]["primary"], "formula": "S_ij = g_ij = m_i m_j / max(d_hyb_ij, 1e-6)^2 (ascending = removed first)",
                  "node_mass_weights": chem["score"]["node_mass_weights"], "role_weights": chem["score"]["role_weights"],
                  "hybrid_lambda": chem["score"]["hybrid_lambda"]},
        "taxonomy": chem["protection"]["taxonomy"],
        "hydrogen_bond_acceptors": chem["protection"]["hydrogen_bond_acceptors"],
        "radius_tolerance_angstrom": chem["bonding"]["radius_tolerance_angstrom"],
        "metals": chem["metals"],
        "gate_order": chem["gate"]["order"],
        "software": {"python": platform.python_version(), "platform": platform.platform(), **{k: version(k) for k in pkgs}},
        "pruning_unit": base["pruning_unit"],
        "representation": base["representation"],
        "qmof_source": {
            "database": "QMOF Database (local copy; no git commit available in ../QMOF)",
            "readme_stated_sources": ["QMOF Database - PBE: https://dx.doi.org/10.17172/NOMAD/2021.10.10-1 (as stated in QMOF README; not independently verified)"],
            "files_sha256": {name: sha(QMOF / name) for name in ["qmof.csv", "qmof.json", "relaxed_structures.zip", "relaxed_structures_full_v1.sha256.json"]},
            "cif_directory": "relaxed_structures_full_v1 (20,372 CIFs; per-file SHA-256 in relaxed_structures_full_v1.sha256.json)",
        },
        "tensor_proof": tensor_proof(),
    }
    (RESUB / "configs/FROZEN.json").write_text(json.dumps(frozen, indent=2, sort_keys=True))
    print(json.dumps(frozen["files_sha256"], indent=1))


if __name__ == "__main__":
    main()
