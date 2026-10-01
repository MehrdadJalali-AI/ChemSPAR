"""Pre-registered final analysis: multiplicity and comparison helpers behave as specified (synthetic data only)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import p4_analyse_final as A  # noqa: E402


def test_holm_matches_hand_computation():
    assert np.allclose(A.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(A.holm([0.5, 0.001]), [0.5, 0.002])


def test_compare_paired_difference_and_verdict():
    rng = np.random.default_rng(0)
    base = rng.normal(0.4, 0.01, 10)
    M = pd.DataFrame({"ChemSPAR-v2@0.2": base + 0.03 + rng.normal(0, 0.002, 10), "Original-4.5": base,
                      "Distance-Chem@0.2": base + rng.normal(0, 0.002, 10)})
    F = A.family(M, [("ChemSPAR-v2@0.2", "Original-4.5"), ("Distance-Chem@0.2", "Original-4.5")], "F3")
    r = F.set_index("A").loc["Gravity-Chem@0.2"]
    assert r.n_pairs == 10 and r.ci_low > 0.02 and r.verdict == "A higher MAE"
    assert F.set_index("A").loc["Distance-Chem@0.2"].verdict == "not distinguishable"


def test_identical_views_excluded_from_holm():
    rng = np.random.default_rng(1)
    base = rng.normal(0.4, 0.01, 10)
    M = pd.DataFrame({"Distance-Chem@0.2": base, "Distance@0.2": base, "Random-Chem@0.2": base + 0.01, "Random@0.2": base + 0.02})
    F = A.family(M, [("Distance-Chem@0.2", "Distance@0.2"), ("Random-Chem@0.2", "Random@0.2")], "F1")
    assert len(F) == 2
    r = F.set_index("A")
    assert r.loc["Distance-Chem@0.2"].verdict.startswith("identical") and np.isnan(r.loc["Distance-Chem@0.2"].p_t_holm)
    assert r.loc["Random-Chem@0.2"].p_t_holm == r.loc["Random-Chem@0.2"].p_t   # family of one after exclusion
