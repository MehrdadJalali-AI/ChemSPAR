# Reproducing the paper

Everything used for the paper is in `paper/`, unchanged from the git tag `paper-v1`; see `paper/README.md` for the
full sequence of scripts. Large outputs (graph views, audit logs, predictions) are archived at
[doi:10.5281/zenodo.23475531](https://doi.org/10.5281/zenodo.23475531).

## Exactness
The package's algorithm functions are the paper's functions; `tests/test_copied_code.py` compares their syntax
trees with `paper/src/chemspar_v2`, and `tests/test_paper_equivalence.py` runs both side by side and requires
identical instances, annotations, removed instances, rejection counts and audit rows.

Bit-identical removal *order* also needs the paper's library versions:

```bash
pip install -r paper/requirements.txt   # Python 3.9-3.12; the paper used 3.9.6
pip install -e . --no-deps
```

With newer pymatgen/numpy, contact distances can change in the last floating-point digit. In symmetric
structures many contacts have tied scores, and a last-digit change can swap their order. Graph construction,
taxonomy, budgets and the algorithm itself are unaffected.
