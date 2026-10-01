# ChemSPAR

**An auditable, chemistry-constrained framework for sparsifying periodic graphs of metal–organic frameworks.**

ChemSPAR prunes the periodic contact graph of a crystal (every atom pair within 4.5 Å, one edge per lattice image) under
an explicit chemistry taxonomy and protection gate. Any edge-ordering policy can be plugged in, and every removal
attempt is written to an audit log. Code for the paper by M. Jalali, B. Vu, S. Chandna and M. H. Nadimi-Shahraki
(manuscript under review; citation to follow).

## What it does
1. **Contact instances.** Each periodic contact `(i, j, image)` is a separate edge, and the model input is built
   directly from the retained instances (no neighbour list is rebuilt).
2. **Taxonomy (first match wins).** Protected: `covalent_radius`, `metal_ligand`, `metal_metal_crystalnn`.
   Removable, audited: `hydrogen_bond_candidate`, `crystalnn_nonmetal`, `metal_contact`, `other_contact`.
   Radius route `d ≤ r_i + r_j + 0.40 Å` (Cordero radii); CrystalNN route via pymatgen (`configs/chemistry_v1.yaml`).
3. **Gate.** In rank order until ⌊τ·|E|⌋ contacts are removed: keep if (1) protected, (2) a non-H atom would lose its
   last contact, (3) the graph would disconnect; otherwise remove. The *ungated* partner omits check (1).
4. **Orderings.** Gravity, distance (longest first), random and an edge-level adaptation of the Black Hole Strategy
   node score (equal weights, as published).

Paper names ↔ internal keys: Gravity-Chem = `ChemSPAR-v2`, Gravity = `ChemSPAR-v2-noGate`, Distance-Chem =
`Distance-Chem`, Distance = `Distance`, Random-Chem = `Random-Chem`, Random = `Random`, BHS-Chem = `BHS-edge-min`,
BHS = `BHS-edge-min-noGate` (also listed in `configs/final_v1.yaml`).

## Installation
```bash
python3.9 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
All results were produced on CPU (training is bitwise reproducible at a fixed thread count).

## Data
Download the QMOF database (Rosen et al., *Matter* 2021; *npj Comput. Mater.* 2022) and set
`QMOF_DIR` to the folder that contains `qmof.csv`, `qmof.json` and `relaxed_structures_full_v1/` (one CIF per
structure). The fixed 5,000-structure cohort and its train/validation/test split are in `data/cohort_5000_v1.csv`.
```bash
export QMOF_DIR=/path/to/qmof_database
```
Configuration hashes and software versions used in the paper are recorded in `configs/FROZEN.json`.

## Reproducing the paper
```bash
python scripts/p2_build_cohort.py                    # cohort and split (already provided in data/)
python scripts/p2i_build_instance_views.py           # contact instances, taxonomy, removal ranks of every policy
python scripts/p2i_analyse.py                        # structural tables (taxonomy, constructions, validity)
python scripts/p3_build_tensors.py                   # tensor store + proofs that each tensor equals its view
python scripts/p4_structural_tradeoff.py 4           # structural endpoints of all views
python scripts/p4_audit_summary.py                   # audit-log summary
python scripts/p4_budget_extension.py Original-4.5   # epoch budget (also ChemSPAR-v2@0.2, ChemSPAR-v2@0.5), then: decide
python scripts/p4_run_final.py preflight             # determinism and smoke checks
python scripts/p4_run_final.py worker 0              # final fits (start workers 0, 1, 2 in parallel)
python scripts/p4_analyse_final.py cgcnn             # pre-specified paired analysis
python scripts/p4_cost_benchmark.py all              # cost benchmark (run on an otherwise idle machine)
python scripts/check_final_numbers.py                # independent recomputation of the paper's tables
```
Figures: `scripts/fig1_export.py`, then open `figures/fig1_build/render.html` through `scripts/fig1_render_server.py`
(download 3Dmol.js into `figures/fig1_build/`, e.g.
`curl -o figures/fig1_build/3Dmol-min.js https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.4.2/3Dmol-min.js`),
then `scripts/fig1_compose.py`, `scripts/ms_make_results.py` and `scripts/fig_graphical_abstract.py`.

## Tests
```bash
python -m pytest -q tests
```
Tests that compare with the earlier, pre-ChemSPAR code are skipped (that code is not part of this repository).

## Large outputs
Graph views, audit logs and per-fit predictions will be deposited on Zenodo; the DOI will be added here.

## License
MIT (see `LICENSE`).
