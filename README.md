# ChemSPAR

**Auditable, chemistry-constrained sparsification of periodic graphs of metal–organic frameworks.**

ChemSPAR prunes the periodic contact graph of a crystal (every atom pair within 4.5 Å, one edge per lattice
image) under an explicit chemistry taxonomy and protection gate. Each contact is assigned one of seven bond
classes through a covalent-radius route and a CrystalNN route; the three bonding classes (covalent, metal–ligand
and CrystalNN metal–metal contacts) are protected. The gate accepts any edge-ordering policy, visits contacts in
that order and rejects a removal if the contact is protected, if a non-hydrogen atom would lose its last contact,
or if the graph would disconnect. Every removal attempt, accepted or rejected, is written to an audit log with
the reason. ChemSPAR is a framework for building controlled, inspectable graph views; it does not claim that any
particular ordering gives more accurate property predictions.

## Installation

```bash
pip install chemspar              # core: graph construction, taxonomy, orderings, gate, audit log
pip install "chemspar[pyg]"       # adds PyTorch Geometric export
```

Python 3.9–3.12 are tested.

## Quick start

```python
from chemspar import build_graph, prune, orderings

g = build_graph("structure.cif", cutoff=4.5)          # periodic contact instances + taxonomy
print(len(g), g.bond_class_counts())

view, log = prune(g, ordering=orderings.distance, tau=0.5, gate=True)
print(len(view), view.protected_retention())          # retained instances; 1.0 under the gate

log.to_parquet("log.parquet")                          # one row per removal attempt
print(log.summary())
data = view.to_pyg()                                   # PyG Data built from the retained instances
```

Any ordering can be plugged in: a registered name (`"gravity"`, `"distance"`, `"random"`, `"bhs"`), a callable
`f(graph)` returning instance keys, a list of instance keys in removal order, a rank array, or scores via
`orderings.from_scores(scores)`. The eight policies of the paper are available by name, e.g.
`chemspar.prune_policy(g, "Distance-Chem", tau=0.5)`.

## Command line

```bash
chemspar prune input.cif --ordering distance --tau 0.5 --out view.json --log log.parquet
chemspar prune input.cif --policy Random --tau 0.2 --log log.csv
chemspar policies        # the paper's named policies
chemspar orderings       # registered orderings
```

## Reproducing the paper

The code, configurations, cohort and scripts used for the paper are in [`paper/`](paper/README.md), unchanged
from the git tag `paper-v1`. Graph views, complete audit logs and per-fit predictions are archived at
[doi:10.5281/zenodo.23475531](https://doi.org/10.5281/zenodo.23475531).

The package reproduces the paper's graph views and audit logs exactly; the test suite checks this against the
paper code on every run. Bit-identical *removal order* also requires the paper's library versions
(`paper/requirements.txt`, notably pymatgen 2024.2.20 and numpy 1.26.4): with other versions, contact distances
can differ in the last floating-point digit, which can swap contacts whose scores are tied by symmetry.

## Limitations

- The connectivity check works on the cell graph and ignores lattice image vectors, so it does not guarantee
  that the periodic network stays connected across cells.
- The taxonomy and gate were developed and validated on metal–organic frameworks (QMOF) only.
- The random ordering is seeded by the structure identifier (the CIF file name by default); pass
  `structure_id=` to `build_graph` to control it.

## Citation

If you use ChemSPAR, please cite the article:

> M. Jalali, B. Vu, S. Chandna, M. H. Nadimi-Shahraki. ChemSPAR: an auditable, chemistry-constrained framework for
> sparsifying periodic graphs of metal–organic frameworks. *Computational Materials Science* ([YEAR]).
> doi:[ARTICLE DOI]

```bibtex
@article{jalali_chemspar,
  author  = {Jalali, M. and Vu, B. and Chandna, S. and Nadimi-Shahraki, M. H.},
  title   = {{ChemSPAR}: an auditable, chemistry-constrained framework for sparsifying periodic graphs of
             metal--organic frameworks},
  journal = {Computational Materials Science},
  year    = {[YEAR]},
  doi     = {[ARTICLE DOI]}
}
```

Machine-readable metadata are in [`CITATION.cff`](CITATION.cff).

## License

MIT (see [`LICENSE`](LICENSE)).
