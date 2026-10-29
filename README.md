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

## Contents
- [Installation](#installation)
- [Quick start: command line](#quick-start-command-line)
- [Quick start: Python](#quick-start-python)
- [Which policy should I use?](#which-policy-should-i-use)
- [Common tasks](#common-tasks)
- [Understanding the outputs](#understanding-the-outputs)
- [FAQ and troubleshooting](#faq-and-troubleshooting)
- [Reproducing the paper](#reproducing-the-paper)

## Installation

```bash
pip install chemspar              # core: graph, taxonomy, orderings, gate, audit log, command line
pip install "chemspar[pyg]"       # adds export to PyTorch Geometric (installs torch and torch_geometric)
```

Python 3.9–3.12 are tested. The latest development version can be installed from GitHub:

```bash
pip install "git+https://github.com/MehrdadJalali-AI/ChemSPAR"
```

Check the installation:

```bash
chemspar --version
chemspar policies
```

## Quick start: command line

Sparsify one structure, removing up to 50 % of its contacts with the gated distance policy:

```bash
chemspar prune my_mof.cif --policy Distance-Chem --tau 0.5 --out view.json --log audit.csv
```

The command prints a summary (contacts attempted, removed, rejected and why) and writes

- `view.json` – the retained contacts (the sparsified graph), and
- `audit.csv` – one row per removal attempt (use `.parquet` for large runs).

Other options:

```bash
chemspar prune my_mof.cif --ordering gravity --tau 0.2              # choose an ordering, gate on (default)
chemspar prune my_mof.cif --ordering distance --tau 0.5 --no-gate   # ungated partner, for comparison
chemspar prune my_mof.cif --ordering random --tau 0.5 --seed 7      # seeded random baseline
chemspar orderings                                                  # list orderings
chemspar prune --help                                               # all options
```

Any structure file that pymatgen can read works (CIF, POSCAR, ...).

## Quick start: Python

```python
from chemspar import build_graph, prune

g = build_graph("my_mof.cif")                         # periodic contact graph + bond classes
print(len(g), "contacts")
print(g.bond_class_counts())                           # e.g. {'covalent_radius': 58, 'metal_ligand': 36, ...}

view, log = prune(g, ordering="distance", tau=0.5)     # sparsify under the protection gate
print(len(view), "contacts kept")
print(view.protected_retention())                      # 1.0: no bond was removed
print(log.summary())                                   # what was removed or rejected, and why

view.to_json("view.json")
log.to_csv("audit.csv")
```

## Which policy should I use?

A policy is an **ordering** (which contacts to try first) plus the **gate** (on or off). `tau` is the share of
all contacts you want removed; the gate may stop earlier if it refuses too many removals.

| policy | ordering | gate | typical use |
|---|---|---|---|
| `Gravity-Chem` | weakest gravity score first | on | ChemSPAR's gravity ordering |
| `Distance-Chem` | longest contact first | on | simple, transparent default |
| `Random-Chem` | seeded random | on | baseline: is the ordering doing anything? |
| `BHS-Chem` | BHS-derived node score | on | Black Hole Strategy baseline |
| `Gravity`, `Distance`, `Random`, `BHS` | as above | off | ungated partners, to measure what the gate changes |

With the gate **on**, covalent bonds, metal–ligand bonds and CrystalNN metal–metal contacts are never removed.
Comparing a policy with its ungated partner shows the effect of the chemistry constraint. The paper reports
results for all eight policies; it does not single out one ordering as the most accurate for prediction.

## Common tasks

**Sparsify a whole folder of CIFs and collect a summary table**

```python
from pathlib import Path
import pandas as pd
from chemspar import build_graph, prune_policy

rows = []
for cif in sorted(Path("cifs").glob("*.cif")):
    g = build_graph(cif)                      # the file name becomes the structure id
    for tau in (0.2, 0.5):
        view, log = prune_policy(g, "Distance-Chem", tau)
        log.to_parquet(f"audit_{cif.stem}_{tau}.parquet")
        rows.append({"structure": cif.stem, "tau": tau, "contacts": len(g), "kept": len(view),
                     "rejected": log.rejected, "protected_retention": view.protected_retention()})
pd.DataFrame(rows).to_csv("summary.csv", index=False)
```

`build_graph` runs CrystalNN, which is the slow step; build each graph once and prune it as often as you like.

**Compare several policies on one structure**

```python
from chemspar import build_graph, prune_policy, PAPER_POLICIES

g = build_graph("my_mof.cif")
for name in PAPER_POLICIES:
    view, log = prune_policy(g, name, tau=0.5)
    print(f"{name:14s} kept={len(view):5d}  protected kept={view.protected_retention():.3f}")
```

**Use your own ordering**

The gate accepts any ordering. The easiest way is a score per contact (lowest score is tried first):

```python
from chemspar import build_graph, prune, orderings

g = build_graph("my_mof.cif")
scores = [g.distance(k) for k in g.instances]           # one score per contact, in g.instances order
view, log = prune(g, ordering=orderings.from_scores(scores), tau=0.3)   # shortest contacts first
```

You can also pass a function `f(graph)` that returns contacts in removal order, a list of contacts, or a rank
array. To reuse an ordering by name, register it:

```python
orderings.register("shortest_first", lambda g, seed: orderings.distance(g)[::-1])
view, log = prune(g, ordering="shortest_first", tau=0.3)
```

**Train a graph neural network on sparsified graphs** (needs `pip install "chemspar[pyg]"`)

```python
from torch_geometric.loader import DataLoader
from chemspar import build_graph, prune_policy

dataset = []
for cif, y in [("mof_a.cif", 1.23), ("mof_b.cif", 0.87)]:        # your structures and target values
    view, _ = prune_policy(build_graph(cif), "Distance-Chem", tau=0.5)
    dataset.append(view.to_pyg(y=y))
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

Each `Data` object has `z` (atomic numbers), `roles` (one-hot element role), `edge_index` (each kept contact as
two directed edges), `edge_dist` (distances in Å) and `edge_image` (lattice image of the target atom). The full,
unpruned graph is `g.to_pyg(y=...)`.

**Work with the graph in networkx**

```python
G = view.to_networkx()          # MultiGraph; nodes carry element, Z, role; edges carry distance, bond_class, ...
```

## Understanding the outputs

**Bond classes.** Every contact gets exactly one class. Protected classes are never removed under the gate.

| class | meaning | protected |
|---|---|---|
| `covalent_radius` | non-metal pair within covalent radii + 0.40 Å | yes |
| `metal_ligand` | metal–non-metal pair, by CrystalNN or covalent radii | yes |
| `metal_metal_crystalnn` | metal–metal pair listed by CrystalNN | yes |
| `hydrogen_bond_candidate` | H to N/O/F/Cl/Br/I, CrystalNN neighbour but not bonded | no |
| `crystalnn_nonmetal` | other CrystalNN non-metal neighbour, not bonded | no |
| `metal_contact` | any other contact involving a metal | no |
| `other_contact` | any other contact | no |

**Audit log** (`log.to_dataframe()`, `.to_csv()`, `.to_parquet()`): one row per attempted removal, in order.
The most useful columns:

| column | meaning |
|---|---|
| `rank` | position of the contact in the ordering |
| `atom_i`, `atom_j`, `image_j`, `element_i`, `element_j`, `distance` | which contact |
| `bond_class`, `protection_route` | its class and which route supports it (`none`, `radius`, `crystalnn`, `both`) |
| `check_protected`, `check_degree`, `check_connectivity` | `pass`, `fail` or `not_evaluated` |
| `decision`, `reason` | `accepted`/`rejected`, and why (`protected:<class>:<route>`, `degree`, `connectivity`) |

For example, to see why removals were refused:

```python
df = log.to_dataframe()
df[df.decision == "rejected"].reason.value_counts()
```

**Graph view** (`view.json`): the structure id, the settings (policy, `tau`, seed), the atoms, the retained
contacts with their distance and class, and the removed contacts in removal order.

## FAQ and troubleshooting

**Fewer contacts were removed than `tau` asked for.** The gate refused the rest: the remaining contacts are
protected, or removing them would isolate an atom or disconnect the graph. `log.summary()["rejected_by_check"]`
shows which check refused them. This is expected at high `tau`.

**`build_graph` is slow.** CrystalNN is computed for every atom; large cells take longer. Build each graph once
and reuse it for all policies and budgets.

**Random results differ between two runs.** They should not: the random ordering is seeded by `seed` (default
42) and the structure id, which is the file name by default. Renaming a file changes the id; pass
`build_graph(..., structure_id="...")` to fix it.

**Can I change the cutoff or the bonding tolerance?** Yes: `build_graph(cif, cutoff=5.0, tolerance=0.25)`. The
paper uses 4.5 Å and 0.40 Å.

**Does it work for materials other than MOFs?** It runs on any periodic structure, but the taxonomy and gate were
validated on MOFs only.

**Are results identical to the paper?** Yes, with the paper's library versions; see
[Reproducing the paper](#reproducing-the-paper).

Questions and bug reports: [GitHub issues](https://github.com/MehrdadJalali-AI/ChemSPAR/issues).

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
