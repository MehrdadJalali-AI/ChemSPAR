# Concepts

## Contact instances
The pruning unit is a **periodic contact instance** `(i, j, image)`: atom `i` in the home cell and atom `j` in
lattice image `image`, closer than the cutoff (4.5 Å). Two images of the same atom pair are two instances.
A graph view is exactly a set of retained instances, and the model input is built from those instances directly.

## Taxonomy and protection
Each instance gets one bond class (first match wins). Two routes decide bonding: the **radius route**
(`d <= r_i + r_j + 0.40 Å`, Cordero covalent radii) and the **CrystalNN route** (either site lists the other as
a CrystalNN neighbour). The audit log records which route(s) supported each contact (`none`, `radius`,
`crystalnn`, `both`).

| class | pair | rule | protected |
|---|---|---|---|
| `covalent_radius` | non-metal pair | radius-bonded | yes |
| `metal_ligand` | metal–non-metal | CrystalNN or radius-bonded | yes |
| `metal_metal_crystalnn` | metal–metal | CrystalNN | yes |
| `hydrogen_bond_candidate` | non-metal pair | CrystalNN, not radius-bonded, H + acceptor | no |
| `crystalnn_nonmetal` | non-metal pair | CrystalNN, not radius-bonded | no |
| `metal_contact` | metal-involving | anything else | no |
| `other_contact` | non-metal pair | anything else | no |

## The gate
Instances are visited once, in ordering rank, until `floor(tau * |E|)` have been removed. A removal is rejected if
(1) the instance is protected, (2) a non-hydrogen endpoint would lose its last contact, or (3) its endpoints would
become disconnected in the cell graph. The ungated partner (`gate=False`) skips check 1 only.

Because the loop is static and deterministic, the instances removed at a smaller budget are always the first
ones removed at a larger budget.

## Orderings
| name | rule |
|---|---|
| `gravity` | ascending gravity score `m_i m_j / d_hyb^2` |
| `distance` | longest contact first |
| `random` | seeded permutation, seed `sha256("<seed>:<structure_id>")` |
| `bhs` | ascending `min(gamma_i, gamma_j)`, an edge-level adaptation of the Black Hole Strategy node score |

Ties keep canonical instance order. Pass your own ordering as a callable, a list of instance keys, a rank array,
or scores via `orderings.from_scores`.

## Paper policies
| paper name | internal key | ordering | gate |
|---|---|---|---|
| Gravity-Chem | ChemSPAR-v2 | gravity | on |
| Gravity | ChemSPAR-v2-noGate | gravity | off |
| Distance-Chem | Distance-Chem | distance | on |
| Distance | Distance | distance | off |
| Random-Chem | Random-Chem | random | on |
| Random | Random | random | off |
| BHS-Chem | BHS-edge-min | bhs | on |
| BHS | BHS-edge-min-noGate | bhs | off |

## Limitations
- The connectivity check ignores lattice image vectors.
- The rules were validated on metal–organic frameworks only.
