# ChemSPAR

ChemSPAR prunes the periodic contact graph of a crystal under an explicit chemistry taxonomy and protection
gate, accepts any edge-ordering policy, and writes every removal attempt to an audit log.

```bash
pip install chemspar            # or: pip install "chemspar[pyg]"
```

```python
from chemspar import build_graph, prune, orderings

g = build_graph("structure.cif")
view, log = prune(g, ordering=orderings.distance, tau=0.5, gate=True)
log.summary()
```

- [Concepts](concepts.md): contact instances, the taxonomy, the gate and the orderings.
- [Quick start notebook](quickstart.ipynb): a complete worked example.
- [Reproducing the paper](reproducing.md).
- [API reference](api.md).
