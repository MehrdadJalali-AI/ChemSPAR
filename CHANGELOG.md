# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [semantic versioning](https://semver.org/).

## [1.0.0] – unreleased

First release as an installable package, accompanying the article in *Computational Materials Science*.

### Added
- `chemspar` package with a public API: `build_graph`, `prune`, `prune_policy`, `orderings`, `AuditLog`,
  `GraphView`, and the eight named policies of the paper (plus the supplementary ablations).
- Custom orderings: callables, instance-key lists, rank arrays, `orderings.from_scores` and `orderings.register`.
- Audit log export to Parquet and CSV, with `summary()`.
- Graph-view export to PyTorch Geometric (`[pyg]` extra), networkx and JSON.
- `chemspar` command-line interface.
- Tests: live equivalence with the paper code, golden fixtures generated in the paper's pinned environment,
  structural invariants, CLI smoke test.

### Unchanged
- The algorithm: taxonomy, protection routes, gate checks, orderings, tie-breaking and audit fields are copied
  from tag `paper-v1`, and the bundled configuration files are byte-identical to the paper's
  (`paper/configs/FROZEN.json`).

### Moved
- The paper's code, scripts, configurations, cohort and tests are in `paper/`, unchanged.
