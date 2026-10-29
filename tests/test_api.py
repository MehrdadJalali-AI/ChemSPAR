"""Public API: custom orderings, policies, audit log, export, configuration and CLI."""
import json

import pytest

import chemspar
from chemspar import AUDIT_FIELDS, build_graph, orderings, prune
from chemspar.cli import main


def test_version():
    assert chemspar.__version__ == "1.0.0"


def test_bundled_config_is_frozen():
    from conftest import ROOT

    from chemspar._config import CHEMISTRY_FILE, FROZEN_SHA256, config_sha256
    for name, h in FROZEN_SHA256.items():
        assert config_sha256(name) == h
    frozen = ROOT / "paper" / "configs" / "FROZEN.json"
    if frozen.exists():
        files = json.loads(frozen.read_text())["files_sha256"]
        assert files["configs/" + CHEMISTRY_FILE] == FROZEN_SHA256[CHEMISTRY_FILE]


def test_build_graph_from_structure_and_path(small_graph):
    from conftest import DATA
    from pymatgen.core import Structure
    p = DATA / "synthetic" / "syn-cu-formate-water.cif"
    g = build_graph(Structure.from_file(str(p)), structure_id="x")
    assert g.instances == small_graph.instances and g.structure_id == "x"
    assert small_graph.structure_id == "syn-cu-formate-water"


# ---------------------------------------------------------------- custom orderings
def test_callable_key_list_rank_array_and_scores_agree(small_graph):
    ref, _ = prune(small_graph, orderings.distance, 0.5)
    keys = orderings.distance(small_graph)
    ranks = [keys.index(k) for k in small_graph.instances]
    dists = [small_graph.distance(k) for k in small_graph.instances]
    for o in (lambda g: orderings.distance(g), keys, ranks, orderings.from_scores(dists, descending=True)):
        assert prune(small_graph, o, 0.5)[0].removed == ref.removed


def test_partial_ordering_only_attempts_listed(small_graph):
    keys = orderings.distance(small_graph)[:5]
    view, log = prune(small_graph, keys, 1.0)
    assert log.attempts <= 5 and set(view.removed) <= set(keys)


def test_register_custom_ordering(small_graph):
    o = orderings.register("shortest_first", lambda g, seed: orderings.distance(g)[::-1], overwrite=True)
    view, log = prune(small_graph, "shortest_first", 0.3)
    assert log.rows[0]["policy"] == "shortest_first" and o.name in orderings.available()


@pytest.mark.parametrize("bad", [[0, 0, 1], "nope"])
def test_invalid_orderings_rejected(small_graph, bad):
    with pytest.raises((ValueError, KeyError)):
        prune(small_graph, bad, 0.5)


def test_duplicate_and_foreign_keys_rejected(small_graph):
    k = small_graph.instances[0]
    with pytest.raises(ValueError):
        prune(small_graph, [k, k], 0.5)
    with pytest.raises(ValueError):
        prune(small_graph, [(0, 999, (0, 0, 0))], 0.5)


def test_tau_bounds(small_graph):
    with pytest.raises(ValueError):
        prune(small_graph, "gravity", 1.5)


def test_policy_lookup_by_paper_name_and_internal_key():
    assert chemspar.get_policy("Gravity-Chem") is chemspar.get_policy("ChemSPAR-v2")
    assert len(chemspar.PAPER_POLICIES) == 8


# ---------------------------------------------------------------- audit log and export
def test_audit_log_io_and_summary(small_graph, tmp_path):
    import pandas as pd
    _, log = prune(small_graph, "gravity", 0.5)
    log.to_parquet(tmp_path / "a.parquet"); log.to_csv(tmp_path / "a.csv")
    df = pd.read_parquet(tmp_path / "a.parquet")
    assert list(df.columns) == list(AUDIT_FIELDS) and len(df) == log.attempts
    assert len(pd.read_csv(tmp_path / "a.csv")) == log.attempts
    s = log.summary()
    assert s["attempts"] == s["removed"] + s["rejected"]
    assert next(iter(log)).decision in ("accepted", "rejected")


def test_json_and_networkx_export(small_graph, tmp_path):
    view, _ = prune(small_graph, "gravity", 0.5)
    d = json.loads(view.to_json(tmp_path / "v.json"))
    assert d["format"] == "chemspar-view" and len(d["retained"]) == len(view)
    assert len(d["retained"]) + len(d["removed"]) == d["n_instances"]
    assert view.to_networkx().number_of_edges() == len(view)


def test_pyg_export_roundtrip(small_graph):
    pytest.importorskip("torch_geometric")
    from chemspar.export import data_instances
    view, _ = prune(small_graph, "gravity", 0.5)
    d = view.to_pyg(y=1.0)
    assert d.edge_index.shape[1] == 2 * len(view) and d.num_nodes == len(small_graph.elements)
    assert data_instances(d) == sorted(view.instances)


# ---------------------------------------------------------------- CLI
def test_cli_prune_smoke(tmp_path, capsys):
    from conftest import DATA
    cif = DATA / "synthetic" / "syn-zn-formate.cif"
    out, log = tmp_path / "view.json", tmp_path / "log.parquet"
    assert main(["prune", str(cif), "--ordering", "distance", "--tau", "0.5", "--out", str(out),
                 "--log", str(log)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["attempts"] == summary["removed"] + summary["rejected"]
    assert out.exists() and log.exists()
    assert main(["prune", str(cif), "--policy", "Random", "--tau", "0.2", "--log", str(tmp_path / "l.csv")]) == 0


def test_cli_lists(capsys):
    assert main(["policies"]) == 0 and "Gravity-Chem" in capsys.readouterr().out
    assert main(["orderings"]) == 0 and "distance" in capsys.readouterr().out
