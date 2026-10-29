"""Properties that must hold for every structure, policy and budget."""
import pytest

from chemspar import PAPER_POLICIES, POLICIES, prune, prune_policy
from chemspar.taxonomy import PROTECTED_CLASSES

TAU_GRID = (0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.0)


@pytest.mark.parametrize("policy", [p for p in PAPER_POLICIES if POLICIES[p].gate])
def test_gate_never_removes_protected(synthetic_graph, policy):
    view, log = prune_policy(synthetic_graph, policy, 1.0)
    assert all(not synthetic_graph.attrs(k)["protected"] for k in view.removed)
    assert view.protected_retention() == 1.0
    for r in log.rows:
        if r["bond_class"] in PROTECTED_CLASSES:
            assert r["decision"] == "rejected" and r["check_protected"] == "fail"


def test_hbprotect_keeps_hydrogen_bond_candidates(synthetic_graph):
    view, _ = prune_policy(synthetic_graph, "Gravity-HBProtect", 1.0)
    assert all(synthetic_graph.attrs(k)["bond_class"] != "hydrogen_bond_candidate" for k in view.removed)


@pytest.mark.parametrize("policy", PAPER_POLICIES)
def test_removed_sets_are_nested_prefixes(synthetic_graph, policy):
    runs = [prune_policy(synthetic_graph, policy, t)[0].removed for t in TAU_GRID]
    for small, large in zip(runs, runs[1:]):
        assert large[:len(small)] == small


@pytest.mark.parametrize("policy", PAPER_POLICIES)
def test_each_instance_visited_at_most_once(synthetic_graph, policy):
    _, log = prune_policy(synthetic_graph, policy, 1.0)
    keys = [(r["atom_i"], r["atom_j"], r["image_j"]) for r in log.rows]
    assert len(keys) == len(set(keys))
    ranks = [r["rank"] for r in log.rows]
    assert ranks == sorted(ranks) and len(ranks) == len(set(ranks))


@pytest.mark.parametrize("tau", TAU_GRID)
@pytest.mark.parametrize("policy", PAPER_POLICIES)
def test_attempts_equal_removed_plus_rejected(synthetic_graph, policy, tau):
    view, log = prune_policy(synthetic_graph, policy, tau)
    assert log.attempts == log.removed + log.rejected
    assert log.removed == len(view.removed) <= view.target == int(tau * len(synthetic_graph))
    assert len(view) + len(view.removed) == len(synthetic_graph)
    assert sum(log.rejections().values()) == log.rejected


@pytest.mark.parametrize("policy", PAPER_POLICIES)
def test_connectivity_and_degree_preserved(synthetic_graph, policy):
    import networkx as nx
    view, _ = prune_policy(synthetic_graph, policy, 1.0)
    base_components = nx.number_connected_components(synthetic_graph.graph)
    assert nx.number_connected_components(view.graph) == base_components
    for n, d in view.graph.nodes(data=True):
        if d["element"] != "H" and synthetic_graph.graph.degree(n) > 0:
            assert view.graph.degree(n) > 0


@pytest.mark.parametrize("ordering", ["gravity", "distance", "random", "bhs"])
def test_deterministic(synthetic_graph, ordering):
    a = prune(synthetic_graph, ordering, 0.5, seed=7)
    b = prune(synthetic_graph, ordering, 0.5, seed=7)
    assert a[0].removed == b[0].removed and a[1].rows == b[1].rows


def test_random_seed_changes_order(synthetic_graph):
    from chemspar import orderings
    assert orderings.random(synthetic_graph, seed=1) != orderings.random(synthetic_graph, seed=2)


def test_ungated_partner_skips_only_the_protection_check(synthetic_graph):
    _, log = prune(synthetic_graph, "distance", 1.0, gate=False)
    assert {r["check_protected"] for r in log.rows} == {"not_evaluated"}
    assert all(not r["reason"].startswith("protected") for r in log.rows)
