"""The package must reproduce the paper code (paper/src/chemspar_v2, = tag paper-v1) exactly.

For every test structure, every policy and tau in {0.2, 0.5}: identical instance set and annotation, identical
removed instances in identical order, identical rejection counts and identical audit rows (all fields).
"""
import sys

import pytest
from conftest import ALL_CIFS, ROOT, cif_id, graph_for
from golden_lib import POLICY_KEYS, TAUS

PAPER_SRC = ROOT / "paper" / "src"
pytestmark = pytest.mark.skipif(not (PAPER_SRC / "chemspar_v2").exists(), reason="paper code not present")

ANNOT = ("distance", "radius_bonded", "crystalnn", "crystalnn_available", "bond_class", "protected",
         "protection_route", "radius_tolerance")


def _paper():
    sys.path.insert(0, str(PAPER_SRC))
    from chemspar_v2 import periodic as P
    from chemspar_v2 import views as V
    return P, V


_cache = {}


def paper_result(path):
    if path not in _cache:
        from pymatgen.core import Structure
        P, V = _paper()
        sid = cif_id(path)
        res = V.compute_structure(Structure.from_file(str(path)), sid, audit=True)
        inst = res["instances"]
        orders = {k: P.order_by(v, inst) for k, v in res["val"].items() if k != "distance"}
        orders["longest"] = P.order_by(res["val"]["distance"], inst, descending=True)
        orders["random"] = P.order_random(inst, 42, sid)
        rows = {}
        for name in POLICY_KEYS:
            okey, mode, extra, tmax = V.POLICIES[name]
            rows[name] = P.prune_instances(res["graph"], tmax, orders[okey], mode, comp=res["comp"], qmof_id=sid,
                                           policy=name, audit=True, extra=extra)["audit"]
        _cache[path] = (res, rows, V)
    return _cache[path]


@pytest.mark.parametrize("path", ALL_CIFS, ids=cif_id)
def test_graph_and_taxonomy_identical(path):
    res, _, _ = paper_result(path)
    g = graph_for(str(path))
    assert g.instances == res["instances"]
    for k in g.instances:
        mine, theirs = g.attrs(k), res["attrs"][k]
        assert {f: mine[f] for f in ANNOT} == {f: theirs[f] for f in ANNOT}, k


@pytest.mark.parametrize("path", ALL_CIFS, ids=cif_id)
def test_orderings_identical(path):
    from chemspar import orderings
    res, _, _ = paper_result(path)
    g = graph_for(str(path))
    inst = res["instances"]
    P, _ = _paper()
    paper_orders = {
        "gravity": P.order_by(res["val"]["gravity"], inst),
        "distance": P.order_by(res["val"]["distance"], inst, descending=True),
        "random": P.order_random(inst, 42, cif_id(path)),
        "bhs": P.order_by(res["val"]["bhs_min"], inst),
        "bhs_prod": P.order_by(res["val"]["bhs_prod"], inst),
        "massfree": P.order_by(res["val"]["massfree"], inst),
        "gravity_chem": P.order_by(res["val"]["gravity_chem"], inst),
        "gravity_topo": P.order_by(res["val"]["gravity_topo"], inst),
    }
    for name, order in paper_orders.items():
        assert orderings.get(name)(g, seed=42) == order, name


@pytest.mark.parametrize("tau", TAUS)
@pytest.mark.parametrize("policy", POLICY_KEYS)
@pytest.mark.parametrize("path", ALL_CIFS, ids=cif_id)
def test_views_and_audit_identical(path, policy, tau):
    from chemspar import prune_policy
    res, rows, V = paper_result(path)
    g = graph_for(str(path))
    view, log = prune_policy(g, policy, tau)

    assert view.instances == V.view_instances(res, f"{policy}@{tau}")
    ranks = res["ranks"][policy]
    assert view.removed == sorted(ranks, key=ranks.get)[:view.target]

    rej = res["rejections"][policy][tau]
    assert view.target == rej["target"] and len(view.removed) == rej["removed"]
    assert log.rejections() == {k: v for k, v in rej.items() if k not in ("target", "removed")}

    expected = [r for r in rows[policy] if r["removed_before_attempt"] < view.target]
    drop = ("tau", "target")  # run-level fields: the paper ran once at tau_max and cut
    assert [{k: v for k, v in r.items() if k not in drop} for r in log.rows] == \
           [{k: v for k, v in r.items() if k not in drop} for r in expected]
