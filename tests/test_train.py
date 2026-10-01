"""Unit tests for the paired training runner (chemspar_v2.train)."""
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from chemspar_v2.train import BACKBONES, FitConfig, fit, instances_to_data, node_features


def _toy(n_struct=40, seed=0, drop_edges=False):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_struct):
        n = int(rng.integers(4, 9))
        z, r = node_features([int(x) for x in rng.choice([1, 6, 8, 30], n)], ["hydrogen"] * n)
        inst = [(i, i + 1, (0, 0, 0)) for i in range(n - 1)] + ([] if drop_edges else [(0, n - 1, (0, 0, 1))])
        out.append(instances_to_data(z, r, inst, [1.5] * len(inst), float(rng.normal()), f"s{k:03d}"))
    return out


def test_edge_index_increments_when_batching():
    data = _toy(3)
    b = next(iter(DataLoader(data, batch_size=3)))
    offs = np.cumsum([0] + [d.num_nodes for d in data[:-1]])
    start = 0
    for d, off in zip(data, offs):
        m = d.edge_index.shape[1]
        assert torch.equal(b.edge_index[:, start:start + m], d.edge_index + int(off))
        assert torch.equal(b.edge_image[start:start + m], d.edge_image)          # images are NOT offset
        start += m


def test_identical_initial_weights_for_same_seed():
    for bb, cls in BACKBONES.items():
        torch.manual_seed(3); a = cls().state_dict()
        torch.manual_seed(3); b = cls().state_dict()
        assert all(torch.equal(a[k], b[k]) for k in a)


def test_same_minibatch_order_across_views():
    v1, v2 = _toy(40, drop_edges=False), _toy(40, drop_edges=True)     # same structures, different edge sets
    ids = []
    for data in (v1, v2):
        gen = torch.Generator().manual_seed(11)
        ids.append([q for b in DataLoader(data, batch_size=8, shuffle=True, generator=gen) for q in b.qmof_id])
    assert ids[0] == ids[1]


def test_fit_is_bitwise_reproducible():
    data = _toy(60)
    tr, va, te = data[:40], data[40:50], data[50:]
    for bb in BACKBONES:
        a = fit(tr, va, te, FitConfig(backbone=bb, epochs=3, batch_size=8), seed=5)
        b = fit(tr, va, te, FitConfig(backbone=bb, epochs=3, batch_size=8), seed=5)
        assert [p["y_pred"] for p in a.predictions] == [p["y_pred"] for p in b.predictions]
        assert a.metrics["epochs_run"] == 3 and len(a.history) == 3
