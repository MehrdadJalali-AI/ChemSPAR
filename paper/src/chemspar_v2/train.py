"""Paired training runner for the frozen ChemSPAR-v2 benchmark (Phase 3 protocol).

Representation (identical for both backbones): a graph view is a set of retained periodic edge INSTANCES
(i, j, image) (see chemspar_v2.periodic). Each instance becomes two directed edges (i->j and j->i; a self-image
contact i->i appears twice, once per image direction). `edge_index` holds exactly these directed edges,
`edge_dist` their distances and `edge_image` the lattice image of the target atom (used only for verification).
  * GraphSAGE uses edge_index only (no edge features).
  * CGCNN-style periodic backbone uses edge_index + Gaussian-expanded edge_dist.
No neighbour list is rebuilt at tensor time; `verify_view_tensor` proves tensor == named view.
Node features are IDENTICAL across views: atomic number (embedding) + 7-way v2 element role.

Pairing: one fixed cohort and split; torch.manual_seed(seed) immediately before model construction (identical
initial weights across views); torch.Generator(seed) for the training loader (identical mini-batch order); fixed
epoch budget (no early stopping; the best-validation epoch's weights are evaluated); identical hyperparameters
and hardware.
"""
from __future__ import annotations

import math
import resource
import time
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, SAGEConv, global_mean_pool

ROLES = ["metal", "oxygen", "nitrogen", "carbon", "hydrogen", "halide", "other"]


# ------------------------------------------------------------------ tensors
def node_features(elements_Z: list[int], roles: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.tensor(elements_Z, dtype=torch.long)
    r = torch.zeros(len(roles), len(ROLES))
    for k, ro in enumerate(roles):
        r[k, ROLES.index(ro)] = 1.0
    return z, r


class ViewData(Data):
    pass


def instances_to_data(z: torch.Tensor, roles: torch.Tensor, instances: list, distances: list[float],
                      y: float, qmof_id: str) -> ViewData:
    """instances: canonical (i, j, img) keys of ONE graph view; distances: matching instance distances."""
    src, dst, dist, img = [], [], [], []
    for (i, j, im), d in zip(instances, distances):
        src += [i, j]; dst += [j, i]; dist += [d, d]
        img += [list(im), [-x for x in im]]
    ei = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros(2, 0, dtype=torch.long)
    return ViewData(z=z, roles=roles, edge_index=ei, edge_dist=torch.tensor(dist, dtype=torch.float32),
                    edge_image=torch.tensor(img, dtype=torch.int16).view(-1, 3), y=torch.tensor([y], dtype=torch.float32),
                    qmof_id=qmof_id, num_nodes=len(z))


def tensor_instances(d: ViewData) -> list:
    """Recover the sorted list of canonical instances encoded in a tensor (each instance appears twice)."""
    from .periodic import canonical
    keys = sorted(canonical(int(a), int(b), tuple(int(x) for x in im))
                  for a, b, im in zip(d.edge_index[0].tolist(), d.edge_index[1].tolist(), d.edge_image.tolist()))
    if len(keys) % 2 or any(keys[2 * t] != keys[2 * t + 1] for t in range(len(keys) // 2)):
        raise ValueError("tensor edges are not paired directed copies of canonical instances")
    return keys[::2]


def verify_view_tensor(d: ViewData, view_instances: list) -> bool:
    """True iff the tensor encodes exactly the retained instances of the named view (no more, no fewer)."""
    return tensor_instances(d) == sorted(view_instances)


# ------------------------------------------------------------------ models
class GaussianExpansion(nn.Module):
    def __init__(self, dmin=0.0, dmax=8.0, step=0.2):
        super().__init__()
        self.register_buffer("centers", torch.arange(dmin, dmax + 1e-9, step))
        self.gamma = 1.0 / step ** 2

    def forward(self, d):
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centers) ** 2)


class SAGEReg(nn.Module):
    """GraphSAGE: 2 x SAGEConv over the view's directed instance edges (no edge features), mean pooling."""
    def __init__(self, hidden=64, zdim=32):
        super().__init__()
        self.emb = nn.Embedding(100, zdim)
        self.c1, self.c2 = SAGEConv(zdim + len(ROLES), hidden), SAGEConv(hidden, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, b):
        x = torch.cat([self.emb(b.z), b.roles], dim=1)
        x = torch.relu(self.c1(x, b.edge_index)); x = torch.relu(self.c2(x, b.edge_index))
        return self.head(global_mean_pool(x, b.batch)).view(-1)


class CGCNNReg(nn.Module):
    """Periodic backbone (CGCNN-style): Z embedding + roles -> 3 x CGConv over the view's periodic edge instances
    with Gaussian-expanded distances (0-8 A, step 0.2 A) -> mean pooling -> 2-layer head with softplus."""
    def __init__(self, hidden=64, zdim=64, n_conv=3):
        super().__init__()
        self.emb = nn.Embedding(100, zdim)
        self.inp = nn.Linear(zdim + len(ROLES), hidden)
        self.rbf = GaussianExpansion()
        edim = self.rbf.centers.numel()
        self.convs = nn.ModuleList([CGConv(hidden, dim=edim, aggr="add", batch_norm=True) for _ in range(n_conv)])
        self.head = nn.Sequential(nn.Linear(hidden, 128), nn.Softplus(), nn.Linear(128, 1))

    def forward(self, b):
        x = self.inp(torch.cat([self.emb(b.z), b.roles], dim=1))
        ea = self.rbf(b.edge_dist)
        for c in self.convs:
            x = c(x, b.edge_index, ea)
        return self.head(global_mean_pool(x, b.batch)).view(-1)


BACKBONES = {"graphsage": SAGEReg, "cgcnn": CGCNNReg}


# ------------------------------------------------------------------ training
@dataclass
class FitConfig:
    backbone: str
    epochs: int
    lr: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 0.0
    device: str = "cpu"


@dataclass
class FitResult:
    metrics: dict
    predictions: list[dict]
    history: list[dict] = field(default_factory=list)
    timing: dict = field(default_factory=dict)


def _peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e6 if r > 1e7 else r / 1e3   # macOS reports bytes, Linux kB


def fit(train: list[Data], val: list[Data], test: list[Data], cfg: FitConfig, seed: int) -> FitResult:
    ytr = torch.tensor([d.y.item() for d in train])
    mu, sd = ytr.mean().item(), ytr.std().item()
    torch.manual_seed(seed)                                      # identical initial weights across views
    model = BACKBONES[cfg.backbone]().to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(seed)                     # identical mini-batch order across views
    tl = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, generator=gen)
    vl = DataLoader(val, batch_size=128, shuffle=False)
    el = DataLoader(test, batch_size=128, shuffle=False)

    def predict(loader):
        model.eval(); out, ids, ys = [], [], []
        with torch.no_grad():
            for b in loader:
                b = b.to(cfg.device)
                out.append(model(b).cpu() * sd + mu); ys.append(b.y.view(-1).cpu()); ids += list(b.qmof_id)
        return torch.cat(out).numpy(), torch.cat(ys).numpy(), ids

    best, best_state, best_ep, hist = math.inf, None, -1, []
    t0 = time.perf_counter()
    for ep in range(1, cfg.epochs + 1):
        model.train(); tot = 0.0
        for b in tl:
            b = b.to(cfg.device); opt.zero_grad()
            loss = nn.functional.mse_loss(model(b), (b.y.view(-1) - mu) / sd)
            loss.backward(); opt.step(); tot += loss.item() * b.num_graphs
        p, y, _ = predict(vl)
        vmae = float(np.abs(p - y).mean())
        hist.append(dict(epoch=ep, train_mse_std=tot / len(train), val_mae=vmae))
        if vmae < best:
            best, best_ep = vmae, ep
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    t_train = time.perf_counter() - t0
    model.load_state_dict(best_state)
    t1 = time.perf_counter(); p, y, ids = predict(el); t_inf = time.perf_counter() - t1
    from scipy.stats import kendalltau, spearmanr
    metrics = dict(MAE=float(np.abs(p - y).mean()), RMSE=float(np.sqrt(((p - y) ** 2).mean())),
                   R2=float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                   Spearman=float(spearmanr(y, p).correlation), Kendall=float(kendalltau(y, p).correlation),
                   best_epoch=best_ep, best_val_mae=best, epochs_run=cfg.epochs, n_train=len(train), n_val=len(val), n_test=len(test))
    preds = [dict(qmof_id=i, y_true=float(a), y_pred=float(b)) for i, a, b in zip(ids, y, p)]
    return FitResult(metrics, preds, hist, dict(train_seconds=t_train, inference_seconds=t_inf, peak_rss_mb=_peak_rss_mb()))
