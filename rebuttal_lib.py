"""
Shared model code for the LoG 2026 #57 rebuttal experiments.

DMoNRefDeep generalizes day7_fairness.DMoNRef to a configurable stack of
official-style GCN layers (shared kernel, learned per-channel skip weight,
selu, no bias), i.e. the official implementation's architecture flag
[64] -> [64, 64] -> [64, 64, 64]. depth=1 with hidden=64, lr=1e-3,
dropout=0.5, epochs=1000 reproduces dmon_ref exactly (same module
structure and seeding path; verified in the pilot).
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.nn import DMoNPooling

COLLAPSE_W = 1.0


def norm_adj_torch(A):
    """D^-1/2 (A + I) D^-1/2 as a torch sparse tensor (official utils)."""
    n = A.shape[0]
    Ai = (A + sp.identity(n, format="csr", dtype=np.float32)).tocsr()
    d = np.asarray(Ai.sum(1)).ravel()
    with np.errstate(divide="ignore"):
        dinv = 1.0 / np.sqrt(d)
    dinv[np.isinf(dinv)] = 0
    D = sp.diags(dinv.astype(np.float32))
    S = (D @ Ai @ D).tocoo()
    idx = torch.from_numpy(np.vstack([S.row, S.col]).astype(np.int64))
    val = torch.from_numpy(S.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, (n, n)).coalesce()


class RefGCN(torch.nn.Module):
    """Official-style GCN layer: shared kernel, per-channel learned
    skip weight, selu, no bias. Identical to day7_fairness.RefGCN."""

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, hidden, bias=False)
        self.skip_w = torch.nn.Parameter(torch.ones(hidden))

    def forward(self, x, adj_norm):
        xw = self.lin(x)
        return F.selu(torch.sparse.mm(adj_norm, xw) + xw * self.skip_w)


class DMoNRefDeep(torch.nn.Module):
    """Reference DMoN with a stack of `depth` RefGCN layers
    (official architecture flag [hidden]*depth)."""

    def __init__(self, in_dim, hidden, k, dropout, depth):
        super().__init__()
        dims = [in_dim] + [hidden] * depth
        self.layers = torch.nn.ModuleList(
            RefGCN(dims[i], dims[i + 1]) for i in range(depth))
        self.pool = DMoNPooling([hidden], k, dropout=dropout)

    def forward(self, x, adj_norm, adj_raw):
        h = x
        for layer in self.layers:
            h = layer(h, adj_norm)
        s, _, _, sp_loss, _ortho, cluster = self.pool(h.unsqueeze(0),
                                                      adj_raw)
        return s.squeeze(0), sp_loss + COLLAPSE_W * cluster


def dmon_deep_labels(A, X, k, seed=0, epochs=1000, hidden=64,
                     lr=1e-3, dropout=0.5, depth=1):
    torch.manual_seed(seed)
    x = torch.from_numpy(X)
    adj_norm = norm_adj_torch(A)
    adj_raw = torch.from_numpy(A.toarray()).float().unsqueeze(0)
    model = DMoNRefDeep(X.shape[1], hidden, k, dropout, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        _, loss = model(x, adj_norm, adj_raw)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        s, _ = model(x, adj_norm, adj_raw)
    return s.argmax(-1).numpy()
