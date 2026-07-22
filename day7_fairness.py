"""
Day 7: the fairness pass.

Rebuilds DMoN to match the OFFICIAL implementation
(google-research/graph_embedding/dmon), which differs from our
fixed-budget version in four ways found by reading the source:

    1. loss = spectral + 1.0 * collapse   (NO orthogonality term;
       we had been summing all three PyG losses)
    2. n_epochs = 1000 (we used 500); lr = 0.001 (same)
    3. dropout = 0.5 on the assignment head (README's Cora setting)
    4. skip connection = shared kernel with a learned per-channel
       skip weight (not a separate Linear); selu; no bias;
       sym-normalized adjacency with self-loops (matches GCNConv)

Runs dmon_ref on: cora, citeseer, photo (3 seeds each) and six
frontier cells of the synthetic map (3 seeds each). Results go to
fairness.csv (resumable). The summary joins realdata.csv and
results.csv if present, so you get ref-config vs fixed-budget vs
baselines side by side without rerunning anything.

Run:
    python3 day7_fairness.py --quick     # ~2 min mechanics check
    caffeinate -i python3 day7_fairness.py   # full, ~5-8 min
"""

import argparse
import csv
import os
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch_geometric.nn import DMoNPooling
from sklearn.metrics import (normalized_mutual_info_score,
                             adjusted_rand_score)

from generator import generate
from day6_realdata import load_dataset

CSV = "fairness.csv"
CELLS = [(0.45, 0.5), (0.35, 0.5), (0.35, 0.35),
         (0.55, 0.2), (0.05, 0.5), (0.95, 0.0)]
HIDDEN = 64          # official flag default
EPOCHS = 1000        # official flag default
LR = 1e-3            # official flag default
DROPOUT = 0.5        # official README example (Cora)
COLLAPSE_W = 1.0     # official flag default


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
    skip weight, selu, no bias."""

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, hidden, bias=False)
        self.skip_w = torch.nn.Parameter(torch.ones(hidden))

    def forward(self, x, adj_norm):
        xw = self.lin(x)
        return F.selu(torch.sparse.mm(adj_norm, xw) + xw * self.skip_w)


class DMoNRef(torch.nn.Module):
    def __init__(self, in_dim, hidden, k, dropout):
        super().__init__()
        self.gcn = RefGCN(in_dim, hidden)
        self.pool = DMoNPooling([hidden], k, dropout=dropout)

    def forward(self, x, adj_norm, adj_raw):
        h = self.gcn(x, adj_norm)
        s, _, _, sp_loss, _ortho, cluster = self.pool(h.unsqueeze(0),
                                                      adj_raw)
        return s.squeeze(0), sp_loss + COLLAPSE_W * cluster


def dmon_ref_labels(A, X, k, seed=0, epochs=EPOCHS):
    torch.manual_seed(seed)
    x = torch.from_numpy(X)
    adj_norm = norm_adj_torch(A)
    adj_raw = torch.from_numpy(A.toarray()).float().unsqueeze(0)
    model = DMoNRef(X.shape[1], HIDDEN, k, DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
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


def load_done():
    keys = set()
    if os.path.exists(CSV):
        with open(CSV) as fh:
            for row in csv.DictReader(fh):
                keys.add((row["target"], row["seed"]))
    return keys


def read_means(path, key_cols):
    """Generic mean-NMI reader: {(key..., method): mean_nmi}."""
    out = {}
    if not os.path.exists(path):
        return out
    acc = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            k = tuple(row[c] for c in key_cols) + (row["method"],)
            acc.setdefault(k, []).append(float(row["nmi"]))
    for k, v in acc.items():
        out[k] = np.mean(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    global CSV
    if args.quick:
        CSV = "fairness_quick.csv"
        datasets, cells, seeds, epochs = ["cora"], [(0.45, 0.5)], [0], 60
    else:
        datasets, cells, seeds, epochs = (["cora", "citeseer", "photo"],
                                          CELLS, [0, 1, 2], EPOCHS)

    done = load_done()
    new_file = not os.path.exists(CSV)
    fh = open(CSV, "a", newline="")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(["target", "seed", "method", "nmi", "ari",
                         "seconds"])
        fh.flush()

    jobs = [("data", name) for name in datasets] + \
           [("cell", c) for c in cells]
    for kind, item in jobs:
        if kind == "data":
            target = item
            try:
                A, X, y, k = load_dataset(item)
            except Exception as exc:
                print(f"SKIP {item}: {exc}")
                continue
        else:
            s_val, f_val = item
            target = f"syn_{s_val}_{f_val}"
        for seed in seeds:
            if (target, str(seed)) in done:
                continue
            if kind == "cell":
                A, X, y = generate(n=1000, k=4, structure=item[0],
                                   feature=item[1], seed=seed)
                k = 4
            t0 = time.time()
            lab = dmon_ref_labels(A, X, k, seed=seed, epochs=epochs)
            dt = time.time() - t0
            writer.writerow([target, seed, "dmon_ref",
                             round(normalized_mutual_info_score(y, lab), 4),
                             round(adjusted_rand_score(y, lab), 4),
                             round(dt, 2)])
            fh.flush()
            print(f"  {target:<14} seed={seed} dmon_ref {dt:7.1f}s")
    fh.close()

    # ----- side-by-side summary -----
    ref = read_means(CSV, ["target"])
    real = read_means("realdata.csv", ["dataset"])
    syn = read_means("results.csv", ["structure", "feature"])

    print("\n===== ref config vs fixed budget vs baselines (mean NMI) =====")
    for name in datasets:
        r = ref.get((name, "dmon_ref"))
        if r is None:
            continue
        fx = real.get((name, "dmon"))
        km = real.get((name, "kmeans"))
        le = real.get((name, "leiden"))
        lo = real.get((name, "louvain"))
        best_cl = max(x for x in [le, lo] if x is not None) \
            if (le or lo) else None
        line = f"  {name:<10} dmon_ref {r:.3f}"
        if fx is not None:
            line += f" | dmon_fixed {fx:.3f}"
        if km is not None:
            line += f" | kmeans {km:.3f}"
        if best_cl is not None:
            line += f" | best classical {best_cl:.3f}"
        print(line)
    for s_val, f_val in cells:
        target = f"syn_{s_val}_{f_val}"
        r = ref.get((target, "dmon_ref"))
        if r is None:
            continue
        fx = syn.get((str(s_val), str(f_val), "dmon"))
        km = syn.get((str(s_val), str(f_val), "kmeans"))
        le = syn.get((str(s_val), str(f_val), "leiden"))
        line = f"  {target:<14} dmon_ref {r:.3f}"
        if fx is not None:
            line += f" | dmon_fixed {fx:.3f}"
        if km is not None:
            line += f" | kmeans {km:.3f}"
        if le is not None:
            line += f" | leiden {le:.3f}"
        print(line)
    print("\nDay 7 fairness pass complete.")


if __name__ == "__main__":
    main()
