"""
Day 3: DMoN enters the arena.

Runs DMoN (1-layer GCN encoder + DMoNPooling head, faithful to the
reference architecture) on the same 3x3 grid as Day 2, 3 seeds per cell,
and compares every cell against the best cheap method from the Day 2
winner map (Louvain / Leiden / k-means, recomputed here so the script
is self-contained).

Outputs:
    1. console table: DMoN NMI vs best cheap method, with the delta
    2. day3_dmon_grid.png:
         left  - DMoN mean NMI heatmap
         right - DMoN minus best-cheap delta map (red = DMoN wins,
                 blue = cheap method wins). This is the prototype of
                 the paper's "what does the GNN actually add?" figure.

Run:  python3 day3_dmon.py        (prints one row per cell as it goes)
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DMoNPooling
from sklearn.metrics import normalized_mutual_info_score as nmi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generator import generate, to_edge_index
from day1_pilot import louvain_labels, leiden_labels, kmeans_labels, K

STRUCTURES = [0.9, 0.4, 0.05]     # rows, top to bottom
FEATURES = [0.0, 0.5, 1.5]        # columns, left to right
SEEDS = [0, 1, 2]
N = 1000
HIDDEN = 64
EPOCHS = 500
LR = 1e-3

CHEAP = {
    "Louvain": lambda A, X, seed: louvain_labels(A, seed=seed),
    "Leiden":  lambda A, X, seed: leiden_labels(A, seed=seed),
    "k-means": lambda A, X, seed: kmeans_labels(X, seed=seed),
}


class DMoNNet(torch.nn.Module):
    """1-layer GCN with a linear skip connection, as in the DMoN paper.

    The skip path (a plain Linear on the raw features) is what lets the
    model keep using feature signal even when the graph is pure noise -
    without it, one round of neighborhood averaging over a random graph
    destroys the feature clusters.
    """

    def __init__(self, in_dim, hidden, k):
        super().__init__()
        self.conv = GCNConv(in_dim, hidden)
        self.skip = torch.nn.Linear(in_dim, hidden)
        self.pool = DMoNPooling([hidden], k)

    def forward(self, x, edge_index, adj):
        h = F.selu(self.conv(x, edge_index) + self.skip(x))
        s, _, _, sp, o, c = self.pool(h.unsqueeze(0), adj)
        return s.squeeze(0), sp + o + c


def dmon_labels(A, X, k=K, seed=0, epochs=EPOCHS):
    torch.manual_seed(seed)
    x = torch.from_numpy(X)
    ei = torch.from_numpy(to_edge_index(A))
    adj = torch.from_numpy(A.toarray()).float().unsqueeze(0)
    model = DMoNNet(X.shape[1], HIDDEN, k)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        _, loss = model(x, ei, adj)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        s, _ = model(x, ei, adj)
    return s.argmax(-1).numpy()


def run():
    dmon_mean = np.zeros((3, 3))
    dmon_std = np.zeros((3, 3))
    delta = np.zeros((3, 3))
    print(f"{'structure':>9} {'feature':>8} | {'DMoN':>15} | "
          f"{'best cheap':>22} | {'delta':>7}")
    for i, s in enumerate(STRUCTURES):
        for j, f in enumerate(FEATURES):
            d_scores = []
            c_scores = {m: [] for m in CHEAP}
            for seed in SEEDS:
                A, X, y = generate(n=N, k=K, structure=s, feature=f,
                                   seed=seed)
                d_scores.append(nmi(y, dmon_labels(A, X, seed=seed)))
                for m, fn in CHEAP.items():
                    c_scores[m].append(nmi(y, fn(A, X, seed)))
            dmon_mean[i, j] = np.mean(d_scores)
            dmon_std[i, j] = np.std(d_scores)
            cheap_means = {m: np.mean(v) for m, v in c_scores.items()}
            best_name = max(cheap_means, key=cheap_means.get)
            best_val = cheap_means[best_name]
            delta[i, j] = dmon_mean[i, j] - best_val
            print(f"{s:>9} {f:>8} | "
                  f"{dmon_mean[i, j]:.3f} +/- {dmon_std[i, j]:.3f} | "
                  f"{best_name:>10} {best_val:.3f}      | "
                  f"{delta[i, j]:+.3f}")
    return dmon_mean, delta


def _decorate(ax):
    ax.set_xticks(range(3), labels=[str(f) for f in FEATURES])
    ax.set_yticks(range(3), labels=[str(s) for s in STRUCTURES])
    ax.set_xlabel("feature signal")
    ax.set_ylabel("structure signal")


def make_figure(dmon_mean, delta):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    ax.imshow(dmon_mean, vmin=0, vmax=1, cmap="viridis")
    ax.set_title("DMoN (mean NMI)")
    _decorate(ax)
    for i in range(3):
        for j in range(3):
            v = dmon_mean[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.6 else "black")

    ax = axes[1]
    ax.imshow(delta, vmin=-0.5, vmax=0.5, cmap="RdBu_r")
    ax.set_title("DMoN minus best cheap method\n(red = DMoN ahead)",
                 fontsize=10)
    _decorate(ax)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{delta[i, j]:+.2f}", ha="center", va="center",
                    color="black")

    fig.suptitle("Does seeing both inputs beat the best single-input method?")
    fig.tight_layout()
    fig.savefig("day3_dmon_grid.png", dpi=150)
    print("\nwrote day3_dmon_grid.png")


if __name__ == "__main__":
    dmon_mean, delta = run()
    make_figure(dmon_mean, delta)
    print("Day 3 DMoN grid complete.")
