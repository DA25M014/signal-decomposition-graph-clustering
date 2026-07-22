"""
Day 2: the 3x3 mini-grid - first rough draft of the money figure.

Crosses the two knobs:
    structure in {0.9, 0.4, 0.05}   (strong / critical / none)
    feature   in {0.0, 0.5, 1.5}    (none / moderate / strong)

and runs the three cheap methods in every cell, 3 seeds each:
    Louvain, Leiden  - see only the graph
    k-means          - sees only the features

Outputs:
    1. console table: mean +/- std NMI for every cell and method
    2. day2_minigrid.png: one heatmap per method, plus a
       "who wins where" map in the fourth panel

Run:  python3 day2_minigrid.py        (about 1-2 minutes)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import normalized_mutual_info_score as nmi

from generator import generate
from day1_pilot import louvain_labels, leiden_labels, kmeans_labels, K

STRUCTURES = [0.9, 0.4, 0.05]     # rows, top to bottom
FEATURES = [0.0, 0.5, 1.5]        # columns, left to right
SEEDS = [0, 1, 2]
N = 1000

METHODS = {
    "Louvain": lambda A, X, seed: louvain_labels(A, seed=seed),
    "Leiden":  lambda A, X, seed: leiden_labels(A, seed=seed),
    "k-means": lambda A, X, seed: kmeans_labels(X, seed=seed),
}


def run_grid():
    mean = {m: np.zeros((3, 3)) for m in METHODS}
    std = {m: np.zeros((3, 3)) for m in METHODS}
    header = " | ".join(f"{m:>15}" for m in METHODS)
    print(f"{'structure':>9} {'feature':>8} | {header}")
    for i, s in enumerate(STRUCTURES):
        for j, f in enumerate(FEATURES):
            scores = {m: [] for m in METHODS}
            for seed in SEEDS:
                A, X, y = generate(n=N, k=K, structure=s, feature=f,
                                   seed=seed)
                for m, fn in METHODS.items():
                    scores[m].append(nmi(y, fn(A, X, seed)))
            cells = []
            for m in METHODS:
                mean[m][i, j] = np.mean(scores[m])
                std[m][i, j] = np.std(scores[m])
                cells.append(f"{mean[m][i, j]:.3f} +/- {std[m][i, j]:.3f}")
            row = " | ".join(f"{c:>15}" for c in cells)
            print(f"{s:>9} {f:>8} | {row}")
    return mean, std


def _decorate(ax):
    ax.set_xticks(range(3), labels=[str(f) for f in FEATURES])
    ax.set_yticks(range(3), labels=[str(s) for s in STRUCTURES])
    ax.set_xlabel("feature signal")
    ax.set_ylabel("structure signal")


def _winner_panel(ax, mean):
    names = list(METHODS)
    cat = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            vals = [mean[m][i, j] for m in names]
            b = int(np.argmax(vals))
            best, second = sorted(vals, reverse=True)[:2]
            if best < 0.1:
                cat[i, j] = 2
                label = "none"
            else:
                cat[i, j] = 0 if b < 2 else 1
                label = f"{names[b]}\n+{best - second:.2f}"
            ax.text(j, i, label, ha="center", va="center",
                    color="black", fontsize=9)
    ax.imshow(cat, cmap=ListedColormap(["#7fb3d5", "#f5b041", "#d5d8dc"]),
              vmin=0, vmax=2)
    _decorate(ax)
    ax.set_title("winner per cell\n(blue = graph method, orange = k-means)",
                 fontsize=10)


def make_figure(mean):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, m in zip(axes.flat[:3], METHODS):
        ax.imshow(mean[m], vmin=0, vmax=1, cmap="viridis")
        ax.set_title(f"{m} (mean NMI)")
        _decorate(ax)
        for i in range(3):
            for j in range(3):
                v = mean[m][i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black")
    _winner_panel(axes.flat[3], mean)
    fig.suptitle("Mini-grid: who recovers the communities, and where?")
    fig.tight_layout()
    fig.savefig("day2_minigrid.png", dpi=150)
    print("\nwrote day2_minigrid.png")


if __name__ == "__main__":
    mean, std = run_grid()
    make_figure(mean)
    print("Day 2 mini-grid complete.")
