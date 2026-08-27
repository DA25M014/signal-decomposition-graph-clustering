"""
Day 5: from results.csv to the analysis figures.

Reads the sweep CSV (any grid size, tolerates partial/incomplete runs)
and produces:

    day5_heatmaps.png - mean NMI heatmap for each of the 5 methods,
                        plus the family winner map:
                        blue   = graph-only methods (Louvain / Leiden)
                        orange = feature-only (k-means)
                        green  = GNN (DMoN / MinCut)
                        grey   = nobody works (best NMI < 0.1)
    day5_deltas.png   - left:  best GNN minus best non-GNN
                                (where do GNNs actually win?)
                        right: DMoN minus k-means
                                (what does the graph add to a
                                 feature clusterer?)

Console: completeness check, family win counts, and the cells with
the largest GNN advantage and largest GNN deficit.

Run:  python3 day5_analysis.py
"""

import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CSV = "results.csv"
METHODS = ["louvain", "leiden", "kmeans", "dmon", "mincut"]
FAMILY = {"louvain": 0, "leiden": 0, "kmeans": 1, "dmon": 2,
          "dmon_ref": 2, "mincut": 2}
FAMILY_NAMES = ["graph-only (Louvain/Leiden)", "feature-only (k-means)",
                "GNN (DMoN/MinCut)", "none (best NMI < 0.1)"]
FAMILY_COLORS = ["#7fb3d5", "#f5b041", "#82c99a", "#d5d8dc"]


def load():
    vals = defaultdict(list)
    with open(CSV) as fh:
        for row in csv.DictReader(fh):
            key = (row["method"], float(row["structure"]),
                   float(row["feature"]))
            vals[key].append(float(row["nmi"]))
    structures = sorted({k[1] for k in vals}, reverse=True)  # top = strong
    features = sorted({k[2] for k in vals})
    n_seeds = max(len(v) for v in vals.values())
    expected = len(structures) * len(features) * len(METHODS) * n_seeds
    got = sum(len(v) for v in vals.values())
    print(f"loaded {got} runs from {CSV} "
          f"({len(structures)}x{len(features)} grid, up to {n_seeds} seeds)")
    if got < expected:
        print(f"NOTE: grid incomplete ({expected - got} runs missing) - "
              f"figures use what exists")
    return vals, structures, features


def grids(vals, structures, features):
    mean = {m: np.full((len(structures), len(features)), np.nan)
            for m in METHODS}
    std = {m: np.full((len(structures), len(features)), np.nan)
           for m in METHODS}
    for m in METHODS:
        for i, s in enumerate(structures):
            for j, f in enumerate(features):
                v = vals.get((m, s, f))
                if v:
                    mean[m][i, j] = np.mean(v)
                    std[m][i, j] = np.std(v)
    return mean, std


def _decorate(ax, structures, features):
    ax.set_xticks(range(len(features)),
                  labels=[f"{f:g}" for f in features], fontsize=7)
    ax.set_yticks(range(len(structures)),
                  labels=[f"{s:g}" for s in structures], fontsize=7)
    ax.set_xlabel("feature signal", fontsize=8)
    ax.set_ylabel("structure signal", fontsize=8)


def heatmap_figure(mean, structures, features):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8),
                             constrained_layout=True)
    im = None
    for ax, m in zip(axes.flat[:5], METHODS):
        im = ax.imshow(mean[m], vmin=0, vmax=1, cmap="viridis")
        ax.set_title(m, fontsize=11)
        _decorate(ax, structures, features)
    fig.colorbar(im, ax=list(axes.flat[:5]), shrink=0.7, label="mean NMI")

    ax = axes.flat[5]
    cat = np.full(mean["louvain"].shape, 3.0)
    for i in range(cat.shape[0]):
        for j in range(cat.shape[1]):
            cell = {m: mean[m][i, j] for m in METHODS
                    if not np.isnan(mean[m][i, j])}
            if not cell:
                continue
            best = max(cell, key=cell.get)
            cat[i, j] = 3 if cell[best] < 0.1 else FAMILY[best]
    ax.imshow(cat, cmap=matplotlib.colors.ListedColormap(FAMILY_COLORS),
              vmin=0, vmax=3)
    ax.set_title("winner per cell (by family)", fontsize=11)
    _decorate(ax, structures, features)
    ax.legend(handles=[Patch(color=c, label=n)
                       for c, n in zip(FAMILY_COLORS, FAMILY_NAMES)],
              fontsize=6, loc="upper left", framealpha=0.9)
    fig.suptitle("Full sweep: who recovers the communities, and where?")
    fig.savefig("day5_heatmaps.png", dpi=150)
    print("wrote day5_heatmaps.png")
    return cat


def delta_figure(mean, structures, features):
    gname = "dmon_ref" if "dmon_ref" in mean else "dmon"
    gnn = np.fmax(mean[gname], mean["mincut"])
    cheap = np.fmax(np.fmax(mean["louvain"], mean["leiden"]),
                    mean["kmeans"])
    d1 = gnn - cheap
    d2 = mean[gname] - mean["kmeans"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             constrained_layout=True)
    for ax, d, title in [
            (axes[0], d1, "best GNN minus best non-GNN\n(red = GNN ahead)"),
            (axes[1], d2, "DMoN minus k-means\n(what the graph adds)")]:
        im = ax.imshow(d, vmin=-0.6, vmax=0.6, cmap="RdBu_r")
        ax.set_title(title, fontsize=10)
        _decorate(ax, structures, features)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                if not np.isnan(d[i, j]):
                    ax.text(j, i, f"{d[i, j]:+.2f}", ha="center",
                            va="center", fontsize=5.5, color="black")
    fig.colorbar(im, ax=list(axes), shrink=0.8, label="delta NMI")
    fig.savefig("day5_deltas.png", dpi=150)
    print("wrote day5_deltas.png")
    return d1


def summary(cat, d1, structures, features):
    n_cells = int(np.sum(~np.isnan(d1)))
    print(f"\nfamily win counts over {n_cells} cells:")
    for fam in range(4):
        print(f"  {FAMILY_NAMES[fam]:<32} {int(np.sum(cat == fam))}")

    flat = [(d1[i, j], structures[i], features[j])
            for i in range(d1.shape[0]) for j in range(d1.shape[1])
            if not np.isnan(d1[i, j])]
    flat.sort(reverse=True)
    print("\nlargest GNN advantage (best GNN minus best non-GNN):")
    for d, s, f in flat[:3]:
        print(f"  structure={s:g} feature={f:g}: {d:+.3f}")
    print("largest GNN deficit:")
    for d, s, f in flat[-3:]:
        print(f"  structure={s:g} feature={f:g}: {d:+.3f}")


if __name__ == "__main__":
    if not os.path.exists(CSV):
        raise SystemExit(f"{CSV} not found - run day4_runner.py first")
    vals, structures, features = load()
    if any(k[0] == "dmon_ref" for k in vals):
        METHODS[:] = ["louvain", "leiden", "kmeans", "dmon_ref", "mincut"]
        print("dmon_ref detected - using it as the DMoN column")
    mean, std = grids(vals, structures, features)
    cat = heatmap_figure(mean, structures, features)
    d1 = delta_figure(mean, structures, features)
    summary(cat, d1, structures, features)
    print("\nDay 5 analysis complete.")
