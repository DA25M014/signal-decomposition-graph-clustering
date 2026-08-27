"""
Paper-ready figures (Fig. 1 and Fig. 2).

Reads results.csv (requires the dmon_ref column from day8) and writes:

    fig1_map.(png|pdf)    - (a) DMoN(ref) mean-NMI heatmap
                            (b) winner-per-regime family map
    fig2_deltas.(png|pdf) - (a) best GNN minus best non-GNN
                            (b) DMoN(ref) minus k-means
                            color scale clipped at +/-0.6; every cell
                            annotated with its exact value

Palette: classical = blue, k-means = orange, GNN = purple, none = grey
(purple instead of green to stay readable under red-green
colorblindness).

Also prints the family win counts and top advantage/deficit cells so
you can cross-check the caption numbers against the frozen results.

Run:  python3 paper_figures.py
"""

import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

CSV = "results.csv"
METHODS = ["louvain", "leiden", "kmeans", "dmon_ref", "mincut"]
FAMILY = {"louvain": 0, "leiden": 0, "kmeans": 1, "dmon_ref": 2,
          "mincut": 2}
FAMILY_NAMES = ["classical (Louvain/Leiden)", "k-means (features only)",
                "GNN (DMoN/MinCut)", "none (best NMI < 0.1)"]
FAMILY_COLORS = ["#6baed6", "#fdae6b", "#9e9ac8", "#d9d9d9"]


def load():
    vals = defaultdict(list)
    with open(CSV) as fh:
        for row in csv.DictReader(fh):
            vals[(row["method"], float(row["structure"]),
                  float(row["feature"]))].append(float(row["nmi"]))
    methods_present = {k[0] for k in vals}
    missing = [m for m in METHODS if m not in methods_present]
    if missing:
        raise SystemExit(f"results.csv is missing {missing} - "
                         f"run day8_refresh.py first")
    structures = sorted({k[1] for k in vals}, reverse=True)
    features = sorted({k[2] for k in vals})
    return vals, structures, features


def grids(vals, structures, features):
    mean = {m: np.full((len(structures), len(features)), np.nan)
            for m in METHODS}
    for m in METHODS:
        for i, s in enumerate(structures):
            for j, f in enumerate(features):
                v = vals.get((m, s, f))
                if v:
                    mean[m][i, j] = np.mean(v)
    return mean


def _decorate(ax, structures, features):
    ax.set_xticks(range(len(features)),
                  labels=[f"{f:g}" for f in features], fontsize=7)
    ax.set_yticks(range(len(structures)),
                  labels=[f"{s:g}" for s in structures], fontsize=7)
    ax.set_xlabel("feature signal", fontsize=8)
    ax.set_ylabel("structure signal", fontsize=8)


def winner_matrix(mean):
    shape = mean["louvain"].shape
    cat = np.full(shape, 3.0)
    for i in range(shape[0]):
        for j in range(shape[1]):
            cell = {m: mean[m][i, j] for m in METHODS
                    if not np.isnan(mean[m][i, j])}
            if not cell:
                continue
            best = max(cell, key=cell.get)
            cat[i, j] = 3 if cell[best] < 0.1 else FAMILY[best]
    return cat


def figure1(mean, structures, features):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9),
                             constrained_layout=True)
    ax = axes[0]
    im = ax.imshow(mean["dmon_ref"], vmin=0, vmax=1, cmap="viridis")
    ax.set_title("(a) DMoN (reference config), mean NMI", fontsize=9)
    _decorate(ax, structures, features)
    fig.colorbar(im, ax=ax, shrink=0.85)

    ax = axes[1]
    cat = winner_matrix(mean)
    ax.imshow(cat, cmap=ListedColormap(FAMILY_COLORS), vmin=0, vmax=3)
    ax.set_title("(b) winner per regime, by family", fontsize=9)
    _decorate(ax, structures, features)
    ax.legend(handles=[Patch(color=c, label=n)
                       for c, n in zip(FAMILY_COLORS, FAMILY_NAMES)],
              fontsize=6.5, loc="upper left", framealpha=0.95)
    for ext in ("png", "pdf"):
        fig.savefig(f"fig1_map.{ext}", dpi=300)
    print("wrote fig1_map.png / .pdf")
    return cat


def figure2(mean, structures, features):
    gnn = np.fmax(mean["dmon_ref"], mean["mincut"])
    cheap = np.fmax(np.fmax(mean["louvain"], mean["leiden"]),
                    mean["kmeans"])
    d1 = gnn - cheap
    d2 = mean["dmon_ref"] - mean["kmeans"]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9),
                             constrained_layout=True)
    im = None
    for ax, d, title in [
            (axes[0], d1, "(a) best GNN minus best non-GNN"),
            (axes[1], d2, "(b) DMoN (ref) minus k-means")]:
        im = ax.imshow(d, vmin=-0.6, vmax=0.6, cmap="RdBu_r")
        ax.set_title(title, fontsize=9)
        _decorate(ax, structures, features)
        for i in range(d.shape[0]):
            for j in range(d.shape[1]):
                v = d[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=5.0,
                        color="#666666" if abs(v) < 0.05 else "black")
    fig.colorbar(im, ax=list(axes), shrink=0.85,
                 label="delta NMI (clipped at +/-0.6)")
    for ext in ("png", "pdf"):
        fig.savefig(f"fig2_deltas.{ext}", dpi=300)
    print("wrote fig2_deltas.png / .pdf")
    return d1


def summary(cat, d1, structures, features):
    print("\nfamily win counts:")
    for fam in range(4):
        print(f"  {FAMILY_NAMES[fam]:<28} {int(np.sum(cat == fam))}")
    flat = [(d1[i, j], structures[i], features[j])
            for i in range(d1.shape[0]) for j in range(d1.shape[1])
            if not np.isnan(d1[i, j])]
    flat.sort(reverse=True)
    print("top GNN advantage:",
          ", ".join(f"({s:g},{f:g}) {d:+.3f}" for d, s, f in flat[:3]))
    print("top GNN deficit:  ",
          ", ".join(f"({s:g},{f:g}) {d:+.3f}" for d, s, f in flat[-3:]))


if __name__ == "__main__":
    if not os.path.exists(CSV):
        raise SystemExit(f"{CSV} not found")
    vals, structures, features = load()
    mean = grids(vals, structures, features)
    cat = figure1(mean, structures, features)
    d1 = figure2(mean, structures, features)
    summary(cat, d1, structures, features)
    print("\nPaper figures complete.")
