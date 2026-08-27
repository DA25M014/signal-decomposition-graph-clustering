"""
Day 1 sanity check for the feature-vs-structure study.

Produces:
  1. day1_check.png - top row: adjacency matrix as the STRUCTURE knob turns
                      bottom row: PCA of features as the FEATURE knob turns
  2. a console table - NMI of structure-only methods (Louvain, Leiden)
                       along the structure axis, and of the graph-blind
                       k-means along the feature axis.

If the plot shows blocks dissolving left to right (top) and clusters
merging left to right (bottom), and both table columns fall smoothly
from ~1.0 toward ~0.0, the instrument works and Day 1 is done.

Run:  python day1_pilot.py        (about 1 minute on CPU)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import igraph as ig
import leidenalg
import community as community_louvain          # pip package: python-louvain
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as nmi

from generator import generate

SEED = 0
K = 4


# ------------- method wrappers (reused by all later scripts) -------------

def louvain_labels(A, seed=SEED):
    G = nx.from_scipy_sparse_array(A)
    part = community_louvain.best_partition(G, random_state=seed)
    return np.array([part[i] for i in range(A.shape[0])])


def leiden_labels(A, seed=SEED):
    coo = sparse.coo_matrix(sparse.triu(A, k=1))
    g = ig.Graph(n=A.shape[0],
                 edges=list(zip(coo.row.tolist(), coo.col.tolist())))
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition,
                                    seed=seed)
    return np.array(part.membership)


def kmeans_labels(X, k=K, seed=SEED):
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)


# ---------------- 1. the picture ----------------

def make_figure():
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    for ax, s in zip(axes[0], [1.0, 0.5, 0.15]):
        A, _, _ = generate(n=600, k=K, structure=s, feature=0.0, seed=SEED)
        ax.spy(A, markersize=0.25)
        ax.set_title(f"structure = {s}")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0][0].set_ylabel("adjacency matrix\n(nodes sorted by community)")

    for ax, f in zip(axes[1], [3.0, 1.0, 0.2]):
        _, X, y = generate(n=600, k=K, structure=0.05, feature=f, seed=SEED)
        Z = PCA(n_components=2).fit_transform(X)
        ax.scatter(Z[:, 0], Z[:, 1], c=y, s=5, cmap="tab10")
        ax.set_title(f"feature = {f}")
        ax.set_xticks([]); ax.set_yticks([])
    axes[1][0].set_ylabel("features, PCA to 2D\n(color = true community)")

    fig.suptitle("The two knobs: structure signal (top) vs feature signal (bottom)")
    fig.tight_layout()
    fig.savefig("day1_check.png", dpi=150)
    print("wrote day1_check.png")


# ---------------- 2. the numbers ----------------

def make_table():
    seeds = [0, 1, 2]

    print("\nSTRUCTURE axis (feature = 0, graph is the only signal), n = 1000")
    print(f"{'structure':>10} | {'Louvain NMI':>15} | {'Leiden NMI':>15}")
    for s in [0.9, 0.6, 0.4, 0.2, 0.05]:
        lou, lei = [], []
        for seed in seeds:
            A, X, y = generate(n=1000, k=K, structure=s, feature=0.0, seed=seed)
            lou.append(nmi(y, louvain_labels(A, seed=seed)))
            lei.append(nmi(y, leiden_labels(A, seed=seed)))
        print(f"{s:>10} | {np.mean(lou):.3f} ± {np.std(lou):.3f} | "
              f"{np.mean(lei):.3f} ± {np.std(lei):.3f}")

    print("\nFEATURE axis (structure = 0.05, features are the only signal)")
    print(f"{'feature':>10} | {'k-means NMI':>15}")
    for f in [3.0, 1.0, 0.5, 0.25, 0.0]:
        km = []
        for seed in seeds:
            A, X, y = generate(n=1000, k=K, structure=0.05, feature=f, seed=seed)
            km.append(nmi(y, kmeans_labels(X, seed=seed)))
        print(f"{f:>10} | {np.mean(km):.3f} ± {np.std(km):.3f}")


if __name__ == "__main__":
    make_figure()
    make_table()
    print("\nDay 1 check complete.")
