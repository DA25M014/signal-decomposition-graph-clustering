"""
LoG 2026 #57 rebuttal: extend the Table-1 battery to two more datasets
(Amazon Computers, Coauthor CS), mirroring realdata.csv + fairness.csv
conventions exactly:

  louvain / leiden / kmeans          (day1 wrappers)
  mincut          fixed budget, 500 epochs   (day4)
  dmon_ref        reference config, 1000 epochs (day7)
  dmonref_noisex  reference config, features -> N(0,1)  (day8 convention)
  dmonref_rewired reference config, graph -> G(n,m)     (day8 convention)

3 seeds each. Writes rebuttal_newdata.csv (resumable), same columns as
realdata.csv. Serial, torch threads capped so it can run alongside the
main rebuttal sweep.

Run:  caffeinate -i python3 rebuttal_newdata.py   # caffeinate: macOS, optional
"""

import csv
import os
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor
from sklearn.metrics import (normalized_mutual_info_score,
                             adjusted_rand_score)

from day1_pilot import louvain_labels, leiden_labels, kmeans_labels
from day4_runner import mincut_labels
from day6_realdata import rewire, noise_like
from day7_fairness import dmon_ref_labels

CSV_PATH = "rebuttal_newdata.csv"
SEEDS = [0, 1, 2]


def load_dataset(name):
    tf = T.NormalizeFeatures()
    if name == "computers":
        ds = Amazon(root="data", name="Computers", transform=tf)
    elif name == "cs":
        ds = Coauthor(root="data", name="CS", transform=tf)
    else:
        raise ValueError(name)
    data = ds[0]
    ei = data.edge_index.numpy()
    n = data.num_nodes
    A = sp.coo_matrix((np.ones(ei.shape[1], dtype=np.float32),
                       (ei[0], ei[1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    X = data.x.numpy().astype(np.float32)
    y = data.y.numpy()
    return A, X, y, int(y.max() + 1)


METHODS = {
    "louvain":        lambda A, X, k, sd: louvain_labels(A, seed=sd),
    "leiden":         lambda A, X, k, sd: leiden_labels(A, seed=sd),
    "kmeans":         lambda A, X, k, sd: kmeans_labels(X, k=k, seed=sd),
    "mincut":         lambda A, X, k, sd: mincut_labels(A, X, k=k, seed=sd,
                                                        epochs=500),
    "dmon_ref":       lambda A, X, k, sd: dmon_ref_labels(A, X, k, seed=sd,
                                                          epochs=1000),
    "dmonref_noisex": lambda A, X, k, sd: dmon_ref_labels(
        A, noise_like(X, sd), k, seed=sd, epochs=1000),
    "dmonref_rewired": lambda A, X, k, sd: dmon_ref_labels(
        rewire(A, sd), X, k, seed=sd, epochs=1000),
}


def main():
    torch.set_num_threads(4)
    done = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH) as fh:
            for r in csv.DictReader(fh):
                done.add((r["dataset"], r["method"], r["seed"]))
    new = not os.path.exists(CSV_PATH)
    fh = open(CSV_PATH, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["dataset", "method", "seed", "nmi", "ari", "seconds"])
        fh.flush()

    for name in ("computers", "cs"):
        A, X, y, k = load_dataset(name)
        print(f"{name}: n={A.shape[0]} m={int(A.nnz // 2)} "
              f"dim={X.shape[1]} k={k}", flush=True)
        for seed in SEEDS:
            for m, fn in METHODS.items():
                if (name, m, str(seed)) in done:
                    continue
                t0 = time.time()
                lab = fn(A, X, k, seed)
                dt = time.time() - t0
                w.writerow([name, m, seed,
                            round(normalized_mutual_info_score(y, lab), 4),
                            round(adjusted_rand_score(y, lab), 4),
                            round(dt, 2)])
                fh.flush()
                print(f"  {name} seed={seed} {m:<16} {dt:8.1f}s "
                      f"nmi={normalized_mutual_info_score(y, lab):.4f}",
                      flush=True)
    fh.close()
    print("new-data battery complete -> " + CSV_PATH, flush=True)


if __name__ == "__main__":
    main()
