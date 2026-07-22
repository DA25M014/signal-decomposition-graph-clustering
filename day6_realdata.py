"""
Day 6: real data - where do Cora, Citeseer, Amazon Photo sit on the map?

For each dataset, runs the 5 sweep methods PLUS two DMoN ablations that
mirror the synthetic axes:

    dmon_noisex   - real graph, features replaced by N(0,1) noise
                    -> the GNN's pure-structure channel
    dmon_rewired  - real features, edges rewired to a same-density
                    random graph -> the GNN's pure-feature channel

If the amplifier story holds on real data, expect:
    dmon_noisex  ~ near zero  (no feature seed -> graph can't act)
    dmon_rewired ~ k-means or below (graph gone or poisonous)
    dmon         > k-means    (real graph amplifies real features)

Writes realdata.csv (resumable - rerun the same command to continue)
and prints a mean +/- std summary per dataset at the end.

Run:
    python3 day6_realdata.py --quick   # Cora only, 1 seed, ~2 min
    caffeinate -i python3 day6_realdata.py   # full, ~30-45 min
"""

import argparse
import csv
import os
import time

import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Amazon
from sklearn.metrics import (normalized_mutual_info_score,
                             adjusted_rand_score)

from day1_pilot import louvain_labels, leiden_labels, kmeans_labels
from day3_dmon import dmon_labels
from day4_runner import mincut_labels

CSV = "realdata.csv"


def load_dataset(name):
    tf = T.NormalizeFeatures()
    if name in ("cora", "citeseer"):
        ds = Planetoid(root="data", name=name.capitalize(), transform=tf)
    elif name == "photo":
        ds = Amazon(root="data", name="Photo", transform=tf)
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


def rewire(A, seed):
    """Random graph with the same node count and edge count."""
    n = A.shape[0]
    m = int(A.nnz // 2)
    G = nx.gnm_random_graph(n, m, seed=seed)
    return nx.to_scipy_sparse_array(G, format="csr", dtype=np.float32)


def noise_like(X, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=X.shape).astype(np.float32)


METHODS = {
    "louvain":      lambda A, X, k, sd, ep: louvain_labels(A, seed=sd),
    "leiden":       lambda A, X, k, sd, ep: leiden_labels(A, seed=sd),
    "kmeans":       lambda A, X, k, sd, ep: kmeans_labels(X, k=k, seed=sd),
    "dmon":         lambda A, X, k, sd, ep: dmon_labels(A, X, k=k, seed=sd,
                                                        epochs=ep),
    "mincut":       lambda A, X, k, sd, ep: mincut_labels(A, X, k=k,
                                                          seed=sd, epochs=ep),
    "dmon_noisex":  lambda A, X, k, sd, ep: dmon_labels(
        A, noise_like(X, sd), k=k, seed=sd, epochs=ep),
    "dmon_rewired": lambda A, X, k, sd, ep: dmon_labels(
        rewire(A, sd), X, k=k, seed=sd, epochs=ep),
}


def load_done():
    keys = set()
    if os.path.exists(CSV):
        with open(CSV) as fh:
            for row in csv.DictReader(fh):
                keys.add((row["dataset"], row["method"], row["seed"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    global CSV
    if args.quick:
        datasets, seeds, epochs = ["cora"], [0], 60
        CSV = "realdata_quick.csv"
    else:
        datasets, seeds, epochs = ["cora", "citeseer", "photo"], [0, 1, 2], 500

    done = load_done()
    new_file = not os.path.exists(CSV)
    fh = open(CSV, "a", newline="")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(["dataset", "method", "seed", "nmi", "ari",
                         "seconds"])
        fh.flush()

    for name in datasets:
        try:
            A, X, y, k = load_dataset(name)
        except Exception as exc:
            print(f"SKIP {name}: could not load ({exc})")
            continue
        print(f"\n{name}: n={A.shape[0]} edges={int(A.nnz // 2)} "
              f"dim={X.shape[1]} k={k}")
        for seed in seeds:
            for m, fn in METHODS.items():
                if (name, m, str(seed)) in done:
                    continue
                t0 = time.time()
                lab = fn(A, X, k, seed, epochs)
                dt = time.time() - t0
                writer.writerow([name, m, seed,
                                 round(normalized_mutual_info_score(y, lab),
                                       4),
                                 round(adjusted_rand_score(y, lab), 4),
                                 round(dt, 2)])
                fh.flush()
                print(f"  {name} seed={seed} {m:<13} {dt:7.1f}s")
    fh.close()

    # ----- summary -----
    vals = {}
    with open(CSV) as f2:
        for row in csv.DictReader(f2):
            vals.setdefault((row["dataset"], row["method"]),
                            []).append(float(row["nmi"]))
    print("\nNMI summary (mean +/- std over seeds):")
    for name in datasets:
        got = [(m, v) for (d, m), v in vals.items() if d == name]
        if not got:
            continue
        print(f"  {name}:")
        for m in METHODS:
            v = dict(got).get(m)
            if v:
                print(f"    {m:<13} {np.mean(v):.3f} +/- {np.std(v):.3f}")
    print("\nDay 6 real-data run complete.")


if __name__ == "__main__":
    main()
