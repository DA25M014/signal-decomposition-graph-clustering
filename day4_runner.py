"""
Day 4: the full sweep runner - the paper's dataset generator.

8x8 grid (structure x feature, denser near both transitions)
x 5 seeds x 5 methods (louvain, leiden, kmeans, dmon, mincut)
-> results.csv, one row per run: nmi, ari, seconds.

RESUMABLE: every finished run is flushed to results.csv immediately.
If the job dies or you stop it, rerun the same command - completed
(cell, seed, method) rows are detected and skipped.

MinCut uses the exact same GCN + skip encoder as DMoN, so any
difference between them is the pooling objective, not the encoder.

Run:
    python3 day4_runner.py --quick        # ~2-3 min smoke test
                                          # (writes results_quick.csv)
    caffeinate -i python3 day4_runner.py  # the real overnight sweep
                                          # (keep lid open, power in)
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, dense_mincut_pool
from sklearn.metrics import (normalized_mutual_info_score,
                             adjusted_rand_score)

from generator import generate, to_edge_index
from day1_pilot import louvain_labels, leiden_labels, kmeans_labels, K
from day3_dmon import dmon_labels

STRUCTURES = [0.05, 0.2, 0.35, 0.45, 0.55, 0.65, 0.8, 0.95]
FEATURES = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0, 1.5]
SEEDS = [0, 1, 2, 3, 4]
N = 1000
HIDDEN = 64
EPOCHS = 500
LR = 1e-3


class MinCutNet(torch.nn.Module):
    """Same encoder as DMoNNet (GCN + linear skip); MinCut pooling head."""

    def __init__(self, in_dim, hidden, k):
        super().__init__()
        self.conv = GCNConv(in_dim, hidden)
        self.skip = torch.nn.Linear(in_dim, hidden)
        self.assign = torch.nn.Linear(hidden, k)

    def forward(self, x, edge_index, adj):
        h = F.selu(self.conv(x, edge_index) + self.skip(x))
        s = self.assign(h)
        _, _, mc, o = dense_mincut_pool(h.unsqueeze(0), adj, s.unsqueeze(0))
        return s, mc + o


def mincut_labels(A, X, k=K, seed=0, epochs=EPOCHS):
    torch.manual_seed(seed)
    x = torch.from_numpy(X)
    ei = torch.from_numpy(to_edge_index(A))
    adj = torch.from_numpy(A.toarray()).float().unsqueeze(0)
    model = MinCutNet(X.shape[1], HIDDEN, k)
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


METHODS = {
    "louvain": lambda A, X, seed, ep: louvain_labels(A, seed=seed),
    "leiden":  lambda A, X, seed, ep: leiden_labels(A, seed=seed),
    "kmeans":  lambda A, X, seed, ep: kmeans_labels(X, seed=seed),
    "dmon":    lambda A, X, seed, ep: dmon_labels(A, X, seed=seed, epochs=ep),
    "mincut":  lambda A, X, seed, ep: mincut_labels(A, X, seed=seed,
                                                    epochs=ep),
}


def load_done(csv_path):
    keys = set()
    if os.path.exists(csv_path):
        with open(csv_path) as fh:
            for row in csv.DictReader(fh):
                keys.add((row["structure"], row["feature"],
                          row["seed"], row["method"]))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke test (separate output file)")
    args = ap.parse_args()

    if args.quick:
        structures, features, seeds, epochs = [0.9, 0.05], [0.0, 1.5], [0], 60
        csv_path = "results_quick.csv"
    else:
        structures, features, seeds, epochs = (STRUCTURES, FEATURES,
                                               SEEDS, EPOCHS)
        csv_path = "results.csv"

    done = load_done(csv_path)
    new_file = not os.path.exists(csv_path)
    fh = open(csv_path, "a", newline="")
    writer = csv.writer(fh)
    if new_file:
        writer.writerow(["structure", "feature", "seed", "method",
                         "nmi", "ari", "seconds"])
        fh.flush()

    total = len(structures) * len(features) * len(seeds) * len(METHODS)
    todo = sum(1 for s in structures for f in features for sd in seeds
               for m in METHODS
               if (str(s), str(f), str(sd), m) not in done)
    print(f"{total} runs in this sweep | {total - todo} already in "
          f"{csv_path} | {todo} to go")

    t0 = time.time()
    ran = 0
    for s in structures:
        for f in features:
            cache = {}
            for seed in seeds:
                for m, fn in METHODS.items():
                    key = (str(s), str(f), str(seed), m)
                    if key in done:
                        continue
                    if seed not in cache:
                        cache[seed] = generate(n=N, k=K, structure=s,
                                               feature=f, seed=seed)
                    A, X, y = cache[seed]
                    t1 = time.time()
                    lab = fn(A, X, seed, epochs)
                    dt = time.time() - t1
                    writer.writerow([s, f, seed, m,
                                     round(normalized_mutual_info_score(
                                         y, lab), 4),
                                     round(adjusted_rand_score(y, lab), 4),
                                     round(dt, 2)])
                    fh.flush()
                    ran += 1
                    eta = (todo - ran) * (time.time() - t0) / ran / 60
                    print(f"[{ran:>4}/{todo}] s={s:<5} f={f:<5} seed={seed} "
                          f"{m:<8} {dt:6.1f}s | eta ~{eta:6.1f} min")
    fh.close()
    print(f"sweep complete -> {csv_path}")


if __name__ == "__main__":
    main()
