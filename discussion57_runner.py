"""
Discussion-stage runs for LoG 2026 submission 57, Reviewer jYQg.

Arm A (Q5): the structureless row at exactly structure = 0.0, the full
feature grid, 5 seeds, all six methods. The paper's grid starts at
s = 0.05; this checks the "structureless graph is actively harmful"
claim at the exact endpoint.

Arm B (Q3): 20 seeds instead of 5 at four decision-critical cells, to
measure how much of the reported spread comes from realized center
separation and graph draw.

Output: discussion57.csv (resumable, same schema as results.csv plus a
realized center-separation column).

Run:  caffeinate -i python3 discussion57_runner.py
"""

import csv
import os
import time

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import (normalized_mutual_info_score,
                             adjusted_rand_score)

from generator import generate
from day1_pilot import louvain_labels, leiden_labels, kmeans_labels, K
from day3_dmon import dmon_labels
from day4_runner import mincut_labels
from day7_fairness import dmon_ref_labels

N = 1000
EPOCHS = 500
REF_EPOCHS = 1000
CSV = "discussion57.csv"

ARM_A = [(0.0, f, s) for f in [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0, 1.5]
         for s in range(5)]
ARM_B = [(st, f, s) for (st, f) in [(0.65, 0.0), (0.45, 0.5),
                                    (0.35, 0.2), (0.95, 0.0)]
         for s in range(20)]

METHODS = {
    "louvain": lambda A, X, sd: louvain_labels(A, seed=sd),
    "leiden": lambda A, X, sd: leiden_labels(A, seed=sd),
    "kmeans": lambda A, X, sd: kmeans_labels(X, seed=sd),
    "dmon": lambda A, X, sd: dmon_labels(A, X, seed=sd, epochs=EPOCHS),
    "mincut": lambda A, X, sd: mincut_labels(A, X, seed=sd, epochs=EPOCHS),
    "dmon_ref": lambda A, X, sd: dmon_ref_labels(A, X, K, seed=sd,
                                                 epochs=REF_EPOCHS),
}


def center_sep(feature, seed, k=K, dim=32):
    """Realized minimum pairwise center distance for this (feature, seed).

    Centers depend only on (feature, seed): the graph draw consumes one
    integer from the stream, so the same unit-normal configuration is
    reused across the whole (structure, feature) grid, scaled by feature.
    """
    rng = np.random.default_rng(seed)
    rng.integers(2 ** 31)
    c = rng.normal(0.0, feature, size=(k, dim))
    d = pdist(c)
    return float(d.min()), float(d.mean())


def main():
    done = set()
    if os.path.exists(CSV):
        with open(CSV) as fh:
            for row in csv.DictReader(fh):
                done.add((row["arm"], row["structure"], row["feature"],
                          row["seed"], row["method"]))
    new = not os.path.exists(CSV) or os.path.getsize(CSV) == 0
    fh = open(CSV, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["arm", "structure", "feature", "seed", "method",
                    "nmi", "ari", "sep_min", "sep_mean", "seconds"])
        fh.flush()

    jobs = [("A", *j) for j in ARM_A] + [("B", *j) for j in ARM_B]
    total = len(jobs) * len(METHODS)
    ran = 0
    t0 = time.time()
    for arm, st, f, seed in jobs:
        A = X = y = None
        smin, smean = center_sep(f, seed)
        for m, fn in METHODS.items():
            if (arm, str(st), str(f), str(seed), m) in done:
                continue
            if A is None:
                A, X, y = generate(n=N, k=K, structure=st, feature=f,
                                   seed=seed)
            t1 = time.time()
            lab = fn(A, X, seed)
            dt = time.time() - t1
            w.writerow([arm, st, f, seed, m,
                        round(normalized_mutual_info_score(y, lab), 4),
                        round(adjusted_rand_score(y, lab), 4),
                        round(smin, 4), round(smean, 4), round(dt, 2)])
            fh.flush()
            ran += 1
            eta = (total - ran) * (time.time() - t0) / ran / 60
            print(f"[{ran:>4}/{total}] {arm} s={st:<5} f={f:<5} "
                  f"seed={seed:<2} {m:<9} {dt:6.1f}s | eta ~{eta:6.1f} min")
    fh.close()
    print("done ->", CSV)


if __name__ == "__main__":
    main()
