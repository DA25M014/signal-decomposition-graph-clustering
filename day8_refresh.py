"""
Day 8: the refresh - the last compute pass behind the released CSVs.

Part 1: runs dmon_ref (the reference-faithful DMoN from day7) over the
        full 8x8 x 5-seed sweep grid and appends rows with
        method='dmon_ref' to results.csv.          (~13 min)
Part 2: reruns the two channel ablations at reference config on the
        real datasets, appending 'dmonref_noisex' / 'dmonref_rewired'
        to realdata.csv.                            (~5 min)

Both parts are resumable: rerun the same command to continue.
Afterwards, rerun day5_analysis.py - it now auto-detects dmon_ref
and uses it as the DMoN column in every figure.

Run:
    python3 day8_refresh.py --quick      # ~1 min mechanics check
    caffeinate -i python3 day8_refresh.py
"""

import argparse
import csv
import os
import time

from sklearn.metrics import (normalized_mutual_info_score,
                             adjusted_rand_score)

from generator import generate
from day4_runner import STRUCTURES, FEATURES, SEEDS
from day6_realdata import load_dataset, noise_like, rewire
from day7_fairness import dmon_ref_labels


def done_keys(path, cols):
    keys = set()
    if os.path.exists(path):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                keys.add(tuple(r[c] for c in cols))
    return keys


def open_append(path, header):
    new = not os.path.exists(path)
    fh = open(path, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(header)
        fh.flush()
    return fh, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        structures, features, seeds, epochs = [0.45], [0.5], [0], 60
        datasets, dseeds = ["cora"], [0]
        syn_csv = real_csv = "refresh_quick.csv"
    else:
        structures, features, seeds, epochs = (STRUCTURES, FEATURES,
                                               SEEDS, 1000)
        datasets, dseeds = ["cora", "citeseer", "photo"], [0, 1, 2]
        syn_csv, real_csv = "results.csv", "realdata.csv"

    # ---- part 1: sweep column ----
    done = done_keys(syn_csv, ["structure", "feature", "seed", "method"]) \
        if os.path.exists(syn_csv) and syn_csv == "results.csv" else set()
    fh, w = open_append(syn_csv, ["structure", "feature", "seed",
                                  "method", "nmi", "ari", "seconds"])
    total = len(structures) * len(features) * len(seeds)
    ran = 0
    t0 = time.time()
    for s in structures:
        for f in features:
            for seed in seeds:
                if (str(s), str(f), str(seed), "dmon_ref") in done:
                    continue
                A, X, y = generate(n=1000, k=4, structure=s, feature=f,
                                   seed=seed)
                t1 = time.time()
                lab = dmon_ref_labels(A, X, 4, seed=seed, epochs=epochs)
                dt = time.time() - t1
                w.writerow([s, f, seed, "dmon_ref",
                            round(normalized_mutual_info_score(y, lab), 4),
                            round(adjusted_rand_score(y, lab), 4),
                            round(dt, 2)])
                fh.flush()
                ran += 1
                eta = (total - ran) * (time.time() - t0) / max(ran, 1) / 60
                print(f"[{ran:>3}/{total}] syn s={s:<5} f={f:<5} "
                      f"seed={seed} {dt:5.1f}s | eta ~{eta:5.1f} min")
    fh.close()

    # ---- part 2: real-data ablations at reference config ----
    done2 = done_keys(real_csv, ["dataset", "method", "seed"]) \
        if os.path.exists(real_csv) and real_csv == "realdata.csv" else set()
    fh, w = open_append(real_csv, ["dataset", "method", "seed",
                                   "nmi", "ari", "seconds"])
    for name in datasets:
        try:
            A, X, y, k = load_dataset(name)
        except Exception as exc:
            print(f"SKIP {name}: {exc}")
            continue
        for seed in dseeds:
            for m in ["dmonref_noisex", "dmonref_rewired"]:
                if (name, m, str(seed)) in done2:
                    continue
                Xi = noise_like(X, seed) if m.endswith("noisex") else X
                Ai = A if m.endswith("noisex") else rewire(A, seed)
                t1 = time.time()
                lab = dmon_ref_labels(Ai, Xi, k, seed=seed, epochs=epochs)
                dt = time.time() - t1
                w.writerow([name, m, seed,
                            round(normalized_mutual_info_score(y, lab), 4),
                            round(adjusted_rand_score(y, lab), 4),
                            round(dt, 2)])
                fh.flush()
                print(f"  {name} seed={seed} {m:<15} {dt:7.1f}s")
    fh.close()
    print("\nDay 8 refresh complete. Now run: python3 day5_analysis.py")


if __name__ == "__main__":
    main()
