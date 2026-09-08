"""Prints every number quoted in the discussion-phase responses (REBUTTAL.md).

Reads discussion57.csv (the discussion-phase runs) and results.csv (the
submitted sweep) and prints, in order:

  0. run counts for every shipped CSV
  1. arm A: the structure = 0 row, means and per-method maxima, with the
     Figure 2a deficit at structure 0 next to the same quantity at 0.05
  2. arm B: the four 20-seed cells, in the paper's Figure 2a convention
     (mean over seeds per method, then max within family) and in the
     per-seed convention used for the range and sign-flip columns
  3. arm B: realized center separation against NMI, per method
  4. arm B seeds 0-4 rejoined with results.csv: the reproducibility check

Run:  python discussion57_analysis.py
"""

import collections
import csv
import math
import statistics as st

GNN = ("dmon_ref", "mincut")          # GNN family of Figure 2a
NON = ("louvain", "leiden", "kmeans")
METHODS = ("louvain", "leiden", "kmeans", "dmon", "mincut", "dmon_ref")
FEATURES = ["0.0", "0.2", "0.35", "0.5", "0.65", "0.8", "1.0", "1.5"]
CELLS = [("0.35", "0.2"), ("0.45", "0.5"), ("0.65", "0.0"), ("0.95", "0.0")]
KEY = ("structure", "feature", "seed", "method")


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def by_method(rows):
    d = collections.defaultdict(list)
    for x in rows:
        d[x["method"]].append(float(x["nmi"]))
    return d


def fig2a(rows):
    """max over GNN of the seed-mean NMI minus max over non-GNN, as in Fig. 2a."""
    m = {k: st.mean(v) for k, v in by_method(rows).items()}
    return max(m[g] for g in GNN) - max(m[c] for c in NON)


def main():
    rows = load("discussion57.csv")
    paper = load("results.csv")
    A = [x for x in rows if x["arm"] == "A"]
    B = [x for x in rows if x["arm"] == "B"]

    print("=== 0. run counts ===")
    for name in ("results.csv", "realdata.csv", "fairness.csv", "rebuttal57.csv",
                 "rebuttal_newdata.csv", "discussion57.csv"):
        try:
            print(f"  {name:<22} {len(load(name)):>5} runs")
        except FileNotFoundError:
            print(f"  {name:<22} not present")
    print(f"  discussion57.csv arm A {len(A)} runs, arm B {len(B)} runs")

    print("\n=== 1. arm A: structure = 0.0 exactly (5 seeds) ===")
    for f in FEATURES:
        cell = [x for x in A if x["feature"] == f]
        old = [x for x in paper if x["structure"] == "0.05" and x["feature"] == f]
        m = {k: st.mean(v) for k, v in by_method(cell).items()}
        best = max(NON, key=lambda c: m[c])
        print(f"  f={f:<5} " + " | ".join(f"{k} {m[k]:.3f}" for k in METHODS)
              + f" || best non-GNN {best} {m[best]:.3f}"
              + f"; deficit at s=0 {fig2a(cell):+.3f}"
              + f"; at s=0.05 {fig2a(old):+.3f}"
              + f"; fixed-budget DMoN {m['dmon'] - m[best]:+.3f}")
    mx = {k: max(v) for k, v in by_method(A).items()}
    print("  per-method maximum over the whole s=0 row: "
          + " | ".join(f"{k} {mx[k]:.4f}" for k in METHODS))

    print("\n=== 2. arm B: 20 seeds at four cells ===")
    for cell in CELLS:
        sub = [x for x in B if (x["structure"], x["feature"]) == cell]
        five = [x for x in sub if int(x["seed"]) < 5]
        per = collections.defaultdict(dict)
        for x in sub:
            per[x["seed"]][x["method"]] = float(x["nmi"])
        seeds = sorted(per, key=int)
        ps = [max(per[s][g] for g in GNN) - max(per[s][c] for c in NON) for s in seeds]
        flips = sum(1 for d in ps if (d > 0) != (st.mean(ps) > 0))
        print(f"  -- s={cell[0]} f={cell[1]}")
        for m in METHODS:
            v20 = [per[s][m] for s in seeds]
            v5 = [per[s][m] for s in seeds if int(s) < 5]
            print(f"     {m:<9} 20-seed {st.mean(v20):.3f}+-{st.pstdev(v20):.3f}"
                  f" [{min(v20):.3f},{max(v20):.3f}]   5-seed {st.mean(v5):.3f}")
        print(f"     Fig.2a delta: 5-seed {fig2a(five):+.3f} | 20-seed {fig2a(sub):+.3f}"
              f" | absolute change {abs(fig2a(sub) - fig2a(five)):.3f}")
        print(f"     per-seed delta: mean {st.mean(ps):+.3f}+-{st.pstdev(ps):.3f}"
              f" range [{min(ps):+.3f},{max(ps):+.3f}] sign flips {flips}/{len(ps)}")

    print("\n=== 3. arm B: realized center separation vs NMI (20 seeds) ===")
    for cell in CELLS:
        sub = [x for x in B if (x["structure"], x["feature"]) == cell]
        per, sep = collections.defaultdict(dict), {}
        for x in sub:
            per[x["seed"]][x["method"]] = float(x["nmi"])
            sep[x["seed"]] = float(x["sep_min"])
        seeds = sorted(per, key=int)
        xs = [sep[s] for s in seeds]
        print(f"  -- s={cell[0]} f={cell[1]} min pairwise center distance "
              f"[{min(xs):.2f},{max(xs):.2f}]")
        if st.pstdev(xs) < 1e-9:
            print("     centers are identical at feature = 0, nothing to correlate")
            continue
        mx_ = st.mean(xs)
        for m in METHODS:
            ys = [per[s][m] for s in seeds]
            if st.pstdev(ys) < 1e-9:
                print(f"     {m:<9} constant")
                continue
            my = st.mean(ys)
            num = sum((a - mx_) * (b - my) for a, b in zip(xs, ys))
            den = math.sqrt(sum((a - mx_) ** 2 for a in xs)
                            * sum((b - my) ** 2 for b in ys))
            print(f"     {m:<9} pearson r = {num / den:+.2f}")

    print("\n=== 4. reproducibility: arm B seeds 0-4 rejoined with results.csv ===")
    old = {tuple(x[k] for k in KEY): float(x["nmi"]) for x in paper}
    gaps = {}
    for x in B:
        if int(x["seed"]) < 5:
            k = tuple(x[k_] for k_ in KEY)
            if k in old:
                gaps[k] = abs(float(x["nmi"]) - old[k])
    diff = {k: v for k, v in gaps.items() if v > 0}
    print(f"  overlapping runs: {len(gaps)} (4 cells x 5 seeds x 6 methods)")
    print(f"  reproduce exactly: {len(gaps) - len(diff)}; differ: {len(diff)}")
    print("  differing by method: "
          + str(dict(collections.Counter(k[3] for k in diff))))
    for k, v in sorted(diff.items(), key=lambda kv: -kv[1]):
        print(f"    s={k[0]} f={k[1]} seed={k[2]} {k[3]}: gap {v:.4f}")
    print(f"  largest gap: {max(diff.values()) if diff else 0.0:.4f}")
    exact = sorted({k[3] for k in gaps} - {k[3] for k in diff})
    print(f"  bit-identical methods: {', '.join(exact)}")


if __name__ == "__main__":
    main()
