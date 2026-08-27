"""
Analysis for the LoG #57 rebuttal experiments. Reads:
    results.csv          (published sweep: louvain/leiden/kmeans/dmon/
                          mincut/dmon_ref, 5 seeds)
    realdata.csv         (published real-data runs)
    fairness.csv         (published dmon_ref real-data runs)
    rebuttal57.csv       (depth / mcdepth / tune / realdepth arms)
    rebuttal_newdata.csv (computers / cs battery)

Prints the headline numbers the rebuttal quotes; the remaining quoted
values are read directly off the CSVs.
"""

import csv
import os
from collections import defaultdict

import numpy as np

REPO = "."
STRUCTURES = [0.05, 0.2, 0.35, 0.45, 0.55, 0.65, 0.8, 0.95]
FEATURES = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0, 1.5]
FAMILY = {"louvain": "classical", "leiden": "classical",
          "kmeans": "kmeans"}


def read_rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def cellmeans(rows, method_col="method", val="nmi",
              keys=("structure", "feature")):
    acc = defaultdict(list)
    for r in rows:
        acc[(r[method_col],) + tuple(float(r[k]) for k in keys)].append(
            float(r[val]))
    return {k: np.mean(v) for k, v in acc.items()}, \
           {k: v for k, v in acc.items()}


def winner_analysis(mean, gnn_methods, cheap_methods, label):
    """Family win counts + GNN-vs-best-cheap deltas, paper_figures logic."""
    order = list(cheap_methods) + list(gnn_methods)   # ties -> cheap first
    counts = defaultdict(int)
    wins = []
    for s in STRUCTURES:
        for f in FEATURES:
            cell = {m: mean[(m, s, f)] for m in order if (m, s, f) in mean}
            best = max(cell, key=cell.get)             # first max wins
            if cell[best] < 0.1:
                counts["none"] += 1
                continue
            fam = FAMILY.get(best, "GNN")
            counts[fam] += 1
            delta = max(cell[m] for m in gnn_methods if m in cell) - \
                max(cell[m] for m in cheap_methods if m in cell)
            if fam == "GNN":
                wins.append((delta, s, f, best))
    wins.sort(reverse=True)
    print(f"\n== {label} ==")
    print("  family wins:", dict(counts))
    if wins:
        print(f"  max GNN win: {wins[0][0]:+.4f} at "
              f"(s={wins[0][1]}, f={wins[0][2]}) [{wins[0][3]}]")
        big = [w for w in wins if w[0] > 0.03]
        print(f"  wins > 0.03: {len(big)}: " +
              ", ".join(f"({s},{f}):{d:+.3f}" for d, s, f, m in big))
        print(f"  all win cells: " +
              ", ".join(f"({s},{f}):{d:+.3f}" for d, s, f, m in wins))
    # worst deficit
    defs_ = []
    for s in STRUCTURES:
        for f in FEATURES:
            g = max(mean[(m, s, f)] for m in gnn_methods
                    if (m, s, f) in mean)
            c = max(mean[(m, s, f)] for m in cheap_methods
                    if (m, s, f) in mean)
            defs_.append((g - c, s, f))
    defs_.sort()
    print(f"  worst GNN deficit: {defs_[0][0]:+.4f} at "
          f"(s={defs_[0][1]}, f={defs_[0][2]})")
    return counts, wins


def col_profile(mean, method, label):
    f0max = max((mean[(method, s, 0.0)], s) for s in STRUCTURES
                if (method, s, 0.0) in mean)
    print(f"  {label}: f=0 column max NMI = {f0max[0]:.4f} at s={f0max[1]}")
    # poison: s=0.05 row, method minus kmeans
    poison = min((mean[(method, 0.05, f)] - mean[("kmeans", 0.05, f)], f)
                 for f in FEATURES if (method, 0.05, f) in mean)
    print(f"  {label}: s=0.05 poison min(method-kmeans) = "
          f"{poison[0]:+.4f} at f={poison[1]}")


def main():
    res = read_rows(os.path.join(REPO, "results.csv"))
    reb = read_rows("rebuttal57.csv")

    base_mean, base_all = cellmeans(res)

    # fold rebuttal synthetic rows in as pseudo-methods
    syn = [r for r in reb if r["exp"] in ("depth", "mcdepth", "tune")]
    for r in syn:
        s, f = r["target"].split("_")
        r["structure"], r["feature"] = s, f
        if r["exp"] == "depth":
            r["method"] = "dmon_" + r["config"]                 # dmon_d2/d3
        elif r["exp"] == "mcdepth":
            r["method"] = "mincut_" + r["config"][-2:]          # mincut_d2/d3
        else:
            r["method"] = "tune|" + r["config"]
    syn_mean, syn_all = cellmeans(syn)
    mean = {**base_mean, **syn_mean}
    allv = {**base_all, **syn_all}

    seeds_check = {k: len(v) for k, v in syn_all.items()}
    bad = {k: n for k, n in seeds_check.items() if n != 5}
    print(f"rebuttal synthetic cells with != 5 seeds: {len(bad)}"
          + (f" e.g. {list(bad.items())[:5]}" if bad else ""))

    cheap = ["louvain", "leiden", "kmeans"]

    # ---------- published map (sanity anchor) ----------
    winner_analysis(mean, ["dmon_ref", "mincut"], cheap,
                    "PUBLISHED map (dmon_ref + mincut d1)")
    col_profile(mean, "dmon_ref", "dmon_ref d1")

    # ---------- depth ----------
    print("\n#################### DEPTH ####################")
    for m in ["dmon_d2", "dmon_d3", "mincut_d2", "mincut_d3"]:
        g = [mean[(m, s, f)] for s in STRUCTURES for f in FEATURES
             if (m, s, f) in mean]
        print(f"  {m}: grid cells {len(g)}, grid-mean NMI {np.mean(g):.4f}"
              f"  (dmon_ref d1 grid-mean "
              f"{np.mean([mean[('dmon_ref', s, f)] for s in STRUCTURES for f in FEATURES]):.4f},"
              f" mincut d1 "
              f"{np.mean([mean[('mincut', s, f)] for s in STRUCTURES for f in FEATURES]):.4f})")
    for m in ["dmon_d2", "dmon_d3", "mincut_d2", "mincut_d3"]:
        col_profile(mean, m, m)
    winner_analysis(mean, ["dmon_d2", "mincut_d2"], cheap,
                    "DEPTH-2 map (dmon_d2 + mincut_d2)")
    winner_analysis(mean, ["dmon_d3", "mincut_d3"], cheap,
                    "DEPTH-3 map (dmon_d3 + mincut_d3)")
    winner_analysis(mean, ["dmon_ref", "mincut", "dmon_d2", "mincut_d2",
                           "dmon_d3", "mincut_d3"], cheap,
                    "BEST-OVER-DEPTHS map (oracle depth per cell)")

    # per-cell d2 vs d1 deltas of dmon
    dd = [(mean[("dmon_d2", s, f)] - mean[("dmon_ref", s, f)], s, f)
          for s in STRUCTURES for f in FEATURES
          if ("dmon_d2", s, f) in mean]
    dd.sort()
    print(f"\n  dmon d2-d1 per-cell delta: mean "
          f"{np.mean([d for d, _, _ in dd]):+.4f}, "
          f"min {dd[0][0]:+.4f} at ({dd[0][1]},{dd[0][2]}), "
          f"max {dd[-1][0]:+.4f} at ({dd[-1][1]},{dd[-1][2]})")
    d3 = [(mean[("dmon_d3", s, f)] - mean[("dmon_ref", s, f)], s, f)
          for s in STRUCTURES for f in FEATURES
          if ("dmon_d3", s, f) in mean]
    d3.sort()
    print(f"  dmon d3-d1 per-cell delta: mean "
          f"{np.mean([d for d, _, _ in d3]):+.4f}, "
          f"min {d3[0][0]:+.4f} at ({d3[0][1]},{d3[0][2]}), "
          f"max {d3[-1][0]:+.4f} at ({d3[-1][1]},{d3[-1][2]})")

    # ---------- tuning ----------
    print("\n#################### TUNING ####################")
    tune_methods = sorted({m for (m, s, f) in syn_mean
                           if m.startswith("tune|")})
    print("  configs:", tune_methods)
    oracle = {}
    oracle_cfg = {}
    for s in STRUCTURES:
        for f in FEATURES:
            cands = {"ref(lr:0.001|do:0.5|h:64)": mean[("dmon_ref", s, f)]}
            for m in tune_methods:
                cands[m] = mean[(m, s, f)]
            bestc = max(cands, key=cands.get)
            oracle[("dmon_oracle", s, f)] = cands[bestc]
            oracle_cfg[(s, f)] = (bestc, cands[bestc],
                                  cands["ref(lr:0.001|do:0.5|h:64)"])
    mean.update(oracle)
    gains = sorted(((v[1] - v[2], s_f[0], s_f[1], v[0])
                    for s_f, v in oracle_cfg.items()), reverse=True)
    print(f"  oracle-minus-ref gain: mean "
          f"{np.mean([g for g, *_ in gains]):+.4f}, max {gains[0][0]:+.4f} "
          f"at ({gains[0][1]},{gains[0][2]}) [{gains[0][3]}], "
          f"cells with gain>0.01: {sum(1 for g, *_ in gains if g > 0.01)}, "
          f">0.05: {sum(1 for g, *_ in gains if g > 0.05)}")
    print("  top-8 oracle gains:")
    for g, s, f, c in gains[:8]:
        print(f"    (s={s}, f={f}): +{g:.4f} via {c}")
    winner_analysis(mean, ["dmon_oracle", "mincut"], cheap,
                    "ORACLE-TUNED map (dmon oracle-cfg + mincut d1)")
    col_profile(mean, "dmon_oracle", "dmon_oracle")
    # which config wins the poison row / f=0 col
    print("  oracle config chosen, s=0.05 row: " +
          "; ".join(f"f={f}:{oracle_cfg[(0.05, f)][0]}" for f in FEATURES))
    print("  oracle config chosen, f=0 col: " +
          "; ".join(f"s={s}:{oracle_cfg[(s, 0.0)][0]}" for s in STRUCTURES))

    # ---------- real-data depth ----------
    print("\n#################### REAL DEPTH ####################")
    fair = read_rows(os.path.join(REPO, "fairness.csv"))
    real = read_rows(os.path.join(REPO, "realdata.csv"))
    rd = [r for r in reb if r["exp"] == "realdepth"]
    for ds in ("cora", "citeseer", "photo"):
        d1 = [float(r["nmi"]) for r in fair
              if r["target"] == ds and r["method"] == "dmon_ref"]
        line = f"  {ds}: d1 {np.mean(d1):.3f}+/-{np.std(d1):.3f}"
        for d in ("d2", "d3"):
            v = [float(r["nmi"]) for r in rd
                 if r["target"] == ds and r["config"] == d]
            if v:
                line += f" | {d} {np.mean(v):.3f}+/-{np.std(v):.3f} (n={len(v)})"
        bc = max(np.mean([float(r["nmi"]) for r in real
                          if r["dataset"] == ds and r["method"] == m])
                 for m in ("louvain", "leiden"))
        line += f" | best classical {bc:.3f}"
        print(line)

    # ---------- new datasets ----------
    print("\n#################### NEW DATASETS ####################")
    if os.path.exists("rebuttal_newdata.csv"):
        nd = read_rows("rebuttal_newdata.csv")
        for ds in ("computers", "cs"):
            rows = [r for r in nd if r["dataset"] == ds]
            if not rows:
                continue
            print(f"  {ds}:")
            for m in ("louvain", "leiden", "kmeans", "mincut", "dmon_ref",
                      "dmonref_noisex", "dmonref_rewired"):
                v = [float(r["nmi"]) for r in rows if r["method"] == m]
                if v:
                    print(f"    {m:<16} {np.mean(v):.3f}+/-{np.std(v):.3f}"
                          f"  (n={len(v)}, per-seed "
                          + ",".join(f"{x:.3f}" for x in v) + ")")
    else:
        print("  rebuttal_newdata.csv not present yet")

    # ---------- map placement of the new datasets ----------
    # Paper logic (Sec. 3): locate a dataset by its two cheap channel
    # measurements, Louvain -> structure axis, k-means -> feature axis.
    print("\n#################### MAP PLACEMENT ####################")
    lou_row = {s: np.mean([mean[("louvain", s, f)] for f in FEATURES])
               for s in STRUCTURES}
    km_col = {f: np.mean([mean[("kmeans", s, f)] for s in STRUCTURES])
              for f in FEATURES}
    print("  louvain row means (s -> NMI): " +
          ", ".join(f"{s}:{v:.3f}" for s, v in lou_row.items()))
    print("  kmeans col means (f -> NMI):  " +
          ", ".join(f"{f}:{v:.3f}" for f, v in km_col.items()))
    if os.path.exists("rebuttal_newdata.csv"):
        nd = read_rows("rebuttal_newdata.csv")
        for ds in ("computers", "cs"):
            L = np.mean([float(r["nmi"]) for r in nd
                         if r["dataset"] == ds and r["method"] == "louvain"])
            K = np.mean([float(r["nmi"]) for r in nd
                         if r["dataset"] == ds and r["method"] == "kmeans"])
            s_star = min(STRUCTURES, key=lambda s: abs(lou_row[s] - L))
            f_star = min(FEATURES, key=lambda f: abs(km_col[f] - K))
            cell = {m: mean[(m, s_star, f_star)]
                    for m in ["louvain", "leiden", "kmeans", "dmon_ref",
                              "mincut"]}
            best = max(cell, key=cell.get)
            fam = FAMILY.get(best, "GNN")
            delta = max(cell["dmon_ref"], cell["mincut"]) - \
                max(cell["louvain"], cell["leiden"], cell["kmeans"])
            print(f"  {ds}: louvain {L:.3f} -> s*={s_star}; "
                  f"kmeans {K:.3f} -> f*={f_star}; map cell winner "
                  f"{best} ({fam}), fig2a delta {delta:+.4f}")


if __name__ == "__main__":
    main()
