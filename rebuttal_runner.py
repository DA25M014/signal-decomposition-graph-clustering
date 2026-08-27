"""
LoG 2026 #57 rebuttal experiments. Writes rebuttal57.csv (resumable).

Arms (all on the paper's 8x8 grid unless noted):
  depth      dmon_ref with 2 and 3 official-style GCN layers, 5 seeds
  mcdepth    mincut (fixed-budget encoder) with 2 and 3 GCN layers, 5 seeds
  tune       dmon_ref, all combos of lr {1e-3,1e-2} x dropout {0,0.5}
             x hidden {64,128} except the reference combo (already in
             results.csv as dmon_ref), 5 seeds
  realdepth  dmon_ref depth 2 and 3 on cora/citeseer/photo, 3 seeds

CSV columns: exp,target,seed,config,nmi,ari,seconds
  target = "s_f" for synthetic cells, dataset name for real
  config = e.g. "d2" | "mc_d2" | "lr:0.01|do:0.0|h:128" | "d2" (realdepth)

Run:  caffeinate -i python3 rebuttal_runner.py   # caffeinate: macOS, optional
"""

import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

CSV_PATH = "rebuttal57.csv"
STRUCTURES = [0.05, 0.2, 0.35, 0.45, 0.55, 0.65, 0.8, 0.95]
FEATURES = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0, 1.5]
SEEDS5 = [0, 1, 2, 3, 4]
SEEDS3 = [0, 1, 2]
WORKERS = 5


def _init_worker():
    import torch
    torch.set_num_threads(2)


@lru_cache(maxsize=None)
def _real(name):
    from day6_realdata import load_dataset
    return load_dataset(name)


def run_task(task):
    kind, target, seed, config = task
    import numpy as np
    from sklearn.metrics import (normalized_mutual_info_score,
                                 adjusted_rand_score)
    from generator import generate
    from rebuttal_lib import dmon_deep_labels

    if kind in ("depth", "mcdepth", "tune"):
        s, f = (float(x) for x in target.split("_"))
        A, X, y = generate(n=1000, k=4, structure=s, feature=f, seed=seed)
        k = 4
    else:
        A, X, y, k = _real(target)

    t0 = time.time()
    if kind in ("depth", "realdepth"):
        depth = int(config[1])
        lab = dmon_deep_labels(A, X, k, seed=seed, epochs=1000, depth=depth)
    elif kind == "mcdepth":
        depth = int(config[-1])
        lab = mincut_deep_labels(A, X, k, seed=seed, epochs=500, depth=depth)
    else:
        parts = dict(p.split(":") for p in config.split("|"))
        lab = dmon_deep_labels(A, X, k, seed=seed, epochs=1000, depth=1,
                               lr=float(parts["lr"]),
                               dropout=float(parts["do"]),
                               hidden=int(parts["h"]))
    dt = time.time() - t0
    return (kind, target, seed, config,
            round(normalized_mutual_info_score(y, lab), 4),
            round(adjusted_rand_score(y, lab), 4), round(dt, 2))


def mincut_deep_labels(A, X, k, seed=0, epochs=500, depth=2, hidden=64,
                       lr=1e-3):
    """day4_runner.MinCutNet generalized to `depth` GCNConv+skip layers."""
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, dense_mincut_pool
    from generator import to_edge_index

    class MinCutDeep(torch.nn.Module):
        def __init__(self, in_dim, hidden, k, depth):
            super().__init__()
            dims = [in_dim] + [hidden] * depth
            self.convs = torch.nn.ModuleList(
                GCNConv(dims[i], dims[i + 1]) for i in range(depth))
            self.skips = torch.nn.ModuleList(
                torch.nn.Linear(dims[i], dims[i + 1]) for i in range(depth))
            self.assign = torch.nn.Linear(hidden, k)

        def forward(self, x, edge_index, adj):
            h = x
            for conv, skip in zip(self.convs, self.skips):
                h = F.selu(conv(h, edge_index) + skip(h))
            s = self.assign(h)
            _, _, mc, o = dense_mincut_pool(h.unsqueeze(0), adj,
                                            s.unsqueeze(0))
            return s, mc + o

    torch.manual_seed(seed)
    x = torch.from_numpy(X)
    ei = torch.from_numpy(to_edge_index(A))
    adj = torch.from_numpy(A.toarray()).float().unsqueeze(0)
    model = MinCutDeep(X.shape[1], hidden, k, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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


def build_tasks():
    tasks = []
    cells = [f"{s}_{f}" for s in STRUCTURES for f in FEATURES]
    for cell in cells:
        for seed in SEEDS5:
            for d in (2, 3):
                tasks.append(("depth", cell, seed, f"d{d}"))
                tasks.append(("mcdepth", cell, seed, f"mc_d{d}"))
            for lr in (1e-3, 1e-2):
                for do in (0.0, 0.5):
                    for h in (64, 128):
                        if (lr, do, h) == (1e-3, 0.5, 64):
                            continue        # = dmon_ref, in results.csv
                        tasks.append(("tune", cell, seed,
                                      f"lr:{lr}|do:{do}|h:{h}"))
    for name in ("cora", "citeseer", "photo"):
        for seed in SEEDS3:
            for d in (2, 3):
                tasks.append(("realdepth", name, seed, f"d{d}"))
    return tasks


def main():
    done = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH) as fh:
            for r in csv.DictReader(fh):
                done.add((r["exp"], r["target"], r["seed"], r["config"]))
    new = not os.path.exists(CSV_PATH)
    fh = open(CSV_PATH, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["exp", "target", "seed", "config", "nmi", "ari",
                    "seconds"])
        fh.flush()

    tasks = [t for t in build_tasks()
             if (t[0], t[1], str(t[2]), t[3]) not in done]
    print(f"{len(tasks)} tasks to run ({len(done)} already done)", flush=True)

    # real datasets download serially first to avoid worker races
    for name in ("cora", "citeseer", "photo"):
        _real(name)

    t0 = time.time()
    ran = 0
    with ProcessPoolExecutor(max_workers=WORKERS,
                             initializer=_init_worker) as ex:
        futs = {ex.submit(run_task, t): t for t in tasks}
        for fut in as_completed(futs):
            kind, target, seed, config, nmi, ari, dt = fut.result()
            w.writerow([kind, target, seed, config, nmi, ari, dt])
            fh.flush()
            ran += 1
            if ran % 25 == 0 or ran == len(tasks):
                eta = (len(tasks) - ran) * (time.time() - t0) / ran / 60
                print(f"[{ran}/{len(tasks)}] last: {kind} {target} "
                      f"seed={seed} {config} nmi={nmi} | eta ~{eta:.0f} min",
                      flush=True)
    fh.close()
    print("rebuttal sweep complete -> " + CSV_PATH, flush=True)


if __name__ == "__main__":
    main()
