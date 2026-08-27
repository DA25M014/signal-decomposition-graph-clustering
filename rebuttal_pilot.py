"""Pilot: verify DMoNRefDeep(depth=1) reproduces dmon_ref, and time depths/widths."""
import time
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi

from generator import generate
from rebuttal_lib import dmon_deep_labels

A, X, y = generate(n=1000, k=4, structure=0.45, feature=0.5, seed=0)

for tag, kw in [
    ("depth1 ref  (expect nmi 0.9491 = dmon_ref@0.45/0.5/s0)", dict(depth=1)),
    ("depth2", dict(depth=2)),
    ("depth3", dict(depth=3)),
    ("depth1 hidden128", dict(depth=1, hidden=128)),
    ("depth1 lr1e-2", dict(depth=1, lr=1e-2)),
    ("depth1 dropout0", dict(depth=1, dropout=0.0)),
]:
    t0 = time.time()
    lab = dmon_deep_labels(A, X, 4, seed=0, epochs=1000, **kw)
    dt = time.time() - t0
    print(f"{tag:<55} nmi={nmi(y, lab):.4f}  {dt:5.1f}s")
