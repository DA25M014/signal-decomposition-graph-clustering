"""
Attributed SBM generator with two independent signal knobs.

This is the instrument for the whole study: it lets us dial how much
community information lives in the GRAPH STRUCTURE and how much lives
in the NODE FEATURES, independently.

structure in [0, 1]:
    1.0 -> every edge stays inside its community (perfect blocks)
    0.0 -> wiring is statistically indistinguishable from a random graph
    Internally mapped to the cross-community edge fraction
    mu = (1 - structure) * (1 - 1/k), the planted-partition mixing rate.

feature >= 0:
    Ratio of between-cluster center spread to within-cluster noise (std 1).
    0.0  -> features carry zero community information
    1.0  -> moderate separation
    3.0+ -> clusters clearly separated in feature space

Usage:
    from generator import generate
    A, X, y = generate(n=1000, k=4, structure=0.6, feature=1.5, seed=0)

    A : scipy.sparse CSR adjacency (n x n, symmetric, unweighted)
    X : float32 feature matrix (n x dim)
    y : int array of ground-truth community labels (n,)
"""

import numpy as np
import networkx as nx
from scipy import sparse


def generate(n=1000, k=4, avg_deg=16, structure=1.0, feature=1.0,
             dim=32, seed=0):
    rng = np.random.default_rng(seed)

    # ----- communities: equal-sized blocks (nodes ordered by block) -----
    sizes = [n // k] * k
    sizes[-1] += n - sum(sizes)                 # absorb rounding remainder
    y = np.repeat(np.arange(k), sizes)

    # ----- graph: planted-partition SBM -----
    mu = (1.0 - structure) * (1.0 - 1.0 / k)    # cross-community edge fraction
    p_in = (1.0 - mu) * avg_deg / (n / k - 1.0)
    p_out = mu * avg_deg / (n - n / k)
    p = np.full((k, k), p_out)
    np.fill_diagonal(p, p_in)
    G = nx.stochastic_block_model(sizes, p.tolist(),
                                  seed=int(rng.integers(2**31)))
    A = nx.to_scipy_sparse_array(G, format="csr", dtype=np.float32)

    # ----- features: Gaussian mixture tied to communities -----
    centers = rng.normal(0.0, feature, size=(k, dim))
    X = (centers[y] + rng.normal(0.0, 1.0, size=(n, dim))).astype(np.float32)

    return A, X, y


def to_edge_index(A):
    """CSR adjacency -> (2, E) int64 array for PyTorch Geometric (Sunday)."""
    coo = sparse.coo_matrix(sparse.triu(A, k=1))
    e = np.vstack([coo.row, coo.col])
    return np.hstack([e, e[::-1]]).astype(np.int64)


if __name__ == "__main__":
    A, X, y = generate(seed=0)
    print("nodes:", A.shape[0], "| edges:", int(A.nnz // 2),
          "| avg degree:", round(A.nnz / A.shape[0], 1))
    print("X:", X.shape, X.dtype, "| communities:", np.bincount(y))
