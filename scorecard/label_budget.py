"""Default prediction when only a small budget of loans can be labelled.

This module re-implements, for use in this repository, the two components of
ADGSSL (https://github.com/TongyuanBai/ADGSSL):

* SJUS key-sample selection (``Code/SJUS.py``): a minimum spanning tree (MST)
  of the pairwise Euclidean distances, spectral clustering of the MST, a
  structural score phi (min-max normalised MST degree), a boundary-uncertainty
  score psi, and budget-aware selection in three phases (joint set, union
  completion, alternating supplementary selection).
* Measure propagation (``Code/label_propagation.py``; Subramanya and Bilmes,
  JMLR 12, 2011) on a K-nearest-neighbour graph with edge weights
  1 / (1 + distance).

The propagation updates are those of the ADGSSL code, written with sparse
matrices so that pools of a few thousand loans run in seconds. Two settings
differ from that code by default, and both can be switched back:

* Graph. ADGSSL uses the directed K-nearest-neighbour graph. In the q-step
  each vertex then averages over its own out-neighbours, whereas minimising
  the measure-propagation objective requires the weights pointing into it;
  the two coincide only for a symmetric graph. ``knn_graph(symmetric=True)``
  keeps an edge if either point is among the other's K nearest neighbours.
* Stopping rule. ADGSSL stops when the relative decrease of its convergence
  measure is at most ``tol``, which also happens when the measure increases.
  The default here stops when q no longer changes (``stop="q_change"``).

In ``sjus_scores``, a zero distance between two distinct rows is replaced by
a negligible positive value so that duplicated rows stay connected in the
minimum spanning tree.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors

SMALL = 1e-10

LOG_FEATURES = ["funded_amnt", "annual_inc", "total_bal_il", "total_il_high_credit_limit", "delinq_amnt"]


# ----------------------------------------------------------------------
# label-free feature matrix
# ----------------------------------------------------------------------
def feature_matrix(df: pd.DataFrame, numeric: list[str], categorical: list[str], reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Numeric variables (log1p for amounts), a missing indicator and median
    imputation for variables with missing values, and dummy variables for
    categorical variables. The medians and the dummy categories are taken
    from ``reference`` (default: ``df``). No outcome information is used."""
    reference = df if reference is None else reference
    out = pd.DataFrame(index=df.index)
    for c in numeric:
        x = df[c].astype(float)
        if reference[c].isna().any() or x.isna().any():
            out[f"{c}_missing"] = x.isna().astype(float)
            x = x.fillna(reference[c].astype(float).median())
        out[c] = np.log1p(x) if c in LOG_FEATURES else x
    for c in categorical:
        levels = sorted(reference[c].astype(str).unique())
        for level in levels[1:]:  # first level is the reference category
            out[f"{c}_{level}"] = (df[c].astype(str) == level).astype(float)
    return out


# ----------------------------------------------------------------------
# SJUS
# ----------------------------------------------------------------------
def sjus_scores(X: np.ndarray, n_clusters: int = 2, random_state: int = 42):
    """Return phi (structural centrality), psi (boundary uncertainty) and the
    spectral cluster labels, as computed in ADGSSL ``Code/SJUS.py``."""
    dist = euclidean_distances(X)
    # scipy treats a zero off-diagonal distance as a missing edge; duplicated
    # rows would disconnect the tree, so they get a negligible distance.
    off_diag = ~np.eye(len(X), dtype=bool)
    dist[off_diag & (dist == 0)] = 1e-12
    mst = minimum_spanning_tree(dist).toarray()
    adj = ((mst + mst.T) > 0).astype(float)

    clusters = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", random_state=random_state).fit_predict(adj)

    degrees = adj.sum(axis=1)
    phi = (degrees - degrees.min()) / (degrees.max() - degrees.min() + 1e-12)

    # ADGSSL uses `sc.embedding_` when it exists and the adjacency matrix
    # otherwise. scikit-learn's SpectralClustering has no `embedding_`
    # attribute, so the adjacency rows are what the published code uses.
    Z = adj
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(Z)
    d = np.sort(cdist(Z, km.cluster_centers_), axis=1)
    psi = np.clip(1.0 - (d[:, 1] - d[:, 0]) / (d[:, 0] + d[:, 1] + 1e-12), 0, 1)
    return phi, psi, clusters


def sjus_select(phi, psi, clusters, budget: int, percentile: float = 95, min_cluster_size: int = 20) -> np.ndarray:
    """Budget-aware selection of ``budget`` sample indices (ADGSSL ``Code/SJUS.py``)."""
    n = len(phi)
    joint, union_rest = [], []
    for k in np.unique(clusters):
        idx = np.flatnonzero(clusters == k)
        pct = percentile if len(idx) >= min_cluster_size else 85
        omega_phi = idx[phi[idx] >= np.percentile(phi[idx], pct)]
        omega_psi = idx[psi[idx] >= np.percentile(psi[idx], pct)]
        cap = np.intersect1d(omega_phi, omega_psi)
        cup = np.union1d(omega_phi, omega_psi)
        joint.extend(cap.tolist())
        union_rest.extend(np.setdiff1d(cup, cap).tolist())

    omega = list(joint)  # phase 1: joint set
    if len(omega) < budget:  # phase 2: union completion
        union_rest = sorted(union_rest, key=lambda i: phi[i] + psi[i], reverse=True)
        omega.extend(union_rest[: budget - len(omega)])
    if len(omega) < budget:  # phase 3: alternating supplementary selection
        chosen = set(omega)
        remaining = [i for i in range(n) if i not in chosen]
        phi_rank = sorted(remaining, key=lambda i: phi[i], reverse=True)
        psi_rank = sorted(remaining, key=lambda i: psi[i], reverse=True)
        p_ptr = s_ptr = 0
        while len(omega) < budget and (p_ptr < len(phi_rank) or s_ptr < len(psi_rank)):
            while p_ptr < len(phi_rank) and phi_rank[p_ptr] in chosen:
                p_ptr += 1
            if p_ptr < len(phi_rank) and len(omega) < budget:
                omega.append(phi_rank[p_ptr])
                chosen.add(phi_rank[p_ptr])
            while s_ptr < len(psi_rank) and psi_rank[s_ptr] in chosen:
                s_ptr += 1
            if s_ptr < len(psi_rank) and len(omega) < budget:
                omega.append(psi_rank[s_ptr])
                chosen.add(psi_rank[s_ptr])
    if len(omega) > budget:  # phase 4: truncation of a large joint set
        omega = sorted(omega, key=lambda i: phi[i] + psi[i], reverse=True)[:budget]
    return np.array(sorted(omega), dtype=int)


# ----------------------------------------------------------------------
# measure propagation
# ----------------------------------------------------------------------
def knn_graph(X: np.ndarray, k: int = 5, symmetric: bool = True) -> sparse.csr_matrix:
    """K-nearest-neighbour graph with weight 1 / (1 + Euclidean distance).

    ``symmetric=False`` returns the directed graph used in ADGSSL (row i holds
    the K nearest neighbours of i); ``symmetric=True`` takes the elementwise
    maximum of that graph and its transpose."""
    n = len(X)
    dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    rows, cols, vals = [], [], []
    for i in range(n):
        keep = idx[i] != i
        js, ds = idx[i][keep][:k], dist[i][keep][:k]
        rows.extend([i] * len(js))
        cols.extend(js.tolist())
        vals.extend((1.0 / (1.0 + ds)).tolist())
    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return W.maximum(W.T).tocsr() if symmetric else W


def measure_propagation(
    W: sparse.csr_matrix,
    labels: np.ndarray,
    mu: float = 0.1,
    nu: float = 0.01,
    stop: str = "q_change",
    tol: float = 0.1,
    eps: float = 1e-6,
    max_iter: int = 1000,
):
    """Alternating minimisation of the measure propagation objective.

    ``labels`` holds a class index for labelled vertices and -1 otherwise.
    Returns the class distributions q (n x classes), the classes and the
    number of iterations.

    Stopping rules:
    - ``"adgssl"``: the rule of the ADGSSL code, which stops when the relative
      decrease of its convergence measure is at most ``tol``. The rule also
      stops when the measure increases, which on the credit data happened
      after two iterations in many runs.
    - ``"q_change"`` (default): stop when no entry of q changes by more than
      ``eps`` between two iterations.
    """
    labelled = labels >= 0
    classes = np.unique(labels[labelled])
    n, c = W.shape[0], len(classes)
    R = np.zeros((n, c))
    R[np.flatnonzero(labelled), np.searchsorted(classes, labels[labelled])] = 1.0
    delta = labelled.astype(float)
    d = np.asarray(W.sum(axis=1)).ravel()
    gamma = nu + mu * d
    q = np.full((n, c), 1.0 / c)
    history = []
    it = 0
    for it in range(max_iter):
        a = (W @ np.log(q + SMALL)) * (mu / gamma)[:, None]
        p = np.exp(a - a.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        q_new = (mu * (W @ p) + R) / (mu * d + delta)[:, None]
        beta = np.log(np.max(q_new / (q + SMALL), axis=1) + SMALL)
        history.append(float(np.sum((delta + d) * beta)))
        max_change = float(np.max(np.abs(q_new - q)))
        q = q_new
        if stop == "adgssl":
            if it > 0 and (history[-2] - history[-1]) / history[-1] <= tol:
                break
        elif max_change < eps:
            break
    return q, classes, it + 1
