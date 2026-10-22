"""
Other criteria that can use the same recursive-Cholesky greedy engine -- tested to justify choosing Fisher.

The engine (shared): add one band at a time; keep the Cholesky factor L of a covariance matrix S of the chosen
bands and extend it by one row per band, so every candidate costs O(k^2).  Any criterion built from
"squared distances measured with S^-1" fits it, because each distance grows by an exact extra term:
    for a vector v (one value per band):  |L^-1 v_S|^2  gains  ((v[x] - y . u)^2) / d^2   when band x is added.

Criteria
  fisher        sum over classes of n_c * |class mean - overall mean|^2  in Sw^-1 units  (= trace Sw^-1 Sb)
                -> the chosen method (identical to fisher_cholesky.fisher_select)
  pairwise_mean mean over all 120 class PAIRS of the Mahalanobis distance between their means (every pair equal,
                so small crops count as much as big ones)
  pairwise_min  the SMALLEST pairwise class distance (maximise the worst-separated pair, e.g. corn vs soybean)
  volume        label-free: maximise the volume (determinant) of the band correlation matrix -- the base paper's
                final step, here run on all 164 bands without its funnel
"""
import numpy as np
from scipy.linalg import solve_triangular


def class_stats(X, y, ridge=0.01):
    classes = np.unique(y); m = X.mean(0)
    n = np.array([(y == c).sum() for c in classes]); M = np.vstack([X[y == c].mean(0) for c in classes])
    W = X - M[np.searchsorted(classes, y)]; Sw = W.T @ W / len(y)
    Sw = Sw + ridge * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    return n, M, m, Sw


def greedy_cholesky(S, V, w, K, agg="sum", first=None):
    """S: bands x bands covariance; V: rows of vectors to separate (one value per band); w: row weights.
    agg = 'sum' (weighted sum of squared distances), 'min' (smallest distance), 'volume' (ignore V; max det S)."""
    n = len(S); rest = list(range(n)); chosen = []
    L = np.zeros((0, 0)); U = np.zeros((len(V), 0)); D = np.zeros(len(V))
    while len(chosen) < K:
        if chosen:
            Y = solve_triangular(L, S[np.ix_(chosen, rest)], lower=True); d2 = S[rest, rest] - (Y ** 2).sum(0)
            num = V[:, rest] - U @ Y
        else:
            Y = np.zeros((0, len(rest))); d2 = S[rest, rest].copy(); num = V[:, rest]
        d = np.sqrt(np.maximum(d2, 1e-12)); comp = num / d; tot = D[:, None] + comp ** 2
        if agg == "volume":
            score = d2 if chosen or first is None else -np.abs(np.array(rest) - first)
        elif agg == "min":
            score = tot.min(0)
        else:
            score = (w[:, None] * tot).sum(0)
        j = int(np.argmax(score)); b = rest[j]
        L = np.block([[L, np.zeros((len(chosen), 1))], [Y[:, j][None, :], d[j:j + 1][None, :]]]) if chosen else d[j:j + 1][None, :]
        U = np.hstack([U, comp[:, j:j + 1]]); D = tot[:, j]
        chosen.append(b); rest.remove(b)
    return chosen


def select(criterion, Xtr, ytr, K):
    if criterion == "volume":
        C = np.corrcoef(Xtr, rowvar=False)
        first = int(np.argmax((C ** 2).sum(0)))                       # start from the most representative band
        return greedy_cholesky(C, np.zeros((0, len(C))), np.zeros(0), K, agg="volume", first=first)
    n, M, m, Sw = class_stats(Xtr, ytr)
    if criterion == "fisher":
        return greedy_cholesky(Sw, M - m, n.astype(float), K, agg="sum")
    i, j = np.triu_indices(len(M), 1); P = M[i] - M[j]                 # all 120 class pairs
    return greedy_cholesky(Sw, P, np.ones(len(P)), K, agg="min" if criterion == "pairwise_min" else "sum")
