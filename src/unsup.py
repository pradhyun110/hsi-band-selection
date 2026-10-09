"""
LABEL-FREE fitness functions for band selection (use only the spectra of the TRAIN pixels; y is never touched).

  U1  reconstruction R^2 : share of the variance of ALL 164 candidate bands that a linear combination of the
        selected bands can reproduce.   R2(S) = tr( C[S,:]^T  C[S,S]^-1  C[S,:] ) / tr(C)   (C = band covariance)
        higher = better; bounded [0,1]; redundant bands add nothing, so redundancy is handled implicitly.
  U2  entropy - redundancy : mean normalised band entropy of S  minus  mean absolute pairwise correlation of S
        (mRMR-style, unsupervised).  higher = better.

Scale calibration: SA's acceptance rule depends on fitness DIFFERENCES, so each unsupervised fitness is affinely
rescaled (rank-preserving, so it changes nothing for GA/GWO/Jaya/DE/PSO) to have the same spread over random
K-subsets as the supervised LDA wrapper has.  Temperatures therefore mean the same thing in every experiment.
"""
import numpy as np
from common import LDAFitness, mask_from_idx


class UnsupBase:
    def __init__(self, setup):
        X = setup["X_train"]; self.n = X.shape[1]
        self.C = np.corrcoef(X, rowvar=False)               # bands are z-scored -> correlation matrix, trace = n
        self.Rabs = np.abs(self.C)
        ent = []
        for j in range(self.n):
            h, _ = np.histogram(X[:, j], bins=256); p = h[h > 0] / h.sum(); ent.append(-(p * np.log2(p)).sum())
        ent = np.array(ent); self.ent = ent / ent.max()
        self.calls = 0

    def r2(self, cols):
        Cs = self.C[np.ix_(cols, cols)] + 1e-6 * np.eye(len(cols))
        B = self.C[cols, :]
        return float((B * np.linalg.solve(Cs, B)).sum() / self.n)

    def u2(self, cols):
        k = len(cols)
        red = (self.Rabs[np.ix_(cols, cols)].sum() - k) / (k * (k - 1)) if k > 1 else 0.0
        return float(self.ent[cols].mean() - red)


class UnsupFitness:
    def __init__(self, base, kind, K, setup, seed=123, n_ref=300):
        self.base, self.kind = base, kind
        self.f = base.r2 if kind == "u1" else base.u2
        rng = np.random.default_rng(seed); n = base.n
        subs = [np.sort(rng.choice(n, K, replace=False)) for _ in range(n_ref)]
        raw = np.array([self.f(c) for c in subs])
        lda = LDAFitness(setup); sup = np.array([lda(mask_from_idx(c, n)) for c in subs])
        self.mu, self.scale = raw.mean(), sup.std() / raw.std()
        self.calls = 0

    def __call__(self, mask):
        self.calls += 1
        cols = np.flatnonzero(mask)
        return (self.f(cols) - self.mu) * self.scale

    def raw(self, mask):
        return self.f(np.flatnonzero(mask))
