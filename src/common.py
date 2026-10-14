"""
Shared pieces for the fixed-K band-selection simulation.

PROTOCOL (no leakage)
  * Outer split: stratified 20% train / 80% test, seed 42 (the agreed protocol).
  * Search / fitness uses TRAIN pixels ONLY: the 2,051 training pixels are cut into two
    stratified halves A/B; fitness = mean validation accuracy of (fit on A, test on B)
    and (fit on B, test on A).  The 8,198 test pixels are touched only to score a FINAL subset.
  * Candidate pool: the 164 entropy-pre-screened bands (Otsu).  An all-200-band baseline is
    reported separately so the pre-screening effect is visible.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from data_setup import (load_fixed_setup, stratified_split, DATA_DIR, RNG_SPLIT_SEED,
                        TEST_FRACTION)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "sim"
OUT.mkdir(parents=True, exist_ok=True)


def load_setup():
    """164-candidate, z-scored (train stats) setup + inner folds."""
    S = load_fixed_setup()
    return S


def load_all200():
    """Same split, same z-scoring recipe, but all 200 raw bands (for the pre-screening baseline)."""
    cube = np.load(DATA_DIR / "indianpinearray.npy").astype(np.float64)
    gt = np.load(DATA_DIR / "IPgt.npy")
    X = cube.reshape(-1, cube.shape[2]); y = gt.reshape(-1)
    m = y != 0
    X, y = X[m], y[m]
    tr, te = stratified_split(y, TEST_FRACTION, RNG_SPLIT_SEED)
    mu, sd = X[tr].mean(0), X[tr].std(0); sd[sd == 0] = 1
    return dict(X_train=(X[tr] - mu) / sd, y_train=y[tr], X_test=(X[te] - mu) / sd, y_test=y[te],
                classes=np.unique(y))


class _Folds:
    """Two stratified halves of the TRAIN set; each is used once for fit, once for validation."""
    def __init__(self, setup, seed=7):
        ytr = setup["y_train"]; Xtr = setup["X_train"]
        a, b = stratified_split(ytr, 0.5, seed)
        self.classes = setup["classes"]
        self.folds = [(Xtr[a], ytr[a], Xtr[b], ytr[b]), (Xtr[b], ytr[b], Xtr[a], ytr[a])]
        self.n = Xtr.shape[1]


class LDAFitness(_Folds):
    """
    Wrapper fitness with a REAL (regularised) LDA classifier trained inside each fold on the selected
    bands only.  Closed form from precomputed class means / pooled covariance, so one evaluation costs
    a k x k solve instead of refitting sklearn.  Returns mean validation overall accuracy (higher=better).
    """
    def __init__(self, setup, ridge=0.01, seed=7):
        super().__init__(setup, seed)
        self.ridge = ridge
        self.pre = []
        C = len(self.classes)
        for Xf, yf, Xv, yv in self.folds:
            M = np.vstack([Xf[yf == c].mean(0) for c in self.classes])               # (C, n)
            Xc = Xf - M[np.searchsorted(self.classes, yf)]
            S = (Xc.T @ Xc) / (len(yf) - C)                                          # pooled covariance
            logp = np.log(np.array([(yf == c).mean() for c in self.classes]))
            self.pre.append((M, S, logp, Xv, np.searchsorted(self.classes, yv)))
        self.calls = 0

    def __call__(self, mask):
        self.calls += 1
        cols = np.flatnonzero(mask)
        k = len(cols)
        if k == 0:
            return 0.0
        accs = 0.0
        for M, S, logp, Xv, yv in self.pre:
            Ss = S[np.ix_(cols, cols)]
            Ss = Ss + self.ridge * (np.trace(Ss) / k) * np.eye(k)
            Mc = M[:, cols]
            W = np.linalg.solve(Ss, Mc.T)                                            # (k, C)
            b = -0.5 * np.einsum("ck,kc->c", Mc, W) + logp
            pred = np.argmax(Xv[:, cols] @ W + b, axis=1)
            accs += (pred == yv).mean()
        return accs / len(self.pre)


class NCCFitness(_Folds):
    """Nearest-centroid wrapper (the cheap hardware-friendly surrogate used earlier), same folds."""
    def __init__(self, setup, seed=7):
        super().__init__(setup, seed)
        self.pre = []
        for Xf, yf, Xv, yv in self.folds:
            M = np.vstack([Xf[yf == c].mean(0) for c in self.classes])
            self.pre.append((M, Xv, np.searchsorted(self.classes, yv)))
        self.calls = 0

    def __call__(self, mask):
        self.calls += 1
        cols = np.flatnonzero(mask)
        if len(cols) == 0:
            return 0.0
        accs = 0.0
        for M, Xv, yv in self.pre:
            C = M[:, cols]; X = Xv[:, cols]
            d = (X * X).sum(1)[:, None] - 2 * X @ C.T + (C * C).sum(1)[None, :]
            accs += (np.argmin(d, 1) == yv).mean()
        return accs / len(self.pre)


def mask_from_idx(idx, n):
    m = np.zeros(n, dtype=bool); m[np.asarray(idx, dtype=int)] = True
    return m


def redundancy(setup, idx):
    idx = np.asarray(idx)
    if len(idx) < 2:
        return 0.0
    C = np.corrcoef(setup["X_train"][:, idx], rowvar=False)
    iu = np.triu_indices_from(C, 1)
    return float(np.abs(C[iu]).mean())
