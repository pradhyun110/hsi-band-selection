"""
Base paper band selector: our re-implementation of MFE-CFSP (Hao et al., IEEE JSTARS 2026),
a CFSP-style version of their own baseline, and the improvement ideas we tested on it.

Everything works on the same 164 Otsu-pre-screened candidate bands as the rest of the project.
Positions returned are indices into those 164 bands.

MFE-CFSP pipeline (paper Sec. II)
  1. gate       block-wise maximum entropy per band; keep bands with E > mean + gamma*std
  2. partition  K contiguous groups, cut at the largest curvature peaks of the mean spectrum
                (finite differences; CFSP used a fitted spline)
  3. features   variance, Sobel edge intensity, SAD texture -> entropy-weighted TOPSIS with L1
                distance -> top eta = 3 bands per group = candidates
  4. final      cosine-similarity pruning, then maximum-ellipsoid-volume forward selection
                with a recursive Cholesky update

Settings the paper does not give, chosen here and stated in the report:
  gamma = -0.5 (keeps 111 of the 164 bands), block size 16 x 16, 256 grey levels,
  cosine similarity on mean-centred band vectors (raw vectors are all > 0.98 alike),
  T_sim = 0.98, first MEV band = highest TOPSIS score, MEV on z-scored pixels of the whole image.

Our ideas (each changes ONE thing in MFE-CFSP so its effect is visible)
  idea 1  Fisher pick     final step maximises class separation trace(Sw^-1 Sb) instead of volume,
                          with the SAME recursive Cholesky update (labels: training pixels only).
                          Applied to the 3K candidates WITHOUT cosine pruning: on Indian Pines the pruning
                          leaves about K bands, so any final step would have nothing to choose from, and the
                          Fisher criterion already ignores bands that add no new separation.
  idea 2  Fisher TOPSIS   a 4th TOPSIS feature: the per-band Fisher score (training pixels only)
  idea 3  Otsu gate       their gate replaced by our parameter-free Otsu entropy pre-screen
                          (= no extra gate on the 164 Otsu bands)
  idea 4  GA on candidates  our genetic algorithm (LDA fitness, same budget) searches only the
                          3K MFE candidates instead of all 164 bands
"""
import numpy as np
from scipy.ndimage import convolve
from scipy.interpolate import UnivariateSpline
from fisher_cholesky import fisher_select   # idea 1 uses the chosen method's engine

GAMMA, BLOCK, LEVELS, ETA, T_SIM = -0.5, 16, 256, 3, 0.98


# ---------------------------------------------------------------- unsupervised band descriptors
def block_max_entropy(img, S=BLOCK, Q=LEVELS):
    lo, hi = img.min(), img.max()
    q = np.floor((img - lo) / (hi - lo + 1e-12) * (Q - 1)).astype(int)
    best = 0.0
    for u in range(0, img.shape[0], S):
        for v in range(0, img.shape[1], S):
            b = q[u:u + S, v:v + S].ravel()
            p = np.bincount(b, minlength=Q) / b.size; p = p[p > 0]
            best = max(best, float(-(p * np.log2(p)).sum()))
    return best


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
NEIGH = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]


def band_features(cube):
    """(L, 3) array: variance, Sobel edge intensity, SAD texture, for each band image (raw values)."""
    F = []
    for i in range(cube.shape[2]):
        X = cube[:, :, i]
        var = X.var()
        edge = (np.abs(convolve(X, SOBEL_X, mode="nearest")) + np.abs(convolve(X, SOBEL_X.T, mode="nearest"))).mean()
        P = np.pad(X, 1, mode="edge")
        sad = sum(np.abs(X - P[1 + dx:1 + dx + X.shape[0], 1 + dy:1 + dy + X.shape[1]]) for dx, dy in NEIGH).mean()
        F.append((var, edge, sad))
    return np.array(F)


def fisher_scores(Xtr, ytr):
    """per-band between-class / within-class variance (training pixels only)."""
    m = Xtr.mean(0); sb = np.zeros(Xtr.shape[1]); sw = np.zeros(Xtr.shape[1])
    for c in np.unique(ytr):
        Xc = Xtr[ytr == c]; sb += len(Xc) * (Xc.mean(0) - m) ** 2; sw += ((Xc - Xc.mean(0)) ** 2).sum(0)
    return sb / (sw + 1e-12)


# ---------------------------------------------------------------- pipeline pieces
def gate(E, gamma=GAMMA):
    return np.flatnonzero(E > E.mean() + gamma * E.std())


def partition(mean_spec, keep, K, spline=False):
    """K contiguous groups of the kept bands, cut at the K-1 largest curvature peaks."""
    y = mean_spec[keep]; y = (y - y.min()) / (y.max() - y.min() + 1e-12)
    if spline:                                                          # CFSP: fit a smooth curve first
        t = np.arange(len(y)); s = UnivariateSpline(t, y, k=3, s=len(y) * 1e-4)
        d1, d2 = s.derivative(1)(t), s.derivative(2)(t)
    else:                                                               # MFE-CFSP: central differences
        d1 = np.gradient(y); d2 = np.zeros_like(y); d2[1:-1] = y[2:] - 2 * y[1:-1] + y[:-2]
    curv = np.abs(d2) / (1 + d1 ** 2) ** 1.5
    peaks = [i for i in range(1, len(y) - 1) if curv[i] >= curv[i - 1] and curv[i] >= curv[i + 1]]
    order = sorted(peaks, key=lambda i: -curv[i]) + sorted(set(range(1, len(y))) - set(peaks), key=lambda i: -curv[i])
    cuts = sorted(order[:K - 1])
    return [keep[a:b] for a, b in zip([0] + cuts, cuts + [len(keep)]) if b > a]


def topsis(F):
    """entropy-weighted TOPSIS with L1 (Manhattan) distance; returns closeness C (higher = better)."""
    Z = (F - F.min(0)) / (F.max(0) - F.min(0) + 1e-12)
    P = Z / (Z.sum(0) + 1e-12)
    e = -(np.where(P > 0, P * np.log(P + 1e-300), 0)).sum(0) / np.log(len(F))
    w = (1 - e) / (1 - e).sum()
    Dp, Dm = (w * np.abs(Z - 1)).sum(1), (w * np.abs(Z)).sum(1)
    return Dm / (Dp + Dm + 1e-12)


def candidates(groups, score, eta=ETA):
    cand = []
    for g in groups:
        cand += list(g[np.argsort(-score[g])[:eta]])
    return sorted(cand, key=lambda b: -score[b])                        # descending score


def cosine_prune(cand, Zc, t=T_SIM):
    """walk the candidates in score order; drop one that is too similar to an already kept band."""
    kept = []
    for b in cand:
        if all(abs(Zc[:, b] @ Zc[:, k]) <= t for k in kept):
            kept.append(b)
    return kept


def mev_select(cand, G, K, recursive=True):
    """maximum-ellipsoid-volume forward selection on Gram matrix G (bands x bands).
    recursive=True : recursive Cholesky update (paper); False: full determinant recomputed (baseline)."""
    cand = list(cand); chosen = [cand[0]]; rest = cand[1:]
    L = np.array([[np.sqrt(G[cand[0], cand[0]])]])
    while len(chosen) < K and rest:
        if recursive:
            Y = np.linalg.solve(L, G[np.ix_(chosen, rest)])            # forward substitution, all candidates
            d2 = G[rest, rest] - (Y ** 2).sum(0)
            j = int(np.argmax(d2)); b = rest[j]
            L = np.block([[L, np.zeros((len(chosen), 1))], [Y[:, j][None, :], np.sqrt(max(d2[j], 1e-12))]])
        else:
            dets = [np.linalg.slogdet(G[np.ix_(chosen + [x], chosen + [x])])[1] for x in rest]
            j = int(np.argmax(dets)); b = rest[j]
        chosen.append(b); rest.remove(b)
    return chosen


def fill(sel, ranked, K):
    """if pruning left fewer than K bands, top up from the remaining candidates in score order."""
    sel = list(sel)
    for b in ranked:
        if len(sel) >= K: break
        if b not in sel: sel.append(b)
    return sel


# ---------------------------------------------------------------- shared precomputation (label-free)
class Image:
    """everything label-free that the selectors need, computed once from the whole 145 x 145 image."""
    def __init__(self, cube164):
        H, W, L = cube164.shape
        self.E = np.array([block_max_entropy(cube164[:, :, i]) for i in range(L)])
        self.F = band_features(cube164)
        X = cube164.reshape(-1, L)
        self.mean_spec = X.mean(0)
        Zc = X - X.mean(0); self.Zc = Zc / np.linalg.norm(Zc, axis=0)   # unit, centred band vectors
        self.G = self.Zc.T @ self.Zc                                     # correlation matrix of all pixels
        self.all = np.arange(L)


def mfe_cfsp(img, K, Xtr=None, ytr=None, use_gate=True, fisher_topsis=False, fisher_pick=False,
             prune=True, recursive=True, spatial=True, spline=False, return_candidates=False):
    keep = gate(img.E) if use_gate else img.all
    groups = partition(img.mean_spec, keep, K, spline=spline)
    F = img.F if spatial else img.F[:, :1]                               # CFSP-style: spectral statistic only
    if fisher_topsis:
        F = np.column_stack([F, fisher_scores(Xtr, ytr)])
    score = np.zeros(len(img.all)); score[keep] = topsis(F[keep])
    cand = candidates(groups, score)
    if return_candidates:
        return cand
    pool = cosine_prune(cand, img.Zc) if prune else cand
    if fisher_pick:
        sel = fisher_select(pool, Xtr, ytr, min(K, len(pool)))
    else:
        sel = mev_select(pool, img.G, min(K, len(pool)), recursive=recursive)
    sel = fill(sel, cand, K)
    return np.array(sorted(sel))


def cfsp_style(img, K):
    """our approximation of the CFSP baseline: spline curvature partition, spectral statistic (variance)
    only, no gate, no pruning, orthogonal-projection (MEV) selection."""
    return mfe_cfsp(img, K, use_gate=False, spatial=False, prune=False, spline=True)


def mfe_from_raw(cube164, K, use_gate=True, spatial=True, prune=True, recursive=True, spline=False):
    """the whole MFE-CFSP pipeline recomputed from the raw cube, so its run time is honest
    (the gate saves feature work).  Used for the ablation (paper Table II) and for timing."""
    L = cube164.shape[2]; X = cube164.reshape(-1, L)
    if use_gate:
        keep = gate(np.array([block_max_entropy(cube164[:, :, i]) for i in range(L)]))
    else:
        keep = np.arange(L)
    Zc = X - X.mean(0); Zc = Zc / np.linalg.norm(Zc, axis=0)
    groups = partition(X.mean(0), keep, K, spline=spline)
    F = band_features(cube164[:, :, keep]) if spatial else X[:, keep].var(0)[:, None]
    score = np.zeros(L); score[keep] = topsis(F)
    cand = candidates(groups, score)
    pool = cosine_prune(cand, Zc) if prune else cand
    G = np.zeros((L, L)); G[np.ix_(pool, pool)] = Zc[:, pool].T @ Zc[:, pool]
    sel = mev_select(pool, G, min(K, len(pool)), recursive=recursive)
    return np.array(sorted(fill(sel, cand, K)))
